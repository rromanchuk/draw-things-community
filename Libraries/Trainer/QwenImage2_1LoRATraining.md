# Qwen Image 2.1 LoRA training in DrawThingsCLI: feasibility and plan

## Context

The user wants `draw-things-cli train` to train LoRAs for Qwen Image 2.1. mflux added this for MLX in mflux-community/mflux PR 973. The question is how much of that work this repo already covers and what remains.

**Verdict: feasible, and most of the pieces exist.** Inference, the VAE, the text encoder, LoRA builders (`LoRAQwenImage2_1Fixed`, `LoRAQwenImage2_1`), LoRA import and the checkpoint lists already handle `.qwenImage2_1`. Three things block training today:

1. `LoRATrainer` has no 2.1 path. `init` calls `fatalError()` for the default VAE and text encoder, and neither dataset preparation nor `train` dispatches 2.1.
2. The CLI tokenizer switch throws `Unsupported LoRA trainer model version` (`Apps/DrawThingsCLI/DrawThingsCLI.swift:4055`), and the CLI has no 2.1 trainable-key list.
3. **The model is split.** Inference runs text tokens through all 32 blocks once in `QwenImage2_1Fixed`, which outputs cached K/V. The denoiser `QwenImage2_1` then attends to that cache (`UNetFixedEncoder.swift:2087-2287`). Both models use the same block weights (`transformer_blocks.N.attn.to_q` …), and inference applies a LoRA to both (`UNetFixedEncoder.swift:2180-2256`). The existing trainers (Krea2, Qwen 1.0) run their fixed encoder frozen and with no LoRA. Copying that pattern here would train a LoRA on image tokens only, and inference would then apply it to text tokens as well, which training never saw. mflux avoids this by training without the K/V cache (`forward_training`), so the gradient reaches the text path too.

## Facts gathered

| Item | Value | Source |
|---|---|---|
| DiT | 32 blocks, 4096 channels, 128-dim heads, single-stream, block-causal | `QwenImage2_1.swift`, `UNetFixedEncoder.swift:2185` |
| Block weights (s4nnc names) | `x_q x_k x_v x_o ffn_up_proj ffn_gate_proj ffn_down_proj` (= `to_q/k/v`, `to_out.0`, `img_mlp.proj/gate_layer/out`) | `QwenImage2_1.swift:202-213` |
| Latent | VAE /16, 64 channels taken directly as tokens (no 2×2 patchify); moments are 128 channels | `FirstStage.swift:1514, 1744-1782`; `LocalImageGenerator.swift:3614-3616` |
| VAE input | RGB, or alpha-first RGBA → RGBA; `qwen_image_2.1_vae_f16.ckpt` | `FirstStage.swift:1749-1758`, `ModelImporter.swift:3018` |
| Text encoder | Qwen3-VL 8B: width 4096, 36 layers, MLP 12288, hidden state 35, no final norm; `qwen_3_vl_8b_instruct_i8x.ckpt` | `TextEncoder.swift:2237-2241`, `ModelImporter.swift:3015` |
| Prompt | `<\|im_start\|>system\nComprehend and analyze the provided prompt.<\|im_end\|>\n<\|im_start\|>user\n{text}<\|im_end\|>\n<\|im_start\|>assistant\n`; the system message plus the newline after it is dropped from the hidden states | `LocalImageGenerator.swift:1677-1691`, `TextEncoder.swift:2137` |
| Timestep | `time` input has one row per sampled timestep plus a final `t=0` row for the text prefix; `timeEmbedding(t*1000, 256)` | `UNetFixedEncoder.swift:2164-2170` |
| Noise / shift | RF, conditionScale 1000, `shiftTerminal 0.02`; `shiftFor` has a 2.1 formula | `ModelZoo.swift:2941`, `ModelZoo.swift:3871` |
| Model files | 2.1 is not in the built-in zoo; users import it with `draw-things-cli import`, which writes a custom spec | `ModelImporter.swift:3013-3028` |

From mflux PR 973:
- Targets `to_q/k/v`, `to_out.0`, `img_mlp.gate_layer/proj/out` on all 32 blocks, rank 16.
- Samples `sigma = sigmoid(N(1, 1))` with no shift. The author found mean 1 learned style far better than mean 0.
- Velocity target, MSE loss, AdamW 1e-4, 5% empty-caption dropout.
- Per-block gradient checkpointing; about 21 GB peak at 512 px.
- Rejects Turbo for training.

## Recommended approach: one training model that shares weights between the text and image streams

Add a training-only builder that does the work of `LoRAQwenImage2_1Fixed` and `LoRAQwenImage2_1` in a single graph. Each `LoRADense` is created once per block and applied to both streams, so gradients reach the text path and the trained LoRA matches what inference does.

### 1. Builder: `Libraries/SwiftDiffusion/Sources/Models/QwenImage2_1.swift`
- Add `public func LoRAQwenImage2_1Unsplit<T>(_:batchSize:height:width:textLength:channels:layers:usesFlashAttention:LoRAConfiguration:)`. The name follows the existing `Tests/QwenImage2_1Unsplit.swift` reference.
  - Inputs: `x [1,h,w,64]`, `context [1,L,4096]`, `time [2,256]` (row 0 = sigma·1000, row 1 = 0), `textRotary`, `imageRotary`.
  - Output: velocity `[1,h,w,64]`.
- Keep the arithmetic of the fixed path (text norm → `context_embedder_0/1`, `t_embedder_0/1`, `x_ada_ln_0..3`, `ada_ln_0`), using row 0 for the image stream and row 1 for the text stream.
- Add a private `LoRATransformerBlockUnsplit`. Per block it:
  - creates `x_q/x_k/x_v/x_o/ffn_*` once;
  - text stream: causal SDPA over text;
  - image stream: SDPA over `[image K/V, text K/V]`, in the same order as `LoRATransformerBlock`;
  - in the last block, the text stream stops after K/V, as `outputKVOnly` does.
- Reuse `QwenImage2_1RotaryEmbedding` and `QwenImage2_1Segment` for positions, building the segments the way `UNetFixedEncoder.swift:2136-2161` does.
- Honour `LoRANetworkConfiguration` gradient-checkpointing flags so `memorySaver` keeps working.
- The weight mapper must give the same parameter names as the split builders, so `store.read("dit", …)` and the existing LoRA importer and loader work unchanged.

### 2. Trainer: `Libraries/Trainer/Sources/LoRATrainer.swift`
- `init`:
  - Replace the `.qwenImage2_1` `fatalError()`s in the VAE and text-encoder defaults with `qwen_image_2.1_vae_f16.ckpt` and `qwen_3_vl_8b_instruct_i8x.ckpt`. The `ModelZoo` spec values still win.
  - Add `.qwenImage2_1` to the `paddedTextEncodingLength = maxTextLength` group.
- `prepareDatasetForQwenImage2_1`, modelled on `prepareDatasetForKrea2` (line 778):
  - **Images:** `FirstStage.encode` and write the 128-channel moments. Use `scale × 64` pixel sizes, which give latent sizes divisible by 4.
  - **Captions:** run `Qwen3` with the 2.1 inference config (width 4096, MLP 12288, output 35, no final norm) on the 2.1 template. Cut the hidden states at `drop` (system message plus newline, found by the first `151_645` + 2), as `encodeQwenImage2_1` does, and keep the user-text and assistant suffix. Write `cond_<path>` with shape `[1, L, 4096]`.
  - **Dataset scope:** text-only. Leave out reference images and vision features (no edit training in v1).
- `trainQwenImage2_1`, modelled on `trainKrea2` (line 3933):
  - Use `LoRAQwenImage2_1Unsplit` with `latents = scale × 4`.
  - Sample the latent from the moments with `firstStage.sampleFromDistribution` (it takes 64 of the 128 channels), and check `shape[3] == 128`.
  - Use the existing uniform-in-`denoisingTimesteps` sampling plus the (resolution-dependent) shift block, so all trainers keep the same config semantics.
  - Build `time` rows `[t·1000, 0]` with `timeEmbedding`.
  - Keep `zt = (1−t)x0 + t·ε`, target `ε − x0`, MSE, AdamW, GradScaler, warmup, cosine restarts and power-EMA unchanged.
  - Batch per token length, as in Krea2. No separate fixed-encode step is needed, because the unsplit model computes its own modulation.
- Dispatch:
  - Add `version == .qwenImage2_1` branches in `prepareDataset` (~2167) and `train` (~6021).
  - Add 2.1 to the "custom path" version lists at 2010/2077/6420/6507/6713/6912 where they gate the generic fallback.
- Reject Turbo: add a `precondition` or early return when the model filename contains `turbo`, matching the `shiftTerminal` check in `ModelImporter.swift:3026`.

### 3. Checkpoint save: `Libraries/Trainer/Sources/LoRATrainerCheckpoint.swift`
- Move `.qwenImage2_1` out of the `fatalError()` groups at 307 and 524.
- Use `modelName = "dit"` and an identity `UNetMapping` over `0..<32`, matching the separate-LoRA mapping in `UNetFixedEncoder.swift:2219`.
- The saved `.ckpt` then loads through the existing LoRA path into both the fixed encoder and the denoiser.

### 4. CLI: `Apps/DrawThingsCLI/DrawThingsCLI.swift`
- Tokenizer switch (4031): add `case .qwenImage2_1: tokenizerStack = [tokenizers.tokenizerQwen3]`.
- Trainable-key switch (4081): add a `qwenImage2_1TrainableKeys(trainableLayers:layerIndices:)` modelled on `qwenImageTrainableKeys` (5814), over 32 layers:
  - `.qkv` → `x_q x_k x_v`
  - `.out` → `x_o`
  - `.feedForward` → `ffn_up_proj ffn_gate_proj ffn_down_proj`
  - `.projectOut` → `linear-0-`
  - `.latentsEmbedder` → `x_embedder-0-`
  - `.contextEmbedder` → `context_embedder_0-0-`, `context_embedder_1-0-`
  - Defaults: the mflux and ai-toolkit set, which is qkv, out and feedForward on all blocks.
- Pass `preserveAlpha: version == .qwenImage2_1` into the training image loader, as generate already does (4499).
- Update the help text and the `.agents/skills/train-lora/SKILL.md` model list.

### 5. Feature gate: `Libraries/FeaturesMatrix/Sources/FeaturesMatrix.swift`
- Turn on `supportsLoRATraining(.qwenImage2_1)` only after it has been verified on the user's Mac. The CLI does not read this gate; the app does.

### Deferred (not in v1)
- **ComfyUI/ai-toolkit LoRA export:** `LoRAExporter.swift:129` calls `fatalError` for 2.1. It would need the fused `img_mlp.gate_up` layout (rank ×2, block-diagonal B) described in the mflux PR. Import already supports 2.1, including published adapters (`Libraries/ModelOp/Tests/QwenImage2_1LoRATests.swift`), so LoRAs trained with mflux or ai-toolkit can be used now.
- **Logit-normal `sigmoid(N(μ, 1))` sampling:** worth a later config option, given the mflux finding. v1 keeps the shared uniform-plus-shift scheme. With the 2.1 shift of about 1.6–2.5, sampling is also biased towards high noise, with a median t of about 0.65–0.7 against 0.73 for mflux.
- **Edit/reference-image training and text-encoder co-training.**

### Fallback if the unsplit builder stalls
Train only the image stream: run `QwenImage2_1Fixed` frozen, with no gradient, and train `LoRAQwenImage2_1`. This needs much less new code. However, inference would still apply the LoRA to text tokens, which training never saw. Use it only for a quick experiment.

## Risks
- **Quantized weights:** Krea2 i8x gave zero or non-finite LoRA updates on some layers (`LoRATrainer.swift:3964`, `DrawThingsCLI.swift:5851`). Test with a q8p or f16 2.1 import first.
- **Memory:** the text stream adds about L tokens of activations per block. At 512 px (1024 image tokens) with L of about 100–300, it is fine on 256 GB. `memorySaver: minimal` enables per-block checkpointing.
- **Build environment:** this Linux container cannot build or run Metal targets, so all verification below runs on the user's Mac.

## Verification (on the Mac)
1. **Parity test:** add `QwenImage2_1Unsplit` coverage to `Libraries/SwiftDiffusion/Tests/QwenImage2_1Tests.swift` or a new `swift_test`.
   - Random weights and a random non-zero LoRA.
   - Compare `LoRAQwenImage2_1Unsplit` output with `LoRAQwenImage2_1Fixed` → `LoRAQwenImage2_1` at the same sigma.
   - Pass condition: relative RMS below about 1e-3.
2. Builds:
   - `bazel build //Libraries/SwiftDiffusion:SwiftDiffusion`
   - `bazel build //Libraries/Trainer:Trainer`
   - `bazel build //Apps:DrawThingsCLI`
3. **Dry run:** `draw-things-cli train --model <imported qwen 2.1 base> --dataset <dir> --dry-run`
4. **Short training run:** 512 px, about 300 steps, rank 16, on a small style dataset (like the mflux test).
   - Loss stays finite and the LoRA up-weights are non-zero.
   - `generate` with the LoRA at 40 steps shows the style on prompts outside the dataset, and the same seed without the LoRA does not.
5. `bazel build //Apps/DrawThings:DrawThings` after the FeaturesMatrix change.

## Handoff: state of the first draft

Written in a Linux container with no Swift toolchain or Bazel. **Nothing here has been compiled or run.** The next agent starts on a Mac.

### Done in this draft
- `Libraries/SwiftDiffusion/Sources/Models/QwenImage2_1.swift`:
  - `QwenImage2_1UnsplitRotaryEmbedding(textLength:height:width:)` (public wrapper over the internal rotary helper).
  - `LoRAQwenImage2_1Unsplit` and the private `LoRATransformerBlockUnsplit`. Text and image streams share each block's layer instances; the text stream of the last block stops after K/V.
  - Only the flash-attention path (`ScaledDotProductAttention`, Float16) is implemented; there is no `usesFlashAttention` parameter. Batch size is 1.
- `Libraries/Trainer/Sources/LoRATrainer.swift`:
  - Default VAE and text encoder for `.qwenImage2_1`; added to the `paddedTextEncodingLength = maxTextLength` group.
  - `prepareDatasetForQwenImage2_1`: VAE moments (128 channels) plus Qwen3-VL 8B hidden state 35, with the system message dropped (`first <|im_end|> + 2`).
  - `trainQwenImage2_1`: per-sample loop (no 32-sample fixed-encode batching), uniform timestep + shift, `time` rows `[t·1000, 0]`, rejects Turbo by filename.
  - Dispatch in `prepareDataset` and `train`.
- `Libraries/Trainer/Sources/LoRATrainerCheckpoint.swift`: `.qwenImage2_1` → `dit` with identity mapping over 32 layers, in both `makeLoRA` paths.
- `Apps/DrawThingsCLI/DrawThingsCLI.swift`: Qwen3 tokenizer, `qwenImage2_1TrainableKeys` (default qkv, out, feedForward over 32 blocks), `preserveAlpha` for training images.

### Next steps, in order
1. Build and fix compile errors:
   - `bazel build //Libraries/SwiftDiffusion:SwiftDiffusion`
   - `bazel build //Libraries/Trainer:Trainer`
   - `bazel build //Apps:DrawThingsCLI`
2. Parity test in `Libraries/SwiftDiffusion/Tests/QwenImage2_1Tests.swift` (target `QwenImage2_1Tests`, manual, requires GPU). Model it on `testCachedPrefixMatchesFullSequence`:
   - small channels (e.g. 256) and 2 layers; context width must stay 4096;
   - compile `LoRAQwenImage2_1Fixed` (timesteps 1, segments `[(L, false)]`, `.scale1`) and `LoRAQwenImage2_1`, write both to a temporary store under `"dit"`, then read that store back into the fixed model and into `LoRAQwenImage2_1Unsplit` so all three share weights;
   - set the `lora_up` weights non-zero, otherwise the LoRA path is not tested;
   - compare outputs at the same timestep; expect relative RMS below about 1e-3.
3. Check that the unsplit model's parameter names match the split builders when read from a real 2.1 checkpoint (`store.read("dit", …)` should report no missing tensors). s4nnc numbers layers per name in graph order; the fixed model's last block has no `x_q`/`x_o`/FFN, which the unsplit model mirrors.
4. `draw-things-cli train --model <imported 2.1 base> --dataset <dir> --dry-run`, then a 512 px, ~300-step run (rank 16). Check loss is finite, LoRA up-weights are non-zero, and `generate` with the LoRA shows the style.
5. Only then enable `FeaturesMatrix.supportsLoRATraining(.qwenImage2_1)` and build `//Apps/DrawThings:DrawThings`.

### Known gaps and things to check
- `memorySaver .balanced` checkpoints whole blocks, like `.minimal`, because the unsplit block has no separate feed-forward sub-model.
- Training ignores the inference `shiftTerminal 0.02`.
- Turbo is detected by `"turbo"` in the model filename only.
- If i8x weights give zero or non-finite LoRA updates (as with Krea2), test with a q8p or f16 import and narrow the trainable keys.
- `.agents/skills/train-lora/SKILL.md` and the CLI help text are not updated yet.
