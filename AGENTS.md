# AGENTS.md

## Project purpose

This repository builds one captioned dataset and trains a Stable Diffusion XL LoRA for
seamless, top-down material textures. The primary target is local Windows execution in
PowerShell on one NVIDIA GeForce RTX 4090 with 24 GB VRAM. There is no notebook UI.

## Architecture

- `texturegen/data.py`: local or Hugging Face ingestion, caption fallback, seam metric,
  deterministic split, and `metadata.jsonl` output.
- `texturegen/config.py`: serializable `TrainingConfig`; add persisted options here.
- `texturegen/train.py`: single-dataset SDXL LoRA training, lightweight checkpoints,
  wrap-around augmentation, and circular convolution padding.
- `texturegen/inference.py`: load the LoRA with matching circular padding and create a
  repeated seam-check preview.
- `scripts/`: executable wrappers only; business logic belongs in the package.

## Implementation notes

- The workflow accepts exactly one prepared dataset root per run. Category directories
  may exist below the raw source folder, but they become one prepared dataset.
- Preserve lightweight imports. Import `torch`, `diffusers`, and `datasets` inside the
  functions that need them.
- Keep RTX 4090-safe defaults: batch 1, gradient accumulation 4, BF16, rank 16, and
  gradient checkpointing. Frozen base weights use reduced precision; trainable LoRA
  weights remain FP32.
- Rely on PyTorch scaled dot-product attention by default. Keep xFormers optional because
  it complicates Windows dependency compatibility.
- Each prepared root contains `train/metadata.jsonl`, `validation/metadata.jsonl`, and
  matching `images/` directories.
- Sidecar `.txt` captions take precedence. Otherwise captions come from relative folder
  names and filenames. Keep the trigger word configurable.
- Do not silently download or scrape third-party assets. Users choose sources and verify
  licenses. Hub ingestion occurs only when explicitly requested.
- Seam error is a diagnostic/filter, not proof that an image tiles perceptually.
- Random periodic translation and circular padding are intentional. If circular padding
  changes in training, update inference to match and document compatibility.
- Final weights remain in standard Diffusers LoRA format
  (`pytorch_lora_weights.safetensors`). Training state may be project-specific.
- Never commit datasets, checkpoints, model weights, access tokens, or cache contents.
- Prefer deterministic preparation and seeded training. Do not destructively replace
  existing output directories.

## Compatibility and validation

- Target 64-bit Python 3.11 or 3.12 on Windows 10/11 and a CUDA-enabled PyTorch build.
- SDXL is the supported architecture. Do not claim SD 1.5, SD 2, SD3, or FLUX support
  without adding and testing an explicit implementation path.
- Keep the PowerShell examples in `README.md` in sync with all CLI changes.
- For data changes, test captions, manifests, rejected images, and seam metrics.
- For inference changes, test preview dimensions without requiring a GPU.
- Before handing off changes, run:

  ```powershell
  python -m compileall -q texturegen scripts tests
  pytest -q
  ```

- Full RTX 4090 training is an integration test. If unavailable, clearly state that and
  at least verify syntax, configuration serialization, and CPU-only tests.
