# texturegen

`texturegen` prepares one captioned image dataset and trains an SDXL LoRA for
top-down, seamless material textures. It is designed to run locally on Windows with
an NVIDIA GeForce RTX 4090.

The source can be a local folder or a standard Hugging Face image dataset. Local
images, including DDS files, do not need existing descriptions: a `.txt` sidecar is
used when present; otherwise, folder names and filenames become the caption.

## What the project does

1. Recursively discovers images, converts them to RGB PNG, measures opposite-edge
   error, and creates deterministic train/validation splits.
2. Builds `metadata.jsonl` manifests with captions such as
   `sks_texture, a seamless texture, stone, rough grey wall`.
3. Trains SDXL UNet LoRA weights with random wrap-around translations, circular
   convolution padding, BF16 mixed precision, gradient checkpointing, and restartable
   checkpoints.
4. Generates a tile and a repeated 2x2 preview for checking its boundaries.

Training on seamless examples does not mathematically guarantee seamless output. Keep
circular padding enabled during training and inference, inspect the repeated preview,
and reject or repair occasional samples with visible boundaries.

## Windows and RTX 4090 setup

Prerequisites:

- Windows 10 or 11
- A current NVIDIA Game Ready or Studio driver
- Git
- 64-bit Python 3.11 or 3.12
- At least 35 GB of free disk space for the environment, SDXL cache, data, and outputs

Open PowerShell in the directory where the repository should be stored. Replace the
example Git URL with this repository's URL:

```powershell
git clone https://github.com/YOUR_ACCOUNT/texturegen.git
Set-Location .\texturegen

py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[test]"
```

The PyTorch CUDA wheel includes the CUDA runtime it needs; a separate CUDA Toolkit is
normally unnecessary. If the `cu128` wheel is no longer the recommended stable build,
select Windows, Pip, Python, and CUDA on the official
[PyTorch installation page](https://pytorch.org/get-started/locally/) and use its command.

Verify that the environment sees the 4090:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')"
```

The output should include `True`, the RTX 4090 name, and approximately 24 GiB.

If a model or Hugging Face dataset is gated, log in once inside the virtual
environment:

```powershell
hf auth login
```

## Prepare the dataset

By default, put the uncaptioned source files in the repository's ignored `.dataset`
folder. Normalize them to RGB PNG without resizing or changing the originals:

```text
.dataset\
├── stone\
│   ├── rough_grey_01.png
│   └── rough_grey_02.dds
└── fabric\
    ├── blue_linen.jpg
    ├── blue_linen.txt       # optional caption for blue_linen.jpg
    └── woven_canvas.webp
```

```powershell
python .\scripts\normalize_dataset.py
```

This writes standardized copies below `.dataset\normalized`. It refuses to replace a
nonempty normalized folder, and it only converts images: it does not generate captions,
manifests, splits, or seam scores. Add one UTF-8 `.txt` description beside each
normalized PNG before preparing the dataset.

Although the source has category subfolders, it becomes one prepared dataset. For
`stone\rough_grey_01.png`, the inferred caption is:

```text
sks_texture, a seamless texture, stone, rough grey 01
```

Use short, factual sidecars when filenames are IDs or otherwise meaningless. Useful
details include material, color, apparent scale, surface character, and viewing
direction. The default `sks_texture` trigger is deliberately rare and should also
appear in generation prompts.

Prepare the normalized, captioned folder from PowerShell:

```powershell
python .\scripts\prepare_dataset.py `
  --output-dir ".\data\prepared-textures"
```

The default source is `.dataset\normalized`. Pass `--source-dir
"D:\TextureSource"` to use a different local folder.

The result contains one dataset with `train` and `validation` subsets:

```text
data\prepared-textures\
├── dataset_info.json
├── train\
│   ├── images\
│   └── metadata.jsonl
└── validation\
    ├── images\
    └── metadata.jsonl
```

To reject obviously non-tiling inputs, add `--max-seam-error 0.12`. This metric is
only a diagnostic: inspect representative source images as repeated 2x2 tiles before
training.

### Optional Hugging Face source

For a standard Hub image dataset with `image` and `text` columns:

```powershell
python .\scripts\prepare_dataset.py `
  --dataset-id "owner/dataset-name" `
  --image-column "image" `
  --caption-column "text" `
  --output-dir ".\data\prepared-textures" `
  --max-images 10000
```

The preparation script does not scrape websites or accept arbitrary download URLs.
Confirm the source license and that model training is allowed before using a dataset.

Ready-made sources worth evaluating:

- [VastTextures](https://huggingface.co/datasets/FlyingFrog/VastTextures) contains CC0
  seamless/PBR archives. It is ZIP-oriented rather than a standard Hub `image` column,
  so extract the desired archive and process it as a local folder.
- [Poly Haven textures](https://polyhaven.com/textures) are high-quality CC0 PBR assets,
  and Poly Haven explicitly permits AI training. Download diffuse/albedo maps and
  process their folder locally. Follow the separate API terms if automating downloads.
- Your own known-seamless images are often the best first run because their provenance
  and tiling quality are known.

Do not mix normal, roughness, displacement, or rendered preview images into a color
texture LoRA. Use only albedo/diffuse images for this dataset.

## Train on the RTX 4090

The defaults are chosen for the card's 24 GB VRAM:

- resolution: 1024
- physical batch size: 1
- gradient accumulation: 4 (effective batch size 4)
- LoRA rank: 16
- precision: BF16
- gradient checkpointing: enabled
- circular padding: enabled

Start training:

```powershell
python .\scripts\train_lora.py `
  --dataset-dir ".\data\prepared-textures" `
  --output-dir ".\outputs\texture-lora" `
  --max-train-steps 3000
```

The first run downloads SDXL to the Hugging Face cache. Training writes
`pytorch_lora_weights.safetensors`, `training_config.json`, TensorBoard logs, and a
checkpoint every 500 optimizer steps.

Monitor utilization in a second PowerShell window:

```powershell
nvidia-smi -l 2
```

To inspect TensorBoard:

```powershell
.\.venv\Scripts\Activate.ps1
tensorboard --logdir .\outputs\texture-lora\logs
```

If training runs out of VRAM, retry at 768 pixels. Do not increase gradient
accumulation to fix an out-of-memory error—it changes the effective batch but does not
reduce the memory used by one image:

```powershell
python .\scripts\train_lora.py `
  --dataset-dir ".\data\prepared-textures" `
  --output-dir ".\outputs\texture-lora-768" `
  --resolution 768 `
  --train-batch-size 1 `
  --gradient-accumulation-steps 4 `
  --max-train-steps 3000
```

PyTorch 2 uses memory-efficient scaled dot-product attention automatically. The
optional `--enable-xformers` switch is therefore not necessary for the normal 4090
setup and can make Windows dependency installation more fragile.

### Resume an interrupted run

Resume from the state file inside a checkpoint:

```powershell
python .\scripts\train_lora.py `
  --dataset-dir ".\data\prepared-textures" `
  --output-dir ".\outputs\texture-lora" `
  --resume-from-checkpoint ".\outputs\texture-lora\checkpoint-1500\training_state.pt"
```

Use the same dataset and training settings when resuming. The saved
`training_config.json` records the original values.

## Generate and inspect a texture

```powershell
python .\scripts\generate_texture.py `
  --lora-dir ".\outputs\texture-lora" `
  --prompt "sks_texture, seamless weathered sandstone, top-down, flat lighting" `
  --output ".\outputs\sample.png" `
  --seed 123
```

This writes `sample.png` and `sample-preview.png`. The preview repeats the tile in a
2x2 grid so the horizontal and vertical joins are visible.

The same operations can be called from Python:

```python
from texturegen import TrainingConfig
from texturegen.train import train_lora

config = TrainingConfig(
    dataset_dir=r"D:\Projects\texturegen\data\prepared-textures",
    output_dir=r"D:\Projects\texturegen\outputs\texture-lora",
    max_train_steps=3000,
)
train_lora(config)
```

## Development checks

```powershell
python -m compileall -q texturegen scripts tests
pytest -q
```

The implementation follows the Diffusers SDXL LoRA approach: SDXL base components are
frozen and PEFT adapters are trained on UNet attention projections. The base weights
use reduced precision to fit the 4090, while trainable LoRA parameters remain FP32.

## References

- [PyTorch local installation](https://pytorch.org/get-started/locally/)
- [Diffusers SDXL LoRA training example](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py)
- [Diffusers LoRA guide](https://huggingface.co/docs/diffusers/main/training/lora)
- [Hugging Face image-dataset layout](https://huggingface.co/docs/hub/datasets-image)
- [Poly Haven license and AI-training permission](https://polyhaven.com/license)
