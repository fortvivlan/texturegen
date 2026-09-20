from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .config import TrainingConfig


def enable_circular_padding(module: Any) -> int:
    """Use periodic padding in every padded Conv2d layer and return the count."""
    import torch

    count = 0
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d) and any(child.padding):
            child.padding_mode = "circular"
            count += 1
    return count


class TextureDataset:
    def __init__(
        self,
        dataset_dir: str,
        resolution: int,
        *,
        random_flip: bool,
        random_roll: bool,
        caption_dropout: float,
    ) -> None:
        self.resolution = resolution
        self.random_flip = random_flip
        self.random_roll = random_roll
        self.caption_dropout = caption_dropout
        self.examples: list[dict[str, str]] = []
        root = Path(dataset_dir).expanduser().resolve() / "train"
        manifest = root / "metadata.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing prepared manifest: {manifest}")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            self.examples.append({"image": str(root / record["file_name"]), "text": record["text"]})
        if not self.examples:
            raise ValueError("The prepared dataset contains no training examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        example = self.examples[index]
        with Image.open(example["image"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        original_size = (image.height, image.width)

        # A random periodic translation exposes every possible tile boundary.
        if self.random_roll:
            array = np.asarray(image)
            array = np.roll(array, random.randrange(image.height), axis=0)
            array = np.roll(array, random.randrange(image.width), axis=1)
            image = Image.fromarray(array)
        if self.random_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)
        if self.random_flip and random.random() < 0.5:
            image = ImageOps.flip(image)

        width, height = image.size
        scale = self.resolution / min(width, height)
        resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
        left = random.randint(0, max(0, resized.width - self.resolution))
        top = random.randint(0, max(0, resized.height - self.resolution))
        image = resized.crop((left, top, left + self.resolution, top + self.resolution))
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        pixels = torch.from_numpy(array).permute(2, 0, 1)
        caption = "" if random.random() < self.caption_dropout else example["text"]
        return {
            "pixel_values": pixels,
            "caption": caption,
            "original_size": original_size,
            "crop_top_left": (top, left),
        }


def _collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in examples]),
        "captions": [item["caption"] for item in examples],
        "original_sizes": [item["original_size"] for item in examples],
        "crop_top_lefts": [item["crop_top_left"] for item in examples],
    }


def _text_encoder_class(model_id: str, subfolder: str) -> type:
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, PretrainedConfig

    architecture = PretrainedConfig.from_pretrained(model_id, subfolder=subfolder).architectures[0]
    classes = {
        "CLIPTextModel": CLIPTextModel,
        "CLIPTextModelWithProjection": CLIPTextModelWithProjection,
    }
    if architecture not in classes:
        raise ValueError(f"Unsupported text encoder architecture: {architecture}")
    return classes[architecture]


def _encode_prompts(captions: list[str], tokenizers: list[Any], text_encoders: list[Any]) -> tuple[Any, Any]:
    import torch

    embeddings = []
    pooled = None
    with torch.no_grad():
        for tokenizer, encoder in zip(tokenizers, text_encoders, strict=True):
            tokens = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(encoder.device)
            output = encoder(tokens, output_hidden_states=True, return_dict=False)
            pooled = output[0]
            embeddings.append(output[-1][-2])
    return torch.cat(embeddings, dim=-1), pooled


def _save_lora(unet: Any, output_dir: Path) -> None:
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=state,
        safe_serialization=True,
    )


def _save_checkpoint(
    path: Path,
    *,
    unet: Any,
    optimizer: Any,
    scheduler: Any,
    global_step: int,
    epoch: int,
) -> None:
    import torch
    from peft.utils import get_peft_model_state_dict

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "lora": {key: value.detach().cpu() for key, value in get_peft_model_state_dict(unet).items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def train_lora(config: TrainingConfig) -> Path:
    """Train a UNet LoRA and return the directory containing Diffusers weights."""
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build and check the NVIDIA driver."
        )
    if config.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; retry with --mixed-precision fp16")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=None if config.report_to == "none" else config.report_to,
        project_dir=str(output_dir / "logs"),
    )
    set_seed(config.seed)
    if accelerator.is_main_process:
        config.save(output_dir / "training_config.json")
    if accelerator.is_main_process and config.report_to != "none":
        accelerator.init_trackers(
            "texturegen", config={key: str(value) for key, value in asdict(config).items()}
        )

    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")
    tokenizer_one = AutoTokenizer.from_pretrained(config.model_id, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(config.model_id, subfolder="tokenizer_2", use_fast=False)
    encoder_one_cls = _text_encoder_class(config.model_id, "text_encoder")
    encoder_two_cls = _text_encoder_class(config.model_id, "text_encoder_2")
    text_encoder_one = encoder_one_cls.from_pretrained(config.model_id, subfolder="text_encoder")
    text_encoder_two = encoder_two_cls.from_pretrained(config.model_id, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet")

    for model in (vae, text_encoder_one, text_encoder_two, unet):
        model.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=config.rank,
            lora_alpha=config.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    if config.circular_padding:
        enable_circular_padding(unet)
        enable_circular_padding(vae)
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if config.enable_xformers:
        unet.enable_xformers_memory_efficient_attention()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # Keep frozen base weights in reduced precision; only LoRA parameters stay fp32.
    # This is the main difference between a comfortable 24 GB run and an OOM-prone one.
    unet.to(accelerator.device, dtype=weight_dtype)
    vae_dtype = torch.float32 if getattr(vae.config, "force_upcast", False) else weight_dtype
    vae.to(accelerator.device, dtype=vae_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    for parameter in unet.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)

    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("Install bitsandbytes to use use_8bit_adam=True") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(trainable, lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)

    dataset = TextureDataset(
        config.dataset_dir,
        config.resolution,
        random_flip=config.random_flip,
        random_roll=config.random_roll,
        caption_dropout=config.caption_dropout,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    updates_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    epochs = math.ceil(config.max_train_steps / updates_per_epoch)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(100, config.max_train_steps // 10),
        num_training_steps=config.max_train_steps,
    )

    global_step = 0
    first_epoch = 0
    if config.resume_from_checkpoint:
        checkpoint = torch.load(config.resume_from_checkpoint, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(unet, checkpoint["lora"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint["global_step"])
        first_epoch = int(checkpoint["epoch"])

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    progress = range(global_step, config.max_train_steps)
    if accelerator.is_local_main_process:
        from tqdm.auto import tqdm

        progress = tqdm(progress, initial=global_step, total=config.max_train_steps, desc="LoRA steps")

    for epoch in range(first_epoch, epochs):
        unet.train()
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixels = batch["pixel_values"].to(accelerator.device, dtype=vae_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    prompt_embeds, pooled_embeds = _encode_prompts(
                        batch["captions"],
                        [tokenizer_one, tokenizer_two],
                        [text_encoder_one, text_encoder_two],
                    )
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device, dtype=torch.long
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                time_ids = torch.tensor(
                    [
                        [*original, *crop, config.resolution, config.resolution]
                        for original, crop in zip(
                            batch["original_sizes"], batch["crop_top_lefts"], strict=True
                        )
                    ],
                    device=latents.device,
                    dtype=prompt_embeds.dtype,
                )
                prediction = unet(
                    noisy_latents.to(dtype=weight_dtype),
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
                    return_dict=False,
                )[0]
                target = noise
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if hasattr(progress, "update"):
                    progress.update(1)
                    progress.set_postfix(loss=f"{loss.detach().item():.4f}")
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)
                if (
                    accelerator.is_main_process
                    and config.checkpointing_steps > 0
                    and global_step % config.checkpointing_steps == 0
                ):
                    raw_unet = accelerator.unwrap_model(unet)
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    _save_lora(raw_unet, checkpoint_dir)
                    _save_checkpoint(
                        checkpoint_dir / "training_state.pt",
                        unet=raw_unet,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        global_step=global_step,
                        epoch=epoch,
                    )
            if global_step >= config.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_lora(accelerator.unwrap_model(unet), output_dir)
        (output_dir / "README.md").write_text(
            f"# Texture LoRA\n\nBase model: `{config.model_id}`\n\n"
            f"Trigger word: `sks_texture` (if retained in the prepared captions).\n",
            encoding="utf-8",
        )
    accelerator.end_training()
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an SDXL LoRA on prepared texture data")
    parser.add_argument("--config", help="JSON TrainingConfig file")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir", default="outputs/texture-lora")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--use-8bit-adam", action="store_true")
    parser.add_argument("--enable-xformers", action="store_true")
    parser.add_argument("--no-circular-padding", action="store_true")
    parser.add_argument("--report-to", default="tensorboard")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        if not args.dataset_dir:
            raise SystemExit("Provide --config or --dataset-dir")
        config = TrainingConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            model_id=args.model_id,
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_train_steps=args.max_train_steps,
            learning_rate=args.learning_rate,
            rank=args.rank,
            mixed_precision=args.mixed_precision,
            checkpointing_steps=args.checkpointing_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
            use_8bit_adam=args.use_8bit_adam,
            enable_xformers=args.enable_xformers,
            circular_padding=not args.no_circular_padding,
            report_to=args.report_to,
        )
    train_lora(config)


if __name__ == "__main__":
    main()
