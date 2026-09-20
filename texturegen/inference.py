from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .train import enable_circular_padding


def tile_preview(image: Image.Image, repetitions: int = 2) -> Image.Image:
    """Create a repeated image that makes boundary artifacts easy to see."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    preview = Image.new(image.mode, (image.width * repetitions, image.height * repetitions))
    for row in range(repetitions):
        for column in range(repetitions):
            preview.paste(image, (column * image.width, row * image.height))
    return preview


def generate_texture(
    prompt: str,
    lora_dir: str | Path,
    output_path: str | Path,
    *,
    model_id: str | None = None,
    negative_prompt: str = "frame, border, text, logo, perspective, horizon, object",
    seed: int = 42,
    steps: int = 35,
    guidance_scale: float = 6.5,
    width: int = 1024,
    height: int = 1024,
    lora_scale: float = 1.0,
    circular_padding: bool = True,
    preview_repetitions: int = 2,
) -> tuple[Path, Path]:
    """Generate one tile and an adjacent repeated preview using a trained LoRA."""
    import torch
    from diffusers import AutoPipelineForText2Image

    lora_path = Path(lora_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if model_id is None:
        config_path = lora_path / "training_config.json"
        if config_path.is_file():
            model_id = json.loads(config_path.read_text(encoding="utf-8"))["model_id"]
        else:
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for practical SDXL inference")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipeline = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
    pipeline.load_lora_weights(str(lora_path))
    if circular_padding:
        enable_circular_padding(pipeline.unet)
        enable_circular_padding(pipeline.vae)
    pipeline.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
        cross_attention_kwargs={"scale": lora_scale},
    ).images[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    preview = output.with_name(f"{output.stem}-preview{output.suffix or '.png'}")
    tile_preview(image, preview_repetitions).save(preview)
    return output, preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a seamless texture with a trained LoRA")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--negative-prompt", default="frame, border, text, logo, perspective, horizon, object")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=6.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()
    tile, preview = generate_texture(
        args.prompt,
        args.lora_dir,
        args.output,
        model_id=args.model_id,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        lora_scale=args.lora_scale,
    )
    print(f"Tile: {tile}\nPreview: {preview}")


if __name__ == "__main__":
    main()

