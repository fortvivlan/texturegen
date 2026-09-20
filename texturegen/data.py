from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image, ImageOps


DEFAULT_RAW_SOURCE_DIR = ".dataset"
DEFAULT_SOURCE_DIR = ".dataset/normalized"
IMAGE_EXTENSIONS = {".bmp", ".dds", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    root: Path
    train_count: int
    validation_count: int
    rejected_count: int


@dataclass(frozen=True, slots=True)
class NormalizedDataset:
    root: Path
    image_count: int


def _words(value: str) -> str:
    value = re.sub(r"[_\-.]+", " ", value)
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def caption_for_path(
    image_path: Path,
    source_root: Path,
    *,
    prefix: str = "a seamless texture",
    trigger_word: str | None = "sks_texture",
) -> str:
    """Use a sidecar caption first, then derive one from folders and filename."""
    sidecar = image_path.with_suffix(".txt")
    if sidecar.is_file():
        description = sidecar.read_text(encoding="utf-8").strip()
    else:
        relative = image_path.relative_to(source_root)
        parts = [_words(part) for part in relative.parent.parts if part not in {".", "images"}]
        filename = _words(image_path.stem)
        description = ", ".join(part for part in [*parts, filename] if part)

    caption_parts = [trigger_word, prefix, description]
    return ", ".join(part.strip(" ,") for part in caption_parts if part and part.strip(" ,"))


def seam_error(image: Image.Image) -> float:
    """Return normalized mean error between opposite edges (0 is a perfect tile)."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    horizontal = np.abs(array[:, 0, :] - array[:, -1, :]).mean()
    vertical = np.abs(array[0, :, :] - array[-1, :, :]).mean()
    return float((horizontal + vertical) / 2.0)


def _is_validation(key: str, fraction: float) -> bool:
    if fraction <= 0:
        return False
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def _safe_name(index: int, original_name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(original_name).stem).strip("-") or "texture"
    return f"{index:07d}-{stem[:80]}.png"


def _write_records(root: Path, split: str, records: list[dict[str, Any]]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_dir / "metadata.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_local_images(
    source_dir: str | Path = DEFAULT_RAW_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_SOURCE_DIR,
) -> NormalizedDataset:
    """Convert local source images to RGB PNG without changing their dimensions."""
    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {source}")
    if output == source:
        raise ValueError("output_dir must be different from source_dir")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace nonempty output folder: {output}")

    paths = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.is_relative_to(output)
    )
    if not paths:
        raise ValueError(f"No supported images found below {source}")

    destinations: dict[Path, Path] = {}
    for path in paths:
        destination = output / path.relative_to(source).with_suffix(".png")
        previous = destinations.get(destination)
        if previous is not None:
            raise ValueError(
                f"Images would map to the same PNG: {previous.relative_to(source)} and "
                f"{path.relative_to(source)}"
            )
        destinations[destination] = path

    for destination, path in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            normalized.save(destination, format="PNG", optimize=True)

    return NormalizedDataset(output, len(destinations))


def _prepare_items(
    items: Iterable[tuple[Image.Image, str, str]],
    output_dir: str | Path,
    *,
    validation_fraction: float,
    min_size: int,
    max_seam_error: float | None,
) -> PreparedDataset:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    rejected = 0

    for index, (raw_image, caption, source_name) in enumerate(items):
        image = ImageOps.exif_transpose(raw_image).convert("RGB")
        error = seam_error(image)
        if min(image.size) < min_size or (max_seam_error is not None and error > max_seam_error):
            rejected += 1
            continue

        split = "validation" if _is_validation(source_name, validation_fraction) else "train"
        filename = _safe_name(index, source_name)
        image_dir = root / split / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image.save(image_dir / filename, format="PNG", optimize=True)
        records[split].append(
            {
                "file_name": f"images/{filename}",
                "text": caption,
                "source": source_name,
                "seam_error": round(error, 6),
                "width": image.width,
                "height": image.height,
            }
        )

    # Very small datasets can hash entirely into validation. Always retain one train item.
    if not records["train"] and records["validation"]:
        record = records["validation"].pop(0)
        source_path = root / "validation" / record["file_name"]
        target_path = root / "train" / record["file_name"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(target_path)
        records["train"].append(record)
    if not records["train"]:
        raise ValueError("No training images were accepted; check the source and filters")
    for split, split_records in records.items():
        _write_records(root, split, split_records)
    summary = {
        "train_count": len(records["train"]),
        "validation_count": len(records["validation"]),
        "rejected_count": rejected,
        "mean_train_seam_error": (
            round(float(np.mean([item["seam_error"] for item in records["train"]])), 6)
            if records["train"]
            else None
        ),
    }
    (root / "dataset_info.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return PreparedDataset(root, len(records["train"]), len(records["validation"]), rejected)


def prepare_local_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    prefix: str = "a seamless texture",
    trigger_word: str | None = "sks_texture",
    validation_fraction: float = 0.1,
    min_size: int = 256,
    max_seam_error: float | None = None,
) -> PreparedDataset:
    """Prepare recursively discovered local images and deterministic captions/splits."""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {source}")
    paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported images found below {source}")

    def items() -> Iterator[tuple[Image.Image, str, str]]:
        for path in paths:
            try:
                with Image.open(path) as image:
                    yield image.copy(), caption_for_path(
                        path, source, prefix=prefix, trigger_word=trigger_word
                    ), path.relative_to(source).as_posix()
            except (OSError, ValueError) as exc:
                print(f"Skipping unreadable image {path}: {exc}")

    return _prepare_items(
        items(), output_dir, validation_fraction=validation_fraction,
        min_size=min_size, max_seam_error=max_seam_error
    )


def prepare_huggingface_dataset(
    dataset_id: str,
    output_dir: str | Path,
    *,
    split: str = "train",
    image_column: str = "image",
    caption_column: str | None = "text",
    prefix: str = "a seamless texture",
    trigger_word: str | None = "sks_texture",
    validation_fraction: float = 0.1,
    min_size: int = 256,
    max_seam_error: float | None = None,
    max_images: int | None = None,
) -> PreparedDataset:
    """Download and normalize an image dataset from the Hugging Face Hub."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    if max_images is not None:
        dataset = dataset.select(range(min(max_images, len(dataset))))

    def items() -> Iterator[tuple[Image.Image, str, str]]:
        for index, row in enumerate(dataset):
            image = row.get(image_column)
            if image is None:
                continue
            description = str(row.get(caption_column, "")).strip() if caption_column else ""
            caption = ", ".join(
                part.strip(" ,") for part in (trigger_word, prefix, description) if part and part.strip(" ,")
            )
            yield image, caption, f"{dataset_id}:{split}:{index}"

    return _prepare_items(
        items(), output_dir, validation_fraction=validation_fraction,
        min_size=min_size, max_seam_error=max_seam_error
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare captioned seamless-texture data")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help=f"local image folder (default: {DEFAULT_SOURCE_DIR})",
    )
    source.add_argument("--dataset-id", help="Hugging Face dataset ID; overrides the local default")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--caption-column", default="text")
    parser.add_argument("--prefix", default="a seamless texture")
    parser.add_argument("--trigger-word", default="sks_texture")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-seam-error", type=float)
    parser.add_argument("--max-images", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    common = dict(
        output_dir=args.output_dir,
        prefix=args.prefix,
        trigger_word=args.trigger_word or None,
        validation_fraction=args.validation_fraction,
        min_size=args.min_size,
        max_seam_error=args.max_seam_error,
    )
    if args.dataset_id:
        result = prepare_huggingface_dataset(
            args.dataset_id,
            split=args.split,
            image_column=args.image_column,
            caption_column=args.caption_column or None,
            max_images=args.max_images,
            **common,
        )
    else:
        result = prepare_local_dataset(args.source_dir, **common)
    print(f"Prepared {result.train_count} train and {result.validation_count} validation images in {result.root}")
    if result.rejected_count:
        print(f"Rejected {result.rejected_count} images")


if __name__ == "__main__":
    main()
