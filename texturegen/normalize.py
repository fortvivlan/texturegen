from __future__ import annotations

import argparse

from texturegen.data import DEFAULT_RAW_SOURCE_DIR, DEFAULT_SOURCE_DIR, normalize_local_images


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert local dataset images to RGB PNG")
    parser.add_argument("--source-dir", default=DEFAULT_RAW_SOURCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_SOURCE_DIR)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = normalize_local_images(args.source_dir, args.output_dir)
    print(f"Converted {result.image_count} images to {result.root}")


if __name__ == "__main__":
    main()
