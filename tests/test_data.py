import json

from PIL import Image

from texturegen.data import (
    _build_parser,
    caption_for_path,
    normalize_local_images,
    prepare_local_dataset,
    seam_error,
)


def test_caption_uses_sidecar_then_path(tmp_path):
    folder = tmp_path / "Rough Stone"
    folder.mkdir()
    image_path = folder / "Grey_Wall-01.png"
    Image.new("RGB", (16, 16), "grey").save(image_path)

    assert caption_for_path(image_path, tmp_path) == (
        "sks_texture, a seamless texture, rough stone, grey wall 01"
    )
    image_path.with_suffix(".txt").write_text("charcoal volcanic rock", encoding="utf-8")
    assert caption_for_path(image_path, tmp_path) == (
        "sks_texture, a seamless texture, charcoal volcanic rock"
    )


def test_prepare_local_dataset_and_uniform_seam(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (32, 32), (20, 40, 60)).save(source / "blue_stone.png")
    output = tmp_path / "prepared"

    result = prepare_local_dataset(
        source, output, validation_fraction=0, min_size=16, max_seam_error=0.01
    )

    assert result.train_count == 1
    assert result.validation_count == 0
    record = json.loads((output / "train" / "metadata.jsonl").read_text(encoding="utf-8"))
    assert record["text"].endswith("blue stone")
    assert record["seam_error"] == 0
    with Image.open(output / "train" / record["file_name"]) as image:
        assert seam_error(image) == 0


def test_prepare_local_dataset_accepts_dds(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (32, 32), (20, 40, 60)).save(source / "blue_stone.dds")
    output = tmp_path / "prepared"

    result = prepare_local_dataset(source, output, validation_fraction=0, min_size=16)

    assert result.train_count == 1
    record = json.loads((output / "train" / "metadata.jsonl").read_text(encoding="utf-8"))
    assert record["source"] == "blue_stone.dds"
    assert record["file_name"].endswith("blue_stone.png")


def test_prepare_cli_defaults_to_local_dataset_folder():
    args = _build_parser().parse_args(["--output-dir", "prepared"])

    assert args.source_dir == ".dataset/normalized"
    assert args.dataset_id is None


def test_normalize_local_images_converts_dds_without_captions(tmp_path):
    source = tmp_path / ".dataset"
    source.mkdir()
    original = source / "blue_stone.dds"
    Image.new("RGBA", (32, 24), (20, 40, 60, 128)).save(original)
    output = source / "normalized"

    result = normalize_local_images(source, output)

    assert result.image_count == 1
    assert original.is_file()
    assert not (output / "blue_stone.txt").exists()
    with Image.open(output / "blue_stone.png") as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (32, 24)


def test_normalize_local_images_refuses_nonempty_output(tmp_path):
    source = tmp_path / ".dataset"
    output = source / "normalized"
    output.mkdir(parents=True)
    (output / "existing.txt").write_text("keep me", encoding="utf-8")
    Image.new("RGB", (16, 16)).save(source / "texture.png")

    try:
        normalize_local_images(source, output)
    except FileExistsError as exc:
        assert "Refusing to replace" in str(exc)
    else:
        raise AssertionError("Expected a nonempty output folder to be rejected")
