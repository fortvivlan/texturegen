from PIL import Image

from texturegen.inference import tile_preview


def test_tile_preview_repeats_dimensions_and_pixels():
    tile = Image.new("RGB", (7, 5), "red")
    preview = tile_preview(tile, 3)
    assert preview.size == (21, 15)
    assert preview.getpixel((0, 0)) == preview.getpixel((20, 14)) == (255, 0, 0)
