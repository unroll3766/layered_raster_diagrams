"""Headless checks for core.py: run with `python test_core.py`."""

import os
import tempfile

import numpy as np
from PIL import Image

import core


def make_bands(n=3, w=64, h=48):
    y, x = np.mgrid[0:h, 0:w]
    return [(x + y + 40.0 * i).astype(np.float64) for i in range(n)]


def test_load_bands_multiband_array():
    arr = np.zeros((12, 10, 3))
    arr[..., 0] = np.arange(120).reshape(12, 10)          # ramp
    arr[..., 1] = 128.0
    arr[..., 2] = np.arange(120).reshape(12, 10) / 2.0
    bands = core.load_bands([arr])
    assert len(bands) == 3, "multiband array should split into 3 bands"
    assert all(b.shape == (12, 10) for b in bands)
    assert bands[0].dtype == np.uint8
    assert bands[0].min() == 0 and bands[0].max() == 255, "ramp should span 0-255"


def test_load_bands_files_and_default_size():
    d = tempfile.mkdtemp(prefix="diagram_test_")
    p1 = os.path.join(d, "a.png")
    p2 = os.path.join(d, "b.tif")
    Image.fromarray((np.random.default_rng(1).random((20, 16)) * 255).astype(np.uint8)).save(p1)
    Image.new("RGB", (8, 6)).save(p2)
    bands = core.load_bands([p1, p2])
    # 1 band from the single-band png + 3 from the RGB tif, resized to first band's size
    assert len(bands) == 4
    assert all(b.shape == (20, 16) for b in bands)
    bands2 = core.load_bands([p1, p2], size=(32, 32))
    assert all(b.shape == (32, 32) for b in bands2)


def test_load_bands_respects_explicit_size():
    bands = core.load_bands([make_bands(1)[0]], size=(10, 7))
    assert bands[0].shape == (7, 10)  # (w, h) -> numpy (h, w)


def test_spacing_errors():
    bands = make_bands(4)
    try:
        core.render(bands, z=20, canvas_size=(64, 48))  # d = 48 - 60 < 0
        raise AssertionError("expected LayerStackError")
    except core.LayerStackError:
        pass
    try:
        core.render(bands, z=10, d=100, canvas_size=(64, 48))  # spacing < z
        raise AssertionError("expected LayerStackError")
    except core.LayerStackError:
        pass


def test_spacing_values():
    assert core.compute_spacing(3, 48, 10) == (10.0, 28.0)
    assert core.compute_spacing(3, 48, 10, layer_height=28) == (10.0, 28.0)
    assert core.compute_spacing(1, 48, 10) == (10.0, 48.0)
    assert core.compute_spacing(1, 48, 10, layer_height=30) == (0.0, 30.0)


def test_geometry_and_corners():
    bands = make_bands(3)
    image, corners = core.render(bands, z=10, canvas_size=(64, 48), mid_left=1 / 3, mid_right=2 / 3)
    assert image.size == (64, 48) and image.mode == "RGBA"
    assert len(corners) == 3
    top0 = corners[0]["top"]
    assert abs(top0[0] - 64 * 2 / 3) < 1e-9 and top0[1] == 0.0
    bottom2 = corners[2]["bottom"]
    assert abs(bottom2[0] - 64 / 3) < 1e-9 and abs(bottom2[1] - (2 * 10 + 28)) < 1e-9
    left0 = corners[0]["left"]
    assert left0 == (0.0, 14.0)
    right0 = corners[0]["right"]
    assert right0 == (64.0, 14.0)
    return image


def test_render_pixels():
    image, _ = core.render(make_bands(3), z=10, canvas_size=(64, 48))
    a = np.asarray(image)[..., 3]
    assert a[0, 0] <= 8, "top-left canvas corner should be transparent"
    assert a[14, 32] == 255, "center of layer 0's diamond should be opaque"
    assert a[34, 32] == 255, "center of layer 2's diamond should be opaque"


def test_single_layer_coverage_area():
    # one layer fills the whole canvas height: diamond area = W*d/2
    image, _ = core.render(make_bands(1), z=10, canvas_size=(64, 48))
    a = np.asarray(image)[..., 3].astype(np.float64) / 255.0
    expected = 64 * 48 / 2
    assert abs(a.sum() - expected) < 0.02 * expected, f"coverage {a.sum():.1f} vs {expected:.1f}"


def test_outline_and_custom_mids():
    base, _ = core.render(make_bands(3), z=10, canvas_size=(64, 48))
    outlined, corners = core.render(
        make_bands(3), z=10, canvas_size=(64, 48), outline=True, mid_left=0.25, mid_right=0.75
    )
    assert np.asarray(outlined).shape == np.asarray(base).shape
    assert abs(corners[0]["top"][0] - 0.75 * 64) < 1e-9
    assert abs(corners[0]["bottom"][0] - 0.25 * 64) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
