"""Core rendering engine for layered raster diagrams.

Turns a stack of raster bands into a single RGBA image where every band is
drawn as a tilted, diamond-shaped layer (exploded-stack view).

Pipeline (see goals.txt):
    1. LOAD     - split inputs into 2-D bands, stretch to 0-255, resize
    2. SPACING  - vertical gap between layers / projected layer height
    3. GEOMETRY - map each band's corners onto a diamond
    4. RENDER   - perspective warp (bilinear, antialiased) and composite
    5. OUTPUT   - RGBA image plus each layer's diamond corners
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

__all__ = [
    "LayerStackError",
    "load_bands",
    "compute_spacing",
    "layer_corners",
    "render",
]


class LayerStackError(ValueError):
    """Raised when the layers cannot fit the canvas with the given gap."""


# --------------------------------------------------------------------------
# 1. LOAD
# --------------------------------------------------------------------------

_ALPHA_MODES = ("RGBA", "LA", "PA", "RGBa", "La")


def _frame_bands(frame: Image.Image):
    """Yield the individual 2-D bands of a single PIL frame as float arrays."""
    frame.load()
    if frame.mode == "P":
        frame = frame.convert("RGBA" if "transparency" in frame.info else "RGB")
    bands = frame.split()
    if frame.mode in _ALPHA_MODES:
        bands = bands[:-1]  # the alpha channel is not a data layer
    for band in bands:
        arr = np.asarray(band)
        if arr.ndim == 2:
            yield arr.astype(np.float64)
        else:
            for c in range(arr.shape[-1]):
                yield arr[..., c].astype(np.float64)


def _input_bands(obj):
    """Yield raw float bands from one input: path, PIL image, or numpy array."""
    if isinstance(obj, np.ndarray):
        arr = obj
        if arr.ndim == 2:
            yield arr.astype(np.float64)
        elif arr.ndim == 3:
            for c in range(arr.shape[-1]):
                yield arr[..., c].astype(np.float64)
        else:
            raise ValueError(f"array input must be 2-D or 3-D, got shape {arr.shape}")
        return
    if isinstance(obj, Image.Image):
        img = obj
    else:
        img = Image.open(obj)
    try:
        for idx in range(getattr(img, "n_frames", 1)):
            if idx:
                img.seek(idx)
            yield from _frame_bands(img)
    finally:
        img.close()


def _stretch(band: np.ndarray, low: float, high: float) -> np.ndarray:
    """Stretch a band to 0-255 using the low-high percentile range."""
    lo, hi = np.percentile(band, (low, high))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.zeros(band.shape, np.uint8)
    scaled = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def load_bands(
    inputs: Sequence,
    size: Optional[tuple] = None,
    low: float = 2.0,
    high: float = 98.0,
) -> list:
    """Split inputs into individual stretched, resized 2-D uint8 bands.

    inputs : any mix of file paths, PIL images, or numpy arrays
             (multiband inputs contribute one band per channel, in order)
    size   : (w, h) every band is resized to; defaults to the first band's size
    low, high : percentile range used for the display stretch
    """
    if not inputs:
        raise ValueError("no input layers given")
    stretched = []
    for obj in inputs:
        for raw in _input_bands(obj):
            stretched.append(_stretch(raw, low, high))
    if not stretched:
        raise ValueError("inputs contained no bands")
    if size is None:
        h, w = stretched[0].shape
        size = (w, h)
    w, h = int(size[0]), int(size[1])
    if w < 2 or h < 2:
        raise ValueError(f"band size too small: {size}")
    out = []
    for band in stretched:
        if band.shape == (h, w):
            out.append(band)
        else:
            out.append(np.asarray(Image.fromarray(band).resize((w, h), Image.BILINEAR)))
    return out


# --------------------------------------------------------------------------
# 2. SPACING
# --------------------------------------------------------------------------

def compute_spacing(n, canvas_h, gap_z, layer_height=None, min_layer_height=1.0):
    """Return (spacing, layer_height d) for n layers on a canvas_h tall canvas.

    gap_z        : minimum vertical gap between layers, in pixels
    layer_height : projected height d of one tilted layer (optional)
    """
    if n < 1:
        raise ValueError("need at least one layer")
    if layer_height is None:
        spacing = float(gap_z)
        d = canvas_h - (n - 1) * spacing
        if d < min_layer_height:
            raise LayerStackError(
                f"too many layers for this gap: {n} layers with z={gap_z:g} px "
                f"leave only {d:g} px of layer height on a {canvas_h:g} px canvas"
            )
    else:
        d = float(layer_height)
        if n == 1:
            spacing = 0.0  # a single layer has no gaps, so z does not apply
        else:
            spacing = (canvas_h - d) / (n - 1)
            if spacing < gap_z:
                raise LayerStackError(
                    f"too many layers for this gap: {n} layers of height {d:g} px "
                    f"give a spacing of {spacing:g} px, below z={gap_z:g} px"
                )
    return spacing, d


# --------------------------------------------------------------------------
# 3. GEOMETRY
# --------------------------------------------------------------------------

def layer_corners(i, spacing, d, canvas_w, mid_left=1 / 3, mid_right=2 / 3):
    """Diamond corners for layer i (i = 0 is the top of the stack).

    Returns {"top": (x, y), "right": (x, y), "bottom": (x, y), "left": (x, y)}.
    The band's corners map: (0,0)->top, (w,0)->right, (w,h)->bottom, (0,h)->left.
    """
    y = i * spacing
    return {
        "top": (mid_right * canvas_w, y),
        "right": (float(canvas_w), y + d / 2),
        "bottom": (mid_left * canvas_w, y + d),
        "left": (0.0, y + d / 2),
    }


def _homography(dst, src):
    """Solve the 8-unknown homography mapping dst points -> src points."""
    A, b = [], []
    for (X, Y), (x, y) in zip(dst, src):
        A.append([X, Y, 1, 0, 0, 0, -X * x, -Y * x])
        b.append(x)
        A.append([0, 0, 0, X, Y, 1, -X * y, -Y * y])
        b.append(y)
    try:
        return np.linalg.solve(np.array(A, float), np.array(b, float))
    except np.linalg.LinAlgError:
        raise ValueError("degenerate diamond geometry (check mid_left / mid_right)")


# --------------------------------------------------------------------------
# 4. RENDER
# --------------------------------------------------------------------------

def _warp_band(band, corners, canvas_w, canvas_h):
    """Warp one band onto its diamond; returns (x0, y0, x1, y1, rgba) or None.

    Bilinear sampling; alpha is a 1-px-wide box-filter coverage ramp so the
    diamond edges come out antialiased. Fully transparent outside the diamond.
    """
    h, w = band.shape
    dst = (corners["top"], corners["right"], corners["bottom"], corners["left"])
    src = ((0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h)))
    a, b, c, d_, e, f, g, k = _homography(dst, src)

    x0 = max(int(np.floor(min(p[0] for p in dst))), 0)
    x1 = min(int(np.ceil(max(p[0] for p in dst))), canvas_w - 1)
    y0 = max(int(np.floor(min(p[1] for p in dst))), 0)
    y1 = min(int(np.ceil(max(p[1] for p in dst))), canvas_h - 1)
    if x1 < x0 or y1 < y0:
        return None

    xs = np.arange(x0, x1 + 1) + 0.5  # sample at pixel centers
    ys = np.arange(y0, y1 + 1) + 0.5
    X, Y = np.meshgrid(xs, ys)
    Xf, Yf = X.ravel(), Y.ravel()

    den = g * Xf + k * Yf + 1.0
    SX = (a * Xf + b * Yf + c) / den
    SY = (d_ * Xf + e * Yf + f) / den

    # bilinear color sample (indices clamped inside the band)
    SXc = np.clip(SX, 0.0, w - 1.0)
    SYc = np.clip(SY, 0.0, h - 1.0)
    ix = np.minimum(np.floor(SXc).astype(np.intp), w - 2)
    iy = np.minimum(np.floor(SYc).astype(np.intp), h - 2)
    fx = SXc - ix
    fy = SYc - iy
    color = (
        band[iy, ix] * (1 - fx) * (1 - fy)
        + band[iy, ix + 1] * fx * (1 - fy)
        + band[iy + 1, ix] * (1 - fx) * fy
        + band[iy + 1, ix + 1] * fx * fy
    )

    # coverage: fraction of the pixel footprint inside the source rectangle
    cov_x = np.clip(np.minimum(SX + 0.5, w) - np.maximum(SX - 0.5, 0.0), 0.0, 1.0)
    cov_y = np.clip(np.minimum(SY + 0.5, h) - np.maximum(SY - 0.5, 0.0), 0.0, 1.0)
    alpha = cov_x * cov_y * 255.0

    region = np.empty((Xf.size, 4), np.uint8)
    region[:, 0] = region[:, 1] = region[:, 2] = np.clip(np.round(color), 0, 255)
    region[:, 3] = np.round(alpha)
    return x0, y0, x1, y1, region.reshape(y1 - y0 + 1, x1 - x0 + 1, 4)


def render(
    bands: Sequence,
    z: float,
    d: Optional[float] = None,
    canvas_size: Optional[tuple] = None,
    mid_left: float = 1 / 3,
    mid_right: float = 2 / 3,
    outline: bool = False,
    outline_color: tuple = (0, 0, 0),
):
    """Render the layered diagram.

    bands       : list of 2-D uint8 arrays (see load_bands), band 0 on top
    z           : minimum vertical gap between layers, in pixels
    d           : projected height of one tilted layer (optional)
    canvas_size : (W, H) of the output; defaults to the bands' size
    mid_left, mid_right : x-positions of the diamond's bottom / top corners,
                          as fractions of W

    Returns (PIL RGBA image, list of per-layer corner dicts).
    """
    if not bands:
        raise ValueError("no bands to render")
    h, w = bands[0].shape
    if canvas_size is None:
        canvas_size = (w, h)
    W, H = int(canvas_size[0]), int(canvas_size[1])
    if W < 2 or H < 2:
        raise ValueError(f"canvas size too small: {canvas_size}")

    n = len(bands)
    spacing, d = compute_spacing(n, H, z, d)
    corners = [layer_corners(i, spacing, d, W, mid_left, mid_right) for i in range(n)]

    canvas = np.zeros((H, W, 4), np.float32)
    for i in range(n - 1, -1, -1):  # bottom of the stack first, top layer last
        res = _warp_band(bands[i], corners[i], W, H)
        if res is None:
            continue
        x0, y0, x1, y1, region = res

        if outline:
            im = Image.fromarray(region)
            pts = [
                (corners[i][k][0] - x0, corners[i][k][1] - y0)
                for k in ("top", "right", "bottom", "left")
            ]
            ImageDraw.Draw(im).line(
                pts + [pts[0]], fill=tuple(outline_color) + (255,), width=1
            )
            region = np.asarray(im)

        sub = canvas[y0 : y1 + 1, x0 : x1 + 1]
        la = region[..., 3].astype(np.float32) / 255.0
        ca = sub[..., 3] / 255.0
        a_out = la + ca * (1.0 - la)
        rgb = (
            region[..., :3].astype(np.float32) * la[..., None]
            + sub[..., :3] * ca[..., None] * (1.0 - la[..., None])
        )
        nz = a_out > 0
        rgb[nz] /= a_out[nz, None]
        sub[..., :3] = rgb
        sub[..., 3] = a_out * 255.0

    out = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA"), corners
