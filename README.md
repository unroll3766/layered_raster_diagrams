# Layered Raster Diagrams

Renders a stack of raster bands as an exploded-stack diagram: every band is
warped onto a tilted diamond and the diamonds are stacked with a vertical gap.
See `goals.txt` for the full specification.

## Run

```bash
conda activate raster_diagrams
python app.py            # launch the GUI
python app.py a.tif b.png  # pre-load layer files
```

Dependencies: `numpy` and `pillow` (tkinter ships with Python). Install with
`conda install numpy pillow` if needed.

## GUI

- **Layers**: add raster files (multiband TIFFs contribute one band per
  channel, in order; multi-page TIFFs contribute every frame). Reorder with
  Move up / Move down; layer 0 renders at the top of the stack.
- **Geometry**: gap `z` (required), optional layer height `d`, and
  `mid_left` / `mid_right` (fractions like `1/3`, default `1/3` and `2/3`).
- **Sizes**: band size `(w, h)` (default: first band's size) and canvas
  `(W, H)` (default: same as band size).
- **Display**: percentile stretch limits (default 2-98) and an optional
  diamond outline.
- The preview shows the RGBA result; layer diamond corners are listed below
  it for placing labels. **Save PNG** writes the rendered image.
- **Demo stack** generates five synthetic bands to try without data.

## Library use

```python
import core

bands = core.load_bands(["a.tif", "b.png"], size=(512, 512))  # or numpy arrays
image, corners = core.render(bands, z=20, mid_left=1/3, mid_right=2/3)
image.save("diagram.png")
```

## Tests

```bash
python test_core.py
```
