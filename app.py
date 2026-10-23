#!/usr/bin/env python3
"""Tkinter GUI for layered raster diagrams.

Renders a stack of raster bands as an exploded-stack diagram: every band is
warped onto a tilted diamond and the diamonds are stacked with a vertical gap
(see goals.txt for the full specification).

Run with:
    python app.py [layer1.tif layer2.png ...]

Dependencies: numpy, Pillow (tkinter is part of the Python distribution).
"""

from __future__ import annotations

import os
import sys
import tempfile
from fractions import Fraction

import numpy as np
from PIL import Image, ImageFilter, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import core

PREVIEW_MAX = (880, 560)

FILETYPES = [
    ("Raster images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.gif *.webp *.ppm"),
    ("All files", "*.*"),
]


def _demo_files() -> list:
    """Write a small stack of distinct single-band PNGs and return the paths."""
    size = 256
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    bands = []
    for i in range(5):
        t = i / 4.0
        if i % 4 == 0:
            band = 255 * (x + y) / (2 * size)
        elif i % 4 == 1:
            band = 255 * (np.sin(x / size * 6 * np.pi + t * 3) * 0.5 + 0.5)
        elif i % 4 == 2:
            cx, cy = size * (0.3 + 0.4 * t), size * 0.5
            band = 255 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * (size / 5) ** 2)))
        else:
            rng = np.random.default_rng(100 + i)
            noise = Image.fromarray((rng.random((size, size)) * 255).astype(np.uint8))
            band = np.asarray(noise.filter(ImageFilter.GaussianBlur(9))).astype(np.float64)
        bands.append(np.clip(band, 0, 255).astype(np.uint8))
    directory = tempfile.mkdtemp(prefix="diagram_demo_")
    paths = []
    for i, band in enumerate(bands):
        path = os.path.join(directory, f"demo_band_{i}.png")
        Image.fromarray(band).save(path)
        paths.append(path)
    return paths


class DiagramApp(tk.Tk):
    def __init__(self, initial_layers=()):
        super().__init__()
        self.title("Layered Raster Diagrams")
        self.geometry("1180x760")
        self.minsize(960, 600)

        self.layers = [p for p in initial_layers if os.path.exists(p)]
        self.result_image = None
        self.corners = []
        self._photo = None

        self._build_ui()
        self._refresh_layer_list()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsw")
        left.columnconfigure(0, weight=1)

        # -- layers ------------------------------------------------------
        layers = ttk.LabelFrame(left, text="Layers (band order)", padding=6)
        layers.grid(row=0, column=0, sticky="nsew")
        layers.columnconfigure(0, weight=1)
        layers.rowconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.layer_list = tk.Listbox(layers, selectmode=tk.EXTENDED, width=34, height=8)
        self.layer_list.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=(0, 6))
        scrollbar = ttk.Scrollbar(layers, orient=tk.VERTICAL, command=self.layer_list.yview)
        self.layer_list.config(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, rowspan=4, sticky="ns")

        ttk.Button(layers, text="Add files…", command=self._add_files).grid(row=0, column=2, sticky="ew")
        ttk.Button(layers, text="Remove", command=self._remove_selected).grid(row=1, column=2, sticky="ew")
        ttk.Button(layers, text="Move up", command=lambda: self._move_selected(-1)).grid(row=2, column=2, sticky="ew")
        ttk.Button(layers, text="Move down", command=lambda: self._move_selected(1)).grid(row=3, column=2, sticky="ew")
        ttk.Button(layers, text="Demo stack", command=self._demo).grid(row=4, column=2, sticky="ew", pady=(6, 0))

        # -- parameters --------------------------------------------------
        def entry(parent, row, label, width=9, default=""):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
            var = tk.StringVar(value=default)
            ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="e", pady=1)
            return var

        geo = ttk.LabelFrame(left, text="Geometry", padding=6)
        geo.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.e_z = entry(geo, 0, "Gap z (px):", default="24")
        self.e_d = entry(geo, 1, "Layer height d (px):")
        self.e_ml = entry(geo, 2, "mid_left:", default="1/3")
        self.e_mr = entry(geo, 3, "mid_right:", default="2/3")

        sizes = ttk.LabelFrame(left, text="Sizes (blank = default)", padding=6)
        sizes.grid(row=2, column=0, sticky="ew", pady=8)
        self.e_w = entry(sizes, 0, "Band width w:")
        self.e_h = entry(sizes, 1, "Band height h:")
        self.e_W = entry(sizes, 2, "Canvas width W:")
        self.e_H = entry(sizes, 3, "Canvas height H:")

        disp = ttk.LabelFrame(left, text="Display", padding=6)
        disp.grid(row=3, column=0, sticky="ew")
        self.e_low = entry(disp, 0, "Stretch low (%):", default="2")
        self.e_high = entry(disp, 1, "Stretch high (%):", default="98")
        self.outline_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(disp, text="Outline diamonds", variable=self.outline_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        self.render_btn = ttk.Button(left, text="Render", command=self._render)
        self.render_btn.grid(row=4, column=0, sticky="ew", pady=(10, 2))
        self.save_btn = ttk.Button(left, text="Save PNG…", command=self._save, state=tk.DISABLED)
        self.save_btn.grid(row=5, column=0, sticky="ew")

        # -- preview -----------------------------------------------------
        right = ttk.Frame(self, padding=(0, 8, 8, 8))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.preview = tk.Label(right, bg="#2e2e2e")
        self.preview.grid(row=0, column=0, sticky="nsew", padx=4, pady=(0, 6))

        corners_frame = ttk.LabelFrame(right, text="Layer corners (top, right, bottom, left)", padding=6)
        corners_frame.grid(row=1, column=0, sticky="ew")
        corners_frame.columnconfigure(0, weight=1)
        self.corners_text = tk.Text(corners_frame, height=6, wrap=tk.NONE, state=tk.DISABLED, font=("TkFixedFont", 9))
        self.corners_text.grid(row=0, column=0, sticky="ew")
        ttk.Scrollbar(corners_frame, orient=tk.HORIZONTAL, command=self.corners_text.xview).grid(
            row=1, column=0, sticky="ew"
        )
        self.corners_text.config(xscrollcommand=corners_frame.grid_slaves(row=1, column=0)[0].set)

        self.status = ttk.Label(self, anchor="w", relief="sunken", padding=(6, 2))
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew")

    # ------------------------------------------------------- helpers

    def _set_status(self, text):
        self.status.config(text=text)

    def _refresh_layer_list(self):
        self.layer_list.delete(0, tk.END)
        for path in self.layers:
            name = os.path.basename(path)
            n_frames = ""
            try:
                with Image.open(path) as img:
                    frames = getattr(img, "n_frames", 1)
                    if frames > 1:
                        n_frames = f"  [{frames} frames]"
            except Exception:
                name += "  (unreadable)"
            self.layer_list.insert(tk.END, name + n_frames)

    def _selected_indices(self):
        return sorted(self.layer_list.curselection(), reverse=True)

    # ------------------------------------------------------- layer actions

    def _add_files(self):
        paths = filedialog.askopenfilenames(title="Add layer files", filetypes=FILETYPES)
        if paths:
            self.layers.extend(paths)
            self._refresh_layer_list()

    def _remove_selected(self):
        for idx in self._selected_indices():
            del self.layers[idx]
        self._refresh_layer_list()

    def _move_selected(self, delta):
        indices = sorted(self.layer_list.curselection())
        if not indices:
            return
        selection = []
        for idx in indices:
            new = idx + delta
            if 0 <= new < len(self.layers):
                self.layers[idx], self.layers[new] = self.layers[new], self.layers[idx]
                selection.append(new)
        self._refresh_layer_list()
        for new in selection:
            self.layer_list.selection_set(new)

    def _demo(self):
        self.layers = _demo_files()
        self._refresh_layer_list()
        self._set_status(f"Loaded demo stack ({len(self.layers)} bands)")

    # ------------------------------------------------------- parsing

    def _opt_int(self, var, name):
        text = var.get().strip()
        if not text:
            return None
        try:
            value = int(float(text))
        except ValueError:
            raise ValueError(f"{name} must be an integer, got {text!r}")
        if value < 2:
            raise ValueError(f"{name} must be at least 2")
        return value

    def _opt_float(self, var, name):
        text = var.get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{name} must be a number, got {text!r}")

    def _opt_fraction(self, var, name, default):
        text = var.get().strip()
        if not text:
            return default
        try:
            return float(Fraction(text))
        except ValueError:
            raise ValueError(f"{name} must be a number or fraction like 1/3, got {text!r}")

    def _pair(self, var_a, var_b, name):
        a, b = self._opt_int(var_a, name + " width"), self._opt_int(var_b, name + " height")
        if (a is None) != (b is None):
            raise ValueError(f"give both {name} width and height, or neither")
        return None if a is None else (a, b)

    # ------------------------------------------------------- render / save

    def _render(self):
        try:
            size = self._pair(self.e_w, self.e_h, "band")
            canvas = self._pair(self.e_W, self.e_H, "canvas")
            z = self._opt_float(self.e_z, "z")
            if z is None:
                raise ValueError("gap z is required")
            d = self._opt_float(self.e_d, "d")
            if d is not None and d < 1:
                raise ValueError("layer height d must be at least 1")
            mid_left = self._opt_fraction(self.e_ml, "mid_left", 1 / 3)
            mid_right = self._opt_fraction(self.e_mr, "mid_right", 2 / 3)
            low = self._opt_float(self.e_low, "stretch low") or 2.0
            high = self._opt_float(self.e_high, "stretch high") or 98.0
            if not 0 <= low < high <= 100:
                raise ValueError("stretch percentiles must satisfy 0 <= low < high <= 100")

            bands = core.load_bands(self.layers, size=size, low=low, high=high)
            image, corners = core.render(
                bands,
                z=z,
                d=d,
                canvas_size=canvas,
                mid_left=mid_left,
                mid_right=mid_right,
                outline=self.outline_var.get(),
            )
        except (core.LayerStackError, ValueError, OSError) as exc:
            messagebox.showerror("Render failed", str(exc))
            self._set_status(f"Render failed: {exc}")
            return

        self.result_image = image
        self.corners = corners
        self._update_preview()
        self._show_corners()
        self.save_btn.config(state=tk.NORMAL)
        self._set_status(
            f"Rendered {len(bands)} bands ({image.width}x{image.height}px, "
            f"spacing {self._spacing_text(len(bands), image.height, z, d)})"
        )

    @staticmethod
    def _spacing_text(n, canvas_h, z, d):
        try:
            spacing, _ = core.compute_spacing(n, canvas_h, z, d)
            return f"{spacing:.1f} px"
        except core.LayerStackError:
            return "?"

    def _update_preview(self):
        image = self.result_image.copy()
        image.thumbnail(PREVIEW_MAX)
        self._photo = ImageTk.PhotoImage(image)
        self.preview.config(image=self._photo, width=image.width, height=image.height)

    def _show_corners(self):
        lines = []
        for i, c in enumerate(self.corners):
            pts = "   ".join(f"{k}=({v[0]:.1f}, {v[1]:.1f})" for k, v in c.items())
            lines.append(f"Layer {i}: {pts}")
        self.corners_text.config(state=tk.NORMAL)
        self.corners_text.delete("1.0", tk.END)
        self.corners_text.insert("1.0", "\n".join(lines))
        self.corners_text.config(state=tk.DISABLED)

    def _save(self):
        if self.result_image is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save diagram",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not path:
            return
        try:
            self.result_image.save(path)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self._set_status(f"Saved {path}")


def main(argv=()):
    app = DiagramApp(initial_layers=argv)
    app.mainloop()


if __name__ == "__main__":
    main(sys.argv[1:])
