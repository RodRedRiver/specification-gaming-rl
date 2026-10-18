"""Raster rendering of the GridWorld (frames for GIFs and the layout diagram)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

COLORS = {
    "bg": (245, 245, 242),
    "grid": (200, 200, 195),
    "hazard": (214, 76, 62),
    "checkpoint": (240, 190, 50),
    "package": (120, 85, 50),
    "destination": (70, 150, 90),
    "start": (215, 215, 230),
    "agent": (40, 90, 200),
    "agent_loaded": (110, 50, 170),
    "text": (30, 30, 30),
}


def draw_grid(env, cell: int = 48, caption: str | None = None) -> np.ndarray:
    h, w = env.n_rows * cell, env.n_cols * cell
    extra = 22 if caption else 0
    img = Image.new("RGB", (w, h + extra), COLORS["bg"])
    d = ImageDraw.Draw(img)

    def box(rc, color, pad=0):
        r, c = rc
        d.rectangle([c * cell + pad, r * cell + pad, (c + 1) * cell - pad - 1, (r + 1) * cell - pad - 1], fill=color)

    box(env.start, COLORS["start"])
    for hz in env.hazards:
        box(hz, COLORS["hazard"])
    box(env.checkpoint, COLORS["checkpoint"])
    box(env.destination, COLORS["destination"])
    if not env.has_package:
        box(env.package_pos, COLORS["package"], pad=cell // 4)
    for i in range(env.n_rows + 1):
        d.line([0, i * cell, w, i * cell], fill=COLORS["grid"])
    for j in range(env.n_cols + 1):
        d.line([j * cell, 0, j * cell, h], fill=COLORS["grid"])
    for rc, label in ((env.checkpoint, "C"), (env.destination, "D"), (env.start, "S")):
        d.text((rc[1] * cell + 4, rc[0] * cell + 2), label, fill=COLORS["text"])

    r, c = env.pos
    color = COLORS["agent_loaded"] if env.has_package else COLORS["agent"]
    p = cell // 5
    d.ellipse([c * cell + p, r * cell + p, (c + 1) * cell - p, (r + 1) * cell - p], fill=color)
    if caption:
        d.text((4, h + 4), caption, fill=COLORS["text"])
    return np.asarray(img)
