"""Circle fitting and edge-based refinement for robust diameter estimation."""

from __future__ import annotations

import math

import cv2
import numpy as np


def fit_circle_kasa(points_xy: np.ndarray) -> tuple[float, float, float]:
    """
    Algebraic (Kåsa) circle fit: minimize sum (x^2+y^2 - a x - b y - c)^2 in least squares.

    Returns (cx, cy, r) in pixel coordinates.
    """
    if points_xy.shape[0] < 3:
        raise ValueError("At least 3 points are required for circle fit")
    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)
    z = x * x + y * y
    a_mat = np.column_stack([x, y, np.ones(len(x))])
    coef, _, rank, _ = np.linalg.lstsq(a_mat, z, rcond=None)
    if rank < 3:
        raise ValueError("Degenerate point set for circle fit")
    cx = float(coef[0] / 2.0)
    cy = float(coef[1] / 2.0)
    r_sq = coef[2] + cx * cx + cy * cy
    if r_sq <= 0:
        raise ValueError("Non-positive radius in algebraic circle fit")
    return cx, cy, float(math.sqrt(r_sq))


def _canny_edges(gray: np.ndarray, low_ratio: float = 0.66, high_ratio: float = 1.33) -> np.ndarray:
    med = float(np.median(gray))
    low = max(1.0, med * low_ratio)
    high = max(low + 1.0, med * high_ratio)
    return cv2.Canny(gray, int(low), int(high))


def refine_circle_from_edges(
    gray: np.ndarray,
    circle: tuple[int, int, int],
    band_px: int,
    min_edge_points: int = 24,
    round_result: bool = True,
) -> tuple[float, float, float] | tuple[int, int, int]:
    """
    Refine an approximate integer circle by fitting to Canny edge pixels in an annulus.

    Falls back to the input circle if too few edge points are found.
    """
    cx_i, cy_i, r_i = circle
    h, w = gray.shape[:2]
    edges = _canny_edges(gray)
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx.astype(np.float64) - float(cx_i)) ** 2 + (yy.astype(np.float64) - float(cy_i)) ** 2)
    inner = max(1.0, float(r_i) - float(band_px))
    outer = float(r_i) + float(band_px)
    mask = (dist >= inner) & (dist <= outer)
    ring = edges.astype(bool) & mask
    pts_y, pts_x = np.where(ring)
    if pts_y.size < min_edge_points:
        return circle
    pts_xy = np.column_stack([pts_x.astype(np.float64), pts_y.astype(np.float64)])
    try:
        cx, cy, r = fit_circle_kasa(pts_xy)
    except ValueError:
        return circle
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(r)):
        return circle
    if r < 2.0 or r > 0.95 * min(h, w) / 2.0:
        return circle
    shift = math.hypot(cx - cx_i, cy - cy_i)
    if shift > 0.35 * r_i:
        return circle
    return (int(round(cx)), int(round(cy)), int(round(r))) if round_result else (cx, cy, r)


def annulus_edge_support_score(
    edges: np.ndarray,
    circle: tuple[int, int, int],
    band_px: int = 1,
) -> float:
    """Mean edge strength on a thin ring (0..1)."""
    height, width = edges.shape[:2]
    x, y, r = circle
    samples: list[float] = []
    for offset in range(-band_px, band_px + 1):
        rr = max(1, r + offset)
        angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        xs = np.clip(np.round(x + rr * np.cos(angles)).astype(int), 0, width - 1)
        ys = np.clip(np.round(y + rr * np.sin(angles)).astype(int), 0, height - 1)
        samples.append(float(edges[ys, xs].mean() / 255.0))
    return float(np.mean(samples))
