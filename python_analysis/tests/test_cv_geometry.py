import math
import unittest
from pathlib import Path
import sys

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvb.cv_geometry import fit_circle_kasa, refine_circle_from_edges


class CvGeometryTests(unittest.TestCase):
    def test_fit_circle_kasa_recover_synthetic_circle(self) -> None:
        cx, cy, r = 120.5, 80.25, 45.0
        angles = np.linspace(0.0, 2.0 * math.pi, 80, endpoint=False)
        x = cx + r * np.cos(angles)
        y = cy + r * np.sin(angles)
        pts = np.column_stack([x, y])
        est_cx, est_cy, est_r = fit_circle_kasa(pts)
        self.assertAlmostEqual(est_cx, cx, delta=0.05)
        self.assertAlmostEqual(est_cy, cy, delta=0.05)
        self.assertAlmostEqual(est_r, r, delta=0.05)

    def test_refine_circle_from_edges_improves_approx(self) -> None:
        size = 400
        gray = np.zeros((size, size), dtype=np.uint8)
        cx, cy, r = 200, 200, 80
        cv2.circle(gray, (cx, cy), r, 200, thickness=2)
        rough = (cx + 3, cy - 2, r - 2)
        refined = refine_circle_from_edges(gray, rough, band_px=6)
        self.assertAlmostEqual(refined[0], cx, delta=2.0)
        self.assertAlmostEqual(refined[1], cy, delta=2.0)
        self.assertAlmostEqual(refined[2], r, delta=2.0)


if __name__ == "__main__":
    unittest.main()
