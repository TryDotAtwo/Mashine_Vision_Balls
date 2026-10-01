import math
import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvb.experiment_math import (
    CellGeometry,
    FovScaleInputs,
    FovScaleUncertainty,
    ReferenceMeasurement,
    diameter_from_pixels_fov,
    diameter_from_reference,
    evaporated_mass_kg,
    mm_per_pixel_from_fov,
    required_cell_tolerance_mm,
)


class ExperimentMathTests(unittest.TestCase):
    def test_reference_measurement_uses_ratio(self) -> None:
        reference = ReferenceMeasurement(name="cell", diameter_mm=40.0, sigma_diameter_mm=0.05)
        result = diameter_from_reference(
            diameter_px=20.0,
            reference_diameter_px=200.0,
            reference=reference,
            sigma_detection_px=1.0,
            sigma_reference_px=1.0,
        )
        self.assertAlmostEqual(result.value_mm, 4.0, places=6)
        self.assertGreater(result.sigma_model_mm, 0.0)

    def test_fov_scale_and_measurement_are_positive(self) -> None:
        inputs = FovScaleInputs(
            distance_to_object_mm=55.0,
            zoom_factor=1.0,
            horizontal_fov_deg=79.0,
            resolution_x_px=2160,
            calibration_coef=1.0,
        )
        uncertainty = FovScaleUncertainty()
        scale = mm_per_pixel_from_fov(inputs)
        result = diameter_from_pixels_fov(diameter_px=120.0, inputs=inputs, uncertainty=uncertainty, sigma_detection_px=1.5)
        self.assertGreater(scale, 0.0)
        self.assertGreater(result.value_mm, 0.0)
        self.assertGreater(result.sigma_model_mm, 0.0)

    def test_evaporated_mass_matches_formula(self) -> None:
        cell = CellGeometry(
            inner_diameter_mm=40.0,
            sigma_inner_diameter_mm=0.05,
            level_drop_mm=10.0,
            sigma_level_mm=0.1,
        )
        mass_kg, sigma_mass_kg = evaporated_mass_kg(cell)
        expected_mass = 808.0 * math.pi * (0.04**2) / 4.0 * 0.01
        self.assertAlmostEqual(mass_kg, expected_mass, places=8)
        self.assertGreater(sigma_mass_kg, 0.0)

    def test_required_tolerance(self) -> None:
        sigma_d = required_cell_tolerance_mm(
            target_relative_error_mass=0.03,
            inner_diameter_mm=40.0,
            level_drop_mm=10.0,
            level_measurement_sigma_mm=0.1,
        )
        self.assertAlmostEqual(sigma_d, 0.5656854249, places=6)


if __name__ == "__main__":
    unittest.main()
