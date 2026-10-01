from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StandGeometry:
    ball_to_cover_mm: float | None = None
    cell_to_cover_mm: float | None = None
    cover_thickness_mm: float | None = None
    cell_cover_gap_mm: float | None = None


@dataclass(frozen=True)
class FovScaleInputs:
    distance_to_object_mm: float
    zoom_factor: float
    horizontal_fov_deg: float
    resolution_x_px: int
    calibration_coef: float = 1.0


@dataclass(frozen=True)
class FovScaleUncertainty:
    sigma_distance_mm: float = 0.5
    sigma_zoom: float = 0.05
    sigma_fov_deg: float = 0.5


@dataclass(frozen=True)
class ReferenceMeasurement:
    name: str
    diameter_mm: float
    sigma_diameter_mm: float


@dataclass(frozen=True)
class SizeMeasurement:
    value_mm: float
    sigma_model_mm: float
    method: str


@dataclass(frozen=True)
class CellGeometry:
    inner_diameter_mm: float
    sigma_inner_diameter_mm: float
    level_drop_mm: float
    sigma_level_mm: float


@dataclass(frozen=True)
class NitrogenModel:
    rho_kg_m3: float = 808.0


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def mm_per_pixel_from_fov(inputs: FovScaleInputs) -> float:
    _require_positive("distance_to_object_mm", inputs.distance_to_object_mm)
    _require_positive("zoom_factor", inputs.zoom_factor)
    _require_positive("resolution_x_px", float(inputs.resolution_x_px))
    theta = math.radians(inputs.horizontal_fov_deg)
    scene_width_mm = 2.0 * inputs.distance_to_object_mm * math.tan(theta / 2.0) / inputs.zoom_factor
    return inputs.calibration_coef * (scene_width_mm / float(inputs.resolution_x_px))


def sigma_mm_per_pixel_from_fov(inputs: FovScaleInputs, uncertainty: FovScaleUncertainty) -> float:
    scale = mm_per_pixel_from_fov(inputs)
    theta = math.radians(inputs.horizontal_fov_deg)
    sigma_theta = math.radians(uncertainty.sigma_fov_deg)
    d_scale_d_distance = scale / inputs.distance_to_object_mm
    d_scale_d_zoom = -scale / inputs.zoom_factor
    d_scale_d_theta = scale / math.sin(theta)
    return math.sqrt(
        (d_scale_d_distance * uncertainty.sigma_distance_mm) ** 2
        + (d_scale_d_zoom * uncertainty.sigma_zoom) ** 2
        + (d_scale_d_theta * sigma_theta) ** 2
    )


def diameter_from_pixels_fov(
    diameter_px: float,
    inputs: FovScaleInputs,
    uncertainty: FovScaleUncertainty,
    sigma_detection_px: float,
) -> SizeMeasurement:
    _require_positive("diameter_px", diameter_px)
    scale = mm_per_pixel_from_fov(inputs)
    sigma_scale = sigma_mm_per_pixel_from_fov(inputs, uncertainty)
    value_mm = diameter_px * scale
    sigma_model_mm = math.sqrt((diameter_px * sigma_scale) ** 2 + (scale * sigma_detection_px) ** 2)
    return SizeMeasurement(value_mm=value_mm, sigma_model_mm=sigma_model_mm, method="fov")


def diameter_from_reference(
    diameter_px: float,
    reference_diameter_px: float,
    reference: ReferenceMeasurement,
    sigma_detection_px: float,
    sigma_reference_px: float,
) -> SizeMeasurement:
    _require_positive("diameter_px", diameter_px)
    _require_positive("reference_diameter_px", reference_diameter_px)
    _require_positive("reference.diameter_mm", reference.diameter_mm)
    value_mm = diameter_px / reference_diameter_px * reference.diameter_mm
    relative_sigma = math.sqrt(
        (sigma_detection_px / diameter_px) ** 2
        + (sigma_reference_px / reference_diameter_px) ** 2
        + (reference.sigma_diameter_mm / reference.diameter_mm) ** 2
    )
    return SizeMeasurement(
        value_mm=value_mm,
        sigma_model_mm=value_mm * relative_sigma,
        method=f"reference:{reference.name}",
    )


def cell_area_m2(cell: CellGeometry) -> tuple[float, float]:
    _require_positive("cell.inner_diameter_mm", cell.inner_diameter_mm)
    diameter_m = cell.inner_diameter_mm / 1000.0
    sigma_diameter_m = cell.sigma_inner_diameter_mm / 1000.0
    area_m2 = math.pi * (diameter_m**2) / 4.0
    d_area_d_diameter = math.pi * diameter_m / 2.0
    sigma_area_m2 = abs(d_area_d_diameter) * sigma_diameter_m
    return area_m2, sigma_area_m2


def relative_mass_uncertainty(cell: CellGeometry) -> float:
    _require_positive("cell.level_drop_mm", cell.level_drop_mm)
    _require_positive("cell.inner_diameter_mm", cell.inner_diameter_mm)
    return math.sqrt(
        (2.0 * cell.sigma_inner_diameter_mm / cell.inner_diameter_mm) ** 2
        + (cell.sigma_level_mm / cell.level_drop_mm) ** 2
    )


def evaporated_mass_kg(cell: CellGeometry, model: NitrogenModel | None = None) -> tuple[float, float]:
    model = model or NitrogenModel()
    area_m2, sigma_area_m2 = cell_area_m2(cell)
    delta_h_m = cell.level_drop_mm / 1000.0
    sigma_h_m = cell.sigma_level_mm / 1000.0
    mass_kg = model.rho_kg_m3 * area_m2 * delta_h_m
    sigma_mass_kg = model.rho_kg_m3 * math.sqrt((delta_h_m * sigma_area_m2) ** 2 + (area_m2 * sigma_h_m) ** 2)
    return mass_kg, sigma_mass_kg


def required_cell_tolerance_mm(
    target_relative_error_mass: float,
    inner_diameter_mm: float,
    level_drop_mm: float,
    level_measurement_sigma_mm: float,
) -> float:
    _require_positive("target_relative_error_mass", target_relative_error_mass)
    _require_positive("inner_diameter_mm", inner_diameter_mm)
    _require_positive("level_drop_mm", level_drop_mm)
    residual = target_relative_error_mass**2 - (level_measurement_sigma_mm / level_drop_mm) ** 2
    if residual <= 0:
        return 0.0
    return 0.5 * inner_diameter_mm * math.sqrt(residual)
