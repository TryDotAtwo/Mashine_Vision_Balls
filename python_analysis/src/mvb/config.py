from __future__ import annotations

from pathlib import Path


CONFIG_TEMPLATE = """# Машинное зрение шариков: шаблон конфигурации стенда
# ВАЖНО:
# 1. ball_to_object_mm, cell_to_cover_mm, cover_thickness_mm и cell_cover_gap_mm
#    описывают геометрию стенда, а НЕ внутренний диаметр ячейки.
# 2. Для расчета массы азота и измерения "по ячейке" внутренний диаметр ячейки
#    должен задаваться отдельно через cell_inner_diameter_mm.

video_path = data/videos/Ball_For_MV.mp4
measurement_mode = fov
output_dir = output/analysis/ball_for_mv
frame_step = 10
max_frames = 0
preview_frames = 6
interactive_preview = false
live_plots = false
live_plot_max_points = 3000
single_ball = false
offline = false
tracking_distance_px = 60.0
tracking_max_missed = 3
min_track_observations = 2

min_ball_diam_mm = 1.0
max_ball_diam_mm = 10.0
expected_ball_diameter_mm = 0.0
sigma_ball_px = 1.5
sigma_reference_px = 1.5

ball_to_object_mm = 55.0
cell_to_cover_mm = 38.13
cover_thickness_mm = 0.85
cell_cover_gap_mm = 6.3
zoom = 1.0
fov_deg = 79.0
calib_coef = 1.0
sigma_distance_mm = 0.5
sigma_zoom = 0.05
sigma_fov_deg = 0.5

reference_name = cell
reference_diameter_mm = 0.0
reference_sigma_mm = 0.0
reference_min_radius_frac = 0.25
reference_max_radius_frac = 0.60

no_edge_refine = false
edge_refine_band_px = 4
no_adaptive_hough = false
min_ball_edge_support = 0.10
ball_candidate_dedupe_px = 14.0
second_ball_score_ratio = 0.92
clear_track_after_lost_frames = 0

cell_inner_diameter_mm = 0.0
cell_inner_sigma_mm = 0.0
delta_level_mm = 0.0
level_sigma_mm = 0.0
target_rel_mass_error = 0.03
"""


KEY_ALIASES = {
    "video": "video_path",
    "distance_mm": "ball_to_object_mm",
    "min_ball_diameter_mm": "min_ball_diam_mm",
    "max_ball_diameter_mm": "max_ball_diam_mm",
}


def _coerce_scalar(raw_value: str) -> str | bool | int | float:
    lowered = raw_value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in lowered for char in (".", "e")):
            return float(lowered)
        return int(lowered)
    except ValueError:
        return raw_value.strip()


def load_key_value_config(path: str | Path) -> dict[str, str | bool | int | float]:
    path = Path(path)
    data: dict[str, str | bool | int | float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid config line: {raw_line!r}")
        key, value = line.split("=", 1)
        normalized_key = KEY_ALIASES.get(key.strip(), key.strip())
        data[normalized_key] = _coerce_scalar(value)
    return data
