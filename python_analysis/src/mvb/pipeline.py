from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from mvb.ball_episodes import advance_ball_id
from mvb.cv_geometry import annulus_edge_support_score, refine_circle_from_edges
from mvb.experiment_math import (
    FovScaleInputs,
    FovScaleUncertainty,
    ReferenceMeasurement,
    SizeMeasurement,
    diameter_from_pixels_fov,
    diameter_from_reference,
    mm_per_pixel_from_fov,
)


MeasurementMode = Literal["fov", "reference"]
DETECTION_MAX_DIMENSION_PX = 1400


@dataclass(frozen=True)
class VideoAnalysisConfig:
    video_path: str
    output_dir: str | None = None
    frame_step: int = 10
    max_frames: int = 0
    preview_frames: int = 6
    interactive_preview: bool = False
    display_scale: float = 0.45
    live_plots: bool = False
    live_plot_max_points: int = 3000


@dataclass(frozen=True)
class DetectionConfig:
    min_ball_diameter_mm: float = 1.0
    max_ball_diameter_mm: float = 10.0
    sigma_ball_px: float = 1.5
    sigma_reference_px: float = 1.5
    reference_min_radius_frac: float = 0.25
    reference_max_radius_frac: float = 0.60
    refine_with_edge_fit: bool = True
    edge_refine_band_px: int = 4
    adaptive_hough: bool = True
    hough_param2_reference: tuple[int, ...] = (22, 26, 28, 32, 36)
    hough_param2_ball: tuple[int, ...] = (12, 16, 18, 22, 26)
    # Gating: no detection if best edge support is below threshold (reduces false positives on empty frames).
    min_ball_edge_support: float = 0.10
    # Merge duplicate Hough detections of the same physical ball.
    ball_candidate_dedupe_px: float = 14.0
    # When two scores are within this ratio of the best, prefer track / ref center (two-ball case).
    second_ball_score_ratio: float = 0.92
    # After this many consecutive missed detections, clear previous_ball (0 = never).
    clear_track_after_lost_frames: int = 0


@dataclass(frozen=True)
class FrameMeasurement:
    frame_index: int
    time_s: float
    ball_center_x_px: float
    ball_center_y_px: float
    ball_radius_px: float
    ball_diameter_px: float
    reference_radius_px: float | None
    reference_diameter_px: float | None
    scale_mm_per_px: float
    diameter_mm: float
    sigma_model_mm: float
    method: str
    ball_edge_support: float = 0.0
    ball_valid_candidates: int = 0
    ball_id: int = 0


@dataclass(frozen=True)
class AnalysisSummary:
    video_path: str
    output_dir: str
    measurement_mode: str
    total_frames: int
    sampled_frames: int
    detections: int
    detection_ratio: float
    fps: float
    width_px: int
    height_px: int
    scale_mm_per_px_mean: float | None
    diameter_mm_mean: float | None
    diameter_mm_median: float | None
    diameter_mm_std: float | None
    diameter_mm_min: float | None
    diameter_mm_max: float | None
    sigma_model_mm_mean: float | None
    combined_sigma_single_mm: float | None
    combined_sigma_mean_mm: float | None
    reference_diameter_px_mean: float | None
    expected_ball_diameter_mm: float | None
    bias_to_expected_mm: float | None


@dataclass(frozen=True)
class VideoAnalysisResult:
    summary: AnalysisSummary
    measurements: list[FrameMeasurement]
    preview_images: list[np.ndarray]


@dataclass(frozen=True)
class BallDetection:
    circle: tuple[int, int, int]
    edge_support: float
    valid_candidates: int


def preprocess_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return cv2.GaussianBlur(equalized, (5, 5), 0)


def _prepare_detection_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    longest_edge = max(height, width)
    scale = min(1.0, DETECTION_MAX_DIMENSION_PX / float(longest_edge))
    if scale < 1.0:
        working_frame = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working_frame = frame
    return preprocess_gray(working_frame), scale


def _rescale_circle(circle: tuple[int, int, int] | None, inverse_scale: float) -> tuple[int, int, int] | None:
    if circle is None:
        return None
    x, y, radius = circle
    return (
        int(round(x * inverse_scale)),
        int(round(y * inverse_scale)),
        int(round(radius * inverse_scale)),
    )


def _detect_circles(gray_frame: np.ndarray, min_radius: int, max_radius: int, param2: int) -> np.ndarray | None:
    circles = cv2.HoughCircles(
        gray_frame,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(gray_frame.shape[:2]) // 5),
        param1=120,
        param2=param2,
        minRadius=max(1, min_radius),
        maxRadius=max(min_radius + 1, max_radius),
    )
    if circles is None:
        return None
    return np.round(circles[0]).astype(int)


def _iter_hough_candidates(
    gray_frame: np.ndarray,
    min_radius: int,
    max_radius: int,
    param2_values: tuple[int, ...],
) -> list[tuple[int, int, int]]:
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[int, int, int]] = []
    for param2 in param2_values:
        circles = _detect_circles(gray_frame, min_radius=min_radius, max_radius=max_radius, param2=param2)
        if circles is None:
            continue
        for x, y, radius in circles:
            key = (int(x), int(y), int(radius))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _prepare_edges_for_scoring(gray_frame: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray_frame, 50, 120)


def _ball_combined_score(
    circle: tuple[int, int, int],
    edge_support: float,
    preferred_center: np.ndarray,
    previous_radius: int | None,
) -> float:
    x, y, radius = circle
    center = np.array([float(x), float(y)])
    distance = float(np.linalg.norm(center - preferred_center))
    if previous_radius is not None:
        return edge_support * 200.0 - distance * 0.8 - abs(radius - previous_radius) * 0.4
    return edge_support * 200.0 - distance * 0.25 + radius * 0.8


def detect_reference_circle(gray_frame: np.ndarray, cfg: DetectionConfig) -> tuple[int, int, int] | None:
    height, width = gray_frame.shape[:2]
    min_radius = int(min(height, width) * cfg.reference_min_radius_frac)
    max_radius = int(min(height, width) * cfg.reference_max_radius_frac)
    param2_list = cfg.hough_param2_reference if cfg.adaptive_hough else (cfg.hough_param2_reference[0],)
    candidates = _iter_hough_candidates(gray_frame, min_radius, max_radius, param2_list)
    if not candidates:
        return None
    edges = _prepare_edges_for_scoring(gray_frame)
    frame_center = np.array([width / 2.0, height / 2.0])
    best = None
    best_score = None
    for x, y, radius in candidates:
        center = np.array([float(x), float(y)])
        distance = np.linalg.norm(center - frame_center)
        edge_support = annulus_edge_support_score(edges, (int(x), int(y), int(radius)))
        score = edge_support * 200.0 - distance * 0.3 + radius * 0.4
        if best_score is None or score > best_score:
            best = (int(x), int(y), int(radius))
            best_score = float(score)
    return best


def detect_ball_circle(
    gray_frame: np.ndarray,
    min_radius: int,
    max_radius: int,
    reference_circle: tuple[int, int, int] | None,
    previous_ball: tuple[int, int, int] | None,
    cfg: DetectionConfig,
) -> BallDetection | None:
    param2_list = cfg.hough_param2_ball if cfg.adaptive_hough else (cfg.hough_param2_ball[0],)
    candidates = _iter_hough_candidates(gray_frame, min_radius, max_radius, param2_list)
    if not candidates:
        return None
    edges = _prepare_edges_for_scoring(gray_frame)
    if previous_ball is not None:
        preferred_center = np.array([previous_ball[0], previous_ball[1]], dtype=float)
        previous_radius = previous_ball[2]
        ref_r = reference_circle[2] if reference_circle is not None else None
    elif reference_circle is not None:
        ref_x, ref_y, ref_r = reference_circle
        preferred_center = np.array([float(ref_x), float(ref_y)], dtype=float)
        previous_radius = None
    else:
        height, width = gray_frame.shape[:2]
        preferred_center = np.array([width / 2.0, height / 2.0], dtype=float)
        ref_r = None
        previous_radius = None

    scored: list[tuple[tuple[int, int, int], float, float]] = []
    for x, y, radius in candidates:
        center = np.array([float(x), float(y)])
        # Same geometry as legacy: distance to preferred_center (track) or ref center, not always ref.
        if reference_circle is not None and ref_r is not None:
            distance_to_ref = float(np.linalg.norm(center - preferred_center))
            if distance_to_ref + radius >= ref_r * 0.90:
                continue
            if radius < max(min_radius, int(ref_r * 0.06)):
                continue
        edge_support = annulus_edge_support_score(edges, (int(x), int(y), int(radius)))
        comb = _ball_combined_score((int(x), int(y), int(radius)), edge_support, preferred_center, previous_radius)
        scored.append(((int(x), int(y), int(radius)), edge_support, comb))

    if not scored:
        return None

    pre_valid = [(c, es, comb) for c, es, comb in scored if es >= cfg.min_ball_edge_support]
    if not pre_valid:
        return None

    pre_valid.sort(key=lambda item: -item[2])
    distinct: list[tuple[tuple[int, int, int], float, float]] = []
    for circle, edge_support, comb in pre_valid:
        if any(
            math.hypot(circle[0] - d[0], circle[1] - d[1]) < cfg.ball_candidate_dedupe_px
            for d, _, _ in distinct
        ):
            continue
        distinct.append((circle, edge_support, comb))

    valid = distinct
    if not valid:
        return None

    valid.sort(key=lambda item: -item[2])
    best_comb = valid[0][2]
    close_threshold = best_comb * cfg.second_ball_score_ratio
    close = [item for item in valid if item[2] >= close_threshold]

    if len(close) >= 2 and previous_ball is not None:
        chosen = min(
            close,
            key=lambda item: math.hypot(item[0][0] - previous_ball[0], item[0][1] - previous_ball[1]),
        )
    elif len(close) >= 2 and reference_circle is not None:
        rx, ry = reference_circle[0], reference_circle[1]
        chosen = min(close, key=lambda item: math.hypot(item[0][0] - rx, item[0][1] - ry))
    else:
        chosen = valid[0]

    return BallDetection(circle=chosen[0], edge_support=chosen[1], valid_candidates=len(valid))


def _maybe_refine_fullres(
    frame_bgr: np.ndarray,
    circle: tuple[int, int, int] | None,
    cfg: DetectionConfig,
) -> tuple[int, int, int] | None:
    if circle is None or not cfg.refine_with_edge_fit:
        return circle
    gray_full = preprocess_gray(frame_bgr)
    return refine_circle_from_edges(gray_full, circle, band_px=cfg.edge_refine_band_px)


def annotate_frame(
    frame: np.ndarray,
    ball_circle: tuple[int, int, int] | None,
    reference_circle: tuple[int, int, int] | None,
    measurement: SizeMeasurement | None,
    scale_mm_per_px: float | None,
    frame_index: int,
    ball_id: int | None = None,
) -> np.ndarray:
    out = frame.copy()
    if reference_circle is not None:
        x, y, radius = reference_circle
        cv2.circle(out, (x, y), radius, (255, 200, 0), 2)
        cv2.putText(out, "reference", (x - radius, max(30, y - radius - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    if ball_circle is not None:
        x, y, radius = ball_circle
        cv2.circle(out, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
    cv2.putText(out, f"frame={frame_index}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if measurement is not None:
        cv2.putText(out, f"D={measurement.value_mm:.3f} mm", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(
            out,
            f"sigma_model={measurement.sigma_model_mm:.3f} mm",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
    if ball_id is not None:
        cv2.putText(out, f"ball_id={ball_id}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 255, 200), 2)
    if scale_mm_per_px is not None:
        cv2.putText(out, f"scale={scale_mm_per_px:.5f} mm/px", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def _make_output_dir(video_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        path = Path("output") / "analysis" / video_path.stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def _expected_ball_radius_range_px(
    measurement_mode: MeasurementMode,
    detection_cfg: DetectionConfig,
    width_px: int,
    fov_inputs: FovScaleInputs | None,
    reference: ReferenceMeasurement | None,
    reference_circle: tuple[int, int, int] | None,
) -> tuple[int, int] | None:
    if measurement_mode == "fov":
        if fov_inputs is None:
            raise ValueError("fov_inputs is required for fov mode")
        inputs = FovScaleInputs(
            distance_to_object_mm=fov_inputs.distance_to_object_mm,
            zoom_factor=fov_inputs.zoom_factor,
            horizontal_fov_deg=fov_inputs.horizontal_fov_deg,
            resolution_x_px=width_px,
            calibration_coef=fov_inputs.calibration_coef,
        )
        mm_per_px = mm_per_pixel_from_fov(inputs)
        min_radius = int((detection_cfg.min_ball_diameter_mm / 2.0) / mm_per_px)
        max_radius = int((detection_cfg.max_ball_diameter_mm / 2.0) / mm_per_px)
        return max(1, min_radius), max(min_radius + 1, max_radius)
    if reference is None or reference_circle is None:
        return None
    reference_radius_px = float(reference_circle[2])
    min_radius = int(reference_radius_px * detection_cfg.min_ball_diameter_mm / reference.diameter_mm)
    max_radius = int(reference_radius_px * detection_cfg.max_ball_diameter_mm / reference.diameter_mm)
    return max(1, min_radius), max(min_radius + 1, max_radius)


def analyze_video(
    video_cfg: VideoAnalysisConfig,
    detection_cfg: DetectionConfig,
    measurement_mode: MeasurementMode,
    fov_inputs: FovScaleInputs | None = None,
    fov_uncertainty: FovScaleUncertainty | None = None,
    reference: ReferenceMeasurement | None = None,
    expected_ball_diameter_mm: float | None = None,
) -> VideoAnalysisResult:
    video_path = Path(video_cfg.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_dir = _make_output_dir(video_path, video_cfg.output_dir)

    measurements: list[FrameMeasurement] = []
    preview_images: list[np.ndarray] = []
    sampled_frames = 0
    previous_ball: tuple[int, int, int] | None = None
    consecutive_misses = 0
    current_ball_id = -1

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(1, video_cfg.frame_step) != 0:
            frame_index += 1
            continue
        sampled_frames += 1
        if video_cfg.max_frames and sampled_frames > video_cfg.max_frames:
            break

        track_empty = previous_ball is None
        gray, detection_scale = _prepare_detection_frame(frame)
        inverse_scale = 1.0 / detection_scale
        reference_circle_scaled = detect_reference_circle(gray, detection_cfg)
        reference_circle = _rescale_circle(reference_circle_scaled, inverse_scale)
        reference_circle = _maybe_refine_fullres(frame, reference_circle, detection_cfg)
        radius_range = _expected_ball_radius_range_px(
            measurement_mode=measurement_mode,
            detection_cfg=detection_cfg,
            width_px=int(round(width_px * detection_scale)),
            fov_inputs=fov_inputs,
            reference=reference,
            reference_circle=reference_circle_scaled,
        )
        measurement: SizeMeasurement | None = None
        scale_mm_per_px: float | None = None
        ball_circle: tuple[int, int, int] | None = None
        ball_det: BallDetection | None = None

        attempted_ball = radius_range is not None
        if radius_range is not None:
            min_radius, max_radius = radius_range
            previous_ball_scaled = _rescale_circle(previous_ball, detection_scale) if previous_ball is not None else None
            ball_det = detect_ball_circle(
                gray,
                min_radius=min_radius,
                max_radius=max_radius,
                reference_circle=reference_circle_scaled,
                previous_ball=previous_ball_scaled,
                cfg=detection_cfg,
            )
            ball_circle_scaled = ball_det.circle if ball_det is not None else None
            ball_circle = _rescale_circle(ball_circle_scaled, inverse_scale)
            ball_circle = _maybe_refine_fullres(frame, ball_circle, detection_cfg)

        if ball_circle is not None:
            consecutive_misses = 0
            previous_ball = ball_circle
            ball_diameter_px = float(ball_circle[2] * 2)
            if measurement_mode == "fov":
                if fov_inputs is None or fov_uncertainty is None:
                    raise ValueError("fov_inputs and fov_uncertainty are required for fov mode")
                dynamic_inputs = FovScaleInputs(
                    distance_to_object_mm=fov_inputs.distance_to_object_mm,
                    zoom_factor=fov_inputs.zoom_factor,
                    horizontal_fov_deg=fov_inputs.horizontal_fov_deg,
                    resolution_x_px=width_px,
                    calibration_coef=fov_inputs.calibration_coef,
                )
                measurement = diameter_from_pixels_fov(
                    diameter_px=ball_diameter_px,
                    inputs=dynamic_inputs,
                    uncertainty=fov_uncertainty,
                    sigma_detection_px=detection_cfg.sigma_ball_px,
                )
                scale_mm_per_px = mm_per_pixel_from_fov(dynamic_inputs)
            else:
                if reference is None or reference_circle is None:
                    measurement = None
                else:
                    reference_diameter_px = float(reference_circle[2] * 2)
                    measurement = diameter_from_reference(
                        diameter_px=ball_diameter_px,
                        reference_diameter_px=reference_diameter_px,
                        reference=reference,
                        sigma_detection_px=detection_cfg.sigma_ball_px,
                        sigma_reference_px=detection_cfg.sigma_reference_px,
                    )
                    scale_mm_per_px = measurement.value_mm / ball_diameter_px

            if measurement is not None:
                current_ball_id = advance_ball_id(track_empty, current_ball_id)
                measurements.append(
                    FrameMeasurement(
                        frame_index=frame_index,
                        time_s=(frame_index / fps) if fps else 0.0,
                        ball_center_x_px=ball_circle[0],
                        ball_center_y_px=ball_circle[1],
                        ball_radius_px=ball_circle[2],
                        ball_diameter_px=ball_diameter_px,
                        reference_radius_px=reference_circle[2] if reference_circle else None,
                        reference_diameter_px=float(reference_circle[2] * 2) if reference_circle else None,
                        scale_mm_per_px=scale_mm_per_px,
                        diameter_mm=measurement.value_mm,
                        sigma_model_mm=measurement.sigma_model_mm,
                        method=measurement.method,
                        ball_edge_support=float(ball_det.edge_support) if ball_det is not None else 0.0,
                        ball_valid_candidates=int(ball_det.valid_candidates) if ball_det is not None else 0,
                        ball_id=current_ball_id,
                    )
                )
        elif attempted_ball:
            consecutive_misses += 1
            if (
                detection_cfg.clear_track_after_lost_frames > 0
                and consecutive_misses >= detection_cfg.clear_track_after_lost_frames
            ):
                previous_ball = None

        if len(preview_images) < max(0, video_cfg.preview_frames):
            preview_images.append(
                annotate_frame(
                    frame,
                    ball_circle,
                    reference_circle,
                    measurement,
                    scale_mm_per_px,
                    frame_index,
                    ball_id=current_ball_id if measurement is not None else None,
                )
            )

        frame_index += 1

    cap.release()

    diameters = [item.diameter_mm for item in measurements]
    sigmas = [item.sigma_model_mm for item in measurements]
    scales = [item.scale_mm_per_px for item in measurements]
    reference_diameters = [item.reference_diameter_px for item in measurements if item.reference_diameter_px is not None]

    diameter_mean = float(np.mean(diameters)) if diameters else None
    diameter_median = float(np.median(diameters)) if diameters else None
    diameter_std = float(np.std(diameters, ddof=1)) if len(diameters) > 1 else (0.0 if diameters else None)
    sigma_model_mean = float(np.mean(sigmas)) if sigmas else None
    combined_sigma_single = (
        math.sqrt((sigma_model_mean or 0.0) ** 2 + (diameter_std or 0.0) ** 2) if sigma_model_mean is not None else None
    )
    combined_sigma_mean = (
        math.sqrt((sigma_model_mean or 0.0) ** 2 + ((diameter_std or 0.0) / math.sqrt(len(diameters))) ** 2)
        if sigma_model_mean is not None and diameters
        else None
    )
    bias_to_expected = (
        diameter_mean - expected_ball_diameter_mm
        if diameter_mean is not None and expected_ball_diameter_mm and expected_ball_diameter_mm > 0
        else None
    )

    summary = AnalysisSummary(
        video_path=str(video_path),
        output_dir=str(output_dir),
        measurement_mode=measurement_mode,
        total_frames=total_frames,
        sampled_frames=sampled_frames,
        detections=len(measurements),
        detection_ratio=(len(measurements) / sampled_frames) if sampled_frames else 0.0,
        fps=fps,
        width_px=width_px,
        height_px=height_px,
        scale_mm_per_px_mean=float(np.mean(scales)) if scales else None,
        diameter_mm_mean=diameter_mean,
        diameter_mm_median=diameter_median,
        diameter_mm_std=diameter_std,
        diameter_mm_min=float(np.min(diameters)) if diameters else None,
        diameter_mm_max=float(np.max(diameters)) if diameters else None,
        sigma_model_mm_mean=sigma_model_mean,
        combined_sigma_single_mm=combined_sigma_single,
        combined_sigma_mean_mm=combined_sigma_mean,
        reference_diameter_px_mean=float(np.mean(reference_diameters)) if reference_diameters else None,
        expected_ball_diameter_mm=expected_ball_diameter_mm if expected_ball_diameter_mm and expected_ball_diameter_mm > 0 else None,
        bias_to_expected_mm=bias_to_expected,
    )
    return VideoAnalysisResult(summary=summary, measurements=measurements, preview_images=preview_images)


def run_live_preview(
    video_cfg: VideoAnalysisConfig,
    detection_cfg: DetectionConfig,
    measurement_mode: MeasurementMode,
    fov_inputs: FovScaleInputs | None = None,
    fov_uncertainty: FovScaleUncertainty | None = None,
    reference: ReferenceMeasurement | None = None,
) -> None:
    video_path = Path(video_cfg.video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    previous_ball: tuple[int, int, int] | None = None
    consecutive_misses = 0
    frame_index = 0
    current_ball_id = -1
    live: object | None = None
    if video_cfg.live_plots:
        from mvb.live_plots import LiveMetricsPlot

        live = LiveMetricsPlot(measurement_mode, max_points=video_cfg.live_plot_max_points)

    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        track_empty = previous_ball is None
        gray, detection_scale = _prepare_detection_frame(frame)
        inverse_scale = 1.0 / detection_scale
        reference_circle_scaled = detect_reference_circle(gray, detection_cfg)
        reference_circle = _rescale_circle(reference_circle_scaled, inverse_scale)
        reference_circle = _maybe_refine_fullres(frame, reference_circle, detection_cfg)
        radius_range = _expected_ball_radius_range_px(
            measurement_mode=measurement_mode,
            detection_cfg=detection_cfg,
            width_px=int(round(width_px * detection_scale)),
            fov_inputs=fov_inputs,
            reference=reference,
            reference_circle=reference_circle_scaled,
        )
        ball_circle = None
        measurement = None
        scale_mm_per_px = None
        attempted = radius_range is not None
        if radius_range is not None:
            previous_ball_scaled = _rescale_circle(previous_ball, detection_scale) if previous_ball is not None else None
            ball_det_preview = detect_ball_circle(
                gray,
                radius_range[0],
                radius_range[1],
                reference_circle_scaled,
                previous_ball_scaled,
                detection_cfg,
            )
            ball_circle_scaled = ball_det_preview.circle if ball_det_preview is not None else None
            ball_circle = _rescale_circle(ball_circle_scaled, inverse_scale)
            ball_circle = _maybe_refine_fullres(frame, ball_circle, detection_cfg)
        if ball_circle is not None:
            consecutive_misses = 0
            previous_ball = ball_circle
            diameter_px = float(ball_circle[2] * 2)
            if measurement_mode == "fov":
                if fov_inputs is None or fov_uncertainty is None:
                    raise ValueError("fov_inputs and fov_uncertainty are required for fov mode")
                measurement = diameter_from_pixels_fov(
                    diameter_px=diameter_px,
                    inputs=FovScaleInputs(
                        distance_to_object_mm=fov_inputs.distance_to_object_mm,
                        zoom_factor=fov_inputs.zoom_factor,
                        horizontal_fov_deg=fov_inputs.horizontal_fov_deg,
                        resolution_x_px=width_px,
                        calibration_coef=fov_inputs.calibration_coef,
                    ),
                    uncertainty=fov_uncertainty,
                    sigma_detection_px=detection_cfg.sigma_ball_px,
                )
                scale_mm_per_px = measurement.value_mm / diameter_px
            elif reference is not None and reference_circle is not None:
                measurement = diameter_from_reference(
                    diameter_px=diameter_px,
                    reference_diameter_px=float(reference_circle[2] * 2),
                    reference=reference,
                    sigma_detection_px=detection_cfg.sigma_ball_px,
                    sigma_reference_px=detection_cfg.sigma_reference_px,
                )
                scale_mm_per_px = measurement.value_mm / diameter_px
            if measurement is not None:
                current_ball_id = advance_ball_id(track_empty, current_ball_id)
        elif attempted:
            consecutive_misses += 1
            if (
                detection_cfg.clear_track_after_lost_frames > 0
                and consecutive_misses >= detection_cfg.clear_track_after_lost_frames
            ):
                previous_ball = None

        if ball_circle is not None and measurement is not None and live is not None and not live.closed:
            time_s = (frame_index / fps) if fps else 0.0
            if measurement_mode == "reference" and reference_circle is not None:
                sec = float(reference_circle[2] * 2)
            else:
                sec = float(scale_mm_per_px) if scale_mm_per_px is not None else 0.0
            live.push(time_s, measurement.value_mm, current_ball_id, sec)
            live.refresh()

        annotated = annotate_frame(
            frame,
            ball_circle,
            reference_circle,
            measurement,
            scale_mm_per_px,
            frame_index,
            ball_id=current_ball_id if measurement is not None else None,
        )
        display = cv2.resize(
            annotated,
            (int(annotated.shape[1] * video_cfg.display_scale), int(annotated.shape[0] * video_cfg.display_scale)),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow("detection", display)
        frame_index += 1
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if live is not None:
        live.close()
