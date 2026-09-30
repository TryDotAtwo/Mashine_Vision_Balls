"""Multi-object video analysis, scalar calibration and reproducible series exports."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
from scipy.optimize import linear_sum_assignment

import cv2
import numpy as np

from mvb.cv_geometry import annulus_edge_support_score, refine_circle_from_edges
from mvb.experiment_math import diameter_from_pixels_fov, diameter_from_reference, mm_per_pixel_from_fov
from mvb.pipeline import (
    FrameMeasurement, _prepare_detection_frame, _rescale_circle,
    _prepare_edges_for_scoring, preprocess_gray, detect_reference_circle,
)
from mvb.reporting import write_measurements_csv, write_preview_sheet


class CircleTracker:
    """Global assignment with constant-velocity prediction and size gating."""
    def __init__(self, max_distance_px=60.0, max_missed=3):
        if not math.isfinite(max_distance_px) or max_distance_px <= 0 or max_missed < 0:
            raise ValueError("Invalid tracking limits")
        self.distance = max_distance_px
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 0
        self.velocities = {}

    def update(self, circles):
        track_ids = sorted(self.tracks)
        cost = np.full((len(track_ids), len(circles) + len(track_ids)), self.distance * 1.6)
        for row_index, tid in enumerate(track_ids):
            old, missed = self.tracks[tid]
            vx, vy = self.velocities.get(tid, (0.0, 0.0))
            predicted = (old[0] + vx * (missed + 1), old[1] + vy * (missed + 1))
            for idx, new in enumerate(circles):
                distance = math.hypot(predicted[0] - new[0], predicted[1] - new[1])
                if missed and distance > self.distance:
                    # A wall bounce during a gap invalidates constant-velocity extrapolation.
                    distance = math.hypot(old[0] - new[0], old[1] - new[1]) + self.distance * 0.25
                radius_ratio = new[2] / old[2]
                cost[row_index, idx] = (distance + self.distance * abs(math.log(radius_ratio))
                                        if distance <= self.distance and 0.65 <= radius_ratio <= 1.5 else self.distance * 1e6)
        assignments = {}
        used_tracks = set()
        if track_ids:
            rows, columns = linear_sum_assignment(cost)
            for row_index, idx in zip(rows, columns):
                tid = track_ids[row_index]
                if idx >= len(circles) or cost[row_index, idx] >= self.distance * 1.6:
                    continue
                assignments[idx] = tid
                used_tracks.add(tid)
        updated = {}
        for tid, (circle, missed) in self.tracks.items():
            if tid not in used_tracks and missed + 1 <= self.max_missed:
                updated[tid] = (circle, missed + 1)
        ids = []
        for idx, circle in enumerate(circles):
            tid = assignments.get(idx)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.velocities[tid] = (0.0, 0.0)
            else:
                old, missed = self.tracks[tid]
                self.velocities[tid] = ((circle[0] - old[0]) / (missed + 1),
                                       (circle[1] - old[1]) / (missed + 1))
            updated[tid] = (circle, 0)
            ids.append(tid)
        self.tracks = updated
        self.velocities = {tid: self.velocities[tid] for tid in updated}
        return ids


def detect_all_circles(gray, min_radius, max_radius, reference, cfg):
    params = cfg.hough_param2_ball if cfg.adaptive_hough else cfg.hough_param2_ball[:1]
    candidates = []
    for threshold in params:
        detected = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=max(4, 2 * min_radius), param1=120, param2=threshold,
                                   minRadius=min_radius, maxRadius=max_radius)
        if detected is not None:
            candidates.extend(tuple(int(v) for v in c) for c in np.round(detected[0]).astype(int))
    edges = _prepare_edges_for_scoring(gray)
    accepted = []
    for circle in candidates:
        if reference is not None:
            if math.hypot(circle[0] - reference[0], circle[1] - reference[1]) + circle[2] > reference[2] + 1:
                continue
        score = annulus_edge_support_score(edges, circle)
        if score >= cfg.min_ball_edge_support:
            accepted.append((circle, score))
    distinct = []
    for circle, score in sorted(accepted, key=lambda item: -item[1]):
        # Suppress concentric detections, preserving nearby distinct small balls.
        if any(math.hypot(circle[0] - old[0], circle[1] - old[1]) <
               max(cfg.ball_candidate_dedupe_px, 0.6 * min(circle[2], old[2]))
               for old, _ in distinct):
            continue
        distinct.append((circle, score))
    return sorted(distinct, key=lambda item: (item[0][0], item[0][1]))


def calibration_from_csv(csv_path, diameter_mm, sigma_mm, output_path, width_px=None):
    """CSV must contain ONLY observations of one known reference object."""
    if not math.isfinite(diameter_mm) or diameter_mm <= 0 or not math.isfinite(sigma_mm) or sigma_mm < 0:
        raise ValueError("Reference diameter must be positive; uncertainty nonnegative")
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pixels = [float(row["ball_diameter_px"]) for row in rows]
    if len(pixels) < 2 or any(not math.isfinite(p) or p <= 0 for p in pixels):
        raise ValueError("Need at least two finite positive reference observations")
    ids = {row.get("ball_id", "0") for row in rows}
    if len(ids) != 1:
        raise ValueError("Calibration CSV must contain one reference track only")
    if width_px is None:
        for filename in ("run.json", "summary.json"):
            metadata = Path(csv_path).parent / filename
            if metadata.exists():
                width_px = json.loads(metadata.read_text(encoding="utf-8")).get("width_px")
                if width_px:
                    break
    if width_px is None or not math.isfinite(width_px) or width_px < 1 or int(width_px) != width_px:
        raise ValueError("Calibration needs image width: supply --calibration-width-px or keep source run.json")
    mean_px = float(np.mean(pixels))
    std_px = float(np.std(pixels, ddof=1))
    scale = diameter_mm / mean_px
    # Keep observed repeatability; do not reduce correlated frames by sqrt(n).
    sigma_scale = scale * math.sqrt((sigma_mm / diameter_mm)**2 + (std_px / mean_px)**2)
    result = {"schema": "mvb.scalar_calibration.v1", "width_px": int(width_px), "scale_mm_per_px": scale,
              "sigma_scale_mm_per_px": sigma_scale, "reference_diameter_mm": diameter_mm,
              "reference_sigma_mm": sigma_mm, "observations": len(pixels),
              "mean_diameter_px": mean_px, "std_diameter_px": std_px,
              "source_csv": str(Path(csv_path).resolve()),
              "source_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
              "limitations": "Scalar scale for unchanged optical geometry; no distortion/refraction correction."}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_calibration(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("schema") != "mvb.scalar_calibration.v1":
        raise ValueError("Unsupported calibration schema")
    scale = float(result["scale_mm_per_px"])
    sigma = float(result["sigma_scale_mm_per_px"])
    width = result.get("width_px", 0)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(sigma) or sigma < 0 or width < 1:
        raise ValueError("Invalid calibration scale or uncertainty")
    return result


def summarize_tracks(measurements):
    grouped = {}
    for row in measurements:
        grouped.setdefault(row.ball_id, []).append(row)
    result = []
    for tid, rows in sorted(grouped.items()):
        ds = [r.diameter_mm for r in rows]
        result.append({"ball_id": tid, "observations": len(rows),
                       "time_start_s": rows[0].time_s, "time_end_s": rows[-1].time_s,
                       "diameter_mm_median": float(np.median(ds)),
                       "diameter_mm_mean": float(np.mean(ds)),
                       "diameter_mm_std": float(np.std(ds, ddof=1)) if len(ds) > 1 else None,
                       "sigma_model_mm_mean": float(np.mean([r.sigma_model_mm for r in rows]))})
    return result


def write_series_plots(output_dir, measurements, tracks):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(11, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for track in tracks:
        rows = [r for r in measurements if r.ball_id == track["ball_id"]]
        axes[0, 0].plot([r.time_s for r in rows], [r.diameter_mm for r in rows], ".-", label=str(track["ball_id"]))
        axes[1, 0].plot([r.time_s for r in rows], [r.sigma_model_mm for r in rows], ".-")
    axes[0, 0].set(xlabel="Time, s", ylabel="Diameter, mm", title="Tracked objects")
    if 0 < len(tracks) <= 10:
        axes[0, 0].legend(title="Track ID")
    axes[0, 1].hist([r["diameter_mm_median"] for r in tracks], bins="auto")
    axes[0, 1].set(xlabel="Median diameter, mm", ylabel="Tracks", title="One observation per track")
    axes[1, 0].set(xlabel="Time, s", ylabel="Model sigma, mm")
    axes[1, 1].plot([r.time_s for r in measurements], [r.scale_mm_per_px for r in measurements], ".")
    axes[1, 1].set(xlabel="Time, s", ylabel="Scale, mm/px", title="Optical scale")
    figure.savefig(str(output_dir / "series_plots.png"), dpi=150)


def analyze_series(video_cfg, detection_cfg, measurement_mode, fov_inputs=None,
                   fov_uncertainty=None, reference=None, calibration_path=None,
                   tracking_distance_px=60.0, tracking_max_missed=3, min_track_observations=2,
                   expected_ball_diameter_mm=None, cell_mask=True, storage=None):
    if storage is not None:
        storage.initialize()
    if min_track_observations < 1:
        raise ValueError("min_track_observations must be positive")
    if video_cfg.frame_step < 1 or video_cfg.max_frames < 0:
        raise ValueError("Invalid frame sampling limits")
    if detection_cfg.min_ball_diameter_mm <= 0 or detection_cfg.max_ball_diameter_mm <= detection_cfg.min_ball_diameter_mm:
        raise ValueError("Invalid diameter range")
    calibration = load_calibration(calibration_path) if calibration_path else None
    if not calibration:
        if measurement_mode == "reference":
            if reference is None or not math.isfinite(reference.diameter_mm) or reference.diameter_mm <= 0:
                raise ValueError("Reference mode requires a positive known diameter")
        elif measurement_mode == "fov":
            if fov_inputs is None or fov_uncertainty is None or not 0 < fov_inputs.horizontal_fov_deg < 180:
                raise ValueError("FOV mode needs geometry and uncertainty; angle must be between 0 and 180 degrees")
        else:
            raise ValueError("Unknown measurement mode")
    source = Path(video_cfg.video_path)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if not math.isfinite(fps) or fps <= 0 or width <= 0:
        cap.release()
        raise ValueError("Video has no valid FPS or width")
    if calibration and calibration["width_px"] != width:
        cap.release()
        raise ValueError("Video width differs from calibration; recalibrate for this resolution")
    out = Path(video_cfg.output_dir or Path("output") / "analysis" / source.stem)
    out.mkdir(parents=True, exist_ok=True)
    tracker = CircleTracker(tracking_distance_px, tracking_max_missed)
    measurements, previews, frame_rows = [], [], []
    frame_index, sampled, detected_frames = 0, 0, 0
    try:
        while not video_cfg.max_frames or sampled < video_cfg.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            index = frame_index
            frame_index += 1
            if index % video_cfg.frame_step:
                continue
            sampled += 1
            gray, factor = _prepare_detection_frame(frame)
            gray_full = preprocess_gray(frame) if detection_cfg.refine_with_edge_fit else None
            def refine(circle):
                if circle is None or gray_full is None:
                    return circle
                return refine_circle_from_edges(gray_full, circle, detection_cfg.edge_refine_band_px, round_result=False)
            ref_small = detect_reference_circle(gray, detection_cfg) if cell_mask or (measurement_mode == "reference" and not calibration) else None
            ref_full = refine(_rescale_circle(ref_small, 1 / factor))
            dynamic = replace(fov_inputs, resolution_x_px=width) if fov_inputs else None
            if calibration:
                scale = calibration["scale_mm_per_px"]
            elif measurement_mode == "reference":
                scale = reference.diameter_mm / (2 * ref_full[2]) if ref_full else None
            else:
                scale = mm_per_pixel_from_fov(dynamic)
            candidates = []
            if scale is not None:
                lo = max(1, int(detection_cfg.min_ball_diameter_mm / scale * factor / 2))
                hi = max(lo + 1, int(detection_cfg.max_ball_diameter_mm / scale * factor / 2))
                candidates = detect_all_circles(gray, lo, hi, ref_small, detection_cfg)
            circles = [refine(_rescale_circle(c, 1 / factor)) for c, _ in candidates]
            ids = tracker.update(circles)
            annotated = frame.copy()
            if circles:
                detected_frames += 1
            frame_rows.append({"frame_index": index, "time_s": index / fps,
                               "objects": len(circles), "scale_available": scale is not None})
            for tid, circle, (_, score) in zip(ids, circles, candidates):
                px = 2.0 * circle[2]
                if calibration:
                    diameter = px * scale
                    sigma = math.hypot(px * calibration["sigma_scale_mm_per_px"], scale * detection_cfg.sigma_ball_px)
                    method = "calibrated"
                else:
                    measured = (diameter_from_reference(px, 2.0 * ref_full[2], reference, detection_cfg.sigma_ball_px,
                                                       detection_cfg.sigma_reference_px) if measurement_mode == "reference"
                                else diameter_from_pixels_fov(px, dynamic, fov_uncertainty, detection_cfg.sigma_ball_px))
                    diameter, sigma, method = measured.value_mm, measured.sigma_model_mm, measured.method
                row = FrameMeasurement(index, index / fps, circle[0], circle[1], circle[2], px,
                                       ref_full[2] if ref_full else None, 2.0 * ref_full[2] if ref_full else None,
                                       scale, diameter, sigma, method, score, len(circles), tid)
                measurements.append(row)
                drawing_circle = tuple(int(round(v)) for v in circle)
                cv2.circle(annotated, drawing_circle[:2], drawing_circle[2], (0, 255, 0), 2)
                cv2.putText(annotated, f"#{tid} {diameter:.3f} mm", (drawing_circle[0], max(20, drawing_circle[1] - drawing_circle[2])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if len(previews) < video_cfg.preview_frames:
                previews.append(annotated)
    finally:
        cap.release()
    tracks = summarize_tracks(measurements)
    eligible = [r for r in tracks if r["observations"] >= min_track_observations]
    values = [t["diameter_mm_median"] for t in eligible]
    summary = {"sampled_frames": sampled, "frames_with_detections": detected_frames,
               "detection_ratio": detected_frames / sampled if sampled else 0,
               "measurements": len(measurements), "tracks": len(tracks), "eligible_tracks": len(eligible),
               "min_track_observations": min_track_observations,
               "track_diameter_median_mm": float(np.median(values)) if values else None,
               "track_diameter_std_mm": float(np.std(values, ddof=1)) if len(values) > 1 else None,
               "limitations": "Track IDs can split/swap at occlusion or crossing; model uncertainty is not validated accuracy."}
    if expected_ball_diameter_mm is not None and expected_ball_diameter_mm > 0:
        summary["expected_ball_diameter_mm"] = expected_ball_diameter_mm
        summary["bias_to_expected_mm"] = float(np.mean(values)) - expected_ball_diameter_mm if values else None
    provenance = {"schema": "mvb.series.v1", "video": str(source.resolve()), "fps": fps, "width_px": width,
                  "video_config": asdict(video_cfg), "detection_config": asdict(detection_cfg),
                  "measurement_mode": "calibrated" if calibration else measurement_mode,
                  "fov_inputs": asdict(fov_inputs) if fov_inputs else None,
                  "fov_uncertainty": asdict(fov_uncertainty) if fov_uncertainty else None,
                  "reference": asdict(reference) if reference else None, "calibration": calibration,
                  "tracking_distance_px": tracking_distance_px, "tracking_max_missed": tracking_max_missed,
                  "min_track_observations": min_track_observations,
                  "cell_mask": cell_mask,
                  "versions": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__}}
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    provenance["video_sha256"] = digest.hexdigest()
    provenance["source_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ("series_analysis.py", "pipeline.py", "cv_geometry.py", "experiment_math.py", "reporting.py", "clickhouse_store.py")}
    if storage is not None:
        run_id = storage.save(measurements, frame_rows, provenance, summary)
        measurements, frame_rows, tracks, summary, provenance = storage.load(run_id)
        eligible = [r for r in tracks if int(r["observations"]) >= min_track_observations]
    else:
        summary["storage"] = "offline"
    export_series_tables(out, measurements, frame_rows, tracks, summary, provenance)
    write_preview_sheet(out / "preview_sheet.jpg", previews)
    write_series_plots(out, measurements, eligible)
    return summary


def export_series_tables(out, measurements, frame_rows, tracks, summary, provenance):
    write_measurements_csv(out / "measurements.csv", measurements)
    for name, rows, fields in [("tracks.csv", tracks, ["ball_id", "observations", "time_start_s", "time_end_s", "diameter_mm_median", "diameter_mm_mean", "diameter_mm_std", "sigma_model_mm_mean"]),
                               ("frames.csv", frame_rows, ["frame_index", "time_s", "objects", "scale_available"])]:
        with (out / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "run.json").write_text(json.dumps(dict(provenance, storage=summary.get("storage"), run_id=summary.get("run_id")), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text("# Анализ серии\n\n" + "\n".join(f"- {k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8")
