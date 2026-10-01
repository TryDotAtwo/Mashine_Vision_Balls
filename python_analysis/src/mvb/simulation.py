"""Literature-informed forward scene model. This is NOT a validated cryostat CFD model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib
import json
import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from mvb.pipeline import DetectionConfig, VideoAnalysisConfig
from mvb.series_analysis import analyze_series, calibration_from_csv


SOURCES = {
    "jinr_2024": {"url": "https://www1.jinr.ru/Preprints/2024/060(P18-2024-60).pdf",
                  "used": "printed pp. 1-2 diameter intervals; eq.13 film thickness; eq.5 cooling; table P.1 material properties"},
    "gauthier_2019": {"url": "https://arxiv.org/html/1901.04218",
                      "used": "eq.5 propulsion/drag model, reflections; fitted alpha=beta=15 and delta_h=1.45 micrometres concern ethanol/silicone oil"},
    "local_echaya": {"path": "Статья в ЭЧАЯ/Using of frozen mesitylene beads with Rupert's droplet properties in cryogenic neutron moderators_M.V.Bulavin_Письма в ЭЧАЯ_final_rus (1).pdf",
                     "used": "pp.1,3-6: millimetre scale, separated cells, motion, internal cavity and light dispersion"},
}


@dataclass
class SceneConfig:
    width: int = 400
    height: int = 400
    fps: float = 30.0
    frames: int = 24
    field_mm: float = 40.0
    distance_mm: float = 50.0
    bath_radius_mm: float = 17.0
    noise_sigma: float = 0.0
    blur_sigma: float = 0.6
    exposure_s: float = 0.002
    scale_drift: float = 0.0
    radial_k1: float = 0.0
    elliptical_ratio: float = 1.0
    highlight: bool = False


@dataclass
class Particle:
    object_id: int
    diameter_mm: float
    x_mm: float
    y_mm: float
    direction: float
    speed_m_s: float = 0.025
    temperature_k: float = 300.0

    def advance(self, dt, bath_radius_mm):
        # JINR table P.1; constants are kept fixed as in its reduced model.
        rho, cp, mu, conductivity = 861.12, 1750.0, 5.52e-6, 7.26e-3
        rho_v, latent, g, bath_temperature = 4.5, 200000.0, 9.81, 77.4
        radius = self.diameter_mm / 2000.0
        delta_t = self.temperature_k - bath_temperature
        film = (9 * conductivity * mu * radius * delta_t / (rho * rho_v * g * latent)) ** 0.25
        mass = rho * 4 * math.pi * radius**3 / 3
        # Cross-material extrapolation of Gauthier eq.5; NOT an empirical mesitylene fit.
        force = 15 * rho * g * radius**2 * 1.45e-6 - 15 * mu * self.speed_m_s * radius**2 / film
        self.speed_m_s = max(0.0, self.speed_m_s + force / mass * dt)
        area = 0.5 * 4 * math.pi * radius**2
        self.temperature_k -= conductivity * area * delta_t / (mass * cp * film) * dt
        velocity = np.array([math.cos(self.direction), math.sin(self.direction)]) * self.speed_m_s * 1000
        position = np.array([self.x_mm, self.y_mm]) + velocity * dt
        allowed = bath_radius_mm - self.diameter_mm / 2
        if np.linalg.norm(position) > allowed:
            normal = position / np.linalg.norm(position)
            position = normal * (2 * allowed - np.linalg.norm(position))
            velocity -= 2 * np.dot(velocity, normal) * normal
            self.direction = math.atan2(velocity[1], velocity[0])
        self.x_mm, self.y_mm = position


def _project(particle, cfg, t):
    depth = cfg.distance_mm * (1 + cfg.scale_drift * math.sin(2 * math.pi * t))
    focal_px = cfg.width * cfg.distance_mm / cfg.field_mm
    xn, yn = particle.x_mm / depth, particle.y_mm / depth
    distortion = 1 + cfg.radial_k1 * (xn*xn + yn*yn)
    x = cfg.width / 2 + focal_px * xn * distortion
    y = cfg.height / 2 + focal_px * yn * distortion
    # Local radial magnification along the radius; small-object approximation.
    radius = focal_px * particle.diameter_mm / (2 * depth) * (1 + 3 * cfg.radial_k1 * (xn*xn + yn*yn))
    return x, y, radius, focal_px / depth


def generate_scene(output_dir, name, seed=900, frames=24):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = SceneConfig(frames=frames)
    if not 12 <= frames <= 60:
        raise ValueError("Use 12..60 frames: this reduced liquid-stage model is restricted to <=2 seconds")
    particles = [Particle(0, 3.7, -7, -4, 0.35)]
    if name in ("two", "crossing", "missing"):
        particles = [Particle(0, 3.2, -8, -1, 0), Particle(1, 4.2, 8, 1, math.pi)]
    if name == "wall":
        particles[0].x_mm, particles[0].y_mm, particles[0].direction = 14, 0, 0
    if name == "noise_blur":
        cfg.noise_sigma, cfg.blur_sigma, cfg.exposure_s, cfg.highlight = 6, 1.2, 0.012, True
    if name == "scale_drift":
        cfg.scale_drift = 0.06
    if name == "deformed":
        cfg.elliptical_ratio = 0.80
    if name == "empty":
        particles = []
    if name == "nuisance":
        cfg.noise_sigma = 3
    if name == "reference":
        particles = [Particle(0, 3.7, 0, 0, 0, speed_m_s=0)]
    if name not in ("single", "two", "crossing", "missing", "wall", "noise_blur", "scale_drift", "deformed", "empty", "nuisance", "reference"):
        raise ValueError("Unknown simulation scenario")
    if name != "reference":
        for particle in particles:
            particle.diameter_mm = float(np.clip(particle.diameter_mm + rng.uniform(-0.04, 0.04), 3.2, 4.2))
            particle.x_mm += rng.uniform(-0.4, 0.4)
            particle.y_mm += rng.uniform(-0.4, 0.4)
            particle.direction += rng.uniform(-0.08, 0.08)
    path = output_dir / "scene.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), cfg.fps, (cfg.width, cfg.height))
    if not writer.isOpened():
        raise RuntimeError("MJPG video writer unavailable")
    yy, xx = np.indices((cfg.height, cfg.width), dtype=float)
    bath_radius_px = cfg.bath_radius_mm * cfg.width / cfg.field_mm
    bath = ((xx - cfg.width/2)**2 + (yy - cfg.height/2)**2) <= bath_radius_px**2
    truth = []
    initial = [asdict(p) for p in particles]
    try:
        for frame_index in range(cfg.frames):
            t = frame_index / cfg.fps
            background = 40 + 12 * xx / cfg.width + 3 * np.sin(xx / 18) * np.cos(yy / 21)
            background[~bath] = 105
            cv2.circle(background, (cfg.width//2, cfg.height//2), int(bath_radius_px), 185, 4)
            label = np.full((cfg.height, cfg.width), -1, dtype=np.int16)
            objects = []
            masks = []
            for particle in particles:
                if name == "crossing":
                    particle.x_mm = (-8 + 22*t) if particle.object_id == 0 else (8 - 22*t)
                    particle.y_mm = -0.4 if particle.object_id == 0 else 0.4
                x, y, radius, pixels_per_mm = _project(particle, cfg, t)
                mask = ((xx - x)**2 + ((yy - y) / cfg.elliptical_ratio)**2) <= radius**2
                hidden = name == "missing" and particle.object_id == 0 and 8 <= frame_index <= 10
                if not hidden:
                    # Anti-aliased boundary plus radial brightness shading.
                    distance = np.sqrt((xx - x)**2 + ((yy - y) / cfg.elliptical_ratio)**2)
                    alpha = np.clip(radius + 0.5 - distance, 0, 1)
                    brightness = 165 + 35 * np.clip(1 - distance / max(radius, 1), 0, 1)
                    background = background * (1-alpha) + brightness * alpha
                    label[mask] = particle.object_id
                    if cfg.highlight:
                        highlight = np.exp(-((xx-x+radius*0.3)**2 + (yy-y+radius*0.35)**2) / (radius*0.18)**2)
                        background += 65 * highlight * mask
                masks.append(mask)
                objects.append({"object_id": particle.object_id, "x_px": x, "y_px": y,
                                "diameter_px": radius * 2, "diameter_mm": particle.diameter_mm,
                                "temperature_k": particle.temperature_k, "speed_m_s": particle.speed_m_s,
                                "pixels_per_mm": pixels_per_mm, "hidden": hidden})
            for obj, mask in zip(objects, masks):
                visible_fraction = float(np.sum(label[mask] == obj["object_id"])) / max(1, int(np.sum(mask)))
                obj["visible_fraction"] = visible_fraction
                obj["score_detection"] = visible_fraction >= 0.65
            if name == "nuisance":
                cv2.circle(background, (120, 120), 9, 175, 2)  # undersized bubble
                cv2.line(background, (90, 280), (180, 300), 160, 2)
            if cfg.blur_sigma:
                background = cv2.GaussianBlur(background, (0, 0), cfg.blur_sigma)
            # Exposure approximation: image displacement from the fastest object.
            motion_px = max((p.speed_m_s for p in particles), default=0) * 1000 * cfg.exposure_s * cfg.width / cfg.field_mm
            if motion_px >= 1:
                kernel = np.ones((1, max(1, int(round(motion_px)))))
                background = cv2.filter2D(background, -1, kernel / kernel.size)
            background += rng.normal(0, cfg.noise_sigma, background.shape)
            frame = np.clip(background, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            truth.append({"frame_index": frame_index, "time_s": t, "objects": objects})
            if frame_index in (0, cfg.frames//2, cfg.frames-1):
                cv2.imencode(".png", frame)[1].tofile(str(output_dir / f"frame_{frame_index:04d}.png"))
                cv2.imencode(".png", (label + 1).astype(np.uint16))[1].tofile(str(output_dir / f"labels_{frame_index:04d}.png"))
            if name != "crossing":
                for _ in range(4):
                    for particle in particles:
                        particle.advance(1 / cfg.fps / 4, cfg.bath_radius_mm)
    finally:
        writer.release()
    metadata = {"schema": "mvb.simulation.v1", "scenario": name, "seed": seed, "config": asdict(cfg),
                "initial_particles": initial, "sources": SOURCES,
                "assumptions": ["Prescribed sphere projections and approximate shading; no optical ray tracing",
                                "Material constants from JINR table P.1 are treated as fixed",
                                "Propulsion fit transferred from other liquids solely to generate test trajectories",
                                "No solidification, adhesion, particle collisions or CFD; crossing is a projection stress test",
                                "Field, FPS, depth, noise, blur and deformation are chosen test parameters, not measured camera data"],
                "visibility_threshold": 0.65, "truth": truth}
    metadata["video_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "truth.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, metadata


def evaluate_measurements(metadata, csv_path):
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    by_frame = {}
    for record in records:
        by_frame.setdefault(int(record["frame_index"]), []).append(record)
    tp = fp = fn = switches = eligible_truth = ignored = 0
    diameter_errors, center_errors = [], []
    previous = {}
    matches = []
    for frame in metadata["truth"]:
        targets = [obj for obj in frame["objects"] if obj["score_detection"]]
        excluded = [obj for obj in frame["objects"] if not obj["score_detection"]]
        predictions = by_frame.get(frame["frame_index"], [])
        eligible_truth += len(targets)
        used = set()
        cost = np.full((len(targets), len(predictions) + len(targets)), 1e3)
        for i, target in enumerate(targets):
            for j, pred in enumerate(predictions):
                distance = math.hypot(float(pred["ball_center_x_px"]) - target["x_px"],
                                      float(pred["ball_center_y_px"]) - target["y_px"])
                cost[i, j] = distance / target["diameter_px"] if distance <= target["diameter_px"] * 0.6 else 1e6
        if targets:
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows, columns):
                if j >= len(predictions) or cost[i, j] >= 1e3:
                    continue
                target, pred = targets[i], predictions[j]
                used.add(j)
                tp += 1
                identity = int(pred["ball_id"])
                if target["object_id"] in previous and previous[target["object_id"]] != identity:
                    switches += 1
                previous[target["object_id"]] = identity
                error = float(pred["diameter_mm"]) - target["diameter_mm"]
                diameter_errors.append(error)
                center_errors.append(cost[i, j] * target["diameter_px"])
                matches.append({"frame_index": frame["frame_index"], "truth_id": target["object_id"],
                                "track_id": identity, "diameter_error_mm": error})
        matched = len(used)
        fn += len(targets) - matched
        for j, pred in enumerate(predictions):
            if j in used:
                continue
            near_occluded = any(math.hypot(float(pred["ball_center_x_px"])-obj["x_px"],
                                          float(pred["ball_center_y_px"])-obj["y_px"]) <= obj["diameter_px"] * 0.6
                                for obj in excluded if not obj["hidden"])
            if near_occluded:
                ignored += 1
            else:
                fp += 1
    return {"tp": tp, "fp": fp, "fn": fn, "visible_truth_observations": eligible_truth,
            "ignored_occluded_predictions": ignored, "id_switches": switches,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "diameter_bias_mm": float(np.mean(diameter_errors)) if diameter_errors else None,
            "diameter_rmse_mm": float(np.sqrt(np.mean(np.square(diameter_errors)))) if diameter_errors else None,
            "center_rmse_px": float(np.sqrt(np.mean(np.square(center_errors)))) if center_errors else None,
            "matched_observations": len(matches), "matches": matches}


def run_validation(output_dir, seed=900, frames=24, edge_fit_method="kasa"):
    root = Path(output_dir)
    # Preserve earlier runs and their manifests.
    if root.exists() and any(root.iterdir()):
        raise ValueError("Simulation output directory is not empty; choose a new run directory")
    root.mkdir(parents=True, exist_ok=True)
    frozen = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ("simulation.py", "series_analysis.py", "cv_geometry.py", "pipeline.py", "experiment_math.py")}
    # Independent reference sequence. Pixel measurements, rather than truth scale,
    # determine the calibration applied to the scored scenes.
    reference_dir = root / "calibration_reference"
    reference_video, _ = generate_scene(reference_dir, "reference", 730, 24)
    virtual_scale = reference_dir / "virtual_scale.json"
    virtual_scale.write_text(json.dumps({"schema": "mvb.scalar_calibration.v1", "width_px": 400,
                                         "scale_mm_per_px": 0.1, "sigma_scale_mm_per_px": 0,
                                         "origin": "Virtual geometry used only for initial reference detection"}), encoding="utf-8")
    reference_analysis = reference_dir / "analysis"
    analyze_series(VideoAnalysisConfig(str(reference_video), str(reference_analysis), frame_step=1),
                   DetectionConfig(min_ball_diameter_mm=2.8, max_ball_diameter_mm=4.6, min_ball_edge_support=0.15,edge_fit_method=edge_fit_method),
                   "fov", calibration_path=str(virtual_scale), cell_mask=True)
    estimated_calibration = root / "estimated_calibration.json"
    calibration = calibration_from_csv(reference_analysis / "measurements.csv", 3.7, 0.0, estimated_calibration)
    cases = ["single", "two", "wall", "crossing", "missing", "noise_blur", "nuisance", "scale_drift", "deformed", "empty"]
    results = []
    for name in cases:
        directory = root / name
        video, metadata = generate_scene(directory, name, seed, frames)
        analysis_dir = directory / "analysis"
        analyze_series(VideoAnalysisConfig(str(video), str(analysis_dir), frame_step=1, preview_frames=3),
                       DetectionConfig(min_ball_diameter_mm=2.8, max_ball_diameter_mm=4.6, min_ball_edge_support=0.15,edge_fit_method=edge_fit_method),
                       "fov", calibration_path=str(estimated_calibration), tracking_distance_px=30,
                       tracking_max_missed=4, cell_mask=True)
        evaluation = evaluate_measurements(metadata, analysis_dir / "measurements.csv")
        (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        results.append({"scenario": name, **{k:v for k,v in evaluation.items() if k != "matches"}})
        print(name, {k: evaluation[k] for k in ("tp", "fp", "fn", "id_switches", "diameter_rmse_mm")}, flush=True)
    report = {"schema": "mvb.simulation_validation.v1", "seed": seed, "frames_per_case": frames,
              "scenarios": results, "sources": SOURCES,
              "calibration": calibration, "calibration_reference_seed": 730,
              "frozen_source_sha256": frozen,
              "scope": "Synthetic image-model validation only; no physical accuracy or thermal-model validation",
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if any(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest for name, digest in frozen.items()):
        raise RuntimeError("Sources changed during validation; results cannot be labelled a frozen run")
    (root / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Проверка алгоритма на симуляции", "", f"Seed: {seed}; кадров в каждом сценарии: {frames}.", "",
             "| Сценарий | TP | FP | FN | Смена ID | RMSE диаметра, мм |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        rmse = f'{row["diameter_rmse_mm"]:.4f}' if row["diameter_rmse_mm"] is not None else "—"
        lines.append(f'| {row["scenario"]} | {row["tp"]} | {row["fp"]} | {row["fn"]} | {row["id_switches"]} | {rmse} |')
    lines.extend(["", "TP/FP/FN считаются по сопоставлению центров с известной разметкой; порог расстояния 0.6 истинного диаметра.",
                  "Истинные объекты включаются при видимой площади не менее 65%; частично скрытые наблюдения учитываются отдельно.",
                  "RMSE и смещение диаметра вычислены только для сопоставленных наблюдений; пропуски отражены в FN.",
                  "Сценарии изменения масштаба и деформации проверяют границы скалярной калибровки и круговой модели.",
                  "Результаты относятся к выбранным синтетическим сценам. Точность камеры и стенда не установлена."])
    (root / "validation.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report
