from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from mvb.pipeline import AnalysisSummary, FrameMeasurement, VideoAnalysisResult


def write_measurements_csv(path: str | Path, measurements: list[FrameMeasurement]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FrameMeasurement.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(asdict(measurement))
    return path


def write_summary_json(path: str | Path, summary: AnalysisSummary) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _markdown_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary_markdown(path: str | Path, result: VideoAnalysisResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = result.summary
    lines = [
        "# Отчет по анализу видео",
        "",
        f"- Видео: `{summary.video_path}`",
        f"- Выходная папка: `{summary.output_dir}`",
        f"- Режим измерения: `{summary.measurement_mode}`",
        f"- Размер кадра: `{summary.width_px}x{summary.height_px}`",
        f"- FPS: `{summary.fps:.3f}`",
        f"- Всего кадров: `{summary.total_frames}`",
        f"- Просэмплировано кадров: `{summary.sampled_frames}`",
        f"- Детекций: `{summary.detections}`",
        f"- Detection ratio: `{summary.detection_ratio:.4f}`",
        "",
        "## Метрики диаметра",
        "",
        f"- Средний диаметр: `{_markdown_value(summary.diameter_mm_mean)}` мм",
        f"- Медиана диаметра: `{_markdown_value(summary.diameter_mm_median)}` мм",
        f"- Стандартное отклонение: `{_markdown_value(summary.diameter_mm_std)}` мм",
        f"- Минимум: `{_markdown_value(summary.diameter_mm_min)}` мм",
        f"- Максимум: `{_markdown_value(summary.diameter_mm_max)}` мм",
        f"- Средняя модельная sigma: `{_markdown_value(summary.sigma_model_mm_mean)}` мм",
        f"- Полная sigma одной реализации: `{_markdown_value(summary.combined_sigma_single_mm)}` мм",
        f"- Sigma среднего по серии: `{_markdown_value(summary.combined_sigma_mean_mm)}` мм",
        f"- Средний масштаб: `{_markdown_value(summary.scale_mm_per_px_mean)}` мм/пикс",
    ]
    if summary.reference_diameter_px_mean is not None:
        lines.append(f"- Средний диаметр референса в кадре: `{summary.reference_diameter_px_mean:.3f}` пикс")
    if summary.expected_ball_diameter_mm is not None:
        lines.append(f"- Ожидаемый диаметр: `{summary.expected_ball_diameter_mm:.6f}` мм")
    if summary.bias_to_expected_mm is not None:
        lines.append(f"- Смещение к ожидаемому: `{summary.bias_to_expected_mm:.6f}` мм")
    lines.extend(
        [
            "",
            "## Артефакты",
            "",
            "- `summary.json` - машинно-читаемая сводка.",
            "- `measurements.csv` - покадровые измерения.",
            "- `preview_sheet.jpg` - набор аннотированных кадров.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_preview_sheet(path: str | Path, preview_images: list[np.ndarray]) -> Path | None:
    if not preview_images:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_width = max(image.shape[1] for image in preview_images)
    max_height = max(image.shape[0] for image in preview_images)
    padded = []
    for image in preview_images:
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        canvas[: image.shape[0], : image.shape[1]] = image
        padded.append(canvas)
    if len(padded) == 1:
        sheet = padded[0]
    else:
        rows = []
        for start in range(0, len(padded), 2):
            row_images = padded[start : start + 2]
            if len(row_images) == 1:
                row_images.append(np.zeros_like(row_images[0]))
            rows.append(np.hstack(row_images))
        sheet = np.vstack(rows)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", sheet)
    if not ok:
        raise RuntimeError(f"Cannot encode preview image: {path}")
    path.write_bytes(encoded.tobytes())
    return path


def write_analysis_bundle(result: VideoAnalysisResult) -> dict[str, str]:
    output_dir = Path(result.summary.output_dir)
    paths = {
        "summary_json": str(write_summary_json(output_dir / "summary.json", result.summary)),
        "measurements_csv": str(write_measurements_csv(output_dir / "measurements.csv", result.measurements)),
        "summary_md": str(write_summary_markdown(output_dir / "summary.md", result)),
    }
    preview_path = write_preview_sheet(output_dir / "preview_sheet.jpg", result.preview_images)
    if preview_path is not None:
        paths["preview_sheet"] = str(preview_path)
    return paths
