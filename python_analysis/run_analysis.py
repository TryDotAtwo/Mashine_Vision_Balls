from __future__ import annotations

import argparse
import sys
import os
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mvb import (
    CONFIG_TEMPLATE,
    CellGeometry,
    DetectionConfig,
    FovScaleInputs,
    FovScaleUncertainty,
    ReferenceMeasurement,
    StandGeometry,
    VideoAnalysisConfig,
    analyze_video,
    evaporated_mass_kg,
    load_key_value_config,
    relative_mass_uncertainty,
    required_cell_tolerance_mm,
    run_live_preview,
    write_analysis_bundle,
)
from mvb.series_analysis import analyze_series, calibration_from_csv


def _preload_defaults(argv: list[str]) -> dict[str, object]:
    scratch = argparse.ArgumentParser(add_help=False)
    scratch.add_argument("--config", type=str)
    scratch.add_argument("--print-config-template", action="store_true")
    known, _ = scratch.parse_known_args(argv)
    defaults: dict[str, object] = {}
    if known.config:
        defaults.update(load_key_value_config(known.config))
    defaults["print_config_template"] = known.print_config_template
    return defaults


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Machine Vision Balls: пакетный анализ видео, расчет допусков ячейки и стендовые артефакты",
    )
    parser.add_argument("--config", type=str, help="Путь к key=value конфигу")
    parser.add_argument("--print-config-template", action="store_true", help="Показать шаблон конфига и выйти")

    parser.add_argument("--video", dest="video_path", type=str, help="Путь к видео")
    parser.add_argument("--output-dir", type=str, default="", help="Папка вывода артефактов")
    parser.add_argument("--measurement-mode", choices=("pixel", "fov", "reference"), default="pixel", help="Режим масштаба")
    parser.add_argument("--detection-max-dimension", type=int, default=1400)
    parser.add_argument("--detector-profile", choices=("hough","dark-ball"), default="hough")
    parser.add_argument("--min-ball-diameter-px", type=float, default=12.0)
    parser.add_argument("--max-ball-diameter-px", type=float, default=500.0)
    parser.add_argument("--dark-contrast-min", type=float, default=8.0)
    parser.add_argument("--dark-angular-contrast-min", type=float, default=0.85)
    parser.add_argument("--roi", type=float, nargs=4, metavar=("LEFT","TOP","RIGHT","BOTTOM"), help="Normalized center-search ROI, 0..1")
    parser.add_argument("--annotated-video", action="store_true", help="Export annotated sampled MP4, max dimension 800px")
    parser.add_argument("--opencv-threads",type=int,default=2,help="OpenCV CPU threads; 0 disables internal parallel execution")
    parser.add_argument("--frame-step", type=int, default=10, help="Брать каждый N-й кадр")
    parser.add_argument("--max-frames", type=int, default=0, help="Ограничить число обработанных сэмплов")
    parser.add_argument("--preview-frames", type=int, default=6, help="Сколько аннотированных кадров сохранить")
    parser.add_argument("--interactive-preview", action="store_true", help="Открыть интерактивный просмотр")
    parser.add_argument(
        "--live-plots",
        action="store_true",
        help="Вместе с --interactive-preview: второе окно matplotlib (D(t), масштаб)",
    )
    parser.add_argument("--live-plot-max-points", type=int, default=3000, help="Длина буфера живых графиков")

    parser.add_argument("--min-ball-diam-mm", type=float, default=1.0, help="Минимальный ожидаемый диаметр шарика")
    parser.add_argument("--max-ball-diam-mm", type=float, default=10.0, help="Максимальный ожидаемый диаметр шарика")
    parser.add_argument("--expected-ball-diameter-mm", type=float, default=0.0, help="Эталонный диаметр для оценки bias")
    parser.add_argument("--sigma-ball-px", type=float, default=1.5, help="Пиксельная sigma детекции шарика")
    parser.add_argument("--sigma-reference-px", type=float, default=1.5, help="Пиксельная sigma детекции референса")
    parser.add_argument("--reference-name", type=str, default="cell", help="Имя референса для отчета")
    parser.add_argument("--reference-diameter-mm", type=float, default=0.0, help="Истинный диаметр референса")
    parser.add_argument("--reference-sigma-mm", type=float, default=0.0, help="Sigma истинного диаметра референса")
    parser.add_argument("--reference-min-radius-frac", type=float, default=0.25, help="Нижняя граница радиуса референса")
    parser.add_argument("--reference-max-radius-frac", type=float, default=0.60, help="Верхняя граница радиуса референса")
    parser.add_argument(
        "--no-edge-refine",
        action="store_true",
        help="Отключить уточнение центра/радиуса по краям Canny на полном разрешении",
    )
    parser.add_argument("--edge-fit-method",choices=("kasa","radial"),default=None)
    parser.add_argument("--radial-polarity",choices=("either","rising","falling"),default="either")
    parser.add_argument("--radial-min-gradient",type=float,default=.2)
    parser.add_argument("--radial-min-coverage",type=float,default=.65)
    parser.add_argument("--radial-max-axis-ratio",type=float,default=1.15)
    parser.add_argument("--edge-refine-band-px", type=int, default=4, help="Толщина кольца вокруг Hough-радиуса для подгонки")
    parser.add_argument("--no-adaptive-hough", action="store_true", help="Отключить перебор param2 HoughCircles")
    parser.add_argument(
        "--min-ball-edge-support",
        type=float,
        default=0.10,
        help="Мин. опора контура шарика на Canny (0..1); ниже — кадр без детекции",
    )
    parser.add_argument(
        "--ball-candidate-dedupe-px",
        type=float,
        default=14.0,
        help="Расстояние для слияния дубликатов Hough (один физический шарик)",
    )
    parser.add_argument(
        "--second-ball-score-ratio",
        type=float,
        default=0.92,
        help="Окно score для выбора между двумя объектами (ближе к треку/центру референса)",
    )
    parser.add_argument(
        "--clear-track-after-lost-frames",
        type=int,
        default=0,
        help="Сброс трека после N пропусков подряд (0 = не сбрасывать)",
    )

    parser.add_argument("--ball-to-object-mm", type=float, default=55.0, help="Расстояние от шарика до окна/камеры")
    parser.add_argument("--cell-to-cover-mm", type=float, default=38.13, help="Расстояние от ячейки до окна/чехла")
    parser.add_argument("--cover-thickness-mm", type=float, default=0.85, help="Толщина окна/чехла")
    parser.add_argument("--cell-cover-gap-mm", type=float, default=6.3, help="Зазор между ячейкой и окном")
    parser.add_argument("--zoom", type=float, default=1.0, help="Оптический zoom factor")
    parser.add_argument("--fov-deg", type=float, default=79.0, help="Горизонтальный FOV")
    parser.add_argument("--calib-coef", type=float, default=1.0, help="Поправочный коэффициент масштаба")
    parser.add_argument("--sigma-distance-mm", type=float, default=0.5, help="Sigma по расстоянию")
    parser.add_argument("--sigma-zoom", type=float, default=0.05, help="Sigma по zoom factor")
    parser.add_argument("--sigma-fov-deg", type=float, default=0.5, help="Sigma по FOV")

    parser.add_argument("--cell-inner-diameter-mm", type=float, default=0.0, help="Внутренний диаметр ячейки")
    parser.add_argument("--cell-inner-sigma-mm", type=float, default=0.0, help="Sigma внутреннего диаметра ячейки")
    parser.add_argument("--delta-level-mm", type=float, default=0.0, help="Изменение уровня азота")
    parser.add_argument("--level-sigma-mm", type=float, default=0.0, help="Sigma измерения уровня")
    parser.add_argument("--target-rel-mass-error", type=float, default=0.03, help="Целевая относительная ошибка массы")
    parser.add_argument("--single-ball", action="store_true", help="Прежний режим одного шарика и интерактивного просмотра")
    parser.add_argument("--calibration", type=str, help="JSON сохранённого масштаба для неизменной оптической геометрии")
    parser.add_argument("--calibrate-csv", type=str, help="CSV наблюдений одного эталона для создания калибровки")
    parser.add_argument("--calibration-diameter-mm", type=float, help="Известный диаметр эталона")
    parser.add_argument("--calibration-sigma-mm", type=float, default=None, help="Неопределённость диаметра эталона")
    parser.add_argument("--provisional-calibration", action="store_true", help="Unconfirmed reference size: metric values are provisional, total sigma stays unknown")
    parser.add_argument("--calibration-output", type=str, default="calibration.json")
    parser.add_argument("--calibration-width-px", type=int, help="Ширина изображения эталонной серии, если нет run.json")
    parser.add_argument("--tracking-distance-px", type=float, default=60.0, help="Макс. перемещение между обработанными кадрами")
    parser.add_argument("--tracking-max-missed", type=int, default=3, help="Допустимое число пропущенных обработанных кадров")
    parser.add_argument("--min-track-observations", type=int, default=2, help="Мин. наблюдений для включения трека в гистограмму")
    parser.add_argument("--no-cell-mask", action="store_true", help="Не ограничивать поиск внутренней областью круглой ячейки")
    parser.add_argument("--simulate-check", action="store_true", help="Создать размеченные синтетические видео и проверить алгоритм")
    parser.add_argument("--simulation-output", default="output/simulation_validation", help="Новая папка результатов симуляции")
    parser.add_argument("--simulation-seed", type=int, default=900)
    parser.add_argument("--simulation-frames", type=int, default=24)
    parser.add_argument("--offline", action="store_true", help="Явный локальный режим без ClickHouse (симуляция/диагностика)")
    parser.add_argument("--clickhouse-url", default=None, help="HTTP(S) ClickHouse; иначе MVB_CLICKHOUSE_URL или localhost:8124")
    parser.add_argument("--clickhouse-database", default=None, help="Отдельная БД анализа; иначе MVB_CLICKHOUSE_DATABASE или mvb_analysis")
    parser.add_argument("--export-run", help="Повторно построить таблицы и графики из ClickHouse по UUID серии")
    parser.add_argument("--bearing-check-run", help="UUID завершённой серии одного подшипника: таблица повторяемости и условной неопределённости")
    parser.add_argument("--bearing-reference-mm", type=float, help="Предварительный размер подшипника для условной проверки")
    parser.add_argument("--bearing-block-frames", type=int, default=15, help="Длина временного блока bootstrap")
    parser.set_defaults(**defaults)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    defaults = _preload_defaults(argv)
    parser = build_parser(defaults)
    args = parser.parse_args(argv)
    args.edge_fit_method=args.edge_fit_method or ("kasa" if args.single_ball or args.no_edge_refine else "radial")
    if args.opencv_threads<0:
        parser.error("--opencv-threads must be nonnegative")
    cv2.setNumThreads(args.opencv_threads)

    if args.print_config_template:
        print(CONFIG_TEMPLATE.rstrip())
        return
    from mvb.clickhouse_store import ClickHouseStore
    storage = None if args.offline else ClickHouseStore(args.clickhouse_url, args.clickhouse_database)
    if args.bearing_check_run:
        if storage is None or not args.output_dir or args.bearing_reference_mm is None:
            parser.error("--bearing-check-run requires ClickHouse, --output-dir and --bearing-reference-mm")
        from mvb.metrology import export_bearing_check
        try:
            print(export_bearing_check(storage,args.bearing_check_run,args.output_dir,
                                      args.bearing_reference_mm,args.bearing_block_frames))
        except ValueError as exc:
            parser.error(str(exc))
        return
    if args.export_run:
        if storage is None or not args.output_dir:
            parser.error("--export-run requires ClickHouse and --output-dir")
        try:
            print(storage.export(args.export_run, args.output_dir))
        except ValueError as exc:
            parser.error(str(exc))
        return
    if args.single_ball and args.edge_fit_method=="radial":
        parser.error("Radial fitting is available in multi-object mode; omit --single-ball")
    if args.single_ball and not args.offline:
        parser.error("Legacy single-ball mode requires --offline; use multi-object mode for ClickHouse")
    if args.simulate_check:
        from mvb.simulation import run_validation
        if not 12 <= args.simulation_frames <= 60:
            parser.error("--simulation-frames must be 12..60 (liquid-stage model, <=2 seconds)")
        try:
            run_validation(args.simulation_output, args.simulation_seed, args.simulation_frames, edge_fit_method=args.edge_fit_method)
        except ValueError as exc:
            parser.error(str(exc))
        return
    if args.calibrate_csv:
        if args.calibration_diameter_mm is None:
            parser.error("--calibration-diameter-mm is required with --calibrate-csv")
        try:
            print(calibration_from_csv(args.calibrate_csv, args.calibration_diameter_mm,
                                       args.calibration_sigma_mm, args.calibration_output, args.calibration_width_px, args.provisional_calibration))
        except ValueError as exc:
            parser.error(str(exc))
        return
    if not args.single_ball and args.interactive_preview:
        parser.error("Interactive preview currently requires --single-ball; multi-object mode exports annotated frames and plots")
    if args.single_ball and args.calibration:
        parser.error("Saved calibration is supported by multi-object analysis; omit --single-ball")

    stand_geometry = StandGeometry(
        ball_to_cover_mm=args.ball_to_object_mm,
        cell_to_cover_mm=args.cell_to_cover_mm,
        cover_thickness_mm=args.cover_thickness_mm,
        cell_cover_gap_mm=args.cell_cover_gap_mm,
    )
    print("Stand geometry:")
    print(f"  ball_to_cover_mm = {stand_geometry.ball_to_cover_mm}")
    print(f"  cell_to_cover_mm = {stand_geometry.cell_to_cover_mm}")
    print(f"  cover_thickness_mm = {stand_geometry.cover_thickness_mm}")
    print(f"  cell_cover_gap_mm = {stand_geometry.cell_cover_gap_mm}")
    print("  note = cell_to_cover_mm is stand geometry, not cell inner diameter")

    if args.cell_inner_diameter_mm > 0 and args.delta_level_mm > 0:
        cell = CellGeometry(
            inner_diameter_mm=args.cell_inner_diameter_mm,
            sigma_inner_diameter_mm=max(args.cell_inner_sigma_mm, 1e-9),
            level_drop_mm=args.delta_level_mm,
            sigma_level_mm=max(args.level_sigma_mm, 1e-9),
        )
        mass_kg, sigma_mass_kg = evaporated_mass_kg(cell)
        rel_sigma = relative_mass_uncertainty(cell)
        required_sigma = required_cell_tolerance_mm(
            target_relative_error_mass=args.target_rel_mass_error,
            inner_diameter_mm=args.cell_inner_diameter_mm,
            level_drop_mm=args.delta_level_mm,
            level_measurement_sigma_mm=max(args.level_sigma_mm, 1e-9),
        )
        print("Cell / LN2 model:")
        print(f"  mass_kg = {mass_kg:.6f}")
        print(f"  sigma_mass_kg = {sigma_mass_kg:.6f}")
        print(f"  relative_sigma = {rel_sigma:.6f}")
        print(f"  required_cell_inner_sigma_mm = {required_sigma:.6f}")

    if not args.video_path:
        if args.cell_inner_diameter_mm > 0 and args.delta_level_mm > 0:
            return
        parser.error("Nothing to do: pass --video or valid cell geometry arguments.")

    if args.measurement_mode == "reference" and args.reference_diameter_mm <= 0 and not args.calibration:
        parser.error("--reference-diameter-mm is required for measurement-mode=reference")

    video_cfg = VideoAnalysisConfig(
        video_path=args.video_path,
        output_dir=args.output_dir or None,
        frame_step=max(1, args.frame_step),
        max_frames=max(0, args.max_frames),
        preview_frames=max(0, args.preview_frames),
        interactive_preview=args.interactive_preview,
        live_plots=bool(args.live_plots),
        live_plot_max_points=max(50, int(args.live_plot_max_points)),
    )
    if args.roi and not (0 <= args.roi[0] < args.roi[2] <= 1 and 0 <= args.roi[1] < args.roi[3] <= 1):
        parser.error("ROI requires 0 <= left < right <= 1, 0 <= top < bottom <= 1")
    if args.single_ball and args.measurement_mode == "pixel":
        parser.error("Pixel mode requires multi-object analysis")
    detection_cfg = DetectionConfig(
        edge_fit_method=args.edge_fit_method, radial_polarity=args.radial_polarity,
        radial_min_gradient=args.radial_min_gradient, radial_min_coverage=args.radial_min_coverage, radial_max_axis_ratio=args.radial_max_axis_ratio,
        max_detection_dimension_px=args.detection_max_dimension,
        detector_profile=args.detector_profile,
        min_ball_diameter_px=args.min_ball_diameter_px,
        max_ball_diameter_px=args.max_ball_diameter_px,
        dark_contrast_min=args.dark_contrast_min,
        dark_angular_contrast_min=args.dark_angular_contrast_min,
        roi=tuple(args.roi) if args.roi else None,
        min_ball_diameter_mm=args.min_ball_diam_mm,
        max_ball_diameter_mm=args.max_ball_diam_mm,
        sigma_ball_px=args.sigma_ball_px,
        sigma_reference_px=args.sigma_reference_px,
        reference_min_radius_frac=args.reference_min_radius_frac,
        reference_max_radius_frac=args.reference_max_radius_frac,
        refine_with_edge_fit=not args.no_edge_refine,
        edge_refine_band_px=max(1, args.edge_refine_band_px),
        adaptive_hough=not args.no_adaptive_hough,
        min_ball_edge_support=max(0.0, min(1.0, float(args.min_ball_edge_support))),
        ball_candidate_dedupe_px=max(1.0, float(args.ball_candidate_dedupe_px)),
        second_ball_score_ratio=max(0.5, min(1.0, float(args.second_ball_score_ratio))),
        clear_track_after_lost_frames=max(0, int(args.clear_track_after_lost_frames)),
    )
    fov_inputs = FovScaleInputs(
        distance_to_object_mm=args.ball_to_object_mm,
        zoom_factor=args.zoom,
        horizontal_fov_deg=args.fov_deg,
        resolution_x_px=1,
        calibration_coef=args.calib_coef,
    )
    fov_uncertainty = FovScaleUncertainty(
        sigma_distance_mm=args.sigma_distance_mm,
        sigma_zoom=args.sigma_zoom,
        sigma_fov_deg=args.sigma_fov_deg,
    )
    reference = ReferenceMeasurement(
        name=args.reference_name,
        diameter_mm=args.reference_diameter_mm,
        sigma_diameter_mm=args.reference_sigma_mm,
    )

    if not args.single_ball:
        try:
            summary = analyze_series(
            video_cfg, detection_cfg, args.measurement_mode,
            fov_inputs=fov_inputs if args.measurement_mode == "fov" else None,
            fov_uncertainty=fov_uncertainty if args.measurement_mode == "fov" else None,
            reference=reference if args.measurement_mode == "reference" else None,
            calibration_path=args.calibration,
            tracking_distance_px=args.tracking_distance_px,
            tracking_max_missed=args.tracking_max_missed,
            min_track_observations=args.min_track_observations,
            expected_ball_diameter_mm=args.expected_ball_diameter_mm,
            cell_mask=not args.no_cell_mask,
            storage=storage,
            annotated_video=args.annotated_video,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print("Series analysis:", summary)
        return

    result = analyze_video(
        video_cfg=video_cfg,
        detection_cfg=detection_cfg,
        measurement_mode=args.measurement_mode,
        fov_inputs=fov_inputs if args.measurement_mode == "fov" else None,
        fov_uncertainty=fov_uncertainty if args.measurement_mode == "fov" else None,
        reference=reference if args.measurement_mode == "reference" else None,
        expected_ball_diameter_mm=args.expected_ball_diameter_mm if args.expected_ball_diameter_mm > 0 else None,
    )
    bundle = write_analysis_bundle(result)
    print("Video analysis:")
    print(f"  detections = {result.summary.detections}/{result.summary.sampled_frames}")
    print(f"  detection_ratio = {result.summary.detection_ratio:.4f}")
    if result.summary.diameter_mm_mean is not None:
        print(f"  diameter_mm_mean = {result.summary.diameter_mm_mean:.6f}")
        print(f"  combined_sigma_single_mm = {result.summary.combined_sigma_single_mm:.6f}")
        print(f"  combined_sigma_mean_mm = {result.summary.combined_sigma_mean_mm:.6f}")
    print("Artifacts:")
    for name, path in bundle.items():
        print(f"  {name} = {path}")

    if args.interactive_preview:
        if args.live_plots:
            try:
                import matplotlib  # noqa: F401
            except ImportError as e:
                raise SystemExit("Для --live-plots установите matplotlib: pip install matplotlib") from e
        run_live_preview(
            video_cfg=video_cfg,
            detection_cfg=detection_cfg,
            measurement_mode=args.measurement_mode,
            fov_inputs=fov_inputs if args.measurement_mode == "fov" else None,
            fov_uncertainty=fov_uncertainty if args.measurement_mode == "fov" else None,
            reference=reference if args.measurement_mode == "reference" else None,
        )


if __name__ == "__main__":
    main()
