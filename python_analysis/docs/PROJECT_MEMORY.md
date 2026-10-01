# PROJECT MEMORY LOG

## 2026-04-02

- Выполнена структурная переработка: введены `src/`, `docs/`, `configs/`.
- Выполнен аудит проекта и кода: добавлен `docs/PROJECT_AUDIT_RU.md`.
- Добавлен единый CLI `run_analysis.py` с batch-режимом и экспортом артефактов серии.
- Добавлен загрузчик `key=value` конфигов: `src/mvb/config.py`.
- Перестроена математика:
  - `fov`-масштаб и его погрешность,
  - измерение по референсу в кадре,
  - масса испарения азота,
  - требуемый допуск ячейки по целевой ошибке,
  - явное разделение геометрии стенда и внутреннего диаметра ячейки.
- Перестроен видеопайплайн:
  - batch-анализ вместо чистого GUI,
  - работа на downscaled-копии кадра ради скорости,
  - вывод `summary.json`, `measurements.csv`, `summary.md`, `preview_sheet.jpg`.
- Добавлены unit-тесты: `tests/test_experiment_math.py`, `tests/test_config.py`.
- Расширена стендовая документация:
  - `docs/EXPERIMENT_MANUAL_RU.md`,
  - `docs/PURCHASE_SPEC_RU.md`,
  - `docs/CELL_TOLERANCE_NOTE_RU.md`,
  - `docs/SERIES_REPORT_TEMPLATE_RU.md`.
- Зафиксировано важное правило памяти: `38.13 мм` - это расстояние `ячейка-чехол`, а не внутренний диаметр ячейки.
- Выполнены smoke-прогоны на `data/videos/Ball_For_MV.mp4` и запуск из `configs/baseline_experiment.txt`.
- Корень репозитория очищен: видео/фото/PDF перенесены в `data/`, старые `.py` — в `legacy/scripts/` (скрипт `tools/reorganize_root.py`; `debug.log` при блокировке можно перенести вручную).
- Уточнена трактовка legacy `MainRadius=22.4`: см. `docs/LEGACY_CALIBRATION_22_4_RU.md` и `legacy/scripts/README.md`.
- Усилен CV-контур: `src/mvb/cv_geometry.py`, адаптивный Hough по нескольким `param2`, уточнение окружностей по Canny на полном разрешении; флаги CLI `--no-edge-refine`, `--no-adaptive-hough`, `--edge-refine-band-px`.
- Добавлены тесты `tests/test_cv_geometry.py`.
- Черновой документ по погрешностям унифицирован в финальный файл `docs/UNCERTAINTY_METHOD_RU.md`.
- Детекция шарика: порог `min_ball_edge_support`, кластеризация кандидатов, разрешение двух объектов (трек/центр референса), сброс трека `clear_track_after_lost_frames`; в `measurements.csv` поля `ball_edge_support`, `ball_valid_candidates`.
- Исправлена ошибка ветвления в `analyze_video` / `run_live_preview`: измерение диаметра снова выполняется только при `ball_circle is not None` (раньше блок ошибочно оказался в `elif attempted_ball`).
- Добавлен отдельный документ `docs/UNCERTAINTY_METHOD_RU.md`: полный список погрешностей, correction по уровню LN2, методика CV-оценки существующей ячейки, оценка достижимости `0.01 мм`.
- Полностью переписан `docs/EXPERIMENT_MANUAL_RU.md` под стендовое использование: установка Python/Git, подключение камеры, требования к ноутбуку, заземление, pump-control note, checklists, troubleshooting.
- Добавлен генератор красивого PDF-руководства `scripts/generate_stand_manual_pdf.py`.
- Сгенерирован артефакт `output/pdf/ivan_stand_manual_ru.pdf` (34 страницы по `pdfinfo`) с фото, таблицами и научной разбивкой.
- Зафиксирован новый ключевой физический вывод: движение шарика по высоте при испарении/подливе LN2 дает заметный scale drift, поэтому диаметр ячейки и `sigma_h` влияют не только на массу, но и на видеометрию.

## 2026-04-02 (manual: серии, геометрия, PDF bold, клонирование)

- Запрос: усилить правило «одна серия = одна конфигурация» пунктом про надежность **между** сериями; явно разделить `38.13 мм` (ячейка–чехол) и внутренний диаметр ячейки (измерение + подтверждение CV); все стендовые числа - перемерять и подставлять фактические; URL клонирования `https://github.com/TryDotAtwo/Mashine_Vision_Balls.git`; правки алгоритма на месте при необходимости; проверка соответствия раздела 11 приложению PDF; вертикальность камеры/ячейки, охлаждение, ячейка в ведре - диаметр ёмкости; неповторяемость уровня, CV-оценка, кривизна опоры ячейки; дорожная карта мультишарика/графиков.
- `docs/EXPERIMENT_MANUAL_RU.md`: правки по всем пунктам; ссылка на `ball_episodes`, `plot_measurements`, `PROJECT_MEMORY`.
- `scripts/generate_stand_manual_pdf.py`: markdown `**полужирный**` конвертируется в ReportLab `<b>` (раньше звездочки уходили в PDF как текст).

## 2026-04-02 (мультишарик, графики, стабильность)

- Запрос: последовательно много шариков на видео — измерять каждый (центры), графики в реальном времени, гистограмма по размерам, время–размер, стабильность размера и стабильность уровня LN2; опционально референс на воде.
- Текущий `pipeline.py`: один трек `previous_ball`, один шарик в кадре; уровень LN2 в коде не извлекается из кадра — только параметры `sigma_h` / `level_drop` в `experiment_math`.
- Добавлено: `src/mvb/ball_episodes.py` — разбиение временного ряда детекций на эпизоды «шарик» по порогу разрыва `gap_threshold_s` (постобработка CSV).
- Добавлено: `scripts/plot_measurements.py` — PNG `plots_stability.png` (время–D, гистограмма, rolling std(D), при `reference` — `reference_diameter_px` как прокси дрейфа масштаба/уровня) и `plots_stability_summary.txt` по эпизодам; зависимость `matplotlib` в `requirements.txt`.
- Сделано позже в тот же день: `ball_id` в `FrameMeasurement` / CSV (`advance_ball_id` при пустом треке в начале кадра); `run_live_preview` + `--live-plots` и `src/mvb/live_plots.py` (D(t) и прокси масштаба); `episode_spans_from_ball_ids` для `plot_measurements.py`; конфиг-ключи `live_plots`, `live_plot_max_points`.
- Роадмап: минимальная длина эпизода; явная детекция уровня жидкости в мм; вода как референс — тот же контур `reference`/`fov`.

## Открытые вопросы

- Уточнить финальные реальные параметры камеры/объектива после поставки.
- Получить серию с подшипником/эталоном для калибровки `measurement_mode=reference`.
- Получить видео с шариками в азоте для валидации на новой конфигурации.
- Получить фактический протокол изготовления новой ячейки и ее внутренний диаметр.
- Отдельно определить метод измерения уровня LN2 и его реальную `sigma_h`, потому что при `h=6.3 мм` и `sigma_h=0.2 мм` цель `3%` по массе недостижима.
- Формализовать интерфейс выдачи сигнала на насос и схему безопасного согласования с I/O камеры / внешним контроллером.


## 2026-09-30: анализ нескольких объектов
* implementation=src/mvb/series_analysis.py; default_entrypoint=run_analysis.py; legacy_mode=--single-ball
* features=несколько_окружностей_трекер_CSV_по_траекториям_графики_калибровка_JSON_проверка_ширины_протокол_SHA256
* config_fix=значения_файла_не_перекрываются_дефолтами_CLI
* validation=12_unittest_passed; synthetic_two_objects_passed; empty_video_passed; unicode_export_passed
* archive=12_frames_without_mask_false_positives_observed; added_cell_mask; tracking_gate_requires_tuning
* limitations=пузырьки_и_перекрытия_требуют_контроля; scalar_calibration_not_distortion; physical_accuracy_unverified; live_preview_single_object



## 2026-09-30: симуляция и независимая проверка
* request=Без камеры: самостоятельно прочитать статьи, сделать симуляцию и проверить алгоритмы.
* implementation=simulation.py; tracking=global_assignment_velocity; geometry=subpixel_circle; near_wall_filter=fixed
* validation=16_unittest_OK; seeds=900,901; scored_frames=480; reference_seed=730; frozen_source_hashes=validation.json
* counts=[(900, 278, 0, 4, 1), (901, 273, 0, 9, 0)]; columns=seed_TP_FP_FN_IDswitch
* method=docs/SIMULATION_METHOD_RU.md; physical_accuracy=unverified; deformation_and_depth_bias=observed; raw_data_preserved=true

* 2026-09-30: Новая инструкция: ../../03_REPORTS/final_nir_2026/stand_manual_v2.tex, версия 2.0 от 30.09.2026; исходная апрельская инструкция сохранена.

* 2026-09-30: ClickHouse implemented: default CLI storage, SQL track summaries, UUID manifest-last completion, reexport; isolated Compose port 8124, durable volumes; 21 tests OK including 4 real-server integrations; manual 2.1 and report updated. Native compiler unavailable; XeLaTeX fallback succeeds.

* 2026-09-30: Published Python analysis/report at TryDotAtwo/Mashine_Vision_Balls PR #1 (draft), head bcad8a61f178df183a1e43fd48da2e0fd6082ce2; original repository preserved, remote SHA verified; report 22 pages, manual 2.1 8 pages. No hardware accuracy claim.

* 2026-09-30: Drive ordinary-ball raw video acquired (41,230,125 bytes, SHA82910bf..., 855 frames). Diagnostic 57 frames in ClickHouse run 27d551ef-7098-4e80-980d-de8592def4ed; 1499 candidates, many non-target circles. Selected frames450/600/750 target matched3/3 plus23/3/5 unmatched candidates. Uncalibrated mm excluded from scientific claims. Report includes source/detection images and limits; hardware materials deferred to next section.

* 2026-09-30: Telegram/Drive intake indexed; six 4K bearing clips from 28.07.2025 processed all1395 frames through ClickHouse, detections1361; provisional3.98mm electronic-caliper reference, unknown sigma NULL, separate first-half pixel calibration per clip; pixel mode default, dark profile angular contrast85%, ROI, annotated MP4, decode status, source freeze audit; 25 tests OK including5 real ClickHouse and incomplete-decode test; reexports18 CSV identical, source hashes verified. Archive liquid full855 frames810 circles/118 episodes; sampled checkpoints3/3+0 extras, not independent accuracy. Negative floor43frames3 residual FP,0 eligible episodes; legacy decodes1960/4234, incomplete. New synthetic902 TP272 FP1 FN8 IDswitch0; old900/901 preserved. Ruler250frames ~52px/mm diagnostic not transferred. Two Drive4K originals acquired; >256MiB originals connector-limited. Chronology/source index prepared for section2, not drafted. Report28pages including complete manual2.2/9pages; native compiler fails platform directories, same-source XeLaTeX succeeds; no raw data deleted.

* 2026-09-30: Published and remotely verified PR1 head01a7a9b93f32ea16165099a5c97e04b66d084ad1 (draft). Source/data line endings preserved for exact-byte hashes. Added earlier liquid-run snapshot with12 matching SHA-256 comparisons; no raw/private exports published. Published report PDF identical to local28-page PDF; complete manual2.2 appendix9pages verified.

* 2026-10-01: Radial gradient/subpixel soft_l1 circle refinement, explicit coverage/residual/ellipticity rejection, PTS-aware tracking, timestamp fallbacks, ClickHouse python_quality_v1 diagnostics and manifest gate implemented. Final six phone runs decode1395 frames, accept1250 observations; approximate3.98mm electronic-caliper reference remains provisional, instrument uncertainty unknown and total sigma NULL. Ordinary full855frames:486 circles/7episodes; legacy only1960/4234 decoded,392sampled,142circles/77episodes (16eligible). Negative controls floor43frames0accepted and vessel26samples0accepted. New simulation903/904:480frames TP510 FP0 FN52 IDswitch0;48 deformed observations rejected; scale-drift RMSE~.16mm.22 synthetic series imported into ClickHouse with max numeric roundtrip difference7.11e-15;88 repeated DB exports byte-identical, plus24phone+8archive+4boiling exports=124 comparisons.29 tests including6 real ClickHouse integration pass in private and publication checkout.
* 2026-10-01: Three new Drive videos registered. V1/V2 are software screen recordings, fully decoded715/1054, excluded as metrology truth. Boiling756/756 completed with350px single-threshold candidate search/no single-cell mask and native resolution refinement;49 accepted circles/25episodes, reviewed frame191 follows cell rim, not confirmed ball; no ball-size distribution claimed. Slow owned process35148 stopped after complete alternative run; partial outputs retained. Raw sources preserved unchanged. Camera SDK, independent hardware accuracy and section2 materials remain separate.
* 2026-10-01: Existing section01 LaTeX edited in place;33-page PDF including complete manual2.3/10pages compiled by existing XeLaTeX and individually reviewed. Built-in compiler returned platform-directory error; existing editor kept open. Writing Style/logika/Technical Style Editor/Visual Research Papers/Academic Writing Toolkit applied, tool heuristic limits recorded. Editable PDF/SVG figures, claim ledger, citation19key audit and source/PDF/page SHA manifests prepared. Published code/report/derived validation only in existing draft PR1; remote head1c85568dd66027ff250a9bd9b87f5b377bffbae7 verified by GitHub, public PDF byte-identical SHA3234c30f8878ffe668864fb87ed8193fa8b961970d96dd9085ceeaf7ecb3e62b. Code/docs diff-check clean; generated CSV/SVG formatting retained for exact provenance. No raw/private exports or secrets published, no merge.

* 2026-10-01: Added closed-boundary support gate (90-degree maximum gap, minimum quadrant share .05), robust inlier refit and metrology.py GUM covariance/blocked bearing holdout exports. Ordinary final855frames472circles8episodes; manual17visibleframes15TP0FP2FN, development-only. Six bearing videos1250accepted; first-half calibration/second-half check, 2000 block bootstrap replicates at5/15/30frames; caliper uncertainty and full physical uncertainty NULL. Final simulations907/908:505TP0FP55FN0IDswitch,48deformation rejections;22CH imports verified. Nine real CH reexports and all available annotated videos verified; six metrology exports byte-identical.36tests pass in local and isolated publication checkout. Report31pages/manual2.4 ninepages compiled with existing XeLaTeX; all31pages visually inspected. Built-in compiler remains unavailable(platform directories). Frozen six-video source bridge verified by originalSHA and AST equality outside unused either branch; original protocol hashes retained. Astra group transmission explicitly authorized by Ivan; tool fetch failed, no delivery or expert answer confirmed. No raw data deleted or published.
