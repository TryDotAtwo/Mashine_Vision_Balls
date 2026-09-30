import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mvb.series_analysis import CircleTracker, analyze_series, calibration_from_csv, load_calibration
from mvb.pipeline import VideoAnalysisConfig, DetectionConfig
from mvb.experiment_math import FovScaleInputs, FovScaleUncertainty


class SeriesTests(unittest.TestCase):
    def test_tracking_order_misses_and_new_object(self):
        tracker = CircleTracker(20, 1)
        self.assertEqual(tracker.update([(10, 10, 5), (100, 10, 6)]), [0, 1])
        self.assertEqual(tracker.update([(102, 10, 6), (12, 10, 5)]), [1, 0])
        tracker.update([])
        self.assertEqual(tracker.update([(14, 10, 5)]), [0])
        tracker.update([])
        tracker.update([])
        self.assertEqual(tracker.update([(14, 10, 5)]), [2])

    def test_config_defaults_and_cli_override(self):
        spec = importlib.util.spec_from_file_location("mvb_cli_test", ROOT / "run_analysis.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parser = module.build_parser({"frame_step": 3, "reference_diameter_mm": 12.5})
        self.assertEqual(parser.parse_args([]).frame_step, 3)
        self.assertEqual(parser.parse_args([]).reference_diameter_mm, 12.5)
        self.assertEqual(parser.parse_args(["--frame-step", "2"]).frame_step, 2)

    def test_calibration_roundtrip_and_reject_mixed_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "reference.csv"
            src.write_text("ball_id,ball_diameter_px\n0,40\n0,40\n", encoding="utf-8")
            out = Path(tmp) / "calibration.json"
            calibration_from_csv(src, 4, 0.01, out, 320)
            cal = load_calibration(out)
            self.assertAlmostEqual(cal["scale_mm_per_px"], 0.1)
            self.assertGreater(cal["sigma_scale_mm_per_px"], 0)
            src.write_text("ball_id,ball_diameter_px\n0,40\n1,40\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                calibration_from_csv(src, 4, 0.01, out)
            out.write_text('{"schema":"mvb.scalar_calibration.v1","scale_mm_per_px":NaN,"sigma_scale_mm_per_px":0}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration(out)

    def test_two_objects_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "two_balls.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
            self.assertTrue(writer.isOpened())
            for i in range(6):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.circle(frame, (70 + 2*i, 110), 20, (255, 255, 255), -1)
                cv2.circle(frame, (240 - 2*i, 110), 25, (255, 255, 255), -1)
                writer.write(frame)
            writer.release()
            output = Path(tmp) / "results"
            summary = analyze_series(VideoAnalysisConfig(str(video), str(output), frame_step=1, max_frames=4),
                                     DetectionConfig(min_ball_diameter_mm=3, max_ball_diameter_mm=6), "fov",
                                     FovScaleInputs(distance_to_object_mm=16, horizontal_fov_deg=90,
                                                    zoom_factor=1, resolution_x_px=320), FovScaleUncertainty())
            self.assertEqual(summary["sampled_frames"], 4)
            self.assertEqual(summary["tracks"], 2)
            self.assertEqual(summary["measurements"], 8)
            self.assertEqual(summary["detection_ratio"], 1)
            with (output / "tracks.csv").open(encoding="utf-8") as stream:
                tracks = list(csv.DictReader(stream))
            self.assertEqual(len(tracks), 2)
            sizes = sorted(float(t["diameter_mm_median"]) for t in tracks)
            self.assertAlmostEqual(sizes[0], 4, delta=0.3)
            self.assertAlmostEqual(sizes[1], 5, delta=0.3)
            for name in ["series_plots.png", "preview_sheet.jpg", "run.json", "frames.csv", "summary.md"]:
                self.assertGreater((output / name).stat().st_size, 0)
            provenance = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(len(provenance["video_sha256"]), 64)
            calibration = Path(tmp) / "scale.json"
            calibration.write_text(json.dumps({"schema": "mvb.scalar_calibration.v1", "width_px": 320,
                                              "scale_mm_per_px": 0.1, "sigma_scale_mm_per_px": 0.001}), encoding="utf-8")
            calibrated = analyze_series(VideoAnalysisConfig(str(video), str(Path(tmp) / "калибровка"), frame_step=1, max_frames=2),
                                        DetectionConfig(min_ball_diameter_mm=3, max_ball_diameter_mm=6), "fov",
                                        calibration_path=str(calibration))
            self.assertEqual(calibrated["measurements"], 4)
            self.assertTrue((Path(tmp) / "калибровка" / "preview_sheet.jpg").exists())
            calibration.write_text(json.dumps({"schema": "mvb.scalar_calibration.v1", "width_px": 321,
                                              "scale_mm_per_px": 0.1, "sigma_scale_mm_per_px": 0.001}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "width differs"):
                analyze_series(VideoAnalysisConfig(str(video)), DetectionConfig(), "fov", calibration_path=str(calibration))

    def test_empty_video_exports_and_reference_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "empty.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(2):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()
            output = Path(tmp) / "empty_results"
            summary = analyze_series(VideoAnalysisConfig(str(video), str(output), frame_step=1),
                                     DetectionConfig(), "fov", FovScaleInputs(16, 1, 90, 320), FovScaleUncertainty())
            self.assertEqual(summary["measurements"], 0)
            self.assertEqual(summary["detection_ratio"], 0)
            self.assertIsNone(summary["track_diameter_median_mm"])
            self.assertIn("ball_id", (output / "measurements.csv").read_text())
            self.assertTrue((output / "series_plots.png").exists())


if __name__ == "__main__":
    unittest.main()
