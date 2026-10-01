import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mvb.clickhouse_store import ClickHouseStore
from mvb.pipeline import FrameMeasurement, VideoAnalysisConfig, DetectionConfig
from mvb.experiment_math import FovScaleInputs, FovScaleUncertainty
from mvb.series_analysis import analyze_series


class ValidationTests(unittest.TestCase):
    def test_reject_sql_identifier_and_credential_url(self):
        for database in ("bad;DROP DATABASE mvb", "x.y", ""):
            if database:
                with self.assertRaises(ValueError):
                    ClickHouseStore(database=database)
        with self.assertRaises(ValueError):
            ClickHouseStore("http://user:secret@localhost:8123")


@unittest.skipUnless(os.environ.get("MVB_CLICKHOUSE_TEST") == "1", "requires local ClickHouse integration server")
class ClickHouseIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.store = ClickHouseStore()

    def test_sql_medians_and_independent_runs(self):
        rows = [FrameMeasurement(i, i/10, 10, 20, 5, 10, None, None, .1, d, .01, "calibrated", .8, 1, tid)
                for i, (tid, d) in enumerate([(0, 1.), (0, 3.), (1, 4.), (1, 5.), (1, 9.)])]
        frames = [{"frame_index": i, "time_s": i/10, "objects": 1, "scale_available": True} for i in range(5)]
        provenance = {"video_sha256": "synthetic", "calibration": {"scale_mm_per_px": .1}}
        summary = {"min_track_observations": 2}
        first = self.store.save(rows, frames, provenance, summary)
        second = self.store.save(rows, frames, provenance, summary)
        self.assertNotEqual(first, second)
        data, fs, tracks, result, meta = self.store.load(first)
        self.assertEqual(len(data), 5)
        self.assertEqual([r["diameter_mm_median"] for r in tracks], [2., 5.])
        self.assertEqual(result["track_diameter_median_mm"], 3.5)
        self.assertEqual(meta["calibration"], provenance["calibration"])
        with tempfile.TemporaryDirectory() as tmp:
            self.store.export(first, tmp)
            self.assertTrue((Path(tmp) / "series_plots.png").exists())
            self.assertEqual(json.loads((Path(tmp) / "summary.json").read_text())["run_id"], first)

    def test_empty_completed_run(self):
        run_id = self.store.save([], [{"frame_index": 0, "time_s": 0., "objects": 0, "scale_available": False}],
                                 {"video_sha256": "empty"}, {"min_track_observations": 2})
        measurements, frames, tracks, summary, meta = self.store.load(run_id)
        self.assertEqual(summary["measurements"], 0)
        self.assertEqual(summary["eligible_tracks"], 0)
        self.assertIsNone(summary["track_diameter_median_mm"])
        self.assertEqual(summary["detection_ratio"], 0)

    def test_video_database_and_reexport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "balls.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
            self.assertTrue(writer.isOpened())
            for i in range(6):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.circle(frame, (70+2*i, 110), 20, (255, 255, 255), -1)
                cv2.circle(frame, (240-2*i, 110), 25, (255, 255, 255), -1)
                writer.write(frame)
            writer.release()
            cal = root / "calibration.json"
            cal.write_text(json.dumps({"schema": "mvb.scalar_calibration.v1", "width_px": 320, "scale_mm_per_px": .1, "sigma_scale_mm_per_px": .001}))
            result = analyze_series(VideoAnalysisConfig(str(video), str(root / "first"), frame_step=1), DetectionConfig(min_ball_diameter_mm=3, max_ball_diameter_mm=6), "fov", calibration_path=cal, cell_mask=False, storage=self.store)
            self.assertEqual(result["storage"], "clickhouse")
            self.assertEqual(result["measurements"], 12)
            exported = self.store.export(result["run_id"], root / "second")
            self.assertEqual(result, exported)
            self.assertEqual((root / "first/measurements.csv").read_bytes(), (root / "second/measurements.csv").read_bytes())
            self.assertEqual(result["tracks"], 2)

    def test_unpublished_partial_run_is_not_readable(self):
        from uuid import uuid4
        run_id = str(uuid4())
        self.store.initialize()
        self.store.request(f"INSERT INTO {self.store.database}.python_frames_v1 FORMAT JSONEachRow", [{"run_id": run_id, "frame_index": 0, "time_s": 0., "objects": 0, "scale_available": False}])
        with self.assertRaisesRegex(ValueError, "Completed"):
            self.store.load(run_id)
