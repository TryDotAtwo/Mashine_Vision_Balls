from pathlib import Path
import csv
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mvb.simulation import Particle, generate_scene, evaluate_measurements
from mvb.series_analysis import CircleTracker


class SimulationTests(unittest.TestCase):
    def test_motion_stays_inside_bath_and_cools(self):
        particle = Particle(0, 3.7, 14, 0, 0)
        for _ in range(240):
            particle.advance(1/120, 17)
            self.assertLessEqual((particle.x_mm**2 + particle.y_mm**2)**0.5, 17 - 3.7/2 + 1e-9)
            self.assertGreater(particle.temperature_k, 77.4)
        self.assertLess(particle.temperature_k, 300)

    def test_global_assignment_beats_greedy(self):
        tracker = CircleTracker(200, 1)
        tracker.update([(0, 0, 5), (5, 0, 5)])
        self.assertEqual(tracker.update([(4, 0, 5), (100, 0, 5)]), [0, 1])

    def test_velocity_retains_ids_after_crossing(self):
        tracker = CircleTracker(30, 2)
        self.assertEqual(tracker.update([(10, 0, 5), (60, 0, 5)]), [0, 1])
        self.assertEqual(tracker.update([(25, 0, 5), (45, 0, 5)]), [0, 1])
        self.assertEqual(tracker.update([(30, 0, 5), (40, 0, 5)]), [1, 0])

    def test_seed_reproducibility_and_scorer_detects_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            video, metadata = generate_scene(Path(tmp)/"first", "single", 41, 12)
            _, repeated = generate_scene(Path(tmp)/"second", "single", 41, 12)
            self.assertEqual(metadata["truth"], repeated["truth"])
            self.assertEqual(metadata["video_sha256"], repeated["video_sha256"])
            _, other = generate_scene(Path(tmp)/"other", "single", 42, 12)
            self.assertNotEqual(metadata["truth"], other["truth"])
            rows = []
            for frame in metadata["truth"][1:]:
                obj = frame["objects"][0]
                rows.append({"frame_index": frame["frame_index"], "ball_center_x_px": obj["x_px"],
                             "ball_center_y_px": obj["y_px"], "diameter_mm": obj["diameter_mm"]+0.1,
                             "ball_id": 0})
            rows.append({**rows[-1], "ball_id": 1})
            path = Path(tmp)/"predictions.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            score = evaluate_measurements(metadata, path)
            self.assertEqual((score["tp"], score["fp"], score["fn"]), (11, 1, 1))
            self.assertAlmostEqual(score["diameter_rmse_mm"], 0.1)


if __name__ == "__main__":
    unittest.main()
