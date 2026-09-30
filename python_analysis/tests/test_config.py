import tempfile
import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvb.config import load_key_value_config


class ConfigTests(unittest.TestCase):
    def test_load_key_value_config(self) -> None:
        text = "\n".join(
            [
                "# comment",
                "video_path = Ball_For_MV.mp4",
                "frame_step = 12",
                "interactive_preview = false",
                "reference_sigma_mm = 0.05",
            ]
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        config = load_key_value_config(temp_path)
        self.assertEqual(config["video_path"], "Ball_For_MV.mp4")
        self.assertEqual(config["frame_step"], 12)
        self.assertIs(config["interactive_preview"], False)
        self.assertAlmostEqual(float(config["reference_sigma_mm"]), 0.05)


if __name__ == "__main__":
    unittest.main()
