"""Blue-dot detection on a synthetic frame (no game screenshots needed)."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pnavshop import detect, pngio, spec  # noqa: E402


class MarkerTests(unittest.TestCase):
    def _frame_with_dot(self, x: int, y: int) -> np.ndarray:
        dot = pngio.load_image(spec.TEMPLATE_DIR / "point_blue.png", cache_dir=None)
        frame = np.full((1080, 1920, 3), 25, np.uint8)
        h, w = dot.shape[:2]
        frame[y:y + h, x:x + w] = dot[:, :, :3]
        return frame

    def test_finds_the_dot(self) -> None:
        result = detect.detect_blue_marker(self._frame_with_dot(900, 500))
        self.assertTrue(result and result.get("found"))
        self.assertAlmostEqual(result["cx"], 923, delta=6)
        self.assertAlmostEqual(result["cy"], 523, delta=6)

    def test_blank_frame_has_no_dot(self) -> None:
        frame = np.full((1080, 1920, 3), 25, np.uint8)
        result = detect.detect_blue_marker(frame)
        self.assertFalse(result and result.get("found"))


if __name__ == "__main__":
    unittest.main()
