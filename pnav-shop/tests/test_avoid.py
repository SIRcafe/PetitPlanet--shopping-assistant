"""Unit tests for the front walkability profile (`pnav/avoid.py`).

Synthetic scenes only: grass ground with (or without) a solid prop standing in
the middle of the path.  Real frames are checked with `scripts/_avoid_probe.py`.
"""

from __future__ import annotations

import sys
import pathlib
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pnavshop import avoid  # noqa: E402

H, W = 540, 960
GRASS = (70, 110, 55)


def scene(prop: bool = False) -> np.ndarray:
    rng = np.random.default_rng(11)
    img = np.zeros((H, W, 3), np.float32)
    # ground: grass with a vertical brightness gradient (distance shading)
    grad = np.linspace(1.25, 0.85, H)[:, None, None]
    img[..., :] = np.array(GRASS, np.float32) * grad
    img += rng.integers(-12, 13, img.shape)
    if prop:
        # a solid object: different colour, hard edges, standing mid-screen
        x0, x1 = int(W * 0.44), int(W * 0.56)
        y0, y1 = int(H * 0.55), int(H * 0.92)
        img[y0:y1, x0:x1] = np.array((40, 60, 95), np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


class ProfileTests(unittest.TestCase):
    def test_open_ground_is_free_everywhere(self) -> None:
        prof = avoid.profile(scene(prop=False))
        frees = [p["free"] for p in prof]
        self.assertTrue(all(f >= avoid.MIN_FREE for f in frees), frees)
        self.assertEqual(avoid.steer(prof, dot_dx=0)["nudge"], 0)

    def test_prop_blocks_the_centre_only(self) -> None:
        prof = avoid.profile(scene(prop=True))
        frees = [p["free"] for p in prof]
        c = avoid.center_index(len(prof))
        self.assertLess(frees[c], avoid.MIN_FREE)
        self.assertGreater(max(frees), avoid.SIDE_FREE)

    def test_steer_goes_around_the_prop(self) -> None:
        prof = avoid.profile(scene(prop=True))
        res = avoid.steer(prof, dot_dx=0)
        self.assertNotEqual(res["nudge"], 0)
        self.assertIn("blocked", res["why"])

    def test_steer_prefers_the_side_the_dot_is_on(self) -> None:
        """With the target to our right, a blocked centre should nudge right."""
        prof = avoid.profile(scene(prop=True))
        right = avoid.steer(prof, dot_dx=+400)
        left = avoid.steer(prof, dot_dx=-400)
        self.assertGreater(right["nudge"], 0)
        self.assertLess(left["nudge"], 0)

    def test_empty_inputs_are_safe(self) -> None:
        self.assertEqual(avoid.steer([], None)["nudge"], 0)
        blank = np.zeros((10, 10, 3), np.uint8)
        self.assertEqual(avoid.profile(blank), [])  # too small to strip

    def test_profile_bins_have_frame_coordinates(self) -> None:
        prof = avoid.profile(scene(prop=False))
        xs = [p["cx"] for p in prof]
        self.assertEqual(xs, sorted(xs))
        self.assertGreater(xs[0], 0)
        self.assertLess(xs[-1], 1920)


if __name__ == "__main__":
    unittest.main()
