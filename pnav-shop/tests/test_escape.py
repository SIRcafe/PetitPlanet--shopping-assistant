"""Unit tests for the stuck diagnosis (`pnav/escape.py`).

Synthetic frames keep the tests independent of any game asset; the real frames
in `games/PetitPlanet Game/pic1/` are used by `scripts/escape_probe.py` instead.
"""

from __future__ import annotations

import sys
import pathlib
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pnavshop import escape, spec  # noqa: E402

H, W = 540, 960


def canvas(color, h: int = H, w: int = W) -> np.ndarray:
    img = np.zeros((h, w, 3), np.float32)
    img[..., :] = color
    return img


def noisy(img: np.ndarray, amp: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.integers(-amp, amp + 1, img.shape), 0, 255).astype(np.uint8)


def wall_frame() -> np.ndarray:
    return noisy(canvas((200, 196, 182)), 3, 1)


def water_frame() -> np.ndarray:
    img = canvas((64, 132, 188))
    # gentle horizontal ripple so it is not a mathematically flat field
    rows = np.sin(np.arange(H) / 9.0)[:, None, None] * 6
    return noisy(img + rows, 5, 2)


def tree_frame() -> np.ndarray:
    # grass with hard high-frequency detail: a canopy / hedge in front of us
    return noisy(canvas((70, 110, 55)), 45, 3)


def step_frame() -> np.ndarray:
    img = canvas((120, 118, 112))
    img[int(H * 0.70):] = np.array((185, 182, 175), np.float32)
    return noisy(img, 3, 4)


def railing_frame() -> np.ndarray:
    img = canvas((150, 120, 90))                      # wooden planks behind
    img = noisy(img, 25, 5)
    for x in range(0, W, 70):
        img[:, x:x + 14] = np.array((58, 110, 168), np.uint8)
    return img


def open_frame() -> np.ndarray:
    img = canvas((110, 105, 95))
    rng = np.random.default_rng(6)
    for _ in range(40):                               # gravel / plaza patches
        y, x = rng.integers(0, H - 60), rng.integers(0, W - 60)
        img[y:y + 60, x:x + 60] = rng.choice([150, 80, 175, 60])
    return noisy(img, 40, 7)


class FeatureTests(unittest.TestCase):
    def test_empty_frame_is_safe(self) -> None:
        self.assertEqual(escape.front_features(np.zeros((0, 0, 3), np.uint8)), {})
        self.assertEqual(escape.describe({}), "(no features)")

    def test_character_silhouette_is_masked(self) -> None:
        """A bright red character in the middle must not colour the statistics."""
        img = canvas((30, 30, 30))
        x0 = int(spec.STUCK_CHAR_BOX[0] * W)
        x1 = int(spec.STUCK_CHAR_BOX[2] * W)
        y0 = int(spec.STUCK_CHAR_BOX[1] * H)
        y1 = int(spec.STUCK_CHAR_BOX[3] * H)
        img[y0:y1, x0:x1] = np.array((220, 60, 40), np.float32)
        img = img.astype(np.uint8)

        masked = escape.front_features(img)
        unmasked = escape.front_features(img, char_box=False)
        self.assertLess(masked["warm"], 0.02)
        self.assertGreater(unmasked["warm"], 0.10)

    def test_water_is_blue_dominant(self) -> None:
        f = escape.front_features(water_frame())
        self.assertGreater(f["water"], 0.9)
        self.assertLess(f["foliage"], 0.1)

    def test_railing_and_water_are_separated(self) -> None:
        rail = escape.front_features(railing_frame())
        water = escape.front_features(water_frame())
        self.assertGreater(rail["rail_bar"], 0.05)
        self.assertLess(water["rail_bar"], 0.01, "water must not read as railing bars")


class ClassifyTests(unittest.TestCase):
    def kind(self, img: np.ndarray) -> str:
        return escape.classify(escape.front_features(img))["kind"]

    def test_wall(self) -> None:
        self.assertEqual(self.kind(wall_frame()), "wall")

    def test_water(self) -> None:
        self.assertEqual(self.kind(water_frame()), "water")

    def test_tree(self) -> None:
        self.assertEqual(self.kind(tree_frame()), "tree")

    def test_step(self) -> None:
        self.assertEqual(self.kind(step_frame()), "step")

    def test_railing(self) -> None:
        self.assertEqual(self.kind(railing_frame()), "railing")

    def test_open(self) -> None:
        self.assertEqual(self.kind(open_frame()), "open")

    def test_confidence_and_scores(self) -> None:
        v = escape.classify(escape.front_features(wall_frame()))
        self.assertGreater(v["confidence"], spec.STUCK_CONF_MIN)
        self.assertEqual(v["rule"], "argmax")
        self.assertEqual(set(v["scores"]),
                         {"wall", "tree", "water", "railing", "step", "open"})

    def test_blank_frame_is_not_a_wall(self) -> None:
        """All black (or all white) is a lost-focus screenshot, not an obstacle."""
        for colour, rule in (((0, 0, 0), "blank-frame"), ((255, 255, 255), "blank-frame")):
            v = escape.classify(escape.front_features(canvas(colour).astype(np.uint8)))
            self.assertEqual(v["kind"], "unknown", colour)
            self.assertEqual(v["rule"], rule, colour)

    def test_classify_without_features(self) -> None:
        self.assertEqual(escape.classify({})["kind"], "unknown")


class PlanTests(unittest.TestCase):
    def test_every_kind_has_a_plan(self) -> None:
        for k in escape.kinds():
            steps = escape.plan(k)
            self.assertTrue(steps, k)
            for s in steps:
                self.assertIn(s["do"], {"back", "turn", "walk", "jump"})

    def test_unknown_plan_is_the_validated_field_rule(self) -> None:
        steps = escape.plan("unknown")
        self.assertEqual(steps, [{"do": "turn", "px": 2222}, {"do": "walk", "ms": 2000}])

    def test_unknown_kind_falls_back(self) -> None:
        self.assertEqual(escape.plan("nonsense"), escape.plan("unknown"))

    def test_mirrored_flips_turns_only(self) -> None:
        right = escape.plan("wall")
        left = escape.plan("wall", mirrored=True)
        for a, b in zip(right, left):
            self.assertEqual(a["do"], b["do"])
            if a["do"] == "turn":
                self.assertEqual(b["px"], -a["px"])
            else:
                self.assertEqual(a, b)

    def test_plan_is_a_copy(self) -> None:
        steps = escape.plan("wall")
        steps[1]["px"] = 1
        self.assertEqual(escape.plan("wall")[1]["px"], 2222)

    def test_water_turns_harder_than_tree(self) -> None:
        def turn_px(kind: str) -> int:
            return sum(abs(s.get("px", 0)) for s in escape.plan(kind))
        self.assertGreater(turn_px("water"), turn_px("tree"))


if __name__ == "__main__":
    unittest.main()
