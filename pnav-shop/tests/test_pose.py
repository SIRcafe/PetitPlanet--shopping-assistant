"""Pose / odometry / shop-localisation tests (pure math, no game)."""

from __future__ import annotations

import math
import sys
import pathlib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pnavshop import pose, spec  # noqa: E402
from pnavshop.pose import Odometry, Pose, ShopEstimator, bearing_to, correct, wrap180


def dot_for(true_pose: Pose, shop, noise_deg: float = 0.0) -> float:
    """What the blue dot would sit at for this pose/shop pair."""
    rel = wrap180(bearing_to(true_pose, shop) - true_pose.theta) + noise_deg
    return 960.0 + rel / spec.BEARING_DEG_PER_PX


class PoseMathTests(unittest.TestCase):
    def test_walk_north(self) -> None:
        o = Odometry()
        p = o.walk(Pose(0, 0, 0), 1000)
        self.assertAlmostEqual(p.y, spec.WALK_SPEED_MPS, places=6)
        self.assertAlmostEqual(p.x, 0.0, places=6)

    def test_walk_east_after_90_deg_turn(self) -> None:
        o = Odometry()
        p = o.turn(Pose(), 2000)          # 90 deg right
        self.assertAlmostEqual(p.theta, 90.0, places=6)
        p = o.walk(p, 1000)
        self.assertAlmostEqual(p.x, spec.WALK_SPEED_MPS, places=6)
        self.assertAlmostEqual(p.y, 0.0, places=6)

    def test_turn_quantisation_round_trip(self) -> None:
        o = Odometry()
        for deg in (-180, -90, -30, 0, 45, 90, 137):
            px = o.turn_px(deg)
            got = o.turn(Pose(), px).theta
            self.assertAlmostEqual(wrap180(got - deg), 0.0, places=1)

    def test_heading_vector(self) -> None:
        self.assertAlmostEqual(Pose(0, 0, 0).heading[1], 1.0, places=6)
        hx, hy = Pose(0, 0, 90).heading
        self.assertAlmostEqual(hx, 1.0, places=6)
        self.assertAlmostEqual(hy, 0.0, places=6)

    def test_bearing_convention(self) -> None:
        self.assertAlmostEqual(bearing_to(Pose(), (0, 10)), 0.0, places=6)
        self.assertAlmostEqual(bearing_to(Pose(), (10, 0)), 90.0, places=6)
        self.assertAlmostEqual(bearing_to(Pose(), (-10, 0)), -90.0, places=6)

    def test_wrap180(self) -> None:
        self.assertEqual(wrap180(190), -170)
        self.assertEqual(wrap180(-190), 170)
        self.assertEqual(wrap180(0), 0)
        self.assertAlmostEqual(wrap180(180), 180)

    def test_screen_offset_is_positive_to_the_right(self) -> None:
        self.assertGreater(pose.screen_bearing_deg(1100), 0)
        self.assertLess(pose.screen_bearing_deg(820), 0)


class ShopEstimatorTests(unittest.TestCase):
    def test_recovers_position_from_bearings(self) -> None:
        shop = (12.0, 25.0)
        est = ShopEstimator()
        for p in (Pose(0, 0, 0), Pose(14, 4, 55), Pose(20, 14, -25)):
            est.add(p, dot_for(p, shop))
        self.assertGreaterEqual(est.parallax_deg(), 10.0)
        sol = est.solve()
        self.assertIsNotNone(sol)
        err = math.hypot(sol["x"] - shop[0], sol["y"] - shop[1])
        self.assertLess(err, 0.5)

    def test_parallax_is_zero_when_standing_still(self) -> None:
        est = ShopEstimator()
        p = Pose(0, 0, 0)
        est.add(p, 1000)
        est.add(p, 1000)
        self.assertEqual(est.parallax_deg(), 0.0)

    def test_not_enough_observations(self) -> None:
        est = ShopEstimator()
        est.add(Pose(), 1000)
        self.assertIsNone(est.solve())


class CorrectionTests(unittest.TestCase):
    def test_heading_is_corrected_by_the_dot(self) -> None:
        shop = (0.0, 30.0)
        drifted = Pose(0, 0, 12.0)          # our estimate says we are rotated
        fixed = correct(drifted, shop, dot_for(drifted, shop))
        self.assertAlmostEqual(wrap180(fixed.theta), wrap180(drifted.theta), places=1)

    def test_position_is_nudged_towards_the_reported_range(self) -> None:
        shop = (0.0, 30.0)
        p = Pose(0.0, 0.0, 0.0)
        moved = correct(p, shop, dot_for(p, shop), meters=20.0)
        self.assertGreater(moved.y, 0.0)
        self.assertLess(moved.y, 20.0)       # half step, not a jump

    def test_tiny_range_error_is_ignored(self) -> None:
        shop = (0.0, 30.0)
        p = Pose(0.0, 0.0, 0.0)
        moved = correct(p, shop, dot_for(p, shop), meters=29.8)
        self.assertAlmostEqual(moved.y, 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
