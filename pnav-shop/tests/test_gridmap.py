"""Occupancy-grid tests: arbitrary shapes, optimistic cost, fan voting."""

from __future__ import annotations

import math
import sys
import pathlib
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pnavshop import spec  # noqa: E402
from pnavshop.gridmap import BLOCKED, FREE, JUMP, UNKNOWN, GridMap  # noqa: E402
from pnavshop.pose import Pose  # noqa: E402


class BasicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gm = GridMap(cell_m=0.4, half_m=10.0)     # 50x50 cells

    def test_index_round_trip(self) -> None:
        for x, y in ((0.0, 0.0), (3.3, -7.1), (-9.5, 9.5)):
            i, j = self.gm.to_cell(x, y)
            wx, wy = self.gm.to_world(i, j)
            self.assertLess(abs(wx - x), self.gm.cell)
            self.assertLess(abs(wy - y), self.gm.cell)

    def test_mark_free_does_not_erase_obstacles(self) -> None:
        self.gm.mark_blocked(1.0, 1.0, radius_m=0.4)
        self.gm.mark_free(1.0, 1.0, radius_m=0.4)
        i, j = self.gm.to_cell(1.0, 1.0)
        self.assertEqual(self.gm.state(i, j), BLOCKED)

    def test_blocked_cost_is_infinite_and_dilated(self) -> None:
        self.gm.mark_blocked(2.0, 2.0, radius_m=0.4)
        cost = self.gm.cost_array(dilate=1)
        i, j = self.gm.to_cell(2.0, 2.0)
        self.assertTrue(np.isinf(cost[i, j]))
        self.assertTrue(np.isinf(cost[i + 1, j]) or np.isinf(cost[i, j + 1]))

    def test_unknown_is_optimistic_not_wall(self) -> None:
        cost = self.gm.cost_array(dilate=0)
        self.assertTrue(np.isfinite(cost).all())
        self.assertAlmostEqual(float(cost[0, 0]), spec.GRID_UNKNOWN_COST, places=6)

    def test_free_is_cheaper_than_unknown(self) -> None:
        self.gm.mark_free(0.0, 0.0, radius_m=2.0)
        cost = self.gm.cost_array(dilate=0)
        i, j = self.gm.to_cell(0.0, 0.0)
        self.assertAlmostEqual(float(cost[i, j]), 1.0, places=6)
        self.assertGreater(float(cost[0, 0]), 1.0)


class MarkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gm = GridMap(cell_m=0.4, half_m=10.0)

    def test_mark_ahead_lands_in_front_not_at_the_pose(self) -> None:
        p = Pose(0.0, 0.0, 0.0)                 # facing north
        x, y = self.gm.mark_ahead(p, dist_m=1.0)
        self.assertGreater(y, 0.5)
        i, j = self.gm.to_cell(x, y)
        self.assertEqual(self.gm.state(i, j), BLOCKED)
        i0, j0 = self.gm.to_cell(0.0, 0.0)
        self.assertNotEqual(self.gm.state(i0, j0), BLOCKED)

    def test_jump_marking(self) -> None:
        self.gm.mark_blocked(1.0, 0.0, jumpable=True)
        i, j = self.gm.to_cell(1.0, 0.0)
        self.assertEqual(self.gm.state(i, j), JUMP)
        cost = self.gm.cost_array(dilate=0)
        self.assertAlmostEqual(float(cost[i, j]), spec.GRID_JUMP_COST, places=6)

    def test_fan_votes_only_on_unknown(self) -> None:
        p = Pose(0.0, 0.0, 0.0)
        fan = [{"free": 0.05}] * 9               # everything blocked ahead
        self.gm.mark_fan(fan, p, near_m=1.0)
        i, j = self.gm.to_cell(0.0, 1.0)
        self.assertEqual(self.gm.state(i, j), BLOCKED)

    def test_line_of_sight(self) -> None:
        self.gm.mark_blocked(0.0, 2.0, radius_m=0.6)
        self.assertFalse(self.gm.line_of_sight((0.0, 0.0), (0.0, 5.0)))
        self.assertTrue(self.gm.line_of_sight((0.0, 0.0), (5.0, 0.0)))

    def test_as_rgb_shape_and_colours(self) -> None:
        self.gm.mark_blocked(0.0, 0.0, radius_m=0.5)
        img = self.gm.as_rgb(pose=Pose(0, 0, 0), goal=(3, 3))
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.shape[0], self.gm.n)


if __name__ == "__main__":
    unittest.main()
