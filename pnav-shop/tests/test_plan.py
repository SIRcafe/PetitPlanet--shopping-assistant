"""Planner tests: optimistic A*, detours, smoothing."""

from __future__ import annotations

import math
import sys
import pathlib
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pnavshop import spec  # noqa: E402
from pnavshop.gridmap import BLOCKED, GridMap  # noqa: E402
from pnavshop.plan import astar, next_waypoint, path_length, plan_route, smooth  # noqa: E402


def wall(gm: GridMap, x0: float, y0: float, x1: float, y1: float,
         gap: tuple[float, float] | None = None, thick: float = 0.4) -> None:
    """Paint a straight wall (with an optional gap along the varying axis)."""
    steps = max(2, int(math.hypot(x1 - x0, y1 - y0) / 0.1))
    horizontal = abs(x1 - x0) >= abs(y1 - y0)
    for k in range(steps + 1):
        t = k / steps
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        along = x if horizontal else y
        if gap is not None and min(gap) <= along <= max(gap):
            continue
        gm.mark_blocked(x, y, radius_m=thick)


class AstarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gm = GridMap(cell_m=0.4, half_m=15.0)

    def test_clear_path_is_the_straight_line(self) -> None:
        path = plan_route(self.gm, (0.0, 0.0), (0.0, 8.0))
        self.assertEqual(len(path), 2)

    def test_wall_with_gap_is_avoided(self) -> None:
        # a wall across y = 4 with a 3 m gap (narrower gaps get closed by the
        # body radius + dilation, which is exactly how the real map behaves)
        wall(self.gm, -6.0, 4.0, 6.0, 4.0, gap=(2.5, 5.5))
        self.assertFalse(self.gm.line_of_sight((0.0, 0.0), (0.0, 8.0)))
        path = plan_route(self.gm, (0.0, 0.0), (0.0, 8.0))
        self.assertIsNotNone(path)
        xs = []
        for a, b in zip(path, path[1:]):
            for k in range(21):
                t = k / 20.0
                xs.append(a[0] + (b[0] - a[0]) * t)
        self.assertGreaterEqual(max(xs), 2.0)          # it had to go sideways
        self.assertGreater(path_length(path), 8.0)
        self.assertLess(path_length(path), 16.0)

    def test_path_never_crosses_a_blocked_cell(self) -> None:
        wall(self.gm, -6.0, 4.0, 6.0, 4.0, gap=(0.5, 1.5))
        path = plan_route(self.gm, (0.0, 0.0), (0.0, 8.0))
        for x, y in path:
            i, j = self.gm.to_cell(x, y)
            self.assertNotEqual(self.gm.state(i, j), BLOCKED)

    def test_enclosed_goal_returns_none(self) -> None:
        for x in np.arange(-2.0, 2.1, 0.3):
            self.gm.mark_blocked(float(x), 6.0, radius_m=0.3)
            self.gm.mark_blocked(float(x), 4.0, radius_m=0.3)
        for y in np.arange(4.0, 6.1, 0.3):
            self.gm.mark_blocked(-2.0, float(y), radius_m=0.3)
            self.gm.mark_blocked(2.0, float(y), radius_m=0.3)
        self.assertIsNone(astar(self.gm, (0.0, 0.0), (0.0, 5.0)))

    def test_jumpable_cell_is_passable(self) -> None:
        # a closed ring of blocked cells around the goal
        for x in np.arange(-1.5, 1.51, 0.2):
            self.gm.mark_blocked(float(x), 4.0, radius_m=0.3)
            self.gm.mark_blocked(float(x), 2.0, radius_m=0.3)
        for y in np.arange(2.0, 4.01, 0.2):
            self.gm.mark_blocked(-1.5, float(y), radius_m=0.3)
            self.gm.mark_blocked(1.5, float(y), radius_m=0.3)
        self.assertIsNone(astar(self.gm, (0.0, 0.0), (0.0, 3.0), dilate=0))
        self.gm.mark_blocked(0.0, 4.0, radius_m=0.35, jumpable=True)
        self.assertIsNotNone(astar(self.gm, (0.0, 0.0), (0.0, 3.0), dilate=0))

    def test_smoothing_reduces_waypoints(self) -> None:
        for y in np.arange(1.0, 6.0, 0.4):
            self.gm.mark_blocked(2.0, float(y), radius_m=0.3)
        raw = astar(self.gm, (0.0, 0.0), (0.0, 7.0))
        self.assertIsNotNone(raw)
        self.assertGreater(len(raw), 4)
        self.assertLess(len(smooth(self.gm, raw)), len(raw))

    def test_next_waypoint_skips_reached_points(self) -> None:
        path = [(0.0, 0.0), (0.3, 0.0), (5.0, 0.0)]
        self.assertEqual(next_waypoint(path, (0.0, 0.0), reach_m=0.6), (5.0, 0.0))
        self.assertEqual(next_waypoint(None, (0.0, 0.0)), None)


if __name__ == "__main__":
    unittest.main()
