r"""Occupancy grid for one planet: arbitrary obstacle shapes, no assumptions.

Nothing here assumes rectangles or any other shape - the only question the map
answers is "has this cell been seen walkable, seen blocked, or not seen yet".
Obstacles are whatever the planet owner placed, so the grid is the honest
representation; a rectangle fit would only be a debugging convenience.

Cells are 0.4 m by default because furniture is often smaller than a metre.
The grid is anchored by the dock: after crossing it the character is at (0,0)
facing north, so world coordinates never need SLAM initialisation.

Four states::

    UNKNOWN  never observed      -> optimistic: passable at a small surcharge
    FREE     walked through / the walkability fan said clear
    BLOCKED  we bumped into it, or the fan said blocked
    JUMP     blocked but a jump badge showed up -> passable with a hop
"""

from __future__ import annotations

import math

import numpy as np

from . import spec
from .pose import Pose, bearing_to

UNKNOWN, FREE, BLOCKED, JUMP = 0, 1, 2, 3


class GridMap:
    def __init__(self, cell_m: float = spec.GRID_CELL_M,
                 half_m: float = spec.GRID_HALF_EXTENT_M) -> None:
        self.cell = float(cell_m)
        self.half = float(half_m)
        self.n = int(round(2.0 * self.half / self.cell))
        self.grid = np.zeros((self.n, self.n), np.int8)
        self._version = 0

    # ------------------------------------------------------------------ index
    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        j = int(math.floor((x + self.half) / self.cell))
        i = int(math.floor((y + self.half) / self.cell))
        return i, j

    def to_world(self, i: int, j: int) -> tuple[float, float]:
        return (-self.half + (j + 0.5) * self.cell,
                -self.half + (i + 0.5) * self.cell)

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.n and 0 <= j < self.n

    def state(self, i: int, j: int) -> int:
        if not self.inside(i, j):
            return BLOCKED          # outside the map counts as a wall
        return int(self.grid[i, j])

    # ------------------------------------------------------------------ edits
    def _paint(self, x: float, y: float, radius_m: float, value: int,
               only_if: tuple[int, ...] = (UNKNOWN, FREE)) -> None:
        ci, cj = self.to_cell(x, y)
        r = max(0, int(math.ceil(radius_m / self.cell)))
        for i in range(ci - r, ci + r + 1):
            for j in range(cj - r, cj + r + 1):
                if not self.inside(i, j):
                    continue
                wx, wy = self.to_world(i, j)
                if math.hypot(wx - x, wy - y) > radius_m:
                    continue
                if self.grid[i, j] in only_if:
                    self.grid[i, j] = value
        self._version += 1

    def mark_free(self, x: float, y: float, radius_m: float = 0.35) -> None:
        self._paint(x, y, radius_m, FREE, only_if=(UNKNOWN,))

    def mark_blocked(self, x: float, y: float,
                     radius_m: float = spec.BODY_RADIUS_M,
                     jumpable: bool = False) -> None:
        # a jump badge *upgrades* a cell we already know is solid: it is low,
        # not impassable, so BLOCKED -> JUMP is allowed in that direction only
        only_if = ((UNKNOWN, FREE, BLOCKED) if jumpable
                   else (UNKNOWN, FREE, JUMP))
        self._paint(x, y, radius_m, JUMP if jumpable else BLOCKED, only_if=only_if)

    def mark_ahead(self, pose: Pose, dist_m: float = 0.8,
                   width_m: float = 0.9, jumpable: bool = False) -> tuple[float, float]:
        """We walked forward and did not move: the blocker is *in front*."""
        hx, hy = pose.heading
        x = pose.x + hx * dist_m
        y = pose.y + hy * dist_m
        self.mark_blocked(x, y, radius_m=max(spec.BODY_RADIUS_M, width_m * 0.5),
                          jumpable=jumpable)
        return x, y

    def mark_fan(self, profile: list[dict], pose: Pose,
                 span: tuple[float, float] = (0.16, 0.84), frame_w: int = 1920,
                 near_m: float = 1.1, blocked_below: float = 0.40,
                 free_above: float = 0.65) -> None:
        """Fold the 9-bin walkability fan (pnav/avoid.py) into the grid.

        The bins are weak evidence, so they only fill UNKNOWN cells: a single
        bump into something stays the strongest observation we have.
        """
        if not profile:
            return
        n = len(profile)
        step = (span[1] - span[0]) / n
        for k, p in enumerate(profile):
            free = float(p.get("free", 0.0))
            frac = span[0] + (k + 0.5) * step
            rel = (frac * frame_w - frame_w / 2.0) * spec.BEARING_DEG_PER_PX
            b = math.radians(pose.theta + rel)
            x = pose.x + math.sin(b) * near_m
            y = pose.y + math.cos(b) * near_m
            if free < blocked_below:
                self._paint(x, y, 0.35, BLOCKED, only_if=(UNKNOWN,))
            elif free > free_above:
                self._paint(x, y, 0.35, FREE, only_if=(UNKNOWN,))

    def mark_jump_here(self, pose: Pose) -> None:
        """A jump badge appeared: the cell in front is low, not solid."""
        hx, hy = pose.heading
        self.mark_blocked(pose.x + hx * 0.6, pose.y + hy * 0.6,
                          radius_m=0.5, jumpable=True)

    # ------------------------------------------------------------------ query
    def cost_array(self, dilate: int = spec.GRID_INFLATE_CELLS,
                   unknown_cost: float = spec.GRID_UNKNOWN_COST,
                   jump_cost: float = spec.GRID_JUMP_COST) -> np.ndarray:
        """Per-cell traversal cost (inf = impassable), blocked cells dilated.

        Dilating the *known* obstacles keeps the body from grazing them; it is
        computed on the fly so the raw observations are never damaged.
        """
        blocked = self.grid == BLOCKED
        if dilate > 0 and blocked.any():
            d = blocked.copy()
            for _ in range(dilate):
                grown = d.copy()
                grown[1:, :] |= d[:-1, :]
                grown[:-1, :] |= d[1:, :]
                grown[:, 1:] |= d[:, :-1]
                grown[:, :-1] |= d[:, 1:]
                d = grown
            blocked = d
        cost = np.full(self.grid.shape, np.inf, np.float32)
        cost[~blocked] = unknown_cost
        cost[self.grid == FREE] = 1.0
        jump_mask = (self.grid == JUMP) & ~blocked
        cost[jump_mask] = jump_cost
        return cost

    def blocked_mask(self, dilate: int = spec.GRID_INFLATE_CELLS) -> np.ndarray:
        return np.isinf(self.cost_array(dilate=dilate))

    def line_of_sight(self, a: tuple[float, float], b: tuple[float, float],
                      step_m: float = 0.25) -> bool:
        """True when the straight segment a->b crosses no known obstacle."""
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(length / step_m) + 1)
        for k in range(n + 1):
            t = k / n
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            i, j = self.to_cell(x, y)
            if self.state(i, j) == BLOCKED:
                return False
        return True

    # ------------------------------------------------------------------ misc
    def known_fraction(self) -> float:
        return float((self.grid != UNKNOWN).mean())

    def as_rgb(self, pose: Pose | None = None, goal: tuple[float, float] | None = None,
               path: list[tuple[float, float]] | None = None) -> np.ndarray:
        """Small RGB debug image: white unknown, green free, red blocked."""
        img = np.zeros((self.n, self.n, 3), np.uint8)
        img[self.grid == UNKNOWN] = (235, 235, 235)
        img[self.grid == FREE] = (70, 170, 90)
        img[self.grid == BLOCKED] = (200, 60, 60)
        img[self.grid == JUMP] = (210, 170, 60)

        def dot(x: float, y: float, colour: tuple[int, int, int], r: int = 2) -> None:
            i, j = self.to_cell(x, y)
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if self.inside(i + di, j + dj):
                        img[i + di, j + dj] = colour

        if goal is not None:
            dot(goal[0], goal[1], (60, 120, 255), r=3)
        if pose is not None:
            dot(pose.x, pose.y, (30, 30, 30), r=2)
        if path:
            for x, y in path:
                dot(x, y, (90, 90, 255), r=0)
        return img
