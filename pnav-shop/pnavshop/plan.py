r"""Optimistic A* over the occupancy grid + string-pulling.

"Optimistic" is the important word: cells we have never seen are given a small
surcharge instead of being treated as walls, so the planner always prefers the
straight line toward the shop and only bends around what we have actually hit
(or what the walkability fan has marked).  With small, arbitrary-shaped clutter
that beats boundary-following by a wide margin; boundary-following stays as the
fallback when the planner cannot find anything.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from . import spec
from .gridmap import BLOCKED, JUMP, GridMap


# --------------------------------------------------------------------------- #
# A*
# --------------------------------------------------------------------------- #
_NEIGHBOURS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
               (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)))


def astar(gm: GridMap, start_xy: tuple[float, float], goal_xy: tuple[float, float],
          dilate: int = spec.GRID_INFLATE_CELLS,
          unknown_cost: float = spec.GRID_UNKNOWN_COST,
          jump_cost: float = spec.GRID_JUMP_COST,
          max_expand: int = 400_000) -> list[tuple[float, float]] | None:
    """Shortest-ish grid path in world coordinates, or None if unreachable."""
    cost = gm.cost_array(dilate=dilate, unknown_cost=unknown_cost, jump_cost=jump_cost)
    si, sj = gm.to_cell(*start_xy)
    gi, gj = gm.to_cell(*goal_xy)
    if not gm.inside(si, sj) or not gm.inside(gi, gj):
        return None
    if gm.state(gi, gj) == BLOCKED:
        # aim at the nearest open cell next to the goal
        gi, gj = _nearest_open(gm, gi, gj) or (gi, gj)

    n = gm.n
    flat_cost = cost.ravel()
    dist = np.full(n * n, np.inf, np.float64)
    prev = np.full(n * n, -1, np.int64)
    start = si * n + sj
    goal = gi * n + gj
    dist[start] = 0.0

    def h(i: int, j: int) -> float:
        return math.hypot(i - gi, j - gj)

    heap = [(h(si, sj), 0.0, start)]
    seen = 0
    while heap:
        _f, g, cur = heapq.heappop(heap)
        if cur == goal:
            break
        if g > dist[cur] + 1e-9:
            continue
        ci, cj = divmod(int(cur), n)
        seen += 1
        if seen > max_expand:
            return None
        for di, dj, step in _NEIGHBOURS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < n and 0 <= nj < n):
                continue
            nxt = ni * n + nj
            c = flat_cost[nxt]
            if not np.isfinite(c):
                continue
            # diagonal moves must not cut a corner
            if di and dj:
                if not np.isfinite(flat_cost[ci * n + nj]) or \
                   not np.isfinite(flat_cost[ni * n + cj]):
                    continue
            ng = g + c * step
            if ng + 1e-9 < dist[nxt]:
                dist[nxt] = ng
                prev[nxt] = cur
                heapq.heappush(heap, (ng + h(ni, nj), ng, nxt))
    if not np.isfinite(dist[goal]):
        return None

    cells = []
    cur = goal
    while cur != -1:
        cells.append(divmod(int(cur), n))
        cur = int(prev[cur])
    cells.reverse()
    return [gm.to_world(i, j) for i, j in cells]


def _nearest_open(gm: GridMap, i: int, j: int, radius: int = 4):
    for r in range(1, radius + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                if gm.inside(i + di, j + dj) and gm.state(i + di, j + dj) != BLOCKED:
                    return i + di, j + dj
    return None


# --------------------------------------------------------------------------- #
# smoothing / waypoint extraction
# --------------------------------------------------------------------------- #
def smooth(gm: GridMap, path: list[tuple[float, float]],
           dilate: int = spec.GRID_INFLATE_CELLS) -> list[tuple[float, float]]:
    """String pulling: keep only the corners needed to stay clear."""
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        # furthest visible waypoint (grid A* is dense, so a coarse scan is fine)
        while j > i + 1:
            if _clear(gm, path[i], path[j], dilate):
                break
            j -= 1
        out.append(path[j])
        i = j
    return out


def _clear(gm: GridMap, a, b, dilate: int, step_m: float = 0.2) -> bool:
    blocked = gm.blocked_mask(dilate=dilate)
    n = gm.n
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    steps = max(2, int(length / step_m) + 1)
    for k in range(steps + 1):
        t = k / steps
        i, j = gm.to_cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if not (0 <= i < n and 0 <= j < n) or blocked[i, j]:
            return False
    return True


def plan_route(gm: GridMap, pose_xy: tuple[float, float],
               goal_xy: tuple[float, float],
               dilate: int = spec.GRID_INFLATE_CELLS,
               unknown_cost: float = spec.GRID_UNKNOWN_COST) -> list[tuple[float, float]] | None:
    """A* + smoothing; falls back to the straight line when that is clear."""
    if gm.line_of_sight(pose_xy, goal_xy):
        return [pose_xy, goal_xy]
    raw = astar(gm, pose_xy, goal_xy, dilate=dilate, unknown_cost=unknown_cost)
    if raw is None:
        return None
    return smooth(gm, raw, dilate=dilate)


def next_waypoint(path: list[tuple[float, float]] | None,
                  pose_xy: tuple[float, float], reach_m: float = 0.6):
    """The first waypoint that is still meaningfully ahead of us."""
    if not path:
        return None
    for p in path:
        if math.hypot(p[0] - pose_xy[0], p[1] - pose_xy[1]) > reach_m:
            return p
    return path[-1]


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def path_length(path: list[tuple[float, float]] | None) -> float:
    if not path or len(path) < 2:
        return 0.0
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def count_states(gm: GridMap) -> dict[str, int]:
    return dict(free=int((gm.grid == 1).sum()),
                blocked=int((gm.grid == BLOCKED).sum()),
                jump=int((gm.grid == JUMP).sum()),
                unknown=int((gm.grid == 0).sum()))
