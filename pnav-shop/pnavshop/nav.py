r"""Navigator v2: grid map + optimistic A*, driven only by what the game shows.

The class is deliberately *pure policy*: it consumes one observation per call
and returns one action.  No screenshots, no Win32, no globals - so it can be
driven by the real pipeline (`scripts/visit_and_shoot.py`) or by the synthetic
world in `pnav/sim.py`, and it can be unit-tested offline.

Observation (what the caller can measure in game)::

    dot_cx    blue-dot centre x in the frame (960 = dead ahead), or None if lost
    meters    the "N米" number under the dot, or None when it is not shown
    fan       pnav/avoid.profile() list (9 bins of walkability), or None
    collided  True when a walk step produced no movement (stuck)
    jump      True when a jump badge was seen (low obstacle we can hop)

Action (what the caller should execute)::

    {"kind": "turn", "px": +-N}         chunked relative mouse move
    {"kind": "walk", "ms": N}           hold forward
    {"kind": "jump"}                    press space (then keep walking)
    {"kind": "done"}                    the shop is reached
    {"kind": "give_up", "why": "..."}   nothing left to try

Field rules kept intact: the distance is the progress signal, "no distance has
ever shown" is not stuck, a hidden dot is not stuck, and the user's escape
sequence (180 deg -> 1 s -> random +-90 deg -> 2 s) runs only when the planner
has nothing to offer.
"""

from __future__ import annotations

import math
import random

from . import spec
from .gridmap import GridMap
from .plan import next_waypoint, plan_route
from .pose import (Odometry, Pose, ShopEstimator, bearing_to, correct,
                   distance_to, screen_bearing_deg, wrap180)


class Navigator:
    def __init__(self, gm: GridMap | None = None, odom: Odometry | None = None,
                 frame_cx: float = 960.0, aim_tol: float = spec.AIM_TOLERANCE_DEG,
                 walk_ms: int = spec.WALK_STEP_MS,
                 arrive_m: float = 1.0, seed: int | None = None) -> None:
        self.gm = gm if gm is not None else GridMap()
        self.odom = odom if odom is not None else Odometry()
        self.frame_cx = float(frame_cx)
        self.aim_tol = float(aim_tol)
        self.walk_ms = int(walk_ms)
        self.arrive_m = float(arrive_m)

        self.pose = Pose()
        self.est = ShopEstimator()
        self.shop: tuple[float, float] | None = None
        # every "bearing + N米" observation yields a shop position directly;
        # keep the last few and take the median (single readings are noisy)
        self._shop_obs: list[tuple[float, float]] = []

        self.stuck = 0
        self.actions = 0
        self.mode = "seek"           # seek | plan | escape
        self._last_walk_ms = 0
        self._escape: list[dict] = []
        self._lost = 0                # consecutive frames without the dot
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ facts
    @property
    def has_shop(self) -> bool:
        return self.shop is not None

    def distance_to_shop(self) -> float | None:
        return None if self.shop is None else distance_to(self.pose, self.shop)

    # ------------------------------------------------------------------ step
    def step(self, obs: dict) -> dict:
        self.actions += 1
        self._absorb(obs)

        if obs.get("jump"):
            self.gm.mark_jump_here(self.pose)
        if obs.get("fan"):
            self.gm.mark_fan(obs["fan"], self.pose)
        self.gm.mark_free(self.pose.x, self.pose.y, 0.45)

        if self._arrived(obs):
            return {"kind": "done"}

        if self._escape:
            return self._escape.pop(0)

        if obs.get("collided"):
            self.gm.mark_ahead(self.pose)
            self.stuck += 1
            self._last_walk_ms = 0
            if self.stuck >= 2:
                self._begin_escape()
                return self._escape.pop(0)
        else:
            self.stuck = 0

        if self.shop is None:
            return self._seek(obs)
        return self._follow_plan()

    # ------------------------------------------------------------------ internals
    def _absorb(self, obs: dict) -> None:
        """Fold the previous action's outcome + this observation into the pose."""
        if obs.get("collided") and self._last_walk_ms:
            # we bumped into something: undo most of the optimistic advance
            back = self.odom.speed * 0.7 * self._last_walk_ms / 1000.0
            self.pose = self.pose.moved(-back)
        self._last_walk_ms = 0

        dot_cx = obs.get("dot_cx")
        meters = obs.get("meters")
        if dot_cx is None:
            return

        # bearing + range places the shop directly (no triangulation needed);
        # this is what makes the goal a *point* instead of only a direction
        b_world = wrap180(self.pose.theta + screen_bearing_deg(dot_cx, self.frame_cx))
        if meters is not None and meters >= 2:
            rad = math.radians(b_world)
            self._shop_obs.append((self.pose.x + math.sin(rad) * meters,
                                   self.pose.y + math.cos(rad) * meters))
            if len(self._shop_obs) > 5:
                self._shop_obs.pop(0)

        self.est.add(self.pose, dot_cx, self.frame_cx, meters)
        if self._shop_obs:
            xs = sorted(o[0] for o in self._shop_obs)
            ys = sorted(o[1] for o in self._shop_obs)
            self.shop = (xs[len(xs) // 2], ys[len(ys) // 2])
            if self.mode == "seek":
                self.mode = "plan"
        elif self.shop is None and len(self.est) >= 2:
            # no range yet: try a bearing-only fix (needs parallax to work)
            if self.est.parallax_deg() >= 10.0:
                sol = self.est.solve()
                if sol is not None and sol["rmse"] < 4.0:
                    self.shop = (sol["x"], sol["y"])
                    self.mode = "plan"

        if self.shop is not None:
            self.pose = correct(self.pose, self.shop, dot_cx, meters,
                                frame_cx=self.frame_cx)

    def _arrived(self, obs: dict) -> bool:
        if self.shop is None:
            return False
        d = distance_to(self.pose, self.shop)
        dot_cx = obs.get("dot_cx")
        aligned = dot_cx is not None and \
            abs(dot_cx - self.frame_cx) * spec.BEARING_DEG_PER_PX < 20.0
        meters = obs.get("meters")
        close = d <= self.arrive_m or (meters is not None and meters <= 1)
        return bool(close and (aligned or meters is not None))

    def _seek(self, obs: dict) -> dict:
        """No shop position yet: pure pursuit on the dot while gathering bearings."""
        dot_cx = obs.get("dot_cx")
        if dot_cx is None:
            # A hidden dot can mean "a prop covers it" (walk on) or "the shop is
            # behind us" (turn and look).  Field rule says walk first; only if
            # that does not reveal it, sweep the horizon - never open the map.
            self._lost += 1
            if self._lost <= 2:
                return self._walk(self.walk_ms)
            if self._lost <= 10:                 # 8 x 45 deg ~ full circle
                return self._turn(45.0)
            self._lost = 0
            return self._walk(self.walk_ms)
        self._lost = 0
        err = (dot_cx - self.frame_cx) * spec.BEARING_DEG_PER_PX
        if abs(err) > self.aim_tol:
            return self._turn(err)
        return self._walk(self.walk_ms)

    def _follow_plan(self) -> dict:
        assert self.shop is not None
        path = plan_route(self.gm, (self.pose.x, self.pose.y), self.shop)
        if path is None:
            self._begin_escape()
            return self._escape.pop(0)
        wp = next_waypoint(path, (self.pose.x, self.pose.y)) or self.shop
        err = wrap180(bearing_to(self.pose, wp) - self.pose.theta)
        if abs(err) > self.aim_tol:
            return self._turn(err)
        return self._walk(self.walk_ms)

    def _turn(self, err_deg: float) -> dict:
        px = self.odom.turn_px(err_deg)
        if abs(px) < 6:
            return self._walk(self.walk_ms)
        self.pose = self.odom.turn(self.pose, px)       # assume it is executed
        return {"kind": "turn", "px": px}

    def _walk(self, ms: int) -> dict:
        self._last_walk_ms = int(ms)
        self.pose = self.odom.walk(self.pose, int(ms))  # optimistic
        return {"kind": "walk", "ms": int(ms)}

    def _begin_escape(self) -> None:
        """User's sequence: 180 deg, walk 1 s, random +-90 deg, walk 2 s."""
        side = 1 if self._rng.random() < 0.5 else -1
        self.mode = "escape"
        self.stuck = 0
        self._last_walk_ms = 0
        self._escape = [
            self._turn(180.0),
            self._walk(1000),
            self._turn(side * 90.0),
            self._walk(2000),
        ]
