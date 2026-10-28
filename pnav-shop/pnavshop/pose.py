r"""Pose: dead reckoning from "walk ms" / "turn px", corrected by the shop.

The dock crossing leaves the character at the map origin facing NORTH, so the
coordinate system is anchored once per planet:

    x = east, y = north, theta = compass degrees clockwise from north
    heading vector = (sin(theta), cos(theta))

Two measurements keep the dead reckoning honest, both cheap:

* the blue dot's screen offset  -> bearing to the shop (relative to heading)
* the "N米" text under the dot   -> range to the shop

With the shop's position known (estimated from a few observations), an
observation yields the heading directly (``theta = bearing - screen_offset``)
and corrects the position along the range direction.  That is the whole
"single landmark SLAM" this game needs: one landmark, two scalars per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from . import spec


# --------------------------------------------------------------------------- #
# pose
# --------------------------------------------------------------------------- #
def wrap180(deg: float) -> float:
    """Wrap an angle to (-180, 180]."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


@dataclass(frozen=True)
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0          # compass degrees: 0 = north, + = clockwise

    @property
    def heading(self) -> tuple[float, float]:
        """Unit vector of where we face."""
        t = math.radians(self.theta)
        return math.sin(t), math.cos(t)

    def moved(self, meters: float) -> "Pose":
        hx, hy = self.heading
        return replace(self, x=self.x + hx * meters, y=self.y + hy * meters)

    def turned(self, degrees: float) -> "Pose":
        return replace(self, theta=wrap180(self.theta + degrees))


def bearing_to(pose: Pose, point: tuple[float, float]) -> float:
    """Compass bearing from `pose` to `point`, in degrees."""
    dx = point[0] - pose.x
    dy = point[1] - pose.y
    return math.degrees(math.atan2(dx, dy))     # atan2(east, north) = compass


def distance_to(pose: Pose, point: tuple[float, float]) -> float:
    return math.hypot(point[0] - pose.x, point[1] - pose.y)


# --------------------------------------------------------------------------- #
# odometry: what the input layer actually gives us
# --------------------------------------------------------------------------- #
class Odometry:
    def __init__(self, speed_mps: float = spec.WALK_SPEED_MPS,
                 px_per_deg: float = spec.TURN_PX_PER_DEG) -> None:
        self.speed = float(speed_mps)
        self.px_per_deg = float(px_per_deg)

    def walk(self, pose: Pose, ms: int) -> Pose:
        return pose.moved(self.speed * ms / 1000.0)

    def turn(self, pose: Pose, px: int) -> Pose:
        return pose.turned(px / self.px_per_deg)

    def walk_meters(self, meters: float) -> int:
        return int(round(meters / self.speed * 1000.0))

    def turn_px(self, degrees: float) -> int:
        return int(round(degrees * self.px_per_deg))


# --------------------------------------------------------------------------- #
# screen offsets -> angles
# --------------------------------------------------------------------------- #
def screen_offset_px(dot_cx: float, frame_cx: float = 960.0) -> float:
    return float(dot_cx) - float(frame_cx)


def screen_bearing_deg(dot_cx: float, frame_cx: float = 960.0,
                       deg_per_px: float = spec.BEARING_DEG_PER_PX) -> float:
    """Relative bearing of the shop: positive = to our right."""
    return screen_offset_px(dot_cx, frame_cx) * deg_per_px


def camera_turn_deg(px: int, deg_per_px: float = spec.CAMERA_DEG_PER_PX) -> float:
    """How much the camera yawed for a relative mouse move (debug/compat)."""
    return px * deg_per_px


# --------------------------------------------------------------------------- #
# shop localisation + pose correction
# --------------------------------------------------------------------------- #
class ShopEstimator:
    """Least-squares position of the shop from bearing (and optional range)."""

    def __init__(self) -> None:
        self.obs: list[tuple[float, float, float, float | None]] = []
        # (x, y, measured_bearing_deg, range_m or None)

    def add(self, pose: Pose, dot_cx: float, frame_cx: float = 960.0,
            meters: float | None = None) -> None:
        self.obs.append((pose.x, pose.y,
                         wrap180(pose.theta + screen_bearing_deg(dot_cx, frame_cx)),
                         float(meters) if meters is not None else None))

    def __len__(self) -> int:
        return len(self.obs)

    def parallax_deg(self) -> float:
        """How much the viewing direction changed: without it a fix is 1-D.

        Bearing-only triangulation needs the two rays to differ; standing still
        and re-reading the dot gives identical rays and a numerically singular
        problem, which is why this guard exists.
        """
        if len(self.obs) < 2:
            return 0.0
        bearings = [o[2] for o in self.obs]
        return max(bearings) - min(bearings) if bearings else 0.0

    def solve(self, guess: tuple[float, float] | None = None,
              iterations: int = 30, bearing_sigma: float = 3.0,
              range_sigma: float = 1.5) -> dict | None:
        """Gauss-Newton on 2 unknowns (shop x, y); returns residual stats."""
        if len(self.obs) < 2:
            return None
        # initial guess: walk out from the first observation
        if guess is None:
            x0, y0, b0, r0 = self.obs[0]
            dist = r0 if r0 is not None else 20.0
            rad = math.radians(b0)
            guess = (x0 + math.sin(rad) * dist, y0 + math.cos(rad) * dist)
        sx, sy = float(guess[0]), float(guess[1])

        def residuals(sx: float, sy: float) -> np.ndarray:
            out = []
            for (x, y, bearing, meters) in self.obs:
                pred = math.degrees(math.atan2(sx - x, sy - y))
                out.append(wrap180(pred - bearing) / bearing_sigma)
                if meters is not None:
                    out.append((math.hypot(sx - x, sy - y) - meters) / range_sigma)
            return np.asarray(out, np.float64)

        for _ in range(iterations):
            r = residuals(sx, sy)
            if r.size == 0:
                return None
            # numerical Jacobian, 2 columns
            eps = 1e-3
            jx = (residuals(sx + eps, sy) - r) / eps
            jy = (residuals(sx, sy + eps) - r) / eps
            jac = np.stack([jx, jy], axis=1)
            try:
                step, *_ = np.linalg.lstsq(jac, -r, rcond=None)
            except np.linalg.LinAlgError:
                return None
            sx += float(step[0])
            sy += float(step[1])
            if abs(step[0]) + abs(step[1]) < 1e-4:
                break

        r = residuals(sx, sy)
        return dict(x=sx, y=sy, rmse=float(np.sqrt((r ** 2).mean())),
                    n=len(self.obs))


def correct(pose: Pose, shop: tuple[float, float], dot_cx: float,
            meters: float | None = None, frame_cx: float = 960.0,
            position_gain: float = 0.5) -> Pose:
    """Fold one observation into the pose (shop position already known).

    * heading: the shop's true bearing minus what the screen says is our error
    * position: slide along the shop direction so the range matches (half step,
      to stay robust against a mis-read "N米")
    """
    b_true = bearing_to(pose, shop)
    theta = wrap180(b_true - screen_bearing_deg(dot_cx, frame_cx))
    out = Pose(pose.x, pose.y, theta)
    if meters is not None and meters > 0:
        dist = distance_to(out, shop)
        err = dist - float(meters)
        if abs(err) > 0.3:                       # ignore sub-metre noise
            b = math.radians(bearing_to(out, shop))
            k = err * position_gain
            out = Pose(out.x + math.sin(b) * k, out.y + math.cos(b) * k, out.theta)
    return out
