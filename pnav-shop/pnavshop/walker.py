"""格连商店助手核心：寻路 + 进店截图。

从主项目里按「格连商店」这一页抽取，逻辑未改：
* 跟随蓝色导航点与 HUD 的「N 米」走到商店门口；
* 看到「F 进入格连商店」提示后按 F 进店，等画面亮起；
* 转圈连拍家具并拼成 `sheet.png`，一张图看全刷新。

不读写游戏内存、不注入、不装钩子：只用系统截图接口 + 模拟按键。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import threading
import time

import numpy as np

from . import avoid, detect, distance, escape, pngio, spec, uid, win32
from . import imgproc as ip

SHADOW_HWND = win32.shadow_from_env()
OUT_ROOT = spec.ROOT / "reports" / "runs"
ESCAPE_STATS = spec.ROOT / "reports" / "stuck_stats.json"
CX = 960
ESCAPE_WALK_MS = 1000

BOUNDS = dict(
    tab=(1783, 861), youhu=(553, 445), starfriends=(481, 36), dots=(1380, 1001),
    interact=(1713, 1005), visit=(1650, 708),
    friend_row0=(300, 190),
    friend_step=80,
    find_btn=(1698, 791),
)


class Walker:
    def __init__(self, a) -> None:
        self.a = a
        self.t0 = time.time()
        if SHADOW_HWND:
            # desktop shadow: the game lives on another desktop, find_game_window
            # enumerates the current one only, so the handle is passed in
            self.hwnd = int(SHADOW_HWND)
        else:
            g = win32.find_game_window()
            if not g:
                raise SystemExit("game window not found")
            self.hwnd = g["hwnd"]
        self.f_prompt = pngio.load_image(spec.TEMPLATE_DIR / spec.TEMPLATES["f_prompt"],
                                         cache_dir=spec.CACHE_DIR)
        self.tpl_avatar = pngio.load_image(spec.TEMPLATE_DIR / spec.TEMPLATES["avatar"],
                                              cache_dir=spec.CACHE_DIR)
        self.tpl_find = pngio.load_image(spec.TEMPLATE_DIR / spec.TEMPLATES["find_btn"],
                                        cache_dir=spec.CACHE_DIR)
        self.out = OUT_ROOT / a.label
        self.out.mkdir(parents=True, exist_ok=True)
        self.escape_stats = self._load_escape_stats()
        self.escape_s = 0.0
        self.bank = uid.load_bank(spec.TEMPLATE_DIR)
        self._avoid_cool = 0
        self._follow_dir = 1        # +1 = follow obstacles on the right
        self._no_gain = 0           # consecutive escapes without a distance drop
        self.blind_walks = 0        # forward steps taken while the dot is hidden
        self._dist_floor = None     # smallest "N米" seen so far, for sanity checks
        self._dist_seen = False     # has the "N米" text ever appeared this walk?
        self._frozen = 0            # consecutive "dot frozen" checks
        self.times: dict[str, float] = {}
        self._stop = None
        self.window_title = ""
        self._capture_mode = "auto"

    def log(self, m: str) -> None:
        print(f"[{time.time() - self.t0:6.1f}s] {m}", flush=True)

    def mark(self, name: str) -> None:
        """Record a stage timestamp so per-planet timings can be averaged."""
        self.times[name] = time.time() - self.t0
        self.log(f"  [t] {name} at {self.times[name]:.1f}s")

    def frame(self) -> np.ndarray:
        mode = getattr(self, "_capture_mode", "auto")
        method = "screen"
        if mode == "printwindow":
            method = "window"
        elif mode == "auto" and win32.foreground_window() != self.hwnd:
            method = "window"        # behind another window: ask DWM for the pixels
        return win32.capture_window(self.hwnd, method=method)

    def cleanup(self) -> None:
        return None

    def shot(self, name: str, img: np.ndarray | None = None) -> pathlib.Path:
        img = self.frame() if img is None else img
        p = self.out / f"{name}.png"
        pngio.save_png(p, img)
        self.log(f"  shot {name} mean={img.mean():.1f}")
        return p

    def click(self, xy: tuple[int, int], hold: int = 180) -> None:
        x, y = int(xy[0]), int(xy[1])
        win32.click(x, y, hold_ms=hold)
        self.log(f"  click {(x, y)}")
        time.sleep(0.45)

    def wait_stable(self, timeout: float = 60.0, tol: float = 1.2) -> bool:
        """Wait until two consecutive frames stop changing (load finished)."""
        prev = self.frame()
        end = time.time() + timeout
        while time.time() < end:
            time.sleep(0.7)
            cur = self.frame()
            diff = float(np.abs(cur.astype(np.int16) - prev.astype(np.int16)).mean())
            if diff < tol:
                self.log(f"  stable (diff={diff:.2f})")
                return True
            prev = cur
        self.log("  WARNING: still changing at timeout")
        return False

    def wait_world(self, timeout: float = 150.0) -> bool:
        """Wait for an actual game scene.

        Planet travel goes  white/loading (flat, bright) -> black fade -> world.
        Neither "stable" nor "bright" alone is sufficient: the loading screen is
        both, and the fade is stable.  Require brightness AND texture.
        """
        end = time.time() + timeout
        while time.time() < end:
            img = self.frame()
            mean = float(img.mean())
            centre = img[200:900, 300:1600].astype(np.float32)
            std = float(centre.std())
            if 45.0 <= mean <= 225.0 and std > 20.0 and min(img[700:950, 0:400].astype("float32").std(), img[620:900, 1520:1920].astype("float32").std()) > 30.0:
                self.log(f"  world up (mean={mean:.0f} std={std:.0f})")
                return True
            time.sleep(0.8)
        self.log("  WARNING: world never settled")
        return False

    def wait_bright(self, min_mean: float = 60.0, timeout: float = 30.0) -> bool:
        """Shop entry fades to black then back in; a black frame is perfectly
        'stable', so stability alone is not enough — wait for brightness too."""
        end = time.time() + timeout
        while time.time() < end:
            img = self.frame()
            if img.mean() >= min_mean:
                self.log(f"  interior lit (mean={img.mean():.0f})")
                return True
            time.sleep(0.5)
        self.log("  WARNING: still dark at timeout")
        return False

    @staticmethod
    def photo_ok(img: np.ndarray) -> tuple[bool, str]:
        """Reject camera-collision close-ups and dark frames.

        A wall-clamped camera fills the middle of the screen with the
        character's head: bright, low contrast.  A real room view has plenty of
        local contrast.
        """
        mean = float(img.mean())
        if mean < 45:
            return False, f"too dark ({mean:.0f})"
        if mean > 215:
            return False, f"washed out ({mean:.0f})"
        centre = img[300:900, 700:1220].astype(np.float32)
        std = float(centre.std())
        if std < 26:
            return False, f"flat centre (std={std:.0f}) - camera clamped"
        return True, f"ok (mean={mean:.0f} std={std:.0f})"

    def turn_px(self, total_px: int, chunk: int = 320, pause: float = 0.06) -> int:
        """Rotate by total_px in small chunks.

        A single huge SendInput delta is clamped by the game per frame, which is
        why a "200 degree" turn only moved a little.  320 px per chunk works.
        """
        sent, n = 0, 0
        while sent != total_px:
            step = chunk if total_px > 0 else -chunk
            if abs(total_px - sent) < chunk:
                step = total_px - sent
            win32.move_rel(step, 0)
            sent += step
            n += 1
            time.sleep(pause)
        self.log(f"    turn {total_px:+d}px in {n} chunk(s) of <= {chunk}px")
        return n

    def go_to_gelian(self, budget_s: float = 600.0, max_steps: int = 90) -> bool:
        """Follow the blue dot with the two recovery rules learned in the field:
        two full turns without the dot -> reopen the map and re-lock 格连;
        stuck against geometry -> back off, turn 45 degrees, retry, then strafe.
        """
        k, deadband = 0.6, 40
        steps, t_end = 0, time.time() + budget_s
        lost_turn = 0.0
        stuck = 0
        side = 1
        prev_dot = None
        TURN_PER_LOST = 700          # ~30 deg per re-acquire attempt
        TURN_45 = 1000               # ~45 deg
        TWO_CIRCLES = 14000          # ~2 x 360 deg

        self.log("walk: right-click to make sure we are running")
        win32.click_right(960, 540)
        time.sleep(0.5)
        self._avoid_cool = 0
        self._frozen = 0

        while time.time() < t_end and steps < max_steps:
            if self._stop is not None and self._stop.is_set():
                self.log("  stopped by user")
                return False
            img = self.frame()

            fp = detect.detect_f_prompt(img, self.f_prompt)
            if fp.get("found"):
                self.log(f"  arrival prompt found score={fp['score']:.3f}")
                self.mark("shop_found")
                return True

            res = detect.detect_blue_marker(img)
            if not res or not res.get("found"):
                # The dot is a UI overlay and props/NPCs do occlude it.  User
                # rule (2026-10-18): NEVER reopen the map to re-track 格连 -
                # clicking 寻找友邻 again actually turns the dot OFF.  Instead
                # keep walking the way we face until the dot shows up again.
                self.blind_walks += 1
                if self.blind_walks > 6:
                    self.log("  dot still hidden after 6 forward steps -> small search turn")
                    win32.move_rel(TURN_PER_LOST, 0)
                    self.blind_walks = 0
                    time.sleep(0.3)
                    continue
                self.log(f"  dot hidden (occluded) -> walk forward to reveal it "
                         f"({self.blind_walks}/6)")
                win32.hold_key(win32.VK["w"], 900)
                time.sleep(0.3)
                continue

            self.blind_walks = 0
            lost_turn = 0.0
            dx = res["cx"] - CX
            dist = self.read_distance(img, res)

            # 1) coarse aim: only rotate while the dot is far off-centre.
            #    Walking is never blocked by steering, which is what previously
            #    made the character stand still and wobble left/right.
            if abs(dx) > 120:
                mv = int(max(-200, min(200, k * dx)))
                win32.move_rel(mv, 0)
                self.log(f"  dot=({res['cx']:.0f},{res['cy']:.0f}) dx={dx:+.0f} "
                         f"{dist}米 aim {mv:+d}")
                time.sleep(0.28)
                continue

            # 2) obstacle check, at most once every few steps.
            #    The game makes every prop solid, so a blocked centre bin means
            #    something is standing on the path: nudge once toward the free
            #    side and then walk on for a few steps (a per-frame bias fights
            #    the dot term and just oscillates).
            if self._avoid_cool <= 0:
                st = avoid.steer(avoid.profile(img), dot_dx=dx)
                if st["nudge"]:
                    self.log(f"  object ahead -> nudge {st['nudge']:+d}px ({st['why']})")
                    self.log(f"    free bins {[round(f, 2) for f in st['frees']]}")
                    win32.move_rel(st["nudge"], 0)
                    self._avoid_cool = 3
                    time.sleep(0.3)
            else:
                self._avoid_cool -= 1

            # TODO(bridge): the colour-ratio test false-positives outdoors
            # (pic1/3.png -> 0.47), so bridge mode stays OFF until a railing
            # template from pic1/14.png is added.  Keeps the validated path.
            bridge, ratio = False, 0.0
            if bridge and abs(dx) <= 260:
                # on the bridge: go straight, no zig-zag into the railings
                win32.hold_key(win32.VK["w"], 1300)
                steps += 1
                self.log(f"  ON BRIDGE (blue={ratio:.3f}) -> straight, step {steps}")
                time.sleep(0.3)
                continue

            jumped, gold = detect.jump_badge(img)
            if jumped and abs(dx) <= 150:
                # confirm on a second frame: flowers and decorations score
                # 0.81-0.87, which used to trigger four jumps in a row
                time.sleep(0.25)
                again, gold2 = detect.jump_badge(self.frame())
                if again and gold2 >= 750:
                    self.log(f"  JUMP BADGE (score={gold2 / 1000.0:.2f}, dx={dx:+.0f}) -> space")
                    win32.key_name("space")
                    time.sleep(1.3)
                    continue
                self.log(f"  jump badge flickered (={gold2 / 1000.0:.2f}) -> ignore")

            win32.hold_key(win32.VK["w"], 1200)
            steps += 1
            time.sleep(0.45)

            # Progress is judged on the "N米" number, NOT on the dot's screen
            # position (the dot freezes when the target is close).  User rule:
            # the distance not shrinking means we are stuck -> act at once,
            # but only when the dot did not move either (otherwise we are just
            # sliding sideways and the next step may already line up).
            img2 = self.frame()
            dot2 = detect.detect_blue_marker(img2)
            dist2 = self.read_distance(img2, dot2, prev=dist)
            if dist is not None and dist2 == dist:
                # confirm once: the HUD may not have refreshed yet
                time.sleep(0.35)
                img2 = self.frame()
                dot2 = detect.detect_blue_marker(img2)
                dist2 = self.read_distance(img2, dot2, prev=dist)

            moved = 0.0
            if dot2 and dot2.get("found"):
                moved = abs(dot2["cx"] - res["cx"]) + abs(dot2["cy"] - res["cy"])

            if dist is not None and dist2 is not None:
                closer = dist2 < dist
                progress = f"{dist}米 -> {dist2}米"
            else:
                closer = moved > 25          # no readable text -> dot movement
                progress = f"{dist}米 -> {dist2}米 (no reading, fallback)"
            if dist2 is not None or dist is not None:
                self._dist_seen = True       # the HUD does show "N米" here
            self.log(f"  step {steps}: {progress} dot moved {moved:.0f}px")

            if closer:
                stuck = 0
                self._no_gain = 0
                self._frozen = 0
                continue

            if moved > 12 and dist2 == dist:
                # walking but not getting closer: usually a steering problem,
                # not a wall - re-aim instead of burning an escape
                stuck = 0
                self._frozen = 0
                self.log("  distance flat but still moving -> re-aim, no escape")
                continue

            self._frozen += 1
            if not self._dist_seen and self._frozen < 2:
                # User rule (2026-10-18 late): no distance reading at all means
                # the shop is simply very far away - 格连 is out of the range
                # where the HUD shows "N米".  Push again before calling it stuck.
                self.log("  no distance has ever shown (shop very far) + dot frozen "
                         "-> push again instead of escaping")
                continue

            stuck += 1
            self.log(f"  NO PROGRESS (stuck #{stuck}) -> hop first, then 100deg right")
            base_sig = detect.distance_signature(img2, dot2)
            if not self.do_escape(img2, dot2, base_sig):
                self.log("  escape budget exhausted -> stop and ask")
                self.shot("STUCK_stop")
                raise SystemExit(7)
            self._avoid_cool = 0
            probe = self.frame()
            dot3 = detect.detect_blue_marker(probe)
            freed = self.read_distance(probe, dot3, prev=dist)
            # Bug2 bookkeeping: did following the boundary actually get us
            # closer?  If this side gives no gain twice in a row, try the other.
            gained = (freed is not None and dist is not None and freed < dist)
            if gained:
                self._no_gain = 0
            else:
                self._no_gain += 1
                if self._no_gain >= 2:
                    self._follow_dir = -self._follow_dir
                    self._no_gain = 0
                    self.log("  no gain from this side -> switch boundary side to "
                             + ("right" if self._follow_dir > 0 else "left"))
            self.log(f"  freed -> resume following the dot (now {freed}米)")
            stuck = 0

        self.log("  gave up walking (budget/steps)")
        return False

    def read_distance(self, img: np.ndarray, dot: dict | None,
                      prev: int | None = None) -> int | None:
        """The "N米" number under the dot, or None when it is not readable.

        With `prev` given, an implausible jump (more than 3x plus 20) is
        treated as a mis-read: the digit templates occasionally glue a stray
        glyph to the number ("16米" read as "613"), and a fake 613 used to
        trigger a pointless escape.
        """
        if not dot or not dot.get("found"):
            return None
        try:
            meters = distance.read(img, dot, self.bank)["meters"]
        except Exception:
            return None
        if meters is None:
            return None
        # both the previous reading and the running minimum must agree, so a
        # bogus big number cannot become the new baseline ("720米" appeared
        # twice on 彩加's planet and used to be treated as real)
        floor = getattr(self, "_dist_floor", None)
        if floor is not None and meters > 3 * floor + 20:
            self.log(f"  (distance read {meters}米 vs floor {floor}米 -> ignored)")
            return None
        if prev is not None and meters > 3 * prev + 20:
            self.log(f"  (distance read {meters}米 looks wrong next to {prev}米 -> ignored)")
            return None
        self._dist_floor = meters if floor is None else min(floor, meters)
        return meters

    def do_escape(self, img: np.ndarray, dot: dict, base_sig) -> bool:
        """Get unstuck instead of blindly turning.

        Modes (--escape-mode):
          jump   (default, user rule 2026-10-18 night): try a hop first, because
                 most "stuck" cases are a kerb/root the character can climb;
                 if the hop does not free us, turn right 100 deg and walk 2 s.
          auto   diagnose wall/tree/water/railing/step, then pick the plan
                 (falls back to the validated 100 deg rule).
          fixed  the old blind "turn right 100 deg + walk 2 s".

        Success is judged on the "N米" text under the dot (bitmap change),
        because the dot's screen position freezes when the target is close.
        """
        self.escape_s = 0.0
        mode = getattr(self.a, "escape_mode", "jump")
        budget = float(getattr(self.a, "escape_budget_s", 60.0))
        diagnosed = "stuck"

        if mode == "jump":
            badge, score = detect.jump_badge(img) if img is not None else (False, 0)
            # boundary-follow direction: keep the same side (Bug2) instead of
            # flipping, and switch sides only when this side keeps failing
            # 2000 px ~= 90 degrees (user rule: first turn is a right 90)
            d = 2000 * (1 if getattr(self, "_follow_dir", 1) >= 0 else -1)
            side = "right" if d > 0 else "left"
            if badge:
                # a badge means the game itself says "you can climb here"
                self.log(f"  STUCK with a jump badge (score={score / 1000.0:.2f}) -> hop")
                attempts = [
                    ("jump", [{"do": "jump"}, {"do": "walk", "ms": 500}]),
                    (f"turn100{side}", [{"do": "turn", "px": d},
                                        {"do": "walk", "ms": ESCAPE_WALK_MS}]),
                ]
            else:
                # no badge: user rule (2026-10-18 latest) = U-turn 180 deg, walk
                # 1 s, then a RANDOM 90 deg left or right, walk 2 s, then go
                # looking for the blue dot again.
                side = random.choice((1, -1))
                side_name = "right" if side > 0 else "left"
                self.log(f"  STUCK, no jump badge -> turn 180deg, walk {ESCAPE_WALK_MS}ms, "
                         f"then random {side_name} 90deg, walk 2000ms")
                attempts = [
                    ("u180_side90", [
                        {"do": "turn", "px": 4000},              # 180 deg
                        {"do": "walk", "ms": ESCAPE_WALK_MS},    # 1 s
                        {"do": "turn", "px": side * 2000},       # random +-90
                        {"do": "walk", "ms": 2000},              # 2 s
                    ]),
                    # persistent: walk a loop around the obstacle
                    ("circle_obstacle", [{"do": "circle", "n": 12, "px": 600,
                                          "ms": 700}]),
                    ("hop_step", [{"do": "jump"}, {"do": "walk", "ms": ESCAPE_WALK_MS}]),
                ]
        elif mode == "auto":
            feats = escape.front_features(img)
            verdict = escape.classify(feats)
            diagnosed = verdict["kind"]
            self.log(f"  STUCK diagnosis: {diagnosed} (conf={verdict['confidence']:.2f})")
            self.log(f"    {escape.describe(feats)}")
            attempts = [(diagnosed, escape.plan(diagnosed))]
            if diagnosed != "unknown":
                attempts.append((f"{diagnosed}>mirror", escape.plan(diagnosed, mirrored=True)))
            attempts.append(("unknown", escape.plan("unknown")))
        else:
            attempts = [("unknown", escape.plan("unknown"))]
            self.log("  STUCK (fixed mode): turn right 100deg + walk 2s")

        for n, (label, steps) in enumerate(attempts, 1):
            if self.escape_s >= budget:
                self.log(f"  escape budget spent ({self.escape_s:.0f}s >= {budget:.0f}s)")
                return False
            self.log(f"    attempt {n}: {label} -> "
                     + " ".join(self._action_text(s) for s in steps))
            t0 = time.time()
            for s in steps:
                self.run_action(s)
            self.escape_s += time.time() - t0

            time.sleep(0.35)
            probe = self.frame()
            diff = float(np.abs(probe.astype(np.int16) - img.astype(np.int16)).mean()) \
                if img is not None and img.shape == probe.shape else 999.0
            if diff < 2.0:
                # nothing on screen changed: the input almost certainly went to
                # another window (a UAC prompt or the console took the focus)
                self.log(f"      screen unchanged (diff={diff:.1f}) -> refocus and retry")
                win32.bring_to_front(win32.find_game_window()["hwnd"])
                time.sleep(0.4)
                for s in steps:
                    self.run_action(s)
                time.sleep(0.35)
                probe = self.frame()

            dot2 = detect.detect_blue_marker(probe)
            sig2 = detect.distance_signature(probe, dot2)
            changed = detect.distance_changed(base_sig, sig2)
            moved = 0.0
            if dot and dot2 and dot.get("found") and dot2.get("found"):
                moved = abs(dot2["cx"] - dot["cx"]) + abs(dot2["cy"] - dot["cy"])
            freed = bool(changed or moved > 25)
            if not freed and not (dot2 and dot2.get("found")):
                # the dot is not visible, so neither "distance changed" nor "dot
                # moved" can be measured.  If the view itself changed we DID
                # move: report success and let the caller walk on and find it,
                # instead of burning every escape variant on a false negative.
                view_moved = diff > 8.0
                self.log(f"      no dot to judge by; view diff={diff:.1f} -> "
                         f"{'assume freed' if view_moved else 'assume stuck'}")
                freed = view_moved
            self.log(f"      distance changed={changed} dot moved={moved:.0f}px -> "
                     f"{'FREED' if freed else 'still stuck'}")
            self._record_escape(diagnosed, label, False, freed)
            if freed:
                return True
            if not (dot2 and dot2.get("found")) and dot and dot.get("found"):
                # we had the dot and lost it: hand back to the caller, whose
                # "walk forward until the dot reappears" rule takes over
                self.log("      dot out of view after the escape -> walk on and look for it")
                return True
        return False

    @staticmethod
    def _action_text(act: dict) -> str:
        what = act["do"]
        if what == "turn":
            return f"turn{int(act['px']):+d}px"
        if what == "jump":
            return "jump"
        return f"{what}{int(act['ms'])}ms"

    def run_action(self, act: dict) -> None:
        """Execute one escape action; turns are always chunked (HANDOFF pit 3)."""
        what = act["do"]
        if what == "turn":
            self.turn_px(int(act["px"]))
        elif what == "back":
            win32.hold_key(win32.VK["s"], int(act["ms"]))
            time.sleep(0.15)
        elif what == "walk":
            win32.hold_key(win32.VK["w"], int(act["ms"]))
            time.sleep(0.15)
        elif what == "jump":
            win32.key_name("space")
            time.sleep(0.9)
        elif what == "circle":
            # walk a loop around the obstacle: small turn + step, repeated.
            # 12 x 600px (~27deg) closes a full circle and, unlike a blind
            # 100deg turn, it *travels*, so concave corners get resolved too.
            n = int(act.get("n", 12))
            step_px = int(act.get("px", 600))
            step_ms = int(act.get("ms", 700))
            self.log(f"    circle the obstacle: {n} x (turn {step_px}px + walk {step_ms}ms)")
            for _ in range(n):
                self.turn_px(step_px)
                win32.hold_key(win32.VK["w"], step_ms)
                time.sleep(0.12)
        else:
            self.log(f"    (ignoring unknown action {what!r})")

    def _load_escape_stats(self) -> dict:
        if ESCAPE_STATS.exists():
            try:
                return json.loads(ESCAPE_STATS.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _record_escape(self, diagnosed: str, tried: str, mirrored: bool,
                       freed: bool) -> None:
        key = f"{diagnosed}>{tried}{'>mirror' if mirrored else ''}"
        row = self.escape_stats.setdefault(key, dict(tries=0, freed=0))
        row["tries"] += 1
        row["freed"] += 1 if freed else 0
        ESCAPE_STATS.parent.mkdir(parents=True, exist_ok=True)
        ESCAPE_STATS.write_text(json.dumps(self.escape_stats, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    def arm_tracking(self, tries: int = 2) -> bool:
        """Arm the 格连 quest: M -> click the avatar -> click 寻找友邻 -> verify dot.

        Verified step by step because the map fades in over ~2-3 s and the
        「寻找友邻」button only exists after a successful selection.  Works on a
        friend's planet too (11.png: the avatar row is the same there).
        """
        for attempt in range(1, tries + 1):
            self.log(f"arm tracking (attempt {attempt}): open map")
            win32.key_name("m")
            time.sleep(1.0)

            avatar = None
            for _ in range(14):                      # up to ~12 s for the fade-in
                cands = detect.detect_avatar_candidates(self.frame(), self.tpl_avatar, top=5)
                if cands:
                    break
                time.sleep(0.85)
            else:
                cands = []
            if not cands:
                self.log("  no 格连 avatar on the map")
                continue

            # the avatar row is a line of look-alike badges, so the best NCC
            # match is often a neighbour; click each candidate until the
            # 「寻找友邻」 button really shows up (2026-10-18: argmax picked
            # x=780 while 格连 sat at x=625 and no button ever appeared)
            self.log("  avatar candidates: " + " ".join(
                f"({c['cx']:.0f},{c['cy']:.0f})={c['score']:.3f}" for c in cands))
            btn = None
            for i, cand in enumerate(cands, 1):
                self.log(f"  try badge #{i} at ({cand['cx']:.0f},{cand['cy']:.0f})")
                self.click((cand["cx"], cand["cy"]))
                time.sleep(0.8)
                btn = self._wait_find_button(tries=5)
                if btn is not None:
                    break
                self.log("    no 寻找友邻 button for this badge")
            if btn is None:
                self.log("  no 寻找友邻 button after selecting any badge")
                continue
            self.log(f"  寻找友邻 at ({btn['cx']:.0f},{btn['cy']:.0f}) "
                     f"score={btn['score']:.3f}")

            self.click((btn["cx"], btn["cy"]))
            time.sleep(2.4)

            for _ in range(6):                       # the blue dot must show up
                dot = detect.detect_blue_marker(self.frame())
                if dot and dot.get("found"):
                    self.log(f"  tracking CONFIRMED at ({dot['cx']:.0f},{dot['cy']:.0f})")
                    return True
                time.sleep(0.8)
            self.log("  armed but no blue dot -> turn to look around")
            win32.move_rel(900, 0)
            time.sleep(0.8)
            dot = detect.detect_blue_marker(self.frame())
            if dot and dot.get("found"):
                self.log(f"  tracking CONFIRMED after turning at ({dot['cx']:.0f},{dot['cy']:.0f})")
                return True
        self.log("  WARNING: could not arm the tracking")
        return False

    def _wait_find_button(self, tries: int = 5) -> dict | None:
        """Poll for the 寻找友邻 button that appears after selecting a badge."""
        for _ in range(tries):
            res = detect.detect_find_button(self.frame(), self.tpl_find)
            if res and res.get("found"):
                return res
            time.sleep(0.6)
        return None

    def phase_walk_only(self) -> dict:
        """Assume we are already past the bridge; just go to the shop door."""
        self.log("=== walk only: head for the 格连 shop ===")
        self.out = OUT_ROOT / "walkonly"
        self.out.mkdir(parents=True, exist_ok=True)
        dot = detect.detect_blue_marker(self.frame())
        if not (dot and dot.get("found")):
            self.log("  no dot on screen -> arm the tracking first")
            self.arm_tracking()
        found = self.go_to_gelian(getattr(self.a, "walk_budget_s", 600.0),
                                  getattr(self.a, "max_walk_steps", 90))
        self.shot("walk_only_end")
        if not found:
            fp = detect.detect_f_prompt(self.frame(), self.f_prompt)
            found = bool(fp.get("found"))
        self.log(f"reach door: {found}")
        return dict(ok=found)

    def phase_shoot_simple(self, enter: bool = True) -> dict:
        """Photograph the furniture inside the shop (the actual product).

        The user's goal is "看哪家的格连商店今天刷了什么", so the screenshots
        have to show the furniture, not the door.  Order matters (README §7):

          1. press F to enter (only when `enter`; pressing F again would leave!)
          2. the interior fades in from black -> wait until it is bright
          3. standing in the doorway frames only the door frame -> step in 1.2 s
          4. sweep the room: shot, turn ~60-70 deg, shot, ... -> `--shots` frames
          5. build `sheet.png` so one look shows the whole selection

        `photo_ok` is advisory (logged, not enforced): a clamped frame is
        reported but still saved, because the user wants the pictures.
        """
        if enter:
            self.log("enter shop (F), then photograph the furniture")
            win32.key_name("f")
            time.sleep(2.5)
        else:
            self.log("already inside: photograph the furniture (no F press)")
        self.wait_bright(min_mean=60.0, timeout=15.0)
        win32.hold_key(win32.VK["w"], 1200)     # leave the doorway / camera clipping
        time.sleep(0.6)

        files: list[str] = []
        notes: list[str] = []
        shots = max(1, int(getattr(self.a, "shots", 6)))
        for i in range(shots):
            img = self.frame()
            ok, why = self.photo_ok(img)
            name = f"inside_{i + 1:02d}.png"
            pngio.save_png(self.out / name, img)
            files.append(name)
            notes.append(why)
            self.log(f"  shot {name} mean={img.mean():.1f} "
                     f"({'ok' if ok else 'check: ' + why})")
            if i + 1 < shots:
                win32.move_rel(self.a.turn, 0)  # 1500px ~ 60-70 deg
                time.sleep(0.7)
        self._montage(files)
        self.log(f"captured {len(files)} furniture screenshots -> {self.out}")
        return dict(arrived=True, photos=len(files), files=files, notes=notes)

    def _montage(self, files: list[str]) -> None:
        """A 2-column contact sheet so one image shows the whole shop."""
        try:
            imgs = [pngio.load_image(self.out / f, cache_dir=spec.CACHE_DIR) for f in files]
            scale = 0.5
            small = [ip.resize_bilinear(im, max(8, int(im.shape[1] * scale)),
                                        max(8, int(im.shape[0] * scale))) for im in imgs]
            cols = 2
            rows = (len(small) + cols - 1) // cols
            cw = max(s.shape[1] for s in small)
            ch = max(s.shape[0] for s in small)
            gap = 6
            canvas = np.full((rows * ch + gap * (rows + 1),
                              cols * cw + gap * (cols + 1), 3), 30, np.uint8)
            for i, s in enumerate(small):
                r, c = divmod(i, cols)
                y = gap + r * (ch + gap)
                x = gap + c * (cw + gap)
                canvas[y:y + s.shape[0], x:x + s.shape[1]] = s
            pngio.save_png(self.out / "sheet.png", canvas)
            self.log(f"  contact sheet -> sheet.png ({canvas.shape[1]}x{canvas.shape[0]})")
        except Exception as exc:                 # never let a preview break a run
            self.log(f"  (contact sheet skipped: {exc})")

    def phase_shoot(self) -> dict:
        """Assume we are already inside the shop: frame the furniture area and
        take a validated sweep."""
        self.log("phase shoot: inside shop, framing")
        self.wait_bright()
        win32.hold_key(win32.VK["w"], 1200)     # leave the doorway / camera clipping
        time.sleep(0.8)
        win32.move_rel(600, 0)                  # turn toward the furniture side
        time.sleep(0.6)

        saved, attempts = [], 0
        while len(saved) < self.a.shots and attempts < self.a.shots * 4:
            attempts += 1
            img = self.frame()
            ok, why = self.photo_ok(img)
            if not ok:
                self.log(f"  attempt {attempts}: reject - {why}")
                win32.move_rel(420, 0)
                time.sleep(0.5)
                continue
            n = len(saved) + 1
            p = self.out / f"photo_{n:02d}.png"
            pngio.save_png(p, img)
            saved.append(p.name)
            self.log(f"  photo {n}/{self.a.shots}  {why}")
            if n < self.a.shots:
                win32.move_rel(self.a.turn, 0)
                time.sleep(0.55)
        self.log(f"captured {len(saved)} photos in {attempts} attempts")
        return dict(photos=len(saved), attempts=attempts, files=saved)

    def phase_shop(self) -> dict:
        """The 「自动截图」 button: enter the shop if we are at its door, then shoot.

        Works from two states: standing at the door (F prompt visible) and
        already inside.  Never presses F twice - a second F would walk out.
        """
        img = self.frame()
        fp = detect.detect_f_prompt(img, self.f_prompt)
        inside = not fp.get("found") and float(img.mean()) > 45.0
        if fp.get("found"):
            self.log(f"door prompt found (score={fp['score']:.3f}) -> enter with F")
            return self.phase_shoot_simple(enter=True)
        if inside:
            self.log("no door prompt -> assuming we are already inside")
            return self.phase_shoot_simple(enter=False)
        self.log("neither at the door nor inside (no F prompt) -> stopping")
        return dict(arrived=False, photos=0, why="no door prompt")


# --------------------------------------------------------------------------- #
# windows / entry points
# --------------------------------------------------------------------------- #
def find_window(manual_hwnd: int | None = None,
                exe_name: str = "PetitPlanet.exe") -> dict | None:
    """The game window: manual handle first, otherwise only PetitPlanet.exe.

    Matching on the title alone grabbed browser tabs named "星布谷地 …" during
    development, which would send keystrokes to the wrong program.
    """
    if manual_hwnd:
        if win32.is_window(manual_hwnd):
            return {"hwnd": int(manual_hwnd), "title": win32.window_text(manual_hwnd)}
        return None
    wanted = exe_name.lower()
    for window in win32.list_windows():
        if str(window.get("exe", "")).lower() != wanted:
            continue
        if int(window.get("w", 0)) < 600 or int(window.get("h", 0)) < 400:
            continue
        if window.get("min"):
            continue
        return window
    return None


def list_game_windows(exe_name: str = "PetitPlanet.exe") -> list[dict]:
    wanted = exe_name.lower()
    return [w for w in win32.list_windows()
            if str(w.get("exe", "")).lower() == wanted]


class ShopOptions:
    """Everything the GUI / CLI passes in (mirrors the 格连商店 page settings)."""

    def __init__(self, label: str = "shop", walk_budget_s: float = 600.0,
                 max_walk_steps: int = 90, shots: int = 6, turn: int = 1500,
                 manual_hwnd: int | None = None, capture: str = "auto",
                 log=print) -> None:
        self.label = label
        self.walk_budget_s = walk_budget_s
        self.max_walk_steps = max_walk_steps
        self.shots = max(1, int(shots))
        self.turn = max(100, int(turn))
        self.manual_hwnd = manual_hwnd
        self.capture = capture
        self.log = log


def _open_game(options: ShopOptions, stop) -> tuple[Walker | None, dict]:
    window = find_window(options.manual_hwnd)
    if not window:
        return None, {"ok": False, "reason": "game window not found"}
    walker = Walker(options)
    walker.hwnd = int(window["hwnd"])
    walker.window_title = str(window.get("title", ""))
    walker._stop = stop
    walker._capture_mode = options.capture
    walker.log(f"window: {walker.window_title} (#{walker.hwnd})")
    return walker, {}


def run_phase(phase: str = "both", options: ShopOptions | None = None,
              stop: threading.Event | None = None) -> dict:
    """Run one of: nav (走到门口) / shop (进店连拍) / both (串起来)."""
    options = options or ShopOptions()
    phase = phase if phase in ("nav", "shop", "both") else "both"
    walker, error = _open_game(options, stop)
    if walker is None:
        return error
    try:
        if phase == "nav":
            return walker.phase_walk_only()
        if phase == "shop":
            return walker.phase_shop()
        arrived = walker.phase_walk_only()
        if stop is not None and stop.is_set():
            return {"ok": False, "reason": "stopped"}
        if not arrived.get("ok"):
            return {"ok": False, "reason": "did not reach the shop door",
                    "nav": arrived}
        time.sleep(0.8)
        shots = walker.phase_shop()
        return {"ok": bool(shots.get("photos")), "nav": arrived, "shots": shots}
    finally:
        walker.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="星布谷地 - 格连商店助手（寻路 + 进店截图）")
    parser.add_argument("--phase", choices=["nav", "shop", "both"], default="both")
    parser.add_argument("--budget-s", type=float, default=600.0)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--shots", type=int, default=6)
    parser.add_argument("--turn", type=int, default=1500)
    parser.add_argument("--label", default="cli")
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args(argv)
    options = ShopOptions(label=args.label, walk_budget_s=args.budget_s,
                          max_walk_steps=args.max_steps, shots=args.shots,
                          turn=args.turn, manual_hwnd=args.window or None)
    result = run_phase(args.phase, options)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok") else 4


if __name__ == "__main__":
    raise SystemExit(main())
