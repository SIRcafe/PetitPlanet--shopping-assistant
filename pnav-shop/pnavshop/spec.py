"""Constants, ROIs, thresholds and fixture expectations (single source of truth).

Values were measured on 1920x1080 game screenshots that are not
distributed (game content); only small UI icons live in ``assets/templates``.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent          # pnav-shop/
TEMPLATE_DIR = ROOT / "assets" / "templates"
CACHE_DIR = ROOT / "_cache"
REPORT_DIR = ROOT / "reports"

DESIGN_W, DESIGN_H = 1920, 1080

# --------------------------------------------------------------------------- #
# the blue dot: fixed-colour UI overlay, identical in day / shadow / night
# --------------------------------------------------------------------------- #
BLUE_RING = (100, 173, 195)
BLUE_DISC = (72, 70, 65)
BLUE_CORE = (234, 231, 214)
BLUE_RING_TOL = 14                 # per-channel tolerance
BLUE_MIN_AREA = 350                # ring pixels only (annulus)
BLUE_MIN_DIAM = 30
BLUE_MAX_DIAM = 60
BLUE_MIN_RATIO = 0.72
BLUE_MAX_RATIO = 1.38
BLUE_SCORE_MIN = 1.0               # needs all three ring checks

# --------------------------------------------------------------------------- #
# UI element colours (for automatic template extraction)
# --------------------------------------------------------------------------- #
CREAM_PILL = (250, 245, 214)       # "寻找友邻" button face
CREAM_PILL_TOL = 16
DARK_PILL = (80, 70, 51)           # "F 进入格连商店" prompt face
DARK_PILL_TOL = 14
PROMPT_TEXT = (218, 215, 196)      # light text on the prompt pill
PROMPT_TEXT_TOL = 26
PROMPT_TEXT_PAD = (14, 12)         # pad from glyph bbox out to the pill edge

# --------------------------------------------------------------------------- #
# ROIs (fractions of the frame) and match thresholds
# --------------------------------------------------------------------------- #
ROI_AVATAR = dict(x0=0.00, x1=1.00, y0=0.80, y1=0.97)
ROI_FIND_BTN = dict(x0=0.70, x1=1.00, y0=0.64, y1=0.82)
ROI_F_PROMPT = dict(x0=0.45, x1=1.00, y0=0.50, y1=0.72)

AVATAR_SCALES = (0.90, 0.95, 1.00, 1.05, 1.10)
UI_SCALES = (0.90, 0.95, 1.00, 1.05, 1.10)
# tile is cropped at 100 px so the gold "selected" ring (r~58) stays outside
AVATAR_TILE_CENTER = (625, 935)
AVATAR_TILE_SIZE = 100
AVATAR_MARGIN_MIN = 0.06           # best tile must beat the runner-up tile

AVATAR_SCORE_MIN = 0.80
AVATAR_COLOR_MAX = 26.0
FIND_BTN_SCORE_MIN = 0.82
F_PROMPT_SCORE_MIN = 0.82
UI_COLOR_MAX = 30.0
MAP_OPEN_SCORE_MIN = 0.80

# --------------------------------------------------------------------------- #
# v2 navigation: global grid map + optimistic A* (arbitrary obstacle shapes)
#
# The world is treated as a flat 2D plane.  Crossing the dock leaves the
# character at (0,0) facing NORTH, which anchors the whole coordinate system
# (x = east, y = north, theta = compass degrees clockwise from north).
# Obstacles are whatever the planet owner placed (furniture / outdoor gear),
# so nothing here assumes a shape: only "is this cell walkable".
# --------------------------------------------------------------------------- #
WALK_SPEED_MPS = 2.5            # 800 ms of walking ~ 2 m (field measurement)
TURN_PX_PER_DEG = 2000 / 90.0   # 2000 mouse px ~ 90 deg (2000/90 ~ 22.2 px/deg)
BEARING_DEG_PER_PX = 0.031      # blue-dot screen offset -> relative bearing (FOV ~60 deg)
CAMERA_DEG_PER_PX = 0.045       # 1500 px ~ 60-70 deg (the shop_photos calibration)

GRID_CELL_M = 0.4               # small furniture still gets a few cells
GRID_HALF_EXTENT_M = 100.0      # +-100 m -> 500 x 500 cells at 0.4 m
GRID_INFLATE_CELLS = 1          # keep a body radius away from known obstacles
GRID_UNKNOWN_COST = 1.10        # optimistic: unknown is walkable, slightly dearer
GRID_JUMP_COST = 1.30           # hop over a low obstacle instead of walking around
BODY_RADIUS_M = 0.5             # used when marking what we bumped into
AIM_TOLERANCE_DEG = 8.0         # "aligned with the waypoint" threshold
WALK_STEP_MS = 1200             # one navigation step

# --------------------------------------------------------------------------- #
# waiting / pacing rules that the user fixed by hand (do not "optimise" them)
# --------------------------------------------------------------------------- #
# Esc opens the phone menu with a fade-in; clicking 友呼 too early closes the
# whole panel.  Also: visiting another friend never needs 「回家」 - the phone
# menu works from anywhere (user rule, 2026-10-18).
PHONE_OPEN_WAIT_S = 5.0
# the fly-over cutscene keeps eating WASD for a while after it is skipped
CUTSCENE_SETTLE_S = 10.0

# --------------------------------------------------------------------------- #
# the bridge you land next to is identical on every planet (user, 2026-10-18),
# so this stays a fixed choreography: left 75deg -> walk -> right 10deg -> walk
# --------------------------------------------------------------------------- #
BRIDGE_PRE_TURN_PX = -1670
BRIDGE_RIGHT10_PX = 230
BRIDGE_WALK_MS = 2000

# --------------------------------------------------------------------------- #
# stuck diagnosis (M5): the ground band in front of the character
# --------------------------------------------------------------------------- #
STUCK_ROI = dict(x0=0.18, x1=0.82, y0=0.52, y1=0.97)
STUCK_CHAR_BOX = (0.40, 0.36, 0.60, 1.00)     # frame fractions, excluded
STUCK_CONF_MIN = 0.42                         # below this -> "unknown" fallback
STUCK_RAIL_MIN = 0.020                        # rail-bar share that wins outright
# the railing colour is also *close* to water (22/22/20 per channel apart), so
# the stuck check needs a tight tolerance while bridge_ratio keeps the loose one
STUCK_RAIL_TOL = 16

# bridge railings (also used by detect.bridge_ratio)
RAIL_BLUE = (58, 110, 168)
RAIL_TOL = 40

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
FIXTURES = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png"]

# expected detections: (test, image) -> expectation dict
#   found: should the detector fire
#   cx/cy/tol: optional position check
EXPECT: dict[tuple[str, str], dict] = {
    ("map_open", "1.png"): dict(found=True),
    ("map_open", "5.png"): dict(found=True),
    ("map_open", "3.png"): dict(found=False),
    ("map_open", "4.png"): dict(found=False),
    ("map_open", "6.png"): dict(found=False),
    ("avatar_gelian", "1.png"): dict(found=True, cx=630, cy=952, tol=70),
    ("avatar_gelian", "5.png"): dict(found=True, cx=600, cy=952, tol=90),
    ("avatar_gelian", "3.png"): dict(found=False),
    ("avatar_gelian", "4.png"): dict(found=False),
    ("avatar_gelian", "6.png"): dict(found=False),
    ("find_neighbor_btn", "5.png"): dict(found=True, cx=1698, cy=787, tol=40),
    ("find_neighbor_btn", "1.png"): dict(found=False),
    ("find_neighbor_btn", "3.png"): dict(found=False),
    ("find_neighbor_btn", "4.png"): dict(found=False),
    ("find_neighbor_btn", "6.png"): dict(found=False),
    ("f_prompt_gelian", "4.png"): dict(found=True, cx=1437, cy=667, tol=40),
    ("f_prompt_gelian", "3.png"): dict(found=False),
    ("f_prompt_gelian", "6.png"): dict(found=False),
    ("blue_marker", "3.png"): dict(found=True, cx=940, cy=152, tol=28),
    ("blue_marker", "4.png"): dict(found=True, cx=940, cy=140, tol=28),
    ("blue_marker", "6.png"): dict(found=True, cx=378, cy=705, tol=28),
    ("blue_marker", "1.png"): dict(found=False),
    ("blue_marker", "5.png"): dict(found=False),
}

TEMPLATES = dict(
    avatar="avatar_gelian.png",
    find_btn="btn_find_neighbor.png",
    f_prompt="prompt_f_gelian.png",
    blue="point_blue.png",
)
