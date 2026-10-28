r"""Win32 layer via ctypes: window lookup, GDI screen capture, SendInput.

No third-party dependency (mss / pydirectinput are not installable here).
Everything is user-space and non-intrusive: we only read screen pixels and
post input events, exactly like a person at the keyboard.

All rects are in *physical* pixels; call `make_dpi_aware()` first.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import numpy as np

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# ---------------------------------------------------------------- structs ---
class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]


INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOUSEEVENTF_MOVE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0001, 0x0002, 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_VIRTUALDESK = 0x8000, 0x4000
KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x0002, 0x0008
PW_RENDERFULLCONTENT = 0x00000002

VK = dict(esc=0x1B, enter=0x0D, space=0x20, tab=0x09, alt=0x12, ctrl=0x11,
          shift=0x10, m=0x4D, f=0x46, w=0x57, a=0x41, s=0x53, d=0x44,
          left=0x25, up=0x26, right=0x27, down=0x28,
          **{f"f{i}": 0x6F + i for i in range(1, 13)})


# ------------------------------------------------------------------- dpi ----
def make_dpi_aware() -> str:
    """Per-monitor DPI awareness so window rects and clicks share one space."""
    try:
        # -4 == PER_MONITOR_AWARE_V2
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor-v2"
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per-monitor"
    except Exception:
        pass
    try:
        user32.SetProcessDPIAware()
        return "system"
    except Exception:
        return "none"


def screen_size() -> tuple[int, int]:
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


# --------------------------------------------------------------- process ----
def list_processes() -> dict[int, str]:
    """pid -> exe name, via Toolhelp snapshot."""
    TH32CS_SNAPPROCESS = 0x00000002

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(wintypes.ULONG)),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_char * 260)]

    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap in (0, -1):
        return {}
    entry = PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
    out = {}
    ok = kernel32.Process32First(snap, ctypes.byref(entry))
    while ok:
        out[entry.th32ProcessID] = entry.szExeFile.decode("mbcs", "replace")
        ok = kernel32.Process32Next(snap, ctypes.byref(entry))
    kernel32.CloseHandle(snap)
    return out


def window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        return (0, 0, 0, 0)
    return (r.left, r.top, r.right - r.left, r.bottom - r.top)


def client_rect(hwnd: int) -> tuple[int, int, int, int]:
    """Client area in *screen* coordinates: (x, y, w, h)."""
    r = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return (pt.x, pt.y, r.right - r.left, r.bottom - r.top)


_TARGET = {"hwnd": 0, "x": 0, "y": 0}


def set_target(hwnd: int) -> tuple[int, int]:
    """Remember where the game's client area sits on screen.

    Everything the callers pass around (BOUNDS, template centres, the 960,540
    screen centre) is in *game* pixels; a windowed game offsets all of them by
    its border + title bar, so clicks are translated here instead of in every
    script.
    """
    x, y, _w, _h = client_rect(hwnd)
    _TARGET.update(hwnd=hwnd, x=x, y=y)
    return x, y


def to_screen(x: int, y: int) -> tuple[int, int]:
    return x + _TARGET["x"], y + _TARGET["y"]


def target_offset() -> tuple[int, int]:
    return _TARGET["x"], _TARGET["y"]


def is_window(hwnd: int) -> bool:
    """IsWindow(hwnd) - cheap validity check for a stored handle."""
    try:
        return bool(ctypes.windll.user32.IsWindow(int(hwnd)))
    except Exception:
        return False


def foreground_window() -> int:
    try:
        return int(ctypes.windll.user32.GetForegroundWindow())
    except Exception:
        return 0


def window_text(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def list_windows(visible_only: bool = True) -> list[dict]:
    out: list[dict] = []
    procs = list_processes()
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lparam):
        if visible_only and not user32.IsWindowVisible(hwnd):
            return True
        title = window_text(hwnd)
        if not title:
            return True
        x, y, w, h = window_rect(hwnd)
        if w <= 0 or h <= 0:
            return True
        pid = window_pid(hwnd)
        out.append(dict(hwnd=hwnd, pid=pid, exe=procs.get(pid, "?"), title=title,
                        cls=class_name(hwnd), x=x, y=y, w=w, h=h,
                        fg=bool(hwnd == user32.GetForegroundWindow()),
                        min=bool(user32.IsIconic(hwnd))))
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return out


def find_game_window(exe_names=("PetitPlanet.exe",), title_keywords=("星布谷地", "Petit Planet")) -> dict | None:
    wins = list_windows()
    for w in wins:
        if w["exe"].lower() in {e.lower() for e in exe_names}:
            set_target(w["hwnd"])
            return w
    for w in wins:
        if any(k.lower() in w["title"].lower() for k in title_keywords):
            set_target(w["hwnd"])
            return w
    return None


# --------------------------------------------------------------- capture ----
def capture_rect(x: int, y: int, w: int, h: int, method: str = "screen") -> np.ndarray:
    """Grab a screen region as (h, w, 3) uint8 RGB.

    method="screen": BitBlt from the desktop DC (works for DWM-composited
    windowed games).  method="window": PrintWindow with PW_RENDERFULLCONTENT,
    which can capture a window even if partially covered.
    """
    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    bmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
    gdi32.SelectObject(hdc_mem, bmp)
    ok = False
    try:
        if method == "window":
            ok = bool(user32.PrintWindow(0, hdc_mem, 0))
        else:
            ok = bool(gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x, y, 0x00CC0020))
        bi = BITMAPINFOHEADER()
        bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.biWidth = w
        bi.biHeight = -h            # top-down
        bi.biPlanes = 1
        bi.biBitCount = 32
        bi.biCompression = 0        # BI_RGB
        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(hdc_mem, bmp, 0, h, buf, ctypes.byref(bi), 0)
        arr = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
        return np.ascontiguousarray(arr[..., 2::-1])   # BGRA -> RGB
    finally:
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)


def capture_window(hwnd: int, method: str = "screen") -> np.ndarray:
    """Grab the game's *client* area as a 1920x1080-class RGB frame.

    The client rect matters: in windowed mode the window rect includes an 8 px
    border and a ~31 px title bar, which would both offset every detected
    coordinate and put the title bar on top of clicks near y=0.
    """
    x, y, w, h = client_rect(hwnd)
    if shadow_target() or method == "window":
        hdc_win = user32.GetWindowDC(hwnd)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_win)
        bmp = gdi32.CreateCompatibleBitmap(hdc_win, w, h)
        gdi32.SelectObject(hdc_mem, bmp)
        try:
            ok = bool(user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT))
            bi = BITMAPINFOHEADER()
            bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bi.biWidth, bi.biHeight = w, -h
            bi.biPlanes, bi.biBitCount, bi.biCompression = 1, 32, 0
            buf = ctypes.create_string_buffer(w * h * 4)
            gdi32.GetDIBits(hdc_mem, bmp, 0, h, buf, ctypes.byref(bi), 0)
            arr = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
            img = np.ascontiguousarray(arr[..., 2::-1])
            return img
        finally:
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(hwnd, hdc_win)
    return capture_rect(x, y, w, h, method="screen")


# ----------------------------------------------------------------- input ----
def _send_inputs(inputs: list[INPUT]) -> None:
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    user32.SendInput(n, arr, ctypes.sizeof(INPUT))


def _mouse_input(flags: int, dx: int = 0, dy: int = 0) -> INPUT:
    inp = INPUT(type=INPUT_MOUSE)
    inp.mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=None)
    return inp


# --------------------------------------------------------------------------- #
# shadow target (桌面分身小窗): PostMessage instead of SendInput
# --------------------------------------------------------------------------- #
_SHADOW = {"hwnd": 0}
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101


def set_shadow_target(hwnd: int | None) -> None:
    """Bind (or clear) the window that receives simulated input."""
    _SHADOW["hwnd"] = int(hwnd or 0)


def shadow_target() -> int:
    return int(_SHADOW["hwnd"])


def shadow_from_env() -> int:
    """Pick up PNAV_SHADOW_HWND (set by the GUI for its child scripts)."""
    import os
    try:
        value = int(os.environ.get("PNAV_SHADOW_HWND", "0") or 0)
    except Exception:
        value = 0
    if value:
        set_shadow_target(value)
    return value


def _post_mouse(hwnd: int, message: int, x: int, y: int, extra: int = 0) -> bool:
    try:
        lparam = ((int(y) & 0xFFFF) << 16) | (int(x) & 0xFFFF)
        return bool(user32.PostMessageW(wintypes.HWND(hwnd), message,
                                       wintypes.WPARAM(extra), wintypes.LPARAM(lparam)))
    except Exception:
        return False


def _post_key(hwnd: int, vk: int, down: bool, scan: int = 0) -> bool:
    try:
        scan = scan or int(ctypes.windll.user32.MapVirtualKeyW(int(vk), 0))
        if down:
            lparam = (scan << 16) | 1
        else:
            lparam = (scan << 16) | 1 | (1 << 30) | (1 << 31)
        message = WM_KEYDOWN if down else WM_KEYUP
        return bool(user32.PostMessageW(wintypes.HWND(hwnd), message,
                                        wintypes.WPARAM(int(vk)), wintypes.LPARAM(lparam)))
    except Exception:
        return False


def move_to(x: int, y: int) -> None:
    """Absolute cursor move.  `x, y` are GAME pixels, translated to the screen."""
    hwnd = shadow_target()
    if hwnd:
        _post_mouse(hwnd, WM_MOUSEMOVE, x, y)
        return
    sx, sy = to_screen(x, y)
    sw, sh = screen_size()
    ax = int(sx * 65535 / max(1, sw - 1))
    ay = int(sy * 65535 / max(1, sh - 1))
    _send_inputs([_mouse_input(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, ax, ay)])


def _send_inputs_checked(inputs: list[INPUT]) -> tuple[int, int]:
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    user32.SendInput.restype = wintypes.UINT
    ret = user32.SendInput(n, arr, ctypes.sizeof(INPUT))
    err = 0 if ret else ctypes.get_last_error()   # only meaningful on failure
    return int(ret), int(err)


def move_to_ex(x: int, y: int) -> dict:
    """Move the cursor (game pixels) and report whether the injection landed."""
    sx, sy = to_screen(x, y)
    sw, sh = screen_size()
    ax = int(sx * 65535 / max(1, sw - 1))
    ay = int(sy * 65535 / max(1, sh - 1))
    before = cursor_pos()
    ret, err = _send_inputs_checked(
        [_mouse_input(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, ax, ay)]
    )
    time.sleep(0.12)
    after = cursor_pos()
    return dict(ret=ret, err=err, before=before, after=after,
                moved=after != before, struct_size=ctypes.sizeof(INPUT))


def wheel(notches: int = 3, up: bool = False) -> dict:
    """Scroll the friend list.  One notch = 120 units; negative scrolls down."""
    delta = 120 * notches * (1 if up else -1)
    inp = _mouse_input(MOUSEEVENTF_WHEEL)
    inp.mi.mouseData = ctypes.c_ulong(delta & 0xFFFFFFFF).value
    ret, err = _send_inputs_checked([inp])
    return dict(ret=ret, err=err, delta=delta)


def move_rel(dx: int, dy: int) -> dict:
    """Relative mouse movement — the ONLY way to turn the camera in this game:
    it locks the cursor to the screen centre, so absolute moves are undone
    immediately while the delta itself is consumed as camera rotation."""
    return dict(zip(("ret", "err"), _send_inputs_checked([_mouse_input(MOUSEEVENTF_MOVE, dx, dy)])))


def hold_key(vk: int, ms: int) -> None:
    """Press and hold a key for `ms` (scan code, so Unity sees it)."""
    hwnd = shadow_target()
    if hwnd:
        _post_key(hwnd, int(vk), True)
        time.sleep(max(0.02, ms / 1000.0))
        _post_key(hwnd, int(vk), False)
        return
    key(vk, hold_ms=ms)


def cursor_pos() -> tuple[int, int]:
    p = POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def click_right(x: int | None = None, y: int | None = None, hold_ms: int = 120) -> None:
    """Right click - used in-game to toggle run."""
    hwnd = shadow_target()
    if hwnd:
        px, py = int(x or 0), int(y or 0)
        _post_mouse(hwnd, WM_MOUSEMOVE, px, py)
        _post_mouse(hwnd, WM_RBUTTONDOWN, px, py, extra=2)
        time.sleep(max(0.02, hold_ms / 1000.0))
        _post_mouse(hwnd, WM_RBUTTONUP, px, py)
        return
    if x is not None and y is not None:
        move_to(x, y)
        time.sleep(0.08)
    _send_inputs([_mouse_input(MOUSEEVENTF_RIGHTDOWN)])
    time.sleep(hold_ms / 1000.0)
    _send_inputs([_mouse_input(MOUSEEVENTF_RIGHTUP)])


def click(x: int | None = None, y: int | None = None, hold_ms: int = 40) -> None:
    hwnd = shadow_target()
    if hwnd:
        px = int(x or 0)
        py = int(y or 0)
        _post_mouse(hwnd, WM_MOUSEMOVE, px, py)
        _post_mouse(hwnd, WM_LBUTTONDOWN, px, py, extra=1)   # MK_LBUTTON
        time.sleep(max(0.02, hold_ms / 1000.0))
        _post_mouse(hwnd, WM_LBUTTONUP, px, py)
        return
    if x is not None and y is not None:
        move_to(x, y)
        time.sleep(0.06)
    _send_inputs([_mouse_input(MOUSEEVENTF_LEFTDOWN)])
    time.sleep(hold_ms / 1000.0)
    _send_inputs([_mouse_input(MOUSEEVENTF_LEFTUP)])


def click_center_of_window(hwnd: int) -> tuple[int, int]:
    x, y, w, h = window_rect(hwnd)
    cx, cy = x + w // 2, y + h // 2
    click(cx, cy)
    return cx, cy


def key(vk: int, hold_ms: int = 40, use_scan: bool = True) -> None:
    """Press a key.  Unity/DirectInput titles usually ignore virtual-key-only
    events, so by default we send a hardware scan code instead."""
    if use_scan:
        user32.MapVirtualKeyW.restype = wintypes.UINT
        sc = int(user32.MapVirtualKeyW(vk, 0)) or vk
        down = INPUT(type=INPUT_KEYBOARD)
        down.ki = KEYBDINPUT(wVk=0, wScan=sc, dwFlags=KEYEVENTF_SCANCODE, time=0, dwExtraInfo=None)
        up = INPUT(type=INPUT_KEYBOARD)
        up.ki = KEYBDINPUT(wVk=0, wScan=sc, dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP,
                           time=0, dwExtraInfo=None)
    else:
        down = INPUT(type=INPUT_KEYBOARD)
        down.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=0, time=0, dwExtraInfo=None)
        up = INPUT(type=INPUT_KEYBOARD)
        up.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)
    _send_inputs([down])
    time.sleep(hold_ms / 1000.0)
    _send_inputs([up])


def key_name(name: str, hold_ms: int = 40, use_scan: bool = True) -> None:
    if name.lower() not in VK:
        raise KeyError(f"unknown key {name!r}; known: {sorted(VK)}")
    key(VK[name.lower()], hold_ms, use_scan=use_scan)


# -------------------------------------------------------------- focus ------
def bring_to_front(hwnd: int) -> bool:
    """Make `hwnd` the foreground window.

    Windows only lets the *foreground* process change the foreground, and the
    game runs at High integrity while this tool may be Medium, so a plain
    SetForegroundWindow is frequently ignored (seen in the field: capture works
    because the window is visible, but keys land in the browser instead).
    Tries the classic tricks in order and reports whether it really worked.
    """
    if shadow_target():
        return False          # another desktop: do not steal focus
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)      # SW_RESTORE
        time.sleep(0.35)
    if user32.GetForegroundWindow() == hwnd:
        return True

    # 1) SwitchToThisWindow: a different code path that works where
    #    SetForegroundWindow is refused
    try:
        user32.SwitchToThisWindow(hwnd, True)
        time.sleep(0.2)
        if user32.GetForegroundWindow() == hwnd:
            return True
    except Exception:
        pass

    # 2) attach to the foreground thread, then bring + activate
    fg = user32.GetForegroundWindow()
    target_tid = user32.GetWindowThreadProcessId(hwnd, None)
    fg_tid = user32.GetWindowThreadProcessId(fg, None)
    attached = False
    if target_tid != fg_tid:
        attached = bool(user32.AttachThreadInput(fg_tid, target_tid, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(fg_tid, target_tid, False)
    time.sleep(0.2)
    if user32.GetForegroundWindow() == hwnd:
        return True

    # 3) tapping Alt unlocks the OS foreground lock
    try:
        key(0x12, hold_ms=40)           # VK_MENU
        time.sleep(0.1)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.2)
        if user32.GetForegroundWindow() == hwnd:
            return True
    except Exception:
        pass

    # 4) last resort: topmost flip
    try:
        HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
        SWP_NOSIZE, SWP_NOMOVE, SWP_SHOWWINDOW = 0x0001, 0x0002, 0x0040
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                            SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW)
        user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0,
                            SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.25)
    except Exception:
        pass
    return user32.GetForegroundWindow() == hwnd


def is_black(img: np.ndarray, thresh: float = 6.0) -> bool:
    return float(img.mean()) < thresh
