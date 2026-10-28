"""桌面分身：把游戏放到 Windows 的第二个桌面上运行。

可行性已验证（见 scripts/experiment_desktop_capture.py）：

* ``CreateDesktopW`` 建桌面、``CreateProcessW(lpDesktop=...)`` 在它上面启动进程；
* 进程窗口用 ``EnumDesktopWindows`` 能枚举到；
* **非活跃桌面上的窗口，``PrintWindow(PW_RENDERFULLCONTENT)`` 能截到完整画面**，
  BitBlt 则全黑 —— 所以分身模式下截图必须用 PrintWindow；
* ``SwitchDesktop`` 可以切过去看，再切回来；
* ``PostMessage`` 跨桌面可用（后台输入不受影响）。

这不是 BetterGI 那种 RDP 多会话方案：同一个 Windows 会话、同一个进程树，
只是多一个桌面，所以权限/输入/截图都在本会话内，简单得多。
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass, field

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

GENERIC_ALL = 0x10000000
DESKTOP_ALL = 0x01FF


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
        ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


def available() -> bool:
    try:
        return hasattr(user32, "CreateDesktopW") and hasattr(user32, "SwitchDesktop")
    except Exception:
        return False


def valid_name(name: str) -> bool:
    """Desktop names must not contain path separators."""
    name = str(name or "")
    return bool(name) and len(name) <= 64 and not any(ch in name for ch in "\\/:*?\"<>|")


@dataclass
class StartedProcess:
    pid: int
    hprocess: int
    name: str


class VirtualDesktop:
    """One extra desktop plus the processes started on it."""

    def __init__(self, name: str = "PNAV_分身") -> None:
        self.name = name if valid_name(name) else "PNAV_分身"
        self.handle = None
        self.processes: list[StartedProcess] = []
        self._previous_desktop = None
        self.active = False
        self.error = ""

    # -- lifecycle ---------------------------------------------------------
    def create(self) -> bool:
        if self.handle:
            return True
        handle = user32.CreateDesktopW(self.name, None, None, 0, GENERIC_ALL, None)
        if not handle:
            self.error = f"CreateDesktopW failed ({kernel32.GetLastError()})"
            return False
        self.handle = handle
        return True

    def close(self) -> None:
        """Kill our processes, then close the desktop."""
        for process in list(self.processes):
            try:
                kernel32.TerminateProcess(process.hprocess, 0)
            except Exception:
                pass
            try:
                kernel32.CloseHandle(process.hprocess)
                kernel32.CloseHandle(process.hprocess)
            except Exception:
                pass
        self.processes.clear()
        if self.active:
            self.switch_back()
        if self.handle:
            try:
                user32.CloseDesktop(self.handle)
            except Exception:
                pass
        self.handle = None

    # -- processes ---------------------------------------------------------
    def start_process(self, executable: str | None, arguments: str = "",
                      cwd: str | None = None, name: str = "") -> int:
        """Start a program on this desktop; returns the pid (0 on failure)."""
        if not self.create():
            return 0
        si = STARTUPINFOW()
        si.cb = ctypes.sizeof(si)
        si.lpDesktop = self.name
        pi = PROCESS_INFORMATION()
        command_line = None
        if executable:
            command_line = f'"{executable}"'
            if arguments:
                command_line += " " + arguments
        ok = kernel32.CreateProcessW(
            executable, command_line, None, None, False, 0, None,
            cwd, ctypes.byref(si), ctypes.byref(pi))
        if not ok:
            self.error = f"CreateProcessW failed ({kernel32.GetLastError()})"
            return 0
        self.processes.append(StartedProcess(int(pi.dwProcessId), int(pi.hProcess),
                                             name or str(executable or "?")))
        return int(pi.dwProcessId)

    def windows(self, min_size: int = 120) -> list[dict]:
        """Top-level windows that live on this desktop."""
        found: list[dict] = []
        if not self.handle:
            return found

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def enum_proc(hwnd, _lparam):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, length + 1)
                rect = wintypes.RECT()
                user32.GetClientRect(hwnd, ctypes.byref(rect))
                width, height = rect.right, rect.bottom
                if width >= min_size and height >= min_size:
                    found.append({"hwnd": int(hwnd), "title": title.value,
                                  "w": width, "h": height})
            return True

        user32.EnumDesktopWindows(self.handle, enum_proc, 0)
        return found

    def wait_for_window(self, timeout: float = 60.0, poll: float = 0.4,
                        title_keywords=None) -> dict | None:
        """Wait until a sized window appears on this desktop."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            for window in self.windows():
                if title_keywords and not any(k in window["title"] for k in title_keywords):
                    continue
                return window
            time.sleep(poll)
        return None

    # -- switching ---------------------------------------------------------
    def switch_to(self) -> bool:
        if not self.handle:
            return False
        try:
            self._previous_desktop = user32.OpenInputDesktop(0, False, DESKTOP_ALL)
        except Exception:
            self._previous_desktop = None
        if user32.SwitchDesktop(self.handle):
            self.active = True
            return True
        self.error = f"SwitchDesktop failed ({kernel32.GetLastError()})"
        return False

    def switch_back(self) -> bool:
        previous = self._previous_desktop
        if not previous:
            return False
        ok = bool(user32.SwitchDesktop(previous))
        try:
            user32.CloseDesktop(previous)
        except Exception:
            pass
        self._previous_desktop = None
        self.active = False
        return ok


def windows_on_desktop(handle: int, min_size: int = 120) -> list[dict]:
    """Standalone helper (same enumeration as VirtualDesktop.windows)."""
    desktop = VirtualDesktop()
    desktop.handle = handle
    return desktop.windows(min_size=min_size)
