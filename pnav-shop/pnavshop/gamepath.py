"""自动寻找星布谷地（PetitPlanet.exe）的安装位置。

查找顺序（找到就停）：
1. 注册表：卸载项 + App Paths 里带「星布谷地 / Petit Planet」的条目；
2. 常见根目录：TapTap 安装目录、本软件所在目录的 games\\、各盘符的游戏目录；
3. 有界的深度搜索（默认 5 层、8 秒上限，跳过系统/缓存目录）。
"""

from __future__ import annotations

import os
import pathlib
import time

TARGET = "petitplanet.exe"
SKIP_NAMES = {
    "windows", "winsxs", "system32", "syswow64", "windows.old", "programdata",
    "$recycle.bin", "node_modules", ".git", ".cache", "__pycache__", "temp",
    "appdata", "onedrive", "recovery", "perflogs", "msocache", "intel",
    "amd", "nvidia", "python313", "python312", "site-packages",
}
MAX_DEPTH = 5
TIME_LIMIT = 8.0


def _is_target(path: pathlib.Path) -> bool:
    return path.name.lower() == TARGET


def registry_hints() -> list[str]:
    """InstallLocation/DisplayIcon of entries whose name mentions the game."""
    hints: list[str] = []
    if os.name != "nt":
        return hints
    try:
        import winreg
    except Exception:
        return hints
    keys = [
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    keywords = ("星布谷地", "petit", "starvalley", "星布谷")
    for root, path in keys:
        try:
            with winreg.OpenKey(root, path) as base:
                count = winreg.QueryInfoKey(base)[0]
                for index in range(count):
                    try:
                        name = winreg.EnumKey(base, index)
                    except OSError:
                        continue
                    try:
                        with winreg.OpenKey(base, name) as entry:
                            display = _reg_value(entry, "DisplayName")
                            if not display or not any(
                                    word in display.lower() for word in keywords):
                                continue
                            location = _reg_value(entry, "InstallLocation")
                            icon = _reg_value(entry, "DisplayIcon")
                            if location:
                                hints.append(str(location).strip('"'))
                            if icon:
                                hints.append(str(icon).split(",")[0].strip('"'))
                    except OSError:
                        continue
        except OSError:
            continue
    try:
        for name in ("PetitPlanet.exe",):
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{name}") as key:
                hints.append(str(winreg.QueryValue(key, None)).strip('"'))
    except OSError:
        pass
    return [hint for hint in hints if hint]


def _reg_value(key, name: str) -> str:
    try:
        import winreg
        value, _kind = winreg.QueryValueEx(key, name)
        return str(value)
    except Exception:
        return ""


def candidate_roots(project_root: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Directories likely to contain the game, cheapest first."""
    roots: list[pathlib.Path] = []
    here = pathlib.Path(__file__).resolve().parents[2]      # pnav/
    project = pathlib.Path(project_root) if project_root else here
    roots += [project / "games", project.parent / "games", project]
    for hint in registry_hints():
        candidate = pathlib.Path(hint)
        roots.append(candidate if candidate.is_dir() else candidate.parent)
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 os.environ.get("LOCALAPPDATA")):
        if base:
            roots.append(pathlib.Path(base) / "TapTap")
    for drive in "CDEFGH":
        roots.append(pathlib.Path(f"{drive}:/TapTap/PC Games"))
        roots.append(pathlib.Path(f"{drive}:/TapTap"))
    seen, unique = set(), []
    for root in roots:
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if root.is_dir():
                unique.append(root)
        except OSError:
            continue
    return unique


def _search_root(root: pathlib.Path, deadline: float) -> str:
    root_depth = len(root.parts)
    for current, dirs, files in os.walk(root):
        if time.monotonic() > deadline:
            return ""
        path = pathlib.Path(current)
        if len(path.parts) - root_depth >= MAX_DEPTH:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_NAMES]
        for name in files:
            if name.lower() == TARGET:
                return str(path / name)
    return ""


def _real_name(directory: pathlib.Path) -> str:
    """Actual file name in the folder (Windows is case-insensitive)."""
    try:
        for entry in directory.iterdir():
            if entry.is_file() and entry.name.lower() == TARGET:
                return entry.name
    except OSError:
        pass
    return TARGET


def find_game_install(extra_roots: list = None, timeout_s: float = TIME_LIMIT,
                      project_root: pathlib.Path | None = None,
                      include_defaults: bool = True) -> str:
    """Return the path of PetitPlanet.exe, or "" when nothing was found."""
    roots = [pathlib.Path(p) for p in (extra_roots or [])]
    if include_defaults:
        roots += candidate_roots(project_root)
    deadline = time.monotonic() + timeout_s
    for root in roots:
        if not root.is_dir():
            continue
        direct = root / _real_name(root)
        if direct.is_file() and direct.name.lower() == TARGET:
            return str(direct)
        found = _search_root(root, deadline)
        if found:
            return found
        if time.monotonic() > deadline:
            break
    return ""
