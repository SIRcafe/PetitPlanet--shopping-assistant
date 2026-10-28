"""Standalone entry point (pythonw friendly)."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pnavshop import gui  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(gui.main())
