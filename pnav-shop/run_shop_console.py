"""Console entry: same pipeline, no GUI."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pnavshop import walker  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(walker.main())
