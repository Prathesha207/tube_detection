"""Print the highest-version head_tail_analyzer wheel in a directory."""

import sys
from pathlib import Path

from packaging.utils import parse_wheel_filename


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: select_latest_analyzer_wheel.py WHEEL_DIRECTORY", file=sys.stderr)
        return 2

    wheel_dir = Path(sys.argv[1])
    wheels = list(wheel_dir.glob("head_tail_analyzer*.whl"))
    if not wheels:
        return 1

    latest = max(wheels, key=lambda path: parse_wheel_filename(path.name)[1])
    print(latest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
