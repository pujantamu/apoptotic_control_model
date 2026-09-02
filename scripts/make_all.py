import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPTS = [
    *(ROOT / "scripts" / "main").glob("figure_*.py"),
    *(ROOT / "scripts" / "si").glob("figure_s*.py"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    for script in sorted(FIGURE_SCRIPTS):
        command = [sys.executable, str(script)]
        if args.quick:
            command.append("--quick")
        print(f"Running {script.relative_to(ROOT)}")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
