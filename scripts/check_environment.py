"""Fast dependency and source-file check for the figure repository."""

from importlib import import_module
from pathlib import Path

REQUIRED = ("numpy", "scipy", "matplotlib", "pandas")


def main() -> None:
    versions = {}
    for package in REQUIRED:
        module = import_module(package)
        versions[package] = getattr(module, "__version__", "unknown")

    root = Path(__file__).resolve().parents[1]
    notebook_source = root / "notebooks" / "original_analysis.py"
    notebook = root / "notebooks" / "original_analysis.ipynb"
    if not notebook_source.is_file():
        raise FileNotFoundError(notebook_source)
    if not notebook.is_file():
        raise FileNotFoundError(notebook)

    print("Environment check passed.")
    for package, version in versions.items():
        print(f"  {package}: {version}")
    print(f"  analysis source: {notebook_source.relative_to(root)}")


if __name__ == "__main__":
    main()
