
from importlib import import_module

REQUIRED = ("numpy", "scipy", "matplotlib", "pandas")


def main() -> None:
    versions = {}

    for package in REQUIRED:
        module = import_module(package)
        versions[package] = getattr(module, "__version__", "unknown")

    print("Environment check passed.")

    for package, version in versions.items():
        print(f"  {package}: {version}")


if __name__ == "__main__":
    main()
