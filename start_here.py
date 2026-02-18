"""Tiny onboarding helper for first-time users of this repo.

Run:
    python start_here.py

It prints a plain-English checklist and checks whether core dependencies are installed.
"""

from __future__ import annotations

import importlib
import platform
import sys

REQUIRED_MODULES = [
    "PySide6",
    "pandas",
    "numpy",
    "scipy",
    "openpyxl",
]


def check_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def print_step(number: int, text: str) -> None:
    print(f"Step {number}) {text}")


def main() -> int:
    print("=" * 72)
    print("Purge Wizard: START HERE (total step-by-step)")
    print("=" * 72)
    print(f"Python: {sys.version.split()[0]} | OS: {platform.system()} {platform.release()}")
    print()

    print_step(1, "Install dependencies (copy/paste this):")
    print("   pip install PySide6 pandas numpy scipy openpyxl")
    print()

    print_step(2, "Run the wizard launcher:")
    print("   python wizard_launcher.py")
    print()

    print_step(3, "In the launcher, click pages in this exact order:")
    print("   1) Pipe & Fluid Inputs")
    print("   2) Profile Loader")
    print("   3) Purge Setup")
    print("   4) Simulation Setup")
    print("   5) Nitrogen Setup")
    print()

    print_step(4, "If something breaks, capture this quick note:")
    print("   - Page:")
    print("   - What I clicked:")
    print("   - What I expected:")
    print("   - What happened:")
    print("   - Error text:")
    print()

    print("Dependency check:")
    missing = []
    for mod in REQUIRED_MODULES:
        ok = check_module(mod)
        print(f"  {'[OK]':<5} {mod}" if ok else f"  {'[MISSING]':<9} {mod}")
        if not ok:
            missing.append(mod)

    print()
    if missing:
        print("Status: not ready yet (dependencies missing).")
        print("Run this exact command:")
        print("   pip install " + " ".join(missing))
        print()
        print("Then run:")
        print("   python wizard_launcher.py")
        return 1

    print("Status: ready ✅")
    print("Next command:")
    print("   python wizard_launcher.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
