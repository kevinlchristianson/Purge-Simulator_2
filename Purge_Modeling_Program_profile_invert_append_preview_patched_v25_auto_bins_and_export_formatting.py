"""Minimal smoke test: importing the monolith must not exit the process.

Run:
    python -m tests.test_import_safe
"""

import importlib

MODULE = "Purge_Modeling_Program_profile_invert_append_preview_patched_v25_auto_bins_and_export_formatting"


def main() -> int:
    try:
        importlib.import_module(MODULE)
    except SystemExit as e:
        print("FAIL: importing monolith raised SystemExit:", e)
        return 1
    except Exception as e:
        print("FAIL: importing monolith raised exception:", repr(e))
        return 1

    print("PASS: monolith import is safe (no SystemExit).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
