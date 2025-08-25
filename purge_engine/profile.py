from __future__ import annotations

from typing import Callable
import numpy as np

try:
    from scipy.interpolate import CubicSpline  # optional
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def make_profile_spline(mileposts: np.ndarray, elevations: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    x = np.asarray(mileposts, dtype=float)
    y = np.asarray(elevations, dtype=float)
    if _HAS_SCIPY and x.size >= 3:
        cs = CubicSpline(x, y, bc_type="natural")
        return lambda xq: cs(xq).astype(float)
    return lambda xq: np.interp(np.asarray(xq, dtype=float), x, y).astype(float)


def to_feet(elev: np.ndarray, units: str) -> np.ndarray:
    if (units or "ft").lower().startswith("m"):
        return elev * 3.28084
    return elev