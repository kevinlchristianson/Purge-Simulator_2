"""
Purge Pipeline Simulator — Engine Module
----------------------------------------
This module contains the physics + control logic used by the desktop UI.
It deliberately mirrors the behavior of your original program (29_07_2025rev2)
—including fluids, roughness options, crude API handling, IPS options, and
pressure/velocity logic—while being UI‑agnostic so it can be imported by
PySide6 (Qt) or any other frontend.

Public entry points
-------------------
- run_simulation(inputs, mileposts, elevations)
    Main routine. Returns a results dict of NumPy arrays.

- simulate_pipeline(inputs, profile_df)
    Convenience wrapper if you have a pandas DataFrame with
    columns ["Milepost", "Elevation"].

Inputs expected (all floats unless noted):
-----------------------------------------
- nps: NPS string (e.g., "6", "10", "24") or fractional like "2 1/2"
- pipe_wt: wall thickness [in]
- roughness_num: 1=new welded steel, 2=corroded steel, 3=HDPE
- fluid_num: 1=Diesel, 2=Gasoline, 3=Crude Oil, 4=Water, 5=NGL (Y1-grade)
- api_gravity: only used if fluid_num==3 (Crude Oil)
- max_drive_pressure: [psi]
- exit_pressure_run: [psi] (target exit pressure during run)
- exit_pressure_end: [psi] (target exit pressure near the end)
- n2_end_pressure: [psi] (used to estimate total SCF at end)
- max_pig_speed: [mph]
- min_pig_speed: [mph]
- purge_start_mp: start milepost of purge window
- purge_end_mp: end milepost of purge window (pig arrival target)
- system_end_mp: final system tie-in milepost (for head calcs)
- throttle_down_miles: miles before purge_end_mp where we switch targets
- hard_cap: bool — if True, cap P_exit to target at high speeds
- taper_down_enabled: bool — allow tapering P_drive near end window
- has_ips: bool — if True, model an intermediate pump station
- ips_mp: milepost of IPS (required if has_ips)
- ips_shutdown_dist: distance (miles) before ips_mp where IPS is shut down
- min_pump_suction_pressure: [psi] — used when has_ips is True
- elevation_units: 'ft' or 'm' (default 'ft')
- n_points: number of calculation points between purge_start_mp and purge_end_mp (default 1000)
- cutoff_volume (optional): nitrogen SCF at which to stop injecting; if not
  provided, a default is computed from line volume and n2_end_pressure.

Returned dict keys (NumPy arrays unless noted):
----------------------------------------------
- purge_mileposts, elevations, elapsed_times [hr], drive_pressures [psi],
  friction_losses [psi], head_losses [psi], exit_pressures [psi],
  injection_rates [scf/min], cumulative_n2 [scf], pig_speeds [mph],
  exceed_points (list of tuples), last_valid_i (int)
"""
from __future__ import annotations

import math
import time
import numpy as np
from typing import Dict, Any, Tuple
try:
    from scipy.interpolate import CubicSpline  # optional; we include a tiny fallback below
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# -----------------------------
# Data tables (mirrors original)
# -----------------------------
# Nominal Pipe Size (outside diameter, inches)
NPS_DATA: Dict[str, Dict[str, float]] = {
    "1/8": {"OD_in": 0.405}, "1/4": {"OD_in": 0.540}, "3/8": {"OD_in": 0.675}, "1/2": {"OD_in": 0.840},
    "3/4": {"OD_in": 1.050}, "1": {"OD_in": 1.315}, "1 1/4": {"OD_in": 1.660}, "1 1/2": {"OD_in": 1.900},
    "2": {"OD_in": 2.375}, "2 1/2": {"OD_in": 2.875}, "3": {"OD_in": 3.500}, "3 1/2": {"OD_in": 4.000},
    "4": {"OD_in": 4.500}, "5": {"OD_in": 5.563}, "6": {"OD_in": 6.625}, "8": {"OD_in": 8.625},
    "10": {"OD_in": 10.750}, "12": {"OD_in": 12.750}, "14": {"OD_in": 14.000}, "16": {"OD_in": 16.000},
    "18": {"OD_in": 18.000}, "20": {"OD_in": 20.000}, "24": {"OD_in": 24.000}, "26": {"OD_in": 26.000},
    "28": {"OD_in": 28.000}, "30": {"OD_in": 30.000}, "32": {"OD_in": 32.000}, "34": {"OD_in": 34.000},
    "36": {"OD_in": 36.000}, "40": {"OD_in": 40.000}, "42": {"OD_in": 42.000}, "44": {"OD_in": 44.000},
    "48": {"OD_in": 48.000},
}

# Pipe roughness (ft)
ROUGHNESS_DATA = {
    1: {"material": "New Welded Steel", "roughness_ft": 0.00015},
    2: {"material": "Rusted/Corroded Welded Steel", "roughness_ft": 0.0005},
    3: {"material": "Welded HDPE", "roughness_ft": 0.000005},
}

# Fluids (SG at ~60F, kinematic viscosity in cSt)
FLUID_DATA = {
    1: {"name": "Diesel", "sg": 0.84, "viscosity_cst": 2.7},
    2: {"name": "Gasoline", "sg": 0.74, "viscosity_cst": 0.6},
    3: {"name": "Crude Oil", "sg": None, "viscosity_cst": None},
    4: {"name": "Water", "sg": 1.0, "viscosity_cst": 1.0},
    5: {"name": "NGL (Y1-grade)", "sg": 0.6, "viscosity_cst": 0.3},
}

# -------------
# Math helpers
# -------------
def _cubic_spline(x: np.ndarray, y: np.ndarray):
    """Return a callable spline(xq)->yq (CubicSpline if available, else linear)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if _HAS_SCIPY and len(x) >= 3:
        return CubicSpline(x, y, bc_type="natural")
    # linear fallback
    def _lin(xq):
        xq = np.atleast_1d(xq)
        return np.interp(xq, x, y)
    return _lin

# -------------------------
# Physics/controls routines
# -------------------------
def calculate_trendline_slope(mileposts: np.ndarray, elevations: np.ndarray, start_mp: float, end_mp: float) -> float:
    mask = (mileposts >= start_mp) & (mileposts <= end_mp)
    sel_x = mileposts[mask]
    sel_y = elevations[mask]
    if len(sel_x) < 2:
        return 0.0
    return float((sel_y[-1] - sel_y[0]) / (sel_x[-1] - sel_x[0]))


def calculate_friction_loss(v: float, L: float, D: float, fluid_density: float, viscosity: float, roughness: float) -> float:
    """Darcy–Weisbach using Haaland (psi).
    v [ft/s], L [ft], D [ft], density [lb/ft^3], viscosity [lb*s/ft^2], roughness [ft].
    Returns deltaP in psi.
    """
    Re = max(fluid_density * v * D / max(viscosity, 1e-12), 1e-6)
    haaland = -1.8 * math.log10((roughness / D / 3.7) ** 1.11 + 6.9 / Re)
    f = (1.0 / haaland) ** 2
    dP_lbf_per_ft2 = f * (L / D) * fluid_density * v * v / (2.0 * 32.174)
    return dP_lbf_per_ft2 / 144.0


def smart_purge_strategy(inputs: Dict[str, Any], position_mp: float, slope_ft_per_mile: float, cumulative_n2_scf: float, cutoff_volume_scf: float) -> float:
    # stop injecting once we've hit cutoff
    if cumulative_n2_scf >= cutoff_volume_scf:
        return 0.0
    # Start tapering near the end window
    if position_mp >= inputs['purge_end_mp'] - inputs['throttle_down_miles']:
        taper_factor = max((inputs['purge_end_mp'] - position_mp) / max(inputs['throttle_down_miles'], 1e-6), 0.0)
        return inputs['max_drive_pressure'] * max(0.7, taper_factor)
    # Uphill bias — keep it simple; you can elaborate later if desired
    if slope_ft_per_mile > 50.0:
        return inputs['max_drive_pressure']
    return inputs['max_drive_pressure']


# ----------------
# Main simulation
# ----------------
def _prep_fluids(inputs: Dict[str, Any]) -> Tuple[float, float]:
    """Return (fluid_density [lb/ft^3], dynamic_viscosity [lb*s/ft^2])"""
    if inputs['fluid_num'] == 3:  # Crude oil from API gravity
        api = float(inputs.get('api_gravity', 0))
        if api <= 0:
            raise ValueError("For Crude Oil, provide a positive API gravity.")
        sg = 141.5 / (131.5 + api)
        # simple correlation used in original code
        viscosity_cst = 10 ** (10 - 0.25 * api)
    else:
        data = FLUID_DATA[inputs['fluid_num']]
        sg = data['sg']
        viscosity_cst = data['viscosity_cst']
    fluid_density = sg * 62.4
    # 1 cSt = 1.076e-5 ft^2/s; mu = rho * nu
    viscosity = fluid_density * viscosity_cst * 1.076e-5
    return fluid_density, viscosity


def _prep_geometry(inputs: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return (pipe_ID_in, pipe_D_ft, area_ft2)."""
    od_in = float(NPS_DATA[inputs['nps']]['OD_in'])
    id_in = od_in - 2.0 * float(inputs['pipe_wt'])
    if id_in <= 0:
        raise ValueError("Pipe wall thickness too large for selected NPS.")
    D_ft = id_in / 12.0
    area = math.pi * (D_ft / 2.0) ** 2
    return id_in, D_ft, area


def _to_feet(elevations: np.ndarray, units: str) -> np.ndarray:
    if (units or 'ft').lower().startswith('m'):
        return elevations * 3.28084
    return elevations


def run_simulation(inputs: Dict[str, Any], mileposts: np.ndarray, elevations: np.ndarray) -> Dict[str, Any]:
    """Core purge simulation. Arrays should span the whole system profile.
    - mileposts: monotonically increasing mile markers (system profile)
    - elevations: same length as mileposts
    """
    # Units & geometry
    elevations = _to_feet(np.asarray(elevations, dtype=float), inputs.get('elevation_units', 'ft'))
    mileposts = np.asarray(mileposts, dtype=float)
    if len(mileposts) != len(elevations) or len(mileposts) < 2:
        raise ValueError("Profile arrays must be same length and have at least 2 points.")

    _, D_ft, area = _prep_geometry(inputs)
    atm = 14.7
    roughness = float(ROUGHNESS_DATA[inputs['roughness_num']]['roughness_ft'])
    rho, mu = _prep_fluids(inputs)

    # Profile splines
    sys_spline = _cubic_spline(mileposts, elevations)
    h_exit = float(sys_spline(inputs['system_end_mp']))

    # Window selection & resampling
    mp0 = float(inputs['purge_start_mp'])
    mp1 = float(inputs['purge_end_mp'])
    n_points = int(inputs.get('n_points', 1000))
    purge_mps = np.linspace(mp0, mp1, n_points)
    elevs = np.atleast_1d(sys_spline(purge_mps)).astype(float)

    # Head losses (psi) at each purge position relative to system end
    head_losses = rho * (h_exit - elevs) / 144.0

    # Nitrogen cutoff volume estimate (SCF)
    purge_len_ft = (inputs['purge_end_mp'] - inputs['purge_start_mp']) * 5280.0
    total_scf_est = area * purge_len_ft * (inputs['n2_end_pressure'] + atm) / atm
    cutoff_scf = float(inputs.get('cutoff_volume', total_scf_est))

    # Arrays
    n = n_points
    distances_ft = purge_mps * 5280.0
    switch_mp = inputs['purge_end_mp'] - inputs['throttle_down_miles']

    elapsed = np.zeros(n)
    P_drive = np.zeros(n)
    P_exit = np.zeros(n)
    P_fric = np.zeros(n)
    inj_rate_scf_min = np.zeros(n)
    cum_n2_scf = np.zeros(n)
    pig_speed_mph = np.zeros(n)

    # IPS handling
    ips_active = bool(inputs.get('has_ips', False))
    current_sys_end_mp = inputs['ips_mp'] if ips_active else inputs['system_end_mp']
    current_exit_target = inputs['min_pump_suction_pressure'] if ips_active else inputs['exit_pressure_run']

    # slope for strategy
    slope = calculate_trendline_slope(mileposts, elevations, inputs['purge_start_mp'], inputs['system_end_mp'])

    # Initial
    v_initial = min(inputs['max_pig_speed'] * 1.46667, inputs['max_pig_speed'] * 1.25 * 1.46667)  # ft/s
    injection_active = True
    x_cutoff_mp = purge_mps[0]
    p_at_cutoff = 0.0

    # Helper to get target exit pressure at a given MP
    def target_exit_pressure(mp: float) -> float:
        return inputs['exit_pressure_end'] if mp >= switch_mp else current_exit_target

    for i in range(n - 1):
        # Switch off IPS when approaching its shutdown zone
        if ips_active and purge_mps[i] >= inputs['ips_mp'] - inputs['ips_shutdown_dist']:
            ips_active = False
            current_sys_end_mp = inputs['system_end_mp']
            current_exit_target = inputs['exit_pressure_run']

        L_remaining_ft = max(current_sys_end_mp * 5280.0 - distances_ft[i], 1.0)
        v_max = inputs['max_pig_speed'] * 1.46667
        target_exit = target_exit_pressure(purge_mps[i])

        # Choose drive pressure (may be zero if cutoff reached)
        Pset = smart_purge_strategy(inputs, purge_mps[i], slope, cum_n2_scf[i], cutoff_scf)
        if Pset == 0 and injection_active:
            # first time we cut off, remember where & at what pressure
            injection_active = False
            x_cutoff_mp = purge_mps[i]
            p_at_cutoff = P_drive[i - 1] if i > 0 else inputs['max_drive_pressure']

        # If we're coasting after cutoff, simple expansion ratio model
        if not injection_active:
            current_dist = max(purge_mps[i] - mp0, 1e-6)
            cutoff_dist = x_cutoff_mp - mp0
            ratio = max(cutoff_dist / current_dist, 0.0)
            Pset = (p_at_cutoff + atm) * ratio - atm
            Pset = max(Pset, inputs['exit_pressure_end'])

        # Find a velocity that satisfies exit pressure (coarse fixed-point loop)
        v = min(v_initial, v_max)
        for _ in range(30):
            dP_f = calculate_friction_loss(v, L_remaining_ft, D_ft, rho, mu, roughness)
            Pexit_calc = Pset - dP_f - head_losses[i]
            if abs(Pexit_calc - target_exit) < 0.01:
                break
            if Pexit_calc < target_exit:
                v *= 0.95
            elif v < v_max:
                v *= 1.05
        # record
        P_fric[i] = dP_f
        pig_speed_mph[i] = v / 1.46667
        # Min speed check (stop if we stall)
        if pig_speed_mph[i] < inputs['min_pig_speed']:
            # truncate here
            last_i = i
            P_drive[:last_i + 1][i] = Pset
            P_exit[:last_i + 1][i] = target_exit
            break

        # Update pressures
        P_drive[i] = Pset
        P_exit[i + 1] = target_exit

        # Time step
        seg_len = distances_ft[i + 1] - distances_ft[i]
        dt_hr = seg_len / max(v, 1e-6) / 3600.0
        elapsed[i + 1] = elapsed[i] + dt_hr

        # Nitrogen accounting
        if injection_active:
            vol_ft3_per_s = area * v
            inj_scf_per_s = vol_ft3_per_s * (Pset + atm) / atm
            inj_rate_scf_min[i] = inj_scf_per_s * 60.0
            added_scf = inj_rate_scf_min[i] * dt_hr * 60.0
            cum_n2_scf[i + 1] = cum_n2_scf[i] + added_scf
        else:
            inj_rate_scf_min[i] = 0.0
            cum_n2_scf[i + 1] = cum_n2_scf[i]
        v_initial = v
    else:
        # finalize last index if we completed the loop
        i = n - 1
        L_remaining_ft = max(current_sys_end_mp * 5280.0 - distances_ft[i], 1.0)
        target_exit = target_exit_pressure(purge_mps[i])
        v = min(v_initial, inputs['max_pig_speed'] * 1.46667)
        dP_f = calculate_friction_loss(v, L_remaining_ft, D_ft, rho, mu, roughness)
        P_fric[i] = dP_f
        pig_speed_mph[i] = v / 1.46667
        P_drive[i] = smart_purge_strategy(inputs, purge_mps[i], slope, cum_n2_scf[i], cutoff_scf)
        P_exit[i] = target_exit
        last_i = i

    # Collect exceed points if hard cap was requested
    exceed_points = []
    if inputs.get('hard_cap', False):
        for i in range(n - 1):
            if purge_mps[i] < switch_mp and P_exit[i] > target_exit_pressure(purge_mps[i]) + 0.01:
                exceed_points.append((float(purge_mps[i]), float(P_exit[i])))

    return {
        'purge_mileposts': purge_mps[: last_i + 1],
        'elevations': elevs[: last_i + 1],
        'elapsed_times': elapsed[: last_i + 1],
        'drive_pressures': P_drive[: last_i + 1],
        'friction_losses': P_fric[: last_i + 1],
        'head_losses': head_losses[: last_i + 1],
        'exit_pressures': P_exit[: last_i + 1],
        'injection_rates': inj_rate_scf_min[: last_i + 1],
        'cumulative_n2': cum_n2_scf[: last_i + 1],
        'pig_speeds': pig_speed_mph[: last_i + 1],
        'exceed_points': exceed_points,
        'last_valid_i': int(last_i),
    }


# ------------------------------
# Pandas-friendly convenience API
# ------------------------------
def simulate_pipeline(inputs: Dict[str, Any], profile_df) -> Dict[str, Any]:
    """profile_df: pandas DataFrame with columns ['Milepost', 'Elevation'].
    This is just a thin wrapper around run_simulation()."""
    mileposts = np.asarray(profile_df['Milepost'].values, dtype=float)
    elevations = np.asarray(profile_df['Elevation'].values, dtype=float)
    return run_simulation(inputs, mileposts, elevations)
