from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple

import numpy as np

from .types import Inputs, Results
from .profile import make_profile_spline, to_feet

# -----------------------------
# Data tables (expanded NPS)
# -----------------------------
NPS_DATA: Dict[str, Dict[str, float]] = {
    "1/8": {"OD_in": 0.405}, "1/4": {"OD_in": 0.540}, "3/8": {"OD_in": 0.675}, "1/2": {"OD_in": 0.840},
    "3/4": {"OD_in": 1.050}, "1": {"OD_in": 1.315}, "1 1/4": {"OD_in": 1.660}, "1 1/2": {"OD_in": 1.900},
    "2": {"OD_in": 2.375}, "2 1/2": {"OD_in": 2.875}, "3": {"OD_in": 3.500}, "3 1/2": {"OD_in": 4.000},
    "4": {"OD_in": 4.500}, "5": {"OD_in": 5.563}, "6": {"OD_in": 6.625}, "8": {"OD_in": 8.625},
    "10": {"OD_in": 10.750}, "12": {"OD_in": 12.750}, "14": {"OD_in": 14.000}, "16": {"OD_in": 16.000},
    "18": {"OD_in": 18.000}, "20": {"OD_in": 20.000}, "24": {"OD_in": 24.000}, "26": {"OD_in": 26.000},
    "28": {"OD_in": 28.000}, "30": {"OD_in": 30.000}, "32": {"OD_in": 32.000}, "34": {"OD_in": 34.000},
    "36": {"OD_in": 36.000}, "40": {"OD_in": 40.000}, "42": {"OD_in": 42.000}, "44": {"OD_in": 44.000},
    "48": {"OD_in": 48.000}, "52": {"OD_in": 52.000}, "56": {"OD_in": 56.000}, "60": {"OD_in": 60.000},
    "64": {"OD_in": 64.000}, "68": {"OD_in": 68.000}, "72": {"OD_in": 72.000}, "80": {"OD_in": 80.000},
}

ROUGHNESS_DATA = {
    1: {"material": "New Welded Steel", "roughness_ft": 0.00015},
    2: {"material": "Rusted/Corroded Welded Steel", "roughness_ft": 0.0005},
    3: {"material": "Welded HDPE", "roughness_ft": 0.000005},
}

FLUID_DATA = {
    1: {"name": "Diesel", "sg": 0.84, "viscosity_cst": 2.7},
    2: {"name": "Gasoline", "sg": 0.74, "viscosity_cst": 0.6},
    3: {"name": "Crude Oil", "sg": None, "viscosity_cst": None},
    4: {"name": "Water", "sg": 1.0, "viscosity_cst": 1.0},
    5: {"name": "NGL (Y1-grade)", "sg": 0.6, "viscosity_cst": 0.3},
}


def _prep_geometry(nps: str, pipe_wt_in: float):
    if nps not in NPS_DATA:
        raise ValueError(f"NPS '{nps}' not in catalog.")
    od_in = float(NPS_DATA[nps]["OD_in"])
    id_in = od_in - 2.0 * float(pipe_wt_in)
    if id_in <= 0:
        raise ValueError("Pipe wall thickness too large for selected NPS.")
    D_ft = id_in / 12.0
    area_ft2 = math.pi * (D_ft / 2.0) ** 2
    return id_in, D_ft, area_ft2


def _prep_fluids(fluid_num: int, api_gravity: float):
    if fluid_num == 3:
        api = float(api_gravity)
        if api <= 0:
            raise ValueError("For Crude Oil, provide a positive API gravity.")
        sg = 141.5 / (131.5 + api)
        viscosity_cst = 10 ** (10 - 0.25 * api)
    else:
        data = FLUID_DATA[fluid_num]
        sg = data["sg"]
        viscosity_cst = data["viscosity_cst"]
    rho = sg * 62.4  # lb/ft^3
    nu_ft2_s = viscosity_cst * 1.076e-5  # 1 cSt = 1.076e-5 ft^2/s
    mu = rho * nu_ft2_s  # lb*s/ft^2
    return rho, mu


def calculate_trendline_slope(mileposts: np.ndarray, elevations: np.ndarray, start_mp: float, end_mp: float) -> float:
    mask = (mileposts >= start_mp) & (mileposts <= end_mp)
    sel_x = mileposts[mask]
    sel_y = elevations[mask]
    if sel_x.size < 2:
        return 0.0
    return float((sel_y[-1] - sel_y[0]) / (sel_x[-1] - sel_x[0]))


def friction_loss_psi(v_ft_s: float, L_ft: float, D_ft: float, rho: float, mu: float, roughness_ft: float) -> float:
    Re = max(rho * v_ft_s * D_ft / max(mu, 1e-12), 1e-6)
    haaland = -1.8 * math.log10((roughness_ft / D_ft / 3.7) ** 1.11 + 6.9 / Re)
    f = (1.0 / haaland) ** 2
    dP_lbf_ft2 = f * (L_ft / D_ft) * rho * v_ft_s * v_ft_s / (2.0 * 32.174)
    return dP_lbf_ft2 / 144.0


def run_simulation(inputs: Inputs, mileposts: np.ndarray, elevations: np.ndarray) -> Results:
    # Prepare arrays and splines
    mileposts = np.asarray(mileposts, dtype=float)
    elevations = to_feet(np.asarray(elevations, dtype=float), inputs.elevation_units)
    if mileposts.size != elevations.size or mileposts.size < 2:
        raise ValueError("Profile arrays must be same length and have at least 2 points.")
    prof = make_profile_spline(mileposts, elevations)

    # Geometry/fluids
    _, D_ft, area_ft2 = _prep_geometry(inputs.nps, inputs.pipe_wt)
    rho, mu = _prep_fluids(inputs.fluid_num, inputs.api_gravity)
    roughness_ft = float(ROUGHNESS_DATA[inputs.roughness_num]["roughness_ft"])
    atm = 14.7

    # Window resampling
    mp0 = float(inputs.purge_start_mp)
    mp1 = float(inputs.purge_end_mp)
    n = int(max(10, inputs.n_points))
    purge_mps = np.linspace(mp0, mp1, n)
    elevs = prof(purge_mps).astype(float)
    distances_ft = purge_mps * 5280.0

    # Strategy helpers
    sw_mp = mp1 - inputs.strategy.throttle_down_miles
    slope = calculate_trendline_slope(mileposts, elevations, mp0, inputs.system_end_mp)

    # IPS state
    ips_active = bool(inputs.ips.has_ips)
    current_end_mp = inputs.ips.ips_mp if ips_active else inputs.system_end_mp
    current_exit_target = (
        inputs.ips.min_pump_suction_pressure if ips_active else inputs.strategy.exit_pressure_run
    )

    # Head against current end: compute on-the-fly since end can change
    def head_loss_psi(position_mp: float, end_mp: float) -> float:
        h_end = float(prof(end_mp))
        h_pos = float(prof(position_mp))
        return rho * (h_end - h_pos) / 144.0

    # Nitrogen cutoff default
    purge_len_ft = (mp1 - mp0) * 5280.0
    total_scf_est = area_ft2 * purge_len_ft * (inputs.strategy.n2_end_pressure + atm) / atm
    cutoff_scf = inputs.strategy.cutoff_volume if inputs.strategy.cutoff_volume is not None else total_scf_est

    # Arrays
    elapsed = np.zeros(n)
    P_drive = np.zeros(n)
    P_exit = np.zeros(n)
    P_fric = np.zeros(n)
    head_vec = np.zeros(n)
    inj_rate_scf_min = np.zeros(n)
    cum_n2_scf = np.zeros(n)
    pig_speed_mph = np.zeros(n)

    # State
    v_prev = inputs.strategy.max_pig_speed * 1.46667
    injection_active = True
    cutoff_mp = mp0
    drive_at_cutoff = 0.0

    def target_exit_pressure(mp: float) -> float:
        local_target = inputs.strategy.exit_pressure_end if mp >= sw_mp else current_exit_target
        return float(local_target)

    def choose_drive_pressure(mp: float, base_max: float) -> float:
        # IPS: force max drive while active
        if ips_active:
            return min(base_max, inputs.strategy.max_drive_pressure)

        # Throttle-down near end if enabled
        if inputs.strategy.taper_down_enabled and mp >= sw_mp and inputs.strategy.throttle_down_miles > 0:
            taper_factor = max((inputs.purge_end_mp - mp) / max(inputs.strategy.throttle_down_miles, 1e-6), 0.0)
            floor = 0.7
            return min(base_max, inputs.strategy.max_drive_pressure * max(floor, taper_factor))

        # Uphill bias (keep simple; keep max)
        if slope > 50.0:
            return min(base_max, inputs.strategy.max_drive_pressure)
        return min(base_max, inputs.strategy.max_drive_pressure)

    last_i = n - 1
    for i in range(n - 1):
        mp = float(purge_mps[i])

        # IPS shutdown handling
        if ips_active and mp >= (inputs.ips.ips_mp - inputs.ips.ips_shutdown_dist):
            ips_active = False
            current_end_mp = inputs.system_end_mp
            current_exit_target = inputs.strategy.exit_pressure_run  # post-shutdown target during run window

        # Remaining length to current endpoint
        L_rem_ft = max(current_end_mp * 5280.0 - distances_ft[i], 1.0)
        H_psi = head_loss_psi(mp, current_end_mp)
        head_vec[i] = H_psi
        target_exit = target_exit_pressure(mp)
        v_max = inputs.strategy.max_pig_speed * 1.46667

        # Determine drive pressure setpoint (may be overridden for coasting)
        Pset = choose_drive_pressure(mp, base_max=inputs.strategy.max_drive_pressure)

        # Nitrogen cutoff/coast model
        if injection_active and cum_n2_scf[i] >= cutoff_scf:
            injection_active = False
            cutoff_mp = mp
            drive_at_cutoff = P_drive[i - 1] if i > 0 else Pset

        if not injection_active:
            # Distance-based decay similar to legacy: pressure decays ~ inverse with distance past cutoff
            current_dist = max(mp - mp0, 1e-6)
            cutoff_dist = max(cutoff_mp - mp0, 1e-6)
            ratio = max(min(cutoff_dist / current_dist, 1.0), 0.0)
            Pset = max(inputs.strategy.exit_pressure_end, (drive_at_cutoff + atm) * ratio - atm)

        # Branching behavior for hard-cap and taper-down
        v = min(v_prev, v_max)

        # First, evaluate at v_max for decision branches
        dP_f_vmax = friction_loss_psi(v_max, L_rem_ft, D_ft, rho, mu, roughness_ft)
        Pexit_vmax = Pset - dP_f_vmax - H_psi

        if inputs.strategy.taper_down_enabled:
            # If at end taper region we want to cap P_exit at target by trimming Pset (within allowed max)
            if mp >= sw_mp and Pexit_vmax > target_exit:
                # Reduce drive to just meet target at v_max
                Pset = min(Pset, target_exit + dP_f_vmax + H_psi)
                Pexit_vmax = target_exit  # reflects trimmed setpoint at v_max

        if inputs.strategy.hard_cap:
            # Hard-cap semantics: if at v_max exit exceeds target, we run at v_max and record exceed
            if Pexit_vmax > target_exit:
                v = v_max
                dP_f = dP_f_vmax
                Pexit = Pexit_vmax
            else:
                # Otherwise iterate v to hit target
                v = min(v_prev, v_max)
                for _ in range(30):
                    dP_f = friction_loss_psi(v, L_rem_ft, D_ft, rho, mu, roughness_ft)
                    Pexit = Pset - dP_f - H_psi
                    if abs(Pexit - target_exit) < 0.01:
                        break
                    if Pexit < target_exit:
                        v *= 0.95
                    else:
                        v = min(v * 1.05, v_max)
        else:
            # No hard-cap: always iterate v to target if possible
            for _ in range(30):
                dP_f = friction_loss_psi(v, L_rem_ft, D_ft, rho, mu, roughness_ft)
                Pexit = Pset - dP_f - H_psi
                if abs(Pexit - target_exit) < 0.01:
                    break
                if Pexit < target_exit:
                    v *= 0.95
                else:
                    v = min(v * 1.05, v_max)

        # Record friction and exit
        P_fric[i] = dP_f
        P_exit[i] = Pexit
        pig_speed_mph[i] = v / 1.46667

        # Stall guard
        if pig_speed_mph[i] < inputs.strategy.min_pig_speed:
            last_i = i
            P_drive[i] = Pset
            break

        # Drive pressure (after potential taper/hard-cap adjustments)
        P_drive[i] = Pset

        # Time step
        seg_len = distances_ft[i + 1] - distances_ft[i]
        dt_hr = seg_len / max(v, 1e-6) / 3600.0
        elapsed[i + 1] = elapsed[i] + dt_hr

        # Nitrogen accounting
        if injection_active:
            vol_ft3_per_s = area_ft2 * v
            inj_scf_per_s = vol_ft3_per_s * (Pset + atm) / atm
            inj_rate_scf_min[i] = inj_scf_per_s * 60.0
            cum_n2_scf[i + 1] = cum_n2_scf[i] + inj_rate_scf_min[i] * dt_hr * 60.0
        else:
            inj_rate_scf_min[i] = 0.0
            cum_n2_scf[i + 1] = cum_n2_scf[i]

        v_prev = v

    else:
        i = n - 1
        last_i = i
        # Final index bookkeeping
        mp = float(purge_mps[i])
        L_rem_ft = max(current_end_mp * 5280.0 - distances_ft[i], 1.0)
        H_psi = head_loss_psi(mp, current_end_mp)
        head_vec[i] = H_psi
        v = min(v_prev, inputs.strategy.max_pig_speed * 1.46667)
        dP_f = friction_loss_psi(v, L_rem_ft, D_ft, rho, mu, roughness_ft)
        P_fric[i] = dP_f
        P_exit[i] = (P_drive[i - 1] if i > 0 else inputs.strategy.max_drive_pressure) - dP_f - H_psi
        pig_speed_mph[i] = v / 1.46667

    # Exceed points (actual computed exit vs target)
    exceed_points: List[Tuple[float, float]] = []
    if inputs.strategy.hard_cap:
        for j, mp in enumerate(purge_mps[: last_i + 1]):
            target = inputs.strategy.exit_pressure_end if mp >= sw_mp else (
                inputs.ips.min_pump_suction_pressure if (inputs.ips.has_ips and mp < (inputs.ips.ips_mp - inputs.ips.ips_shutdown_dist)) else inputs.strategy.exit_pressure_run
            )
            if P_exit[j] > target + 0.01 and mp < sw_mp:
                exceed_points.append((float(mp), float(P_exit[j])))

    return Results(
        purge_mileposts=purge_mps[: last_i + 1],
        elevations=elevs[: last_i + 1],
        elapsed_times=elapsed[: last_i + 1],
        drive_pressures=P_drive[: last_i + 1],
        friction_losses=P_fric[: last_i + 1],
        head_losses=head_vec[: last_i + 1],
        exit_pressures=P_exit[: last_i + 1],
        injection_rates=inj_rate_scf_min[: last_i + 1],
        cumulative_n2=cum_n2_scf[: last_i + 1],
        pig_speeds=pig_speed_mph[: last_i + 1],
        exceed_points=exceed_points,
        last_valid_i=int(last_i),
    )