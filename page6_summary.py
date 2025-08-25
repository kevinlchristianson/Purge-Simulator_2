"""
Thin wrapper around the new purge_engine, preserving the old function signature.
UI can continue to call run_simulation(inputs: dict, mileposts: np.ndarray, elevations: np.ndarray).
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from purge_engine.types import Inputs, StrategyConfig, IPSConfig
from purge_engine.engine import run_simulation as _run_sim


def _dict_to_inputs(d: Dict[str, Any]) -> Inputs:
    strategy = StrategyConfig(
        max_drive_pressure=float(d.get("max_drive_pressure", 0.0)),
        exit_pressure_run=float(d.get("exit_pressure_run", 0.0)),
        exit_pressure_end=float(d.get("exit_pressure_end", 0.0)),
        n2_end_pressure=float(d.get("n2_end_pressure", 0.0)),
        max_pig_speed=float(d.get("max_pig_speed", 0.0)),
        min_pig_speed=float(d.get("min_pig_speed", 0.0)),
        throttle_down_miles=float(d.get("throttle_down_miles", 0.0)),
        hard_cap=bool(d.get("hard_cap", False)),
        taper_down_enabled=bool(d.get("taper_down_enabled", True)),
        cutoff_volume=(None if d.get("cutoff_volume") is None else float(d.get("cutoff_volume"))),
    )
    ips = IPSConfig(
        has_ips=bool(d.get("has_ips", False)),
        ips_mp=float(d.get("ips_mp", 0.0)),
        ips_shutdown_dist=float(d.get("ips_shutdown_dist", 0.0)),
        min_pump_suction_pressure=float(d.get("min_pump_suction_pressure", 0.0)),
    )
    return Inputs(
        nps=str(d.get("nps")),
        pipe_wt=float(d.get("pipe_wt")),
        roughness_num=int(d.get("roughness_num", 1)),
        fluid_num=int(d.get("fluid_num", 4)),
        api_gravity=float(d.get("api_gravity", 0.0)),
        purge_start_mp=float(d.get("purge_start_mp")),
        purge_end_mp=float(d.get("purge_end_mp")),
        system_end_mp=float(d.get("system_end_mp")),
        strategy=strategy,
        ips=ips,
        elevation_units=str(d.get("elevation_units", "ft")),
        n_points=int(d.get("n_points", 1000)),
    )


def run_simulation(inputs: Dict[str, Any], mileposts: np.ndarray, elevations: np.ndarray) -> Dict[str, Any]:
    """Adapt dict inputs to the new engine and return results as a dict of arrays."""
    in_dataclass = _dict_to_inputs(inputs)
    res = _run_sim(in_dataclass, np.asarray(mileposts, dtype=float), np.asarray(elevations, dtype=float))
    return {
        "purge_mileposts": res.purge_mileposts,
        "elevations": res.elevations,
        "elapsed_times": res.elapsed_times,
        "drive_pressures": res.drive_pressures,
        "friction_losses": res.friction_losses,
        "head_losses": res.head_losses,
        "exit_pressures": res.exit_pressures,
        "injection_rates": res.injection_rates,
        "cumulative_n2": res.cumulative_n2,
        "pig_speeds": res.pig_speeds,
        "exceed_points": res.exceed_points,
        "last_valid_i": res.last_valid_i,
    }


def simulate_pipeline(inputs: Dict[str, Any], profile_df) -> Dict[str, Any]:
    """Pandas-friendly wrapper; profile_df must have ['Milepost', 'Elevation'] columns."""
    mileposts = np.asarray(profile_df['Milepost'].values, dtype=float)
    elevations = np.asarray(profile_df['Elevation'].values, dtype=float)
    return run_simulation(inputs, mileposts, elevations)