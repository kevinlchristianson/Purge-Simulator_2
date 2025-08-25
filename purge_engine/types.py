from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class IPSConfig:
    has_ips: bool = False
    ips_mp: float = 0.0
    ips_shutdown_dist: float = 0.0
    min_pump_suction_pressure: float = 0.0  # psi


@dataclass
class StrategyConfig:
    max_drive_pressure: float = 0.0  # psi
    exit_pressure_run: float = 0.0   # psi
    exit_pressure_end: float = 0.0   # psi
    n2_end_pressure: float = 0.0     # psi
    max_pig_speed: float = 0.0       # mph
    min_pig_speed: float = 0.0       # mph
    throttle_down_miles: float = 0.0
    hard_cap: bool = False
    taper_down_enabled: bool = True
    cutoff_volume: Optional[float] = None  # scf


@dataclass
class Inputs:
    # Geometry
    nps: str
    pipe_wt: float  # inches
    roughness_num: int  # 1=new steel, 2=corroded steel, 3=HDPE

    # Fluid
    fluid_num: int  # 1=Diesel, 2=Gasoline, 3=Crude, 4=Water, 5=NGL
    api_gravity: float = 0.0  # only if crude

    # Window and endpoints
    purge_start_mp: float = 0.0
    purge_end_mp: float = 0.0
    system_end_mp: float = 0.0

    # Strategy
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # IPS
    ips: IPSConfig = field(default_factory=IPSConfig)

    # Misc
    elevation_units: str = "ft"
    n_points: int = 1000


@dataclass
class Results:
    purge_mileposts: np.ndarray
    elevations: np.ndarray
    elapsed_times: np.ndarray
    drive_pressures: np.ndarray
    friction_losses: np.ndarray
    head_losses: np.ndarray
    exit_pressures: np.ndarray
    injection_rates: np.ndarray
    cumulative_n2: np.ndarray
    pig_speeds: np.ndarray
    exceed_points: List[Tuple[float, float]]
    last_valid_i: int