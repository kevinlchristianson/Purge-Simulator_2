# New engine module (UI-agnostic)

This introduces a modular purge engine under `purge_engine/` that mirrors the legacy engine's control logic while staying independent of any UI toolkit.

Highlights:
- Hard-cap and taper-down enforced inside the solver (not just flags)
- IPS handling: force max drive before shutdown, then switch endpoint/targets
- Real exit pressure computed, with exceed_points capture
- Nitrogen cutoff and coasting
- Optional SciPy spline with safe linear fallback
- Dataclass inputs/results for clarity

Usage:

```python
import numpy as np
from purge_engine.types import Inputs, StrategyConfig, IPSConfig
from purge_engine.engine import run_simulation

# Profile
mileposts = np.array([...], dtype=float)
elevations = np.array([...], dtype=float)

inputs = Inputs(
  nps="24",
  pipe_wt=0.375,
  roughness_num=1,
  fluid_num=3,
  api_gravity=35.0,
  purge_start_mp=0.0,
  purge_end_mp=10.0,
  system_end_mp=12.0,
  strategy=StrategyConfig(
    max_drive_pressure=600.0,
    exit_pressure_run=250.0,
    exit_pressure_end=150.0,
    n2_end_pressure=500.0,
    max_pig_speed=8.0,
    min_pig_speed=0.5,
    throttle_down_miles=1.0,
    hard_cap=True,
    taper_down_enabled=True,
  ),
  ips=IPSConfig(
    has_ips=True,
    ips_mp=6.0,
    ips_shutdown_dist=0.5,
    min_pump_suction_pressure=80.0,
  ),
  elevation_units="ft",
  n_points=1000,
)

results = run_simulation(inputs, mileposts, elevations)
```

To retire `page6_summary.py`, we've replaced it with a thin wrapper that adapts the existing dict-based inputs to the new engine. The UI can keep calling `run_simulation(inputs_dict, mileposts, elevations)` as before.