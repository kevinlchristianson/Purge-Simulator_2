# Pipeline Pigging Simulation Program with Smart Enhancements (v8g_fix3)
import numpy as np
import pandas as pd
import zipfile
from lxml import etree as ET
from scipy.interpolate import CubicSpline
from geopy.distance import geodesic
import tkinter as tk
from tkinter import messagebox, filedialog, ttk, simpledialog
import matplotlib.pyplot as plt
import time
import logging
import unittest
import os
import sys
import math

# Set up logging with console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('purge_simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Test tkinter availability
try:
    tk.Tk().destroy()
    logging.info("Tkinter initialized successfully")
except Exception as e:
    logging.error(f"Tkinter initialization failed: {str(e)}")
    print(f"Error: Tkinter initialization failed: {str(e)}")
    sys.exit(1)

# NPS data (Outer Diameter in inches)
nps_data = {
    "1/8": {"OD_in": 0.405}, "1/4": {"OD_in": 0.540}, "3/8": {"OD_in": 0.675}, "1/2": {"OD_in": 0.840},
    "3/4": {"OD_in": 1.050}, "1": {"OD_in": 1.315}, "1 1/4": {"OD_in": 1.660}, "1 1/2": {"OD_in": 1.900},
    "2": {"OD_in": 2.375}, "2 1/2": {"OD_in": 2.875}, "3": {"OD_in": 3.500}, "3 1/2": {"OD_in": 4.000},
    "4": {"OD_in": 4.500}, "5": {"OD_in": 5.563}, "6": {"OD_in": 6.625}, "8": {"OD_in": 8.625},
    "10": {"OD_in": 10.750}, "12": {"OD_in": 12.750}, "14": {"OD_in": 14.000}, "16": {"OD_in": 16.000},
    "18": {"OD_in": 18.000}, "20": {"OD_in": 20.000}, "24": {"OD_in": 24.000}, "26": {"OD_in": 26.000},
    "28": {"OD_in": 28.000}, "30": {"OD_in": 30.000}, "32": {"OD_in": 32.000}, "34": {"OD_in": 34.000},
    "36": {"OD_in": 36.000}, "40": {"OD_in": 40.000}, "42": {"OD_in": 42.000}, "44": {"OD_in": 44.000},
    "48": {"OD_in": 48.000}, "52": {"OD_in": 52.000}, "56": {"OD_in": 56.000}, "60": {"OD_in": 60.000},
    "64": {"OD_in": 64.000}, "68": {"OD_in": 68.000}, "72": {"OD_in": 72.000}, "80": {"OD_in": 80.000},}

# Roughness data (in feet)
roughness_data = {
    1: {"material": "New Welded Steel", "roughness_ft": 0.00015},
    2: {"material": "Rusted/Corroded Welded Steel", "roughness_ft": 0.0005},
    3: {"material": "Welded HDPE", "roughness_ft": 0.000005}
}

# Fluid data
fluid_data = {
    1: {"name": "Diesel", "sg": 0.84, "viscosity_cst": 2.7},
    2: {"name": "Gasoline", "sg": 0.74, "viscosity_cst": 0.6},
    3: {"name": "Crude Oil", "sg": None, "viscosity_cst": None},
    4: {"name": "Water", "sg": 1.0, "viscosity_cst": 1.0},
    5: {"name": "NGL (Y1-grade)", "sg": 0.6, "viscosity_cst": 0.3}
}

# Nitrogen properties (used for gas-side calculations)
# We treat N2 as a real gas via a Peng–Robinson EOS for Z-factor.
N2_MW_KG_PER_MOL = 0.0280134
N2_CRIT_T_K = 126.192
N2_CRIT_P_PA = 3.3958e6  # ~492.3 psia
N2_ACENTRIC = 0.0372

# Assumed nitrogen temperature in the pipe (ground temperature), per user convention.
N2_TEMP_F_DEFAULT = 45.0

# Dynamic viscosity of nitrogen (Pa*s). A constant is adequate for purge-speed screening.
N2_VISCOSITY_PA_S = 1.75e-5
SCF_TO_FT3 = 1.0  # 1 SCF == 1 ft³ at STP
ATM_PSI = 14.7

# ------------------------ Small utility helpers ------------------------
def resolve_nps_key(nps_val):
    """Map GUI 'nps' (str like '16' or float 16.0) to a key present in nps_data."""
    # Exact match first
    if nps_val in nps_data:
        return nps_val
    # Try string->float->string conversions
    try:
        as_float = float(nps_val)
        # Try exact string of int
        cand = str(int(round(as_float)))
        if cand in nps_data:
            return cand
        # Try scanning keys numerically
        for k in nps_data.keys():
            try:
                if abs(float(k) - as_float) < 1e-9:
                    return k
            except Exception:
                continue
    except Exception:
        pass
    # As a last resort just string it
    k = str(nps_val)
    if k in nps_data:
        return k
    raise KeyError(f"NPS '{nps_val}' not found in nps_data; available keys: {list(nps_data.keys())}")

def print_inputs(inputs):
    print("\n=== Simulation Inputs ===")
    for key, value in sorted(inputs.items()):
        print(f"{key}: {value}")
    print("=== End Inputs ===\n")

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def calculate_trendline_slope(mileposts, elevations, start_mp, end_mp):
    logging.debug(f"Calculating trendline slope from {start_mp} to {end_mp}")
    mask = (mileposts >= start_mp) & (mileposts <= end_mp)
    selected_mileposts = mileposts[mask]
    selected_elevations = elevations[mask]
    if len(selected_mileposts) < 2:
        logging.warning("Insufficient points for trendline slope calculation")
        return 0
    slope = (selected_elevations[-1] - selected_elevations[0]) / (selected_mileposts[-1] - selected_mileposts[0])
    logging.debug(f"Calculated slope: {slope}")
    return slope

def show_help():
    logging.info("Displaying help guide")
    help_text = """
Pipeline Pigging Simulation Program
==================================
- This version auto-adjusts N2 injection rate to maximize pig speed (<= Max) at a fixed exit pressure.
- Exit pressure tapers from 'run' to 'end' over the last N miles (Throttle Down).
- Friction uses Darcy–Weisbach (Swamee–Jain/Haaland), applied to remaining liquid slug length.
- Export fixes: NPS key is resolved robustly, avoiding the 16.0 KeyError.
    """
    try:
        help_window = tk.Toplevel()
        help_window.title("User Help")
        help_window.update()
        text = tk.Text(help_window, height=20, width=80)
        text.insert(tk.END, help_text)
        text.pack(padx=10, pady=10)
        tk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=5)
        logging.info("Help guide displayed")
    except Exception as e:
        logging.error(f"Failed to display help guide: {str(e)}")
        print(f"Error: Failed to display help guide: {str(e)}")
        raise

def get_user_inputs(dialog_root, profile_start_mp, profile_end_mp):
    logging.info("Entering get_user_inputs")
    inputs = {}
    def validate_and_submit():
        try:
            nonlocal inputs
            logging.debug("Validating user inputs")
            nps = nps_var.get()
            if not nps:
                raise ValueError("Nominal Pipe Size must be selected.")
            pipe_wt = wt_entry.get().strip()
            if not pipe_wt:
                raise ValueError("Wall thickness must be provided.")
            pipe_wt = float(pipe_wt)
            pipe_od = nps_data[resolve_nps_key(nps)]["OD_in"]
            if pipe_wt <= 0 or pipe_wt >= pipe_od / 2:
                raise ValueError("Wall thickness must be positive and less than half the OD.")
            roughness_num = int(material_var.get().split(':')[0])
            if roughness_num not in roughness_data:
                raise ValueError("Invalid pipe material selected.")
            fluid_num = int(fluid_var.get().split(':')[0])
            if fluid_num not in fluid_data:
                raise ValueError("Invalid fluid selected.")
            api_gravity = float(api_entry.get()) if fluid_num == 3 else None
            viscosity_cst = float(viscosity_entry.get()) if fluid_num == 3 else None
            if fluid_num == 3:
                if api_gravity <= 0:
                    raise ValueError("API Gravity must be positive.")
                if viscosity_cst <= 0:
                    raise ValueError("Viscosity must be positive.")
            # Optional max cap on N2 rate
            max_n2_rate_scfm = float(max_rate_entry.get()) if max_rate_entry.get().strip() else None
            n2_cutoff_scf = float(n2_cutoff_entry.get()) if n2_cutoff_entry.get().strip() else None
            exit_run = float(exit_run_entry.get())
            if exit_run < 0:
                raise ValueError("Minimum Exit Pressure (Run) must be non-negative.")
            exit_end = float(exit_end_entry.get())
            if exit_end < 0:
                raise ValueError("Exit Pressure (End) must be non-negative.")
            n2_end = float(n2_end_entry.get())
            if n2_end < 0:
                raise ValueError("Estimated nitrogen pressure must be non-negative.")
            max_speed = float(max_speed_entry.get())
            if max_speed <= 0:
                raise ValueError("Max Pig Speed must be positive.")
            min_speed = float(min_speed_entry.get())
            if min_speed < 0:
                raise ValueError("Min Pig Speed must be non-negative.")
            if max_speed <= min_speed / 1.25:
                raise ValueError("Max Pig Speed must be greater than Min Pig Speed / 1.25.")
            target_speed = float(target_speed_entry.get())
            if target_speed <= 0 or target_speed > max_speed:
                raise ValueError("Target Pig Speed must be positive and <= Max Pig Speed.")
            purge_start = float(purge_start_entry.get())
            purge_end = float(purge_end_entry.get())
            system_end = float(system_end_entry.get())
            if not (purge_start < purge_end <= system_end):
                raise ValueError("Mileposts must satisfy: Start < End <= System End.")
            if purge_start < profile_start_mp:
                logging.warning(f"Purge Start Milepost {purge_start:.1f} is below profile range {profile_start_mp:.1f}. Adjusting to {profile_start_mp:.1f}.")
                purge_start = profile_start_mp
            if system_end > profile_end_mp:
                logging.warning(f"System Endpoint Milepost {system_end:.1f} exceeds profile range {profile_end_mp:.1f}. Adjusting to {profile_end_mp:.1f}.")
                system_end = profile_end_mp
            if purge_end > system_end:
                logging.warning(f"Purge End Milepost {purge_end:.1f} exceeds System Endpoint {system_end:.1f}. Adjusting to {system_end:.1f}.")
                purge_end = system_end
            throttle_down = float(throttle_down_entry.get())
            if throttle_down < 0 or throttle_down > (purge_end - purge_start):
                raise ValueError(f"Throttle Down Point must be between 0 and {purge_end - purge_start:.1f} miles.")
            
            has_ips = ips_var.get()
            ips_mp = None
            ips_shutdown_dist = None
            min_pump_suction_pressure = None
            if has_ips:
                ips_mp = ips_mp_entry.get().strip()
                if not ips_mp:
                    raise ValueError("IPS milepost must be provided.")
                ips_mp = float(ips_mp)
                if not (purge_start < ips_mp < system_end):
                    raise ValueError("IPS milepost must be between Purge Start and System End.")
                ips_shutdown_dist = ips_shutdown_entry.get().strip()
                if not ips_shutdown_dist:
                    raise ValueError("IPS shutdown distance must be provided.")
                ips_shutdown_dist = float(ips_shutdown_dist)
                if ips_shutdown_dist <= 0:
                    raise ValueError("IPS shutdown distance must be positive.")
                min_pump_suction_pressure = min_pump_suction_entry.get().strip()
                if not min_pump_suction_pressure:
                    raise ValueError("Minimum Pump Suction Pressure must be provided.")
                min_pump_suction_pressure = float(min_pump_suction_pressure)
                if min_pump_suction_pressure < 0:
                    raise ValueError("Minimum Pump Suction Pressure must be non-negative.")
            
            resolution = int(resolution_entry.get()) if resolution_entry.get().strip() else 500
            if resolution < 50 or resolution > 2000:
                raise ValueError("Resolution must be between 50 and 2000 points.")
            
            # "Max Nitrogen Pressure" is the maximum allowable nitrogen pressure at the injection gauge.
            # For backwards compatibility with older files/exports, we also store it under
            # the legacy key name "max_drive_pressure".
            max_nitrogen_pressure = float(max_nitrogen_pressure_entry.get())

            inputs.update({
                'nps': nps, 'pipe_wt': pipe_wt, 'roughness_num': roughness_num, 'fluid_num': fluid_num,
                'api_gravity': api_gravity, 'viscosity_cst': viscosity_cst, 'max_n2_rate_scfm': max_n2_rate_scfm,
                'exit_pressure_run': exit_run, 'exit_pressure_end': exit_end, 'n2_end_pressure': n2_end,
                'max_pig_speed': max_speed, 'min_pig_speed': min_speed, 'target_pig_speed': target_speed,
                'purge_start_mp': purge_start, 'purge_end_mp': purge_end, 'system_end_mp': system_end,
                'throttle_down_miles': throttle_down, 'hard_cap': hard_cap_var.get(),
                'taper_down_enabled': taper_down_var.get(), 'elevation_format': elev_format_var.get(),
                'elevation_units': elev_units_var.get(), 'has_ips': has_ips, 'ips_mp': ips_mp,
                'ips_shutdown_dist': ips_shutdown_dist, 'min_pump_suction_pressure': min_pump_suction_pressure,
                'resolution': resolution,
                'n2_cutoff_scf': n2_cutoff_scf,
                'max_nitrogen_pressure': max_nitrogen_pressure,
                'max_drive_pressure': max_nitrogen_pressure
            })
            logging.info(f"Validated inputs: NPS={nps}, Max N2 Rate={(max_n2_rate_scfm if max_n2_rate_scfm is not None else 'auto')} SCFM, Target Speed={target_speed} mph, Purge Start={purge_start}, Purge End={purge_end}, Resolution={resolution}")
            root.destroy()
        except ValueError as e:
            logging.error(f"Input validation failed: {str(e)}")
            messagebox.showerror("Input Error", str(e), parent=root)
    
    try:
        logging.info("Initializing Tkinter root for input GUI")
        root = tk.Toplevel(dialog_root)
        logging.info("Tkinter input GUI created")
        root.title("Pipeline Pigging Simulation Inputs")
        root.lift()
        root.attributes('-topmost', True)
        root.after(100, lambda: root.attributes('-topmost', False))
        root.update()
        messagebox.showinfo("Tip", "Copy files to C:\\Users\\YourName\\Documents to avoid network drive hangs.", parent=root)
    except Exception as e:
        logging.error(f"Failed to initialize Tkinter input GUI: {str(e)}")
        print(f"Error: Failed to initialize Tkinter input GUI: {str(e)}")
        raise
    
    row = 0
    tk.Label(root, text="Nominal Pipe Size (NPS):").grid(row=row, column=0, sticky='e')
    nps_var = tk.StringVar(value="16")
    ttk.Combobox(root, textvariable=nps_var, values=[""] + list(nps_data.keys())).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Wall Thickness (in):").grid(row=row, column=0, sticky='e')
    wt_entry = tk.Entry(root)
    wt_entry.insert(0, "0.280")
    wt_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Pipe Material:").grid(row=row, column=0, sticky='e')
    material_var = tk.StringVar(value="2: Rusted/Corroded Welded Steel")
    material_options = [f"{k}: {v['material']}" for k, v in roughness_data.items()]
    ttk.Combobox(root, textvariable=material_var, values=material_options).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Fluid Type:").grid(row=row, column=0, sticky='e')
    fluid_var = tk.StringVar(value="3: Crude Oil")
    fluid_options = [f"{k}: {v['name']}" for k, v in fluid_data.items()]
    ttk.Combobox(root, textvariable=fluid_var, values=fluid_options).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="API Gravity (if Crude Oil):").grid(row=row, column=0, sticky='e')
    api_entry = tk.Entry(root)
    api_entry.insert(0, "25")
    api_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Viscosity (cSt) (if Crude Oil):").grid(row=row, column=0, sticky='e')
    viscosity_entry = tk.Entry(root)
    viscosity_entry.insert(0, "200")
    viscosity_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Min Exit Pressure (Run) (psi):").grid(row=row, column=0, sticky='e')
    exit_run_entry = tk.Entry(root)
    exit_run_entry.insert(0, "100")
    exit_run_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Exit Pressure (End) (psi):").grid(row=row, column=0, sticky='e')
    exit_end_entry = tk.Entry(root)
    exit_end_entry.insert(0, "100")
    exit_end_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Est. N2 Pressure at End (psi):").grid(row=row, column=0, sticky='e')
    n2_end_entry = tk.Entry(root)
    n2_end_entry.insert(0, "300")
    n2_end_entry.grid(row=row, column=1)
    row += 1

    # This is the maximum nitrogen pressure allowed at the injection gauge.
    tk.Label(root, text="Max Nitrogen Pressure (psi):").grid(row=row, column=0, sticky='e')
    max_nitrogen_pressure_entry = tk.Entry(root)
    max_nitrogen_pressure_entry.insert(0, "400")
    max_nitrogen_pressure_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Max N2 Rate (SCFM) [optional]:").grid(row=row, column=0, sticky='e')
    max_rate_entry = tk.Entry(root)
    max_rate_entry.insert(0, "")
    max_rate_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="N2 Cutoff Volume (SCF) [optional]:").grid(row=row, column=0, sticky='e')
    n2_cutoff_entry = tk.Entry(root)
    n2_cutoff_entry.insert(0, "")
    n2_cutoff_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Max Pig Speed (mph):").grid(row=row, column=0, sticky='e')
    max_speed_entry = tk.Entry(root)
    max_speed_entry.insert(0, "4")
    max_speed_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Min Pig Speed (mph):").grid(row=row, column=0, sticky='e')
    min_speed_entry = tk.Entry(root)
    min_speed_entry.insert(0, "1")
    min_speed_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Target Pig Speed (mph):").grid(row=row, column=0, sticky='e')
    target_speed_entry = tk.Entry(root)
    target_speed_entry.insert(0, "2")
    target_speed_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text=f"Purge Start Milepost (min {profile_start_mp:.1f}):").grid(row=row, column=0, sticky='e')
    purge_start_entry = tk.Entry(root)
    purge_start_entry.insert(0, f"{profile_start_mp:.1f}")
    purge_start_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text=f"Purge End Milepost (max {profile_end_mp:.1f}):").grid(row=row, column=0, sticky='e')
    purge_end_entry = tk.Entry(root)
    purge_end_entry.insert(0, f"{profile_end_mp:.1f}")
    purge_end_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text=f"System Endpoint Milepost (max {profile_end_mp:.1f}):").grid(row=row, column=0, sticky='e')
    system_end_entry = tk.Entry(root)
    system_end_entry.insert(0, f"{profile_end_mp:.1f}")
    system_end_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Throttle Down Point (miles from end):").grid(row=row, column=0, sticky='e')
    throttle_down_entry = tk.Entry(root)
    throttle_down_entry.insert(0, "9.6")
    throttle_down_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Hard Cap on Max Speed?").grid(row=row, column=0, sticky='e')
    hard_cap_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, variable=hard_cap_var).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Enable Taper Down Phase?").grid(row=row, column=0, sticky='e')
    taper_down_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, variable=taper_down_var).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Is there an Intermediate Pump Station?").grid(row=row, column=0, sticky='e')
    ips_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, variable=ips_var).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="IPS Milepost:").grid(row=row, column=0, sticky='e')
    ips_mp_entry = tk.Entry(root)
    ips_mp_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="IPS Shutdown Distance (miles):").grid(row=row, column=0, sticky='e')
    ips_shutdown_entry = tk.Entry(root)
    ips_shutdown_entry.insert(0, "1")
    ips_shutdown_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Minimum Pump Suction Pressure (psi):").grid(row=row, column=0, sticky='e')
    min_pump_suction_entry = tk.Entry(root)
    min_pump_suction_entry.insert(0, "50")
    min_pump_suction_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Elevation Profile Format:").grid(row=row, column=0, sticky='e')
    elev_format_var = tk.StringVar(value="TXT")
    ttk.Combobox(root, textvariable=elev_format_var, values=["KMZ/KML", "Excel", "TXT"]).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Elevation Units:").grid(row=row, column=0, sticky='e')
    elev_units_var = tk.StringVar(value="Feet")
    ttk.Combobox(root, textvariable=elev_units_var, values=["Feet", "Meters"]).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Resolution (points):").grid(row=row, column=0, sticky='e')
    resolution_entry = tk.Entry(root)
    resolution_entry.insert(0, "500")
    resolution_entry.grid(row=row, column=1)
    row += 1
    
    tk.Button(root, text="Submit", command=validate_and_submit).grid(row=row, column=0, columnspan=2)
    tk.Button(root, text="Help", command=show_help).grid(row=row, column=1, sticky='e')
    
    logging.info("Starting Tkinter wait for input GUI")
    root.grab_set()
    start_time = time.time()
    while root.winfo_exists():
        dialog_root.update()
        root.update()
        if time.time() - start_time > 60:
            logging.warning("Input GUI timeout after 60 seconds")
            raise ValueError("Input GUI timed out. Try running from Command Prompt: cd C:\\Users\\YourName\\Documents && python Purge_Modeling_Program_22_09_2025.py")
    logging.info("Exiting get_user_inputs")
    return inputs

def calculate_friction_loss(v, L, D, fluid_density, viscosity, roughness, purged_fraction=0):
    # Convert "fluid_density" (weight density, lb/ft^3) to mass density (slug/ft^3)
    rho_m = fluid_density / 32.174
    # NOTE: Downstream of a perfectly-sealing pig we assume the segment is liquid-full.
    # Therefore, do NOT blend viscosity/density toward nitrogen based on purged fraction.
    mu_eff = viscosity
    try:
        Re = rho_m * v * D / max(mu_eff, 1e-12)
        # Haaland friction factor
        haaland = -1.8 * np.log10((roughness / D / 3.7)**1.11 + 6.9 / max(Re, 1e-12))
        f = (1.0 / haaland)**2
        # Darcy–Weisbach pressure drop (psi)
        dp_psf = f * (L / D) * 0.5 * rho_m * v * v
        friction_loss = dp_psf / 144.0
        return friction_loss
    except Exception as e:
        logging.error(f"Friction loss calculation failed: {str(e)}")
        raise ValueError(f"Error in friction loss calculation: {str(e)}")

def target_exit_pressure(mp, inputs, purge_start, purge_end, throttle_down):
    distance_from_end = purge_end - mp
    if throttle_down > 0 and distance_from_end <= throttle_down:
        taper_factor = max(0.0, distance_from_end / throttle_down)
        return inputs['exit_pressure_run'] * taper_factor + inputs['exit_pressure_end'] * (1 - taper_factor)
    return inputs['exit_pressure_run']

# === SI-based slug friction helper (Darcy–Weisbach over remaining liquid) ===
def slug_friction_psi_SI(L_ft, D_ft, v_fts, api_gravity, viscosity_cst, eps_ft, sg_fallback=0.85):
    """
    Darcy–Weisbach dp across the *liquid slug ahead of the pig*.
    Imperial inputs, internal calc in SI, returns psi.
    """
    try:
        if L_ft <= 0 or v_fts <= 0 or D_ft <= 0:
            return 0.0
        # Imperial -> SI
        ft_to_m = 0.3048
        L = L_ft * ft_to_m
        D = D_ft * ft_to_m
        v = v_fts * ft_to_m         # ft/s -> m/s
        eps = eps_ft * ft_to_m

        # Fluid properties
        if api_gravity is None and sg_fallback is not None:
            SG = sg_fallback
        elif api_gravity is None:
            SG = 0.85
        else:
            SG = 141.5 / (api_gravity + 131.5)   # crude SG from API
        rho = 999.0 * SG                     # kg/m^3  (~water*SG)
        nu  = (viscosity_cst if viscosity_cst else 1.0) * 1.0e-6   # m^2/s   (cSt -> m^2/s)

        # Reynolds number and friction factor (Swamee-Jain)
        Re = (v * D) / max(1e-12, nu)
        if Re < 2300.0:
            f = 64.0 / max(1.0, Re)
        else:
            f = 0.25 / (math.log10((eps/(3.7*D)) + (5.74/(Re**0.9))))**2

        # Darcy–Weisbach dp (Pa) and convert to psi
        dp_Pa = f * (L / D) * 0.5 * rho * v * v
        dp_psi = dp_Pa / 6894.757
        return float(dp_psi)
    except Exception:
        return 0.0


# === Real-gas nitrogen helpers (Peng–Robinson) ===
def _f_to_k(t_f: float) -> float:
    return (t_f - 32.0) * (5.0 / 9.0) + 273.15

def _psia_to_pa(p_psia: float) -> float:
    return float(p_psia) * 6894.757

def _pa_to_psia(p_pa: float) -> float:
    return float(p_pa) / 6894.757

def z_factor_n2_pr(p_psia: float, t_f: float) -> float:
    """Return nitrogen compressibility factor Z using Peng–Robinson EOS."""
    p_pa = max(1.0, _psia_to_pa(p_psia))
    t_k = max(1.0, _f_to_k(t_f))
    R = 8.314462618  # J/mol/K

    # Peng–Robinson parameters
    kappa = 0.37464 + 1.54226 * N2_ACENTRIC - 0.26992 * (N2_ACENTRIC ** 2)
    alpha = (1.0 + kappa * (1.0 - math.sqrt(t_k / N2_CRIT_T_K))) ** 2
    a = 0.45724 * (R ** 2) * (N2_CRIT_T_K ** 2) / N2_CRIT_P_PA
    b = 0.07780 * R * N2_CRIT_T_K / N2_CRIT_P_PA
    A = a * alpha * p_pa / (R ** 2 * t_k ** 2)
    B = b * p_pa / (R * t_k)

    # PR cubic: Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0
    c3 = 1.0
    c2 = -(1.0 - B)
    c1 = A - 3.0 * B * B - 2.0 * B
    c0 = -(A * B - B * B - B ** 3)

    roots = np.roots([c3, c2, c1, c0])
    # Take largest real root (gas phase)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]
    if not real_roots:
        return 1.0
    Z = max(real_roots)
    # Guardrails
    if not np.isfinite(Z) or Z <= 0:
        return 1.0
    return float(Z)


def n2_density_kg_m3(p_psia: float, t_f: float) -> float:
    """Nitrogen density from PR EOS: rho = P*MW/(Z*R*T)."""
    p_pa = _psia_to_pa(p_psia)
    t_k = _f_to_k(t_f)
    Z = z_factor_n2_pr(p_psia, t_f)
    R = 8.314462618
    rho = p_pa * N2_MW_KG_PER_MOL / (max(1e-12, Z) * R * t_k)
    return float(rho)


def n2_moles_from_scf(scf: float, t_std_f: float = 60.0, z_std: float = 1.0) -> float:
    """Convert SCF at standard conditions into moles."""
    # Standard conditions: 14.7 psia, 60F by convention
    p_std_pa = _psia_to_pa(ATM_PSI)
    t_std_k = _f_to_k(t_std_f)
    V_m3 = float(scf) * 0.028316846592  # ft^3 -> m^3
    R = 8.314462618
    n = p_std_pa * V_m3 / (max(1e-12, z_std) * R * t_std_k)
    return float(n)


def n2_pressure_psia_from_moles(n_mol: float, V_ft3: float, t_f: float) -> float:
    """Solve for nitrogen pressure (psia) given moles, volume, temperature using PR EOS."""
    V_m3 = max(1e-12, float(V_ft3) * 0.028316846592)
    t_k = _f_to_k(t_f)
    R = 8.314462618
    # Ideal initial guess
    p_pa = max(1.0, n_mol * R * t_k / V_m3)
    # Fixed-point iteration with Z(P)
    for _ in range(20):
        p_psia = _pa_to_psia(p_pa)
        Z = z_factor_n2_pr(p_psia, t_f)
        p_new = n_mol * R * t_k * Z / V_m3
        if abs(p_new - p_pa) / max(1.0, p_pa) < 1e-8:
            p_pa = p_new
            break
        p_pa = p_new
    return float(_pa_to_psia(p_pa))


def gas_friction_loss_psi_SI(L_ft: float, D_ft: float, v_fts: float, p_avg_psia: float, t_f: float, eps_ft: float) -> float:
    """Darcy–Weisbach dp (psi) for nitrogen over length L at average pressure p_avg."""
    if L_ft <= 0 or v_fts <= 0 or D_ft <= 0:
        return 0.0
    ft_to_m = 0.3048
    L = L_ft * ft_to_m
    D = D_ft * ft_to_m
    v = v_fts * ft_to_m
    eps = eps_ft * ft_to_m

    rho = n2_density_kg_m3(p_avg_psia, t_f)
    mu = N2_VISCOSITY_PA_S
    Re = rho * v * D / max(1e-12, mu)
    if Re < 2300.0:
        f = 64.0 / max(1.0, Re)
    else:
        f = 0.25 / (math.log10((eps/(3.7*D)) + (5.74/(Re**0.9))))**2

    dp_pa = f * (L / D) * 0.5 * rho * v * v
    return float(dp_pa / 6894.757)


def required_injection_pressure_psig(p_behind_pig_psig: float, L_gas_ft: float, D_ft: float, v_fts: float, eps_ft: float, t_f: float) -> tuple:
    """Return (p_inj_psig, dp_gas_psi) required to supply the pressure immediately behind the pig."""
    p_behind_pig_psia = float(p_behind_pig_psig) + ATM_PSI
    if L_gas_ft <= 0 or v_fts <= 0:
        return float(p_behind_pig_psig), 0.0
    # Iterate because density depends on pressure and dp depends on density.
    p_inj_psia = p_behind_pig_psia
    for _ in range(20):
        p_avg = 0.5 * (p_behind_pig_psia + p_inj_psia)
        dp = gas_friction_loss_psi_SI(L_gas_ft, D_ft, v_fts, p_avg, t_f, eps_ft)
        p_new = p_behind_pig_psia + dp
        if abs(p_new - p_inj_psia) / max(1.0, p_inj_psia) < 1e-8:
            p_inj_psia = p_new
            break
        p_inj_psia = p_new
    dp_gas = max(0.0, p_inj_psia - p_behind_pig_psia)
    return float(p_inj_psia - ATM_PSI), float(dp_gas)


def scfm_required_from_velocity(v_fts: float, area_ft2: float, p_inj_psig: float, p_behind_pig_psig: float, t_f: float) -> float:
    """Compute nitrogen SCFM required to sustain volumetric displacement at pig speed."""
    if v_fts <= 0 or area_ft2 <= 0:
        return 0.0
    # Use mass flow at average pressure
    p_inj_psia = float(p_inj_psig) + ATM_PSI
    p_behind_pig_psia = float(p_behind_pig_psig) + ATM_PSI
    p_avg = 0.5 * (p_inj_psia + p_behind_pig_psia)
    rho_avg = n2_density_kg_m3(p_avg, t_f)
    ft_to_m = 0.3048
    area_m2 = float(area_ft2) * (ft_to_m ** 2)
    v_m_s = float(v_fts) * ft_to_m
    m_dot = rho_avg * v_m_s * area_m2  # kg/s

    # Standard density at 14.7 psia, 60F
    rho_std = n2_density_kg_m3(ATM_PSI, 60.0)
    q_std_m3_s = m_dot / max(1e-12, rho_std)
    scfm = q_std_m3_s * 35.3146667 * 60.0
    return float(scfm)


def run_simulation(dialog_root, inputs, purge_mileposts, elevations, system_mileposts, system_elevations):
    # Work on a copy so we don't mutate inputs passed to export_results
    cfg = dict(inputs)

    # ---- normalize numeric inputs to floats (robust to Tk string values) ----
    def _to_float(val, default=None):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    _numeric_fields = [
        'pipe_wt','viscosity_cst','api_gravity','max_nitrogen_pressure','max_drive_pressure',
        'exit_pressure_run','exit_pressure_end','purge_end_mp','purge_start_mp',
        'roughness_num','max_n2_rate_scfm','max_pig_speed','min_pig_speed',
        'n2_end_pressure','resolution','system_end_mp','target_pig_speed'
    ]
    for _k in _numeric_fields:
        if _k in cfg:
            _v = _to_float(cfg[_k], None)
            if _v is not None:
                cfg[_k] = _v

    # Resolve NPS
    nps_key = resolve_nps_key(cfg['nps'])
    pipe_od = nps_data[nps_key]["OD_in"]
    pipe_id = pipe_od - 2 * cfg['pipe_wt']
    if pipe_id <= 0:
        raise ValueError(f"Computed ID <= 0. Check NPS {nps_key} and wall {cfg['pipe_wt']}.")
    pipe_diameter = pipe_id / 12.0  # ft
    area = np.pi * (pipe_diameter / 2) ** 2
    roughness = roughness_data[int(cfg['roughness_num'])]['roughness_ft']
    
    # Fluid properties
    if int(cfg['fluid_num']) == 3:
        specific_gravity = 141.5 / (131.5 + cfg['api_gravity'])  # SG from API
        viscosity_cst = cfg['viscosity_cst']
    else:
        specific_gravity = fluid_data[int(cfg['fluid_num'])]['sg']
        viscosity_cst = fluid_data[int(cfg['fluid_num'])]['viscosity_cst']
    # Use weight density (lb/ft^3) for hydrostatic, convert to mass density when needed
    gamma = specific_gravity * 62.4
    fluid_density = gamma  # keep legacy name for compatibility
    viscosity = (gamma / 32.174) * (viscosity_cst * 1.076e-5)  # dynamic mu (lb*s/ft^2)

    purge_length = (cfg['purge_end_mp'] - cfg['purge_start_mp']) * 5280.0
    total_volume_ft3 = area * purge_length
    total_volume_scf = total_volume_ft3 / SCF_TO_FT3

    # --- Nitrogen temperature & real-gas Z-factor ---
    # Per user convention we assume pipeline/gas temperature is approximately ground temperature.
    n2_temp_f = float(cfg.get('n2_temp_f', N2_TEMP_F_DEFAULT) or N2_TEMP_F_DEFAULT)

    # Compute the automatic nitrogen cut‑off volume (SCF @ 14.7 psia, 60F) based on
    # the desired end pressure *throughout the purged volume*.
    # Real-gas correction: V_std = V_pipe * (P_end/P_std) * (T_std/T_pipe) * (Z_std/Z_end)
    P_end_abs = float(cfg['n2_end_pressure']) + ATM_PSI
    Z_end = z_factor_n2_pr(P_end_abs, n2_temp_f)
    Z_std = z_factor_n2_pr(ATM_PSI, 60.0)
    T_std_R = 60.0 + 459.67
    T_pipe_R = n2_temp_f + 459.67
    cutoff_volume = total_volume_ft3 * (P_end_abs / ATM_PSI) * (T_std_R / max(1e-12, T_pipe_R)) * (Z_std / max(1e-12, Z_end))
    logging.info(
        f"Total pipe volume: {total_volume_ft3:.0f} ft³, {total_volume_scf:.0f} SCF, "
        f"Cutoff (N2) volume: {cutoff_volume:.0f} SCF"
    )
    
    n_points = len(purge_mileposts)
    distances = purge_mileposts * 5280.0

    # Static head: positive means uphill (pressure you must overcome), negative downhill (assist)
    cs = CubicSpline(system_mileposts, system_elevations)
    h_exit = cs(cfg['system_end_mp'])
    head_losses = fluid_density * (h_exit - elevations) / 144.0  # psi

    elapsed_times = np.zeros(n_points, dtype=np.float64)
    # Pressure immediately behind the pig (psig). Historically this was called
    # "drive pressure" in the program, but "pressure behind pig" is clearer.
    behind_pig_pressures = np.zeros(n_points, dtype=np.float64)
    # Inlet injection pressure at purge start (psig)
    injection_pressures = np.zeros(n_points, dtype=np.float64)
    # Gas friction dp from injection point -> pig (psi)
    gas_dp_losses = np.zeros(n_points, dtype=np.float64)

    friction_losses = np.zeros(n_points, dtype=np.float64)
    exit_pressures = np.zeros(n_points, dtype=np.float64)
    injection_rates = np.zeros(n_points, dtype=np.float64)
    cumulative_n2 = np.zeros(n_points, dtype=np.float64)
    pig_speeds = np.zeros(n_points, dtype=np.float64)
    exceed_points = []
    
    cumulative_n2[0] = 0.0
    cumulative_distance = 0.0
    last_valid_i = n_points - 1
    v_max = cfg['max_pig_speed'] * 1.46667  # ft/s
    v_target = float(cfg.get('target_pig_speed', cfg['max_pig_speed']) or cfg['max_pig_speed']) * 1.46667
    max_inj_psig = float(cfg.get('max_nitrogen_pressure', cfg.get('max_drive_pressure', 400.0)) or 400.0)  # inlet cap

    # Max N2 rate: None -> unlimited ("auto")
    max_rate_in = cfg.get('max_n2_rate_scfm', None)
    try:
        max_rate = float(max_rate_in) if (max_rate_in is not None and float(max_rate_in) > 0) else math.inf
    except Exception:
        max_rate = math.inf

    # Optional nitrogen cutoff (SCF) -- stop injection and coast using stored gas
    # Determine nitrogen cut‑off volume.  If the user specified a
    # manual cut‑off (n2_cutoff_scf) use it; otherwise fall back to
    # the automatically computed cutoff_volume.  A non‑positive
    # user‑provided value is ignored.
    n2_cutoff = cfg.get('n2_cutoff_scf', None)
    if n2_cutoff is not None:
        try:
            n2_cutoff = float(n2_cutoff)
            if n2_cutoff <= 0:
                n2_cutoff = None
        except Exception:
            n2_cutoff = None
    # Use automatic cut‑off if no valid manual value is provided
    cutoff_value = n2_cutoff if n2_cutoff is not None else cutoff_volume
    coast_mode = False
    n2_moles_total = None   # mol (at standard, injected)
    V_gas_ft3 = None        # ft^3 (geometric volume behind pig)

    for i in range(n_points - 1):
        segment_length_ft = max(1e-6, distances[i + 1] - distances[i])

        # Exit pressure for this MP (with taper)
        P_exit = target_exit_pressure(
            purge_mileposts[i], cfg, cfg['purge_start_mp'], cfg['purge_end_mp'], cfg['throttle_down_miles']
        )
        head = head_losses[i]

        # Remaining liquid slug length ahead of pig (ft)
        slug_length_ft = max(0.0, (cfg['system_end_mp'] - purge_mileposts[i]) * 5280.0)
        # Gas length behind pig (ft) from *purge start MP* (injection point)
        L_gas_ft = max(0.0, (purge_mileposts[i] - cfg['purge_start_mp']) * 5280.0)

        # --- Available pressure behind pig in coast mode ---
        P_trap_psig = None
        if coast_mode and n2_moles_total is not None and V_gas_ft3 is not None:
            p_psia = n2_pressure_psia_from_moles(n2_moles_total, V_gas_ft3, n2_temp_f)
            P_trap_psig = max(0.0, p_psia - ATM_PSI)

        # --- Helper to evaluate feasibility at a trial speed ---
        def evaluate(v_try: float):
            # Liquid friction (psi) at v_try
            api_val = cfg.get('api_gravity', None)
            if api_val is None and int(cfg['fluid_num']) != 3:
                sg = fluid_data[int(cfg['fluid_num'])]['sg']
                if sg and sg > 0:
                    api_val = 141.5 / sg - 131.5

            fr_liq = slug_friction_psi_SI(slug_length_ft, pipe_diameter, max(1e-6, v_try), api_val, viscosity_cst, roughness, sg_fallback=specific_gravity)
            # Required pressure immediately behind the pig (psig)
            P_behind_pig_req = max(P_exit, P_exit + head + fr_liq)

            if coast_mode:
                # No flow in the trapped gas column => no inlet→pig gas DP while coasting
                cap = float(P_trap_psig) if P_trap_psig is not None else 0.0
                ok = (P_behind_pig_req <= cap + 1e-9)
                # For output we report the trapped pressure (what the gauge would read)
                P_behind_pig_out = cap
                P_inj_out = cap
                dp_gas = 0.0
                q_req = 0.0
                return ok, P_behind_pig_req, P_behind_pig_out, P_inj_out, dp_gas, q_req, fr_liq

            # Injection mode: include gas DP from injection point (purge start) to pig
            P_inj_req, dp_gas = required_injection_pressure_psig(P_behind_pig_req, L_gas_ft, pipe_diameter, v_try, roughness, n2_temp_f)
            if P_inj_req > max_inj_psig + 1e-9:
                return False, P_behind_pig_req, P_behind_pig_req, P_inj_req, dp_gas, 0.0, fr_liq

            q_req = scfm_required_from_velocity(v_try, area, P_inj_req, P_behind_pig_req, n2_temp_f)
            if q_req > max_rate + 1e-9:
                return False, P_behind_pig_req, P_behind_pig_req, P_inj_req, dp_gas, q_req, fr_liq

            return True, P_behind_pig_req, P_behind_pig_req, P_inj_req, dp_gas, q_req, fr_liq

        # Determine speed target for this step
        if coast_mode:
            v_hi = v_max
        else:
            v_hi = min(v_max, v_target) if v_target and v_target > 0 else v_max

        # Quick stall check at near-zero speed
        ok0, P_behind_pig_req0, _, P_inj0, _, _, fr0 = evaluate(1e-6)
        if not ok0:
            pig_speeds[i] = 0.0
            friction_losses[i] = fr0
            behind_pig_pressures[i] = 0.0 if (P_trap_psig is None) else float(P_trap_psig)
            injection_pressures[i] = 0.0 if (P_trap_psig is None) else float(P_trap_psig)
            gas_dp_losses[i] = 0.0
            exit_pressures[i] = P_exit
            injection_rates[i] = 0.0
            last_valid_i = i
            msg = f"Stall at MP {purge_mileposts[i]:.2f}: cannot meet even minimal flow."
            if not coast_mode:
                msg += f" Required injection {P_inj0:.1f} psig exceeds cap {max_inj_psig:.1f} psig."
            else:
                msg += f" Required behind-pig {P_behind_pig_req0:.1f} psig exceeds trapped {float(P_trap_psig or 0):.1f} psig."
            logging.warning(msg)
            break

        # If v_hi feasible, use it (keeps injection mode pinned to target when possible)
        ok_hi, P_behind_pig_req, P_behind_pig_out, P_inj_out, dp_gas, q_use, fr_use = evaluate(v_hi)
        if not ok_hi:
            # Binary search for max feasible speed up to v_hi
            v_lo = 0.0
            best = (0.0, P_behind_pig_req0, 0.0, P_inj0, 0.0, 0.0, fr0)
            for _ in range(30):
                v_mid = 0.5 * (v_lo + v_hi)
                ok, Pdr, Pdo, Pinj, dpg, q, fr = evaluate(v_mid)
                if ok:
                    best = (v_mid, Pdr, Pdo, Pinj, dpg, q, fr)
                    v_lo = v_mid
                else:
                    v_hi = v_mid
            v, P_behind_pig_req, P_behind_pig_out, P_inj_out, dp_gas, q_use, fr_use = best
        else:
            v = v_hi

        # Record step values
        pig_speeds[i] = v / 1.46667
        friction_losses[i] = fr_use
        behind_pig_pressures[i] = P_behind_pig_out
        injection_pressures[i] = P_inj_out
        gas_dp_losses[i] = dp_gas
        exit_pressures[i] = P_exit
        injection_rates[i] = 0.0 if coast_mode else q_use

        # Time for this segment; if v too small, stop
        if v <= 1e-8:
            last_valid_i = i
            logging.warning(f"Near-zero velocity at MP {purge_mileposts[i]:.2f}; stopping.")
            break
        time_step_hours = segment_length_ft / v / 3600.0
        elapsed_times[i + 1] = elapsed_times[i] + time_step_hours

        # Integrate nitrogen (SCF @ std) during injection
        if coast_mode and cutoff_value is not None:
            cumulative_n2[i] = cutoff_value
        n2_added_scf = 0.0 if coast_mode else (q_use * time_step_hours * 60.0)
        cumulative_n2[i + 1] = cumulative_n2[i] + n2_added_scf

        # Update trapped gas volume (coast) after moving
        if coast_mode and V_gas_ft3 is not None:
            V_gas_ft3 += area * segment_length_ft

        # Switch to coasting for next steps once cutoff reached
        if (not coast_mode) and (cutoff_value is not None) and (cumulative_n2[i + 1] >= cutoff_value):
            cumulative_n2[i + 1] = cutoff_value
            coast_mode = True
            n2_moles_total = n2_moles_from_scf(cutoff_value)
            # Initialize gas volume behind pig at the *next* position
            L_gas_next_ft = max(1e-6, (purge_mileposts[i + 1] - cfg['purge_start_mp']) * 5280.0)
            V_gas_ft3 = max(1e-6, area * L_gas_next_ft)

        cumulative_distance += segment_length_ft

    # --- Populate the final row (pig at purge end) so exports/tables don't show zeros ---
    j = min(last_valid_i, n_points - 1)
    try:
        mp_j = purge_mileposts[j]
        P_exit_j = target_exit_pressure(mp_j, cfg, cfg['purge_start_mp'], cfg['purge_end_mp'], cfg['throttle_down_miles'])
        head_j = head_losses[j]
        # With pig stopped, flowing friction is ~0; static requirement is exit + head (if uphill)
        P_behind_pig_static = max(P_exit_j, P_exit_j + head_j)

        if coast_mode and (n2_moles_total is not None) and (V_gas_ft3 is not None):
            p_psia = n2_pressure_psia_from_moles(n2_moles_total, V_gas_ft3, n2_temp_f)
            P_trap_psig = max(0.0, p_psia - ATM_PSI)
            behind_pig_pressures[j] = P_trap_psig
            injection_pressures[j] = P_trap_psig
            gas_dp_losses[j] = 0.0
        else:
            behind_pig_pressures[j] = P_behind_pig_static
            injection_pressures[j] = P_behind_pig_static
            gas_dp_losses[j] = 0.0

        exit_pressures[j] = P_exit_j
        friction_losses[j] = 0.0
        pig_speeds[j] = 0.0
        injection_rates[j] = 0.0
        if coast_mode and cutoff_value is not None:
            cumulative_n2[j] = cutoff_value
    except Exception:
        pass

    logging.info("Exiting run_simulation")
    return {
        'purge_mileposts': purge_mileposts[:last_valid_i + 1],
        'elevations': elevations[:last_valid_i + 1],
        'elapsed_times': elapsed_times[:last_valid_i + 1],
        # Preferred naming (used throughout UI/output):
        'behind_pig_pressures': behind_pig_pressures[:last_valid_i + 1],  # psig immediately behind pig
        # Backward-compatible alias (internal legacy name):
        'drive_pressures': behind_pig_pressures[:last_valid_i + 1],
        'injection_pressures': injection_pressures[:last_valid_i + 1],
        'gas_dp_losses': gas_dp_losses[:last_valid_i + 1],
        'friction_losses': friction_losses[:last_valid_i + 1],
        'head_losses': head_losses[:last_valid_i + 1],
        'exit_pressures': exit_pressures[:last_valid_i + 1],
        'injection_rates': injection_rates[:last_valid_i + 1],
        'cumulative_n2': cumulative_n2[:last_valid_i + 1],
        'pig_speeds': pig_speeds[:last_valid_i + 1],
        'exceed_points': [],
        'last_valid_i': last_valid_i
    }

def visualize_results(results, inputs):
    logging.info("Entering visualize_results")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))
    
    ax1.plot(results['purge_mileposts'] - inputs['purge_start_mp'], results['elevations'], 'b-')
    ax1.set_title('Elevation vs. Milepost')
    ax1.set_xlabel('Miles')
    ax1.set_ylabel('Elevation (ft)')
    
    ax2.plot(results['purge_mileposts'] - inputs['purge_start_mp'], results['pig_speeds'], 'r-')
    ax2.axhline(y=inputs['max_pig_speed'] * 1.25, linestyle='--', label='Max Speed × 1.25')
    ax2.axhline(y=inputs['target_pig_speed'], linestyle=':', label='Target Speed')
    ax2.set_title('Pig Speed vs. Milepost')
    ax2.set_xlabel('Miles')
    ax2.set_ylabel('Speed (mph)')
    ax2.legend()
    
    ax3.plot(results['purge_mileposts'] - inputs['purge_start_mp'], results['injection_pressures'], label='Injection Pressure (inlet)')
    behind = results.get('behind_pig_pressures', results.get('drive_pressures'))
    ax3.plot(results['purge_mileposts'] - inputs['purge_start_mp'], behind, label='Pressure Behind Pig')
    ax3.plot(results['purge_mileposts'] - inputs['purge_start_mp'], results['exit_pressures'], label='Exit Pressure (target/taper)')
    ax3.plot(results['purge_mileposts'] - inputs['purge_start_mp'], results['gas_dp_losses'], label='Gas DP inlet→pig')
    ax3.set_title('Pressure vs. Milepost')
    ax3.set_xlabel('Miles')
    ax3.set_ylabel('Pressure (psi)')
    ax3.legend()
    
    ax4.plot(results['purge_mileposts'] - inputs['purge_start_mp'], results['cumulative_n2'], 'c-')
    ax4.set_title('Cumulative Nitrogen vs. Milepost')
    ax4.set_xlabel('Miles')
    ax4.set_ylabel('N2 Volume (SCF)')
    
    plt.tight_layout()
    plt.show()
    logging.info("Exiting visualize_results")

def print_condensed_table(results, inputs):
    logging.info("Printing condensed table")
    # pipe area for throughput
    nps_key = resolve_nps_key(inputs['nps'])
    pipe_od = nps_data[nps_key]["OD_in"]
    pipe_id = pipe_od - 2 * float(inputs['pipe_wt'])
    pipe_diameter_ft = pipe_id / 12.0
    area = math.pi * (pipe_diameter_ft / 2) ** 2
    print("\nCondensed Table (50 evenly spaced points):")
    print(
        "Miles\tElevation (ft)\tElapsed Time (hr)\tInjection P (psi)\tGas DP (psi)\tBehind Pig P (psi)\t"
        "Liquid Fric (psi)\tHead (psi)\tExit P (psi)\tInjection Rate (SCFM)\tCumulative N2 (SCF)\t"
        "Pig Speed (mph)\tMiles to Outlet\tBarrels/hr"
    )
    
    n_points = len(results['purge_mileposts'])
    indices = np.linspace(0, results['last_valid_i'], min(50, n_points), dtype=int)
    
    for i in indices:
        miles = results['purge_mileposts'][i] - inputs['purge_start_mp']
        elevation = results['elevations'][i]
        time_hr = results['elapsed_times'][i]
        behind = results.get('behind_pig_pressures', results.get('drive_pressures'))
        inj_p = results.get('injection_pressures', np.zeros_like(behind))[i]
        gas_dp = results.get('gas_dp_losses', np.zeros_like(behind))[i]
        behind_p = behind[i]
        friction = results['friction_losses'][i]
        head = results['head_losses'][i]
        exit_p = results['exit_pressures'][i]
        inj_rate = results['injection_rates'][i]
        n2_vol = results['cumulative_n2'][i]
        speed = results['pig_speeds'][i]
        miles_to_outlet = inputs['system_end_mp'] - results['purge_mileposts'][i]
        bbl_hr = speed * 5280.0 * area / 5.614583
        print(
            f"{miles:.2f}\t{elevation:.2f}\t{time_hr:.2f}\t\t"
            f"{inj_p:.0f}\t\t{gas_dp:.1f}\t\t{behind_p:.0f}\t\t"
            f"{friction:.2f}\t\t{head:.0f}\t\t{exit_p:.0f}\t\t"
            f"{int(inj_rate):,}\t\t{int(n2_vol):,}\t\t{speed:.2f}\t\t{miles_to_outlet:.2f}\t\t{bbl_hr:.0f}"
        )

def export_results(dialog_root, inputs, results):
    logging.info("Entering export_results")
    # Resolve NPS key safely (fixes '16.0' KeyError)
    try:
        nps_key = resolve_nps_key(inputs['nps'])
    except Exception:
        # fall back to string
        nps_key = str(inputs['nps'])

    inputs_data = {
        "Parameter": [
            "Nominal Pipe Size (NPS)", "Outside Diameter (in)", "Wall Thickness (in)", "Pipe Material", "Fluid Type",
            "API Gravity (if Crude)", "Viscosity (cSt) (if Crude)", "Max Nitrogen Pressure (psi)", "Max N2 Rate (SCFM) [optional]", "Min Exit Pressure (Run) (psi)", "Exit Pressure (End) (psi)",
            "Max Pig Speed (mph)", "Min Pig Speed (mph)", "Target Pig Speed (mph)", "Purge Start Milepost", "Purge End Milepost",
            "System Endpoint Milepost", "Throttle Down Point (miles from end)", "Estimated N2 Pressure at End (psi)",
            "Has Intermediate Pump Station", "IPS Milepost", "IPS Shutdown Distance (miles)", "Minimum Pump Suction Pressure (psi)",
            "Resolution (points)"
        ],
        "Value": [
            inputs['nps'], nps_data[nps_key]["OD_in"], inputs['pipe_wt'],
            roughness_data[int(inputs['roughness_num'])]['material'], fluid_data[int(inputs['fluid_num'])]['name'],
            inputs.get('api_gravity', "N/A"), inputs.get('viscosity_cst', "N/A"),
            inputs.get('max_nitrogen_pressure', inputs.get('max_drive_pressure', None)),
            inputs.get('max_n2_rate_scfm', None),
            inputs['exit_pressure_run'], inputs['exit_pressure_end'], inputs['max_pig_speed'], inputs['min_pig_speed'], inputs['target_pig_speed'],
            inputs['purge_start_mp'], inputs['purge_end_mp'], inputs['system_end_mp'], inputs['throttle_down_miles'],
            inputs['n2_end_pressure'], inputs['has_ips'], inputs['ips_mp'] or "N/A",
            inputs['ips_shutdown_dist'] or "N/A", inputs['min_pump_suction_pressure'] or "N/A", inputs['resolution']
        ]
    }
    inputs_df = pd.DataFrame(inputs_data)
    
    # area for bbl/hr and miles to outlet
    nps_key2 = resolve_nps_key(inputs['nps'])
    pipe_od2 = nps_data[nps_key2]["OD_in"]
    pipe_id2 = pipe_od2 - 2 * float(inputs['pipe_wt'])
    pipe_diameter_ft2 = pipe_id2 / 12.0
    area2 = math.pi * (pipe_diameter_ft2 / 2) ** 2
    miles_to_outlet = inputs['system_end_mp'] - results['purge_mileposts']
    bbl_hr = results['pig_speeds'] * 5280.0 * area2 / 5.614583

    full_results_df = pd.DataFrame({
        "Miles": results['purge_mileposts'] - inputs['purge_start_mp'],
        "Elevation (ft)": results['elevations'],
        "Elapsed Time (hours)": results['elapsed_times'],
        "Injection Pressure (psi)": results.get('injection_pressures', results.get('behind_pig_pressures', results['drive_pressures'])),
        "Gas DP Inlet→Pig (psi)": results.get('gas_dp_losses', np.zeros_like(results.get('behind_pig_pressures', results['drive_pressures']))),
        "Pressure Behind Pig (psi)": results.get('behind_pig_pressures', results['drive_pressures']),
        "Friction Loss (psi)": results['friction_losses'],
        "Head Pressure (psi)": results['head_losses'],
        "Exit Pressure (psi)": results['exit_pressures'],
        "Injection Rate (SCFM)": results['injection_rates'],
        "Cumulative N2 (SCF)": results['cumulative_n2'],
        "Pig Speed (mph)": results['pig_speeds'],
        "Miles to Outlet": miles_to_outlet,
        "Barrels per hour": bbl_hr
    })
    
    n_points = len(results['purge_mileposts'])
    condensed_indices = np.linspace(0, results['last_valid_i'], min(50, n_points), dtype=int)
    condensed_results_df = full_results_df.iloc[condensed_indices]
    
    try:
        dialog_root.update()
        start_time = time.time()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            parent=dialog_root
        )
        dialog_root.update()
        if time.time() - start_time > 45:
            logging.warning("Export file dialog timeout after 45 seconds")
            raise ValueError("Export file dialog timed out. Try running from Command Prompt: cd C:\\Users\\YourName\\Documents && python Purge_Modeling_Program_22_09_2025.py")
        logging.info(f"Selected output file: {file_path}")
        if file_path:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                inputs_df.to_excel(writer, sheet_name="Simulation Inputs", index=False)
                full_results_df.to_excel(writer, sheet_name="Full Results", index=False)
                condensed_results_df.to_excel(writer, sheet_name="Condensed Results", index=False)
            messagebox.showinfo("Success", f"Results exported to {file_path}", parent=dialog_root)
        else:
            logging.warning("No output file selected")
            messagebox.showwarning("File Selection", "No output file selected. Results not saved.", parent=dialog_root)
    except Exception as e:
        logging.error(f"Failed to export results: {str(e)}")
        raise ValueError(f"Error exporting results: {str(e)}")
    logging.info("Exiting export_results")

def process_elevation_profile(dialog_root, initial_inputs):
    logging.info("Processing elevation profile")
    elevation_units = initial_inputs['elevation_units']
    
    try:
        dialog_root.update()
        start_time = time.time()
        files = filedialog.askopenfilenames(
            title="Select Elevation Profile Files (Multiple OK)",
            filetypes=[("All Supported", "*.kmz *.kml *.xlsx *.xls *.txt"), ("KMZ/KML", "*.kmz *.kml"), ("Excel", "*.xlsx *.xls"), ("TXT", "*.txt")],
            parent=dialog_root
        )
        dialog_root.update()
        if time.time() - start_time > 45:
            logging.warning("File dialog timeout after 45 seconds")
            raise ValueError("File dialog timed out. Try copying files to C:\\Users\\YourName\\Documents and running from Command Prompt.")
        logging.info(f"Selected files: {files}")
    except Exception as e:
        logging.error(f"Failed to open file dialog: {str(e)}")
        messagebox.showerror("File Dialog Error", f"Failed to open file dialog: {str(e)}. Try copying files to C:\\Users\\YourName\\Documents.", parent=dialog_root)
        raise ValueError(f"Failed to open file dialog: {str(e)}")
    
    if not files:
        logging.warning("No elevation profile files selected")
        messagebox.showerror("File Selection Error", "No elevation profile files selected.", parent=dialog_root)
        raise ValueError("No elevation profile files selected.")
    
    mileposts = []
    elevations = []
    current_mp = 0.0
    
    for file_path in files:
        logging.debug(f"Processing file: {file_path}")
        if file_path.lower().endswith(('.kmz', '.kml')):
            try:
                if file_path.lower().endswith('.kmz'):
                    with zipfile.ZipFile(file_path, 'r') as z:
                        kml_files = [f for f in z.namelist() if f.endswith('.kml')]
                        if not kml_files:
                            raise ValueError(f"No KML found in KMZ: {file_path}")
                        with z.open(kml_files[0]) as kml_file:
                            tree = ET.parse(kml_file)
                else:
                    tree = ET.parse(file_path)
                
                root = tree.getroot()
                ns = {'kml': 'http://www.opengis.net/kml/2.2', 'gx': 'http://www.google.com/kml/ext/2.2'}
                coords = []
                
                coord_elements = root.xpath('.//kml:coordinates|.//gx:coordinates', namespaces=ns)
                if not coord_elements:
                    coord_elements = root.xpath('.//*[local-name()="coordinates"]/text()', namespaces=ns)
                
                for coord_str in coord_elements:
                    if isinstance(coord_str, str):
                        coord_text = coord_str.strip()
                    else:
                        coord_text = coord_str.text.strip() if coord_str.text else ''
                    if coord_text:
                        for c in coord_text.split():
                            try:
                                parts = c.split(',')
                                if len(parts) >= 2:
                                    lon, lat = map(float, parts[:2])
                                    elev = float(parts[2]) if len(parts) > 2 else 0.0
                                    coords.append((lat, lon, elev))
                                else:
                                    logging.warning(f"Skipping invalid coordinate in {file_path}: {c}")
                            except (ValueError, IndexError):
                                logging.warning(f"Invalid coordinate format in {file_path}: {c}")
                
                if not coords:
                    raise ValueError(f"No valid coordinates found in {file_path}")
                
                file_mileposts = [current_mp]
                file_elevations = [coords[0][2]]
                for j in range(1, len(coords)):
                    try:
                        dist_miles = geodesic(coords[j-1][:2], coords[j][:2]).miles
                    except Exception as e:
                        logging.warning(f"Geodesic failed at point {j}: {str(e)}, using Haversine")
                        dist_miles = haversine_distance(coords[j-1][0], coords[j-1][1], coords[j][0], coords[j][1])
                    current_mp += dist_miles
                    file_mileposts.append(current_mp)
                    file_elevations.append(coords[j][2])
                
                mileposts.extend(file_mileposts)
                elevations.extend(file_elevations)
            except Exception as e:
                logging.error(f"Failed to parse KMZ/KML {file_path}: {str(e)}")
                raise ValueError(f"Failed to parse KMZ/KML {file_path}: {str(e)}")
        
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file_path)
                if 'Milepost' not in df.columns or 'Elevation' not in df.columns:
                    raise ValueError(f"Excel {file_path} must have 'Milepost' and 'Elevation' columns.")
                file_mileposts = df['Milepost'].values + current_mp
                file_elevations = df['Elevation'].values
                current_mp = file_mileposts[-1]
                mileposts.extend(file_mileposts)
                elevations.extend(file_elevations)
            except Exception as e:
                logging.error(f"Failed to parse Excel {file_path}: {str(e)}")
                raise ValueError(f"Failed to parse Excel {file_path}: {str(e)}")
        
        elif file_path.lower().endswith('.txt'):
            try:
                df = pd.read_csv(file_path, sep=None, engine='python')
                if 'latitude' not in df.columns or 'longitude' not in df.columns or 'altitude (ft)' not in df.columns:
                    raise ValueError(f"TXT {file_path} must have 'latitude', 'longitude', and 'altitude (ft)' columns.")
                coords = list(zip(df['latitude'], df['longitude'], df['altitude (ft)']))
                file_mileposts = [current_mp]
                file_elevations = [coords[0][2]]
                for j in range(1, len(coords)):
                    try:
                        dist_miles = geodesic((coords[j-1][0], coords[j-1][1]), (coords[j][0], coords[j][1])).miles
                    except Exception as e:
                        logging.warning(f"Geodesic failed at point {j}: {str(e)}, using Haversine")
                        dist_miles = haversine_distance(coords[j-1][0], coords[j-1][1], coords[j][0], coords[j][1])
                    current_mp += dist_miles
                    file_mileposts.append(current_mp)
                    file_elevations.append(coords[j][2])
                mileposts.extend(file_mileposts)
                elevations.extend(file_elevations)
            except Exception as e:
                logging.error(f"Failed to parse TXT {file_path}: {str(e)}")
                raise ValueError(f"Failed to parse TXT {file_path}: {str(e)}")
        
        else:
            logging.warning(f"Unsupported file type: {file_path}")
            raise ValueError(f"Unsupported file type: {file_path}")
    
    if not mileposts:
        raise ValueError("No valid data extracted from files.")
    
    elevations = np.array(elevations)
    if elevation_units == "Meters":
        elevations *= 3.28084
    
    mileposts = np.array(mileposts)
    sorted_idx = np.argsort(mileposts)
    mileposts = mileposts[sorted_idx]
    elevations = elevations[sorted_idx]
    unique_mask = np.diff(mileposts, prepend=mileposts[0] - 1) > 0
    mileposts = mileposts[unique_mask]
    elevations = elevations[unique_mask]
    
    profile_start_mp = mileposts.min()
    profile_end_mp = mileposts.max()
    
    purge_mileposts = mileposts.copy()
    system_mileposts = mileposts.copy()
    system_elevations = elevations.copy()
    
    logging.info(f"Profile processed: Start MP={profile_start_mp:.1f}, End MP={profile_end_mp:.1f}, Points={len(mileposts)}")
    return purge_mileposts, elevations, system_mileposts, system_elevations, profile_start_mp, profile_end_mp

def main():
    logging.info("Starting Pipeline Pigging Simulation")
    print("Starting Pipeline Pigging Simulation")
    try:
        dialog_root = tk.Tk()
        dialog_root.withdraw()
        initial_inputs = {
            'elevation_format': 'TXT',
            'elevation_units': 'Feet',
            'purge_start_mp': 0
        }
        logging.info("Calling process_elevation_profile")
        purge_mileposts, elevations, system_mileposts, system_elevations, profile_start_mp, profile_end_mp = process_elevation_profile(dialog_root, initial_inputs)
        logging.info("Calling get_user_inputs")
        inputs = get_user_inputs(dialog_root, profile_start_mp, profile_end_mp)
        inputs['elevation_format'] = initial_inputs['elevation_format']
        inputs['elevation_units'] = initial_inputs['elevation_units']
        
        print_inputs(inputs)
        
        if len(system_mileposts) > inputs['resolution'] * 2:
            new_system_mp = np.linspace(profile_start_mp, profile_end_mp, inputs['resolution'] * 2)
            cs_system = CubicSpline(system_mileposts, system_elevations)
            system_elevations = cs_system(new_system_mp)
            system_mileposts = new_system_mp
        
        mask = (purge_mileposts >= inputs['purge_start_mp']) & (purge_mileposts <= inputs['purge_end_mp'])
        purge_mileposts = purge_mileposts[mask]
        elevations = elevations[mask]
        
        if len(purge_mileposts) > inputs['resolution']:
            new_purge_mp = np.linspace(inputs['purge_start_mp'], inputs['purge_end_mp'], inputs['resolution'])
            cs_purge = CubicSpline(system_mileposts, system_elevations)
            elevations = cs_purge(new_purge_mp)
            purge_mileposts = new_purge_mp
        else:
            cs = CubicSpline(system_mileposts, system_elevations)
            if purge_mileposts[0] > inputs['purge_start_mp']:
                purge_mileposts = np.insert(purge_mileposts, 0, inputs['purge_start_mp'])
                elevations = np.insert(elevations, 0, cs(inputs['purge_start_mp']))
            if purge_mileposts[-1] < inputs['purge_end_mp']:
                purge_mileposts = np.append(purge_mileposts, inputs['purge_end_mp'])
                elevations = np.append(elevations, cs(inputs['purge_end_mp']))
        
        while True:
            logging.info("Running simulation")
            # Optional N2 cutoff default (None = no cutoff)
            inputs.setdefault('n2_optional_cutoff_scf', None)

            results = run_simulation(dialog_root, inputs, purge_mileposts, elevations, system_mileposts, system_elevations)
            visualize_results(results, inputs)
            print_condensed_table(results, inputs)
            
            if messagebox.askyesno("Output Satisfactory", "Is the output satisfactory?", parent=dialog_root):
                export_results(dialog_root, inputs, results)
                break
            else:
                try:
                    dialog_root.update()
                    new_n2_rate = tk.simpledialog.askfloat(
                        "Input", f"New Max N2 Rate (SCFM) [optional] (Last={inputs.get('max_n2_rate_scfm', None)}):",
                        parent=dialog_root, minvalue=0.1
                    )
                    if new_n2_rate is not None:
                        inputs['max_n2_rate_scfm'] = new_n2_rate
                    new_exit_run = tk.simpledialog.askfloat(
                        "Input", f"New Min Exit Pressure (Run) (Last={inputs['exit_pressure_run']:.1f}):",
                        parent=dialog_root, minvalue=0
                    )
                    if new_exit_run is not None:
                        inputs['exit_pressure_run'] = new_exit_run
                    new_exit_end = tk.simpledialog.askfloat(
                        "Input", f"New Exit Pressure (End) (Last={inputs['exit_pressure_end']:.1f}):",
                        parent=dialog_root, minvalue=0
                    )
                    if new_exit_end is not None:
                        inputs['exit_pressure_end'] = new_exit_end
                    new_target_speed = tk.simpledialog.askfloat(
                        "Input", f"New Target Pig Speed (mph) (Last={inputs['target_pig_speed']:.1f}):",
                        parent=dialog_root, minvalue=0.1, maxvalue=inputs['max_pig_speed']
                    )
                    if new_target_speed is not None:
                        inputs['target_pig_speed'] = new_target_speed
                    dialog_root.update()
                    print_inputs(inputs)
                    continue
                except ValueError as e:
                    logging.error(f"Input revision error: {str(e)}")
                    messagebox.showerror("Input Error", f"Invalid input: {str(e)}", parent=dialog_root)
    
    except Exception as e:
        error_msg = f"Simulation failed: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        try:
            messagebox.showerror("Error", error_msg, parent=dialog_root)
        except Exception as tk_e:
            logging.error(f"Failed to show error messagebox: {str(tk_e)}")
            print(f"Error: Failed to show error messagebox: {str(tk_e)}")
        raise
    finally:
        dialog_root.destroy()

class TestPurgeSimulation(unittest.TestCase):
    def test_friction_loss(self):
        v, L, D, rho, mu, eps = 10, 1000, 0.5, 62.4, 1e-5, 0.00015
        loss = calculate_friction_loss(v, L, D, rho, mu, eps)
        self.assertGreater(loss, 0, "Friction loss should be positive.")
    
    def test_trendline_slope(self):
        mileposts = np.array([0, 1, 2])
        elevations = np.array([100, 110, 120])
        slope = calculate_trendline_slope(mileposts, elevations, 0, 2)
        self.assertEqual(slope, 10, "Slope should be 10 ft/mile.")

if __name__ == "__main__":
    main()