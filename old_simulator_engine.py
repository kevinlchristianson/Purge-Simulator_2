# Pipeline Pigging Simulation Program with Smart Enhancements
import numpy as np
import pandas as pd
import zipfile
from xml.etree import ElementTree as ET
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
    "64": {"OD_in": 64.000}, "68": {"OD_in": 68.000}, "72": {"OD_in": 72.000}, "80": {"OD_in": 80.000}
}

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
1. Inputs:
   - Nominal Pipe Size (NPS): Select from predefined sizes. Default: Blank.
   - Wall Thickness: Enter in inches. Default: Blank.
   - Pipe Material: Choose from New Welded Steel, Rusted Steel, or HDPE. Default: Rusted Steel.
   - Fluid Type: Select Diesel, Gasoline, Crude Oil, Water, or NGL. Default: Diesel.
   - API Gravity: Required for Crude Oil. Default: 40.
   - Max Drive Pressure: Default: 400 psi (recommended for downhill profiles).
   - Min Exit Pressure (Run): Default: 100 psi.
   - Exit Pressure (End): Default: 100 psi.
   - Est. N2 Pressure at End: Default: 300 psi.
   - Max Pig Speed: Default: 4 mph. Ensure > Min Pig Speed / 1.25.
   - Min Pig Speed: Default: 1 mph.
   - Hard Cap: Enforce max speed. Default: Checked (recommended).
   - Taper Down: Enable pressure tapering. Default: Unchecked.
   - Purge Start Milepost: Default: Minimum profile milepost (e.g., 2.8).
   - Purge End Milepost: Default: Profile end (e.g., 52.5 miles).
   - System Endpoint Milepost: Default: Profile end.
   - Throttle Down: Default: 9.6 miles.
   - Intermediate Pump Station (IPS): Check to enable. Specify milepost, shutdown distance (default: 1 mile), and minimum pump suction pressure (default: 50 psi).
   - Elevation Profile: Select multiple KMZ/KML, Excel, or TXT files to append. Confirm order to ensure sequential mileposts (e.g., Goldsmith to Huntsman, then Huntsman to Airport).
   - Elevation Units: Default: Feet.
   - Resolution: For large profiles (e.g., 1418 points), use 100–1000 points (default: 500).

2. Simulation:
   - Applies max drive pressure (400 psi) until throttle-down or for steep uphill (>50 ft/mile).
   - Tapers to 70% near end.
   - Stops if pig speed is too low (<1 mph) or too high (>5 mph), with options to adjust.

3. Outputs:
   - 50-point condensed table in console.
   - Plots for elevation, speed, pressure, and nitrogen volume.
   - Excel with inputs, full results, and condensed results.

4. Troubleshooting:
   - Run in Command Prompt (not IDLE) to avoid Tkinter hangs: `cd C:\\Users\\kchristianson\\Documents && python "R:\\PPS\\PPS PROJECTS\\PROJECT FOLDERS\\2025 -Tenders and Customer PO's\\P66\\MX-30 Purge\\3.0 Engineering\\Calculations\\Purge_Modeling_Program_29_07_2025.py"`.
   - Copy files to a local drive (e.g., C:\\Users\\kchristianson\\Documents) if R:\\ is slow or causes hangs.
   - Check purge_simulation.log for errors.
   - Use resolution 100–1000 to avoid memory issues.
   - If stalled, note last log entry and any error pop-ups.
   - For speed excursions, use Max Pig Speed=4 mph, Max Drive Pressure=400 psi, Hard Cap enabled.
   - For appending, confirm file order to ensure sequential mileposts (e.g., Goldsmith to Huntsman before Huntsman to Airport).
   - If milepost range errors occur, ensure Purge Start Milepost matches the profile’s minimum (e.g., 2.8).
    """
    try:
        help_window = tk.Toplevel()
        help_window.title("User Help")
        help_window.update()
        text = tk.Text(help_window, height=30, width=80)
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
            logging.debug("Validating user inputs")
            nps = nps_var.get()
            if not nps:
                raise ValueError("Nominal Pipe Size must be selected.")
            pipe_wt = wt_entry.get().strip()
            if not pipe_wt:
                raise ValueError("Wall thickness must be provided.")
            pipe_wt = float(pipe_wt)
            pipe_od = nps_data[nps]["OD_in"]
            if pipe_wt <= 0 or pipe_wt >= pipe_od / 2:
                raise ValueError("Wall thickness must be positive and less than half the OD.")
            roughness_num = int(material_var.get().split(':')[0])
            if roughness_num not in roughness_data:
                raise ValueError("Invalid pipe material selected.")
            fluid_num = int(fluid_var.get().split(':')[0])
            if fluid_num not in fluid_data:
                raise ValueError("Invalid fluid selected.")
            api_gravity = float(api_entry.get()) if fluid_num == 3 else None
            if fluid_num == 3 and api_gravity <= 0:
                raise ValueError("API Gravity must be positive.")
            max_drive = float(max_drive_entry.get())
            if max_drive <= 0:
                raise ValueError("Max Drive Pressure must be positive.")
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
            
            inputs.update({
                'nps': nps, 'pipe_wt': pipe_wt, 'roughness_num': roughness_num, 'fluid_num': fluid_num,
                'api_gravity': api_gravity, 'max_drive_pressure': max_drive, 'exit_pressure_run': exit_run,
                'exit_pressure_end': exit_end, 'n2_end_pressure': n2_end, 'max_pig_speed': max_speed,
                'min_pig_speed': min_speed, 'purge_start_mp': purge_start, 'purge_end_mp': purge_end,
                'system_end_mp': system_end, 'throttle_down_miles': throttle_down, 'hard_cap': hard_cap_var.get(),
                'taper_down_enabled': taper_down_var.get(), 'elevation_format': elev_format_var.get(),
                'elevation_units': elev_units_var.get(), 'has_ips': has_ips, 'ips_mp': ips_mp,
                'ips_shutdown_dist': ips_shutdown_dist, 'min_pump_suction_pressure': min_pump_suction_pressure
            })
            logging.info(f"Validated inputs: NPS={nps}, Max Drive={max_drive} psi, Max Speed={max_speed} mph, Min Speed={min_speed} mph, Purge Start={purge_start}, Purge End={purge_end}")
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
    except Exception as e:
        logging.error(f"Failed to initialize Tkinter input GUI: {str(e)}")
        print(f"Error: Failed to initialize Tkinter input GUI: {str(e)}")
        raise
    
    row = 0
    tk.Label(root, text="Nominal Pipe Size (NPS):").grid(row=row, column=0, sticky='e')
    nps_var = tk.StringVar(value="")
    ttk.Combobox(root, textvariable=nps_var, values=[""] + list(nps_data.keys())).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Wall Thickness (in):").grid(row=row, column=0, sticky='e')
    wt_entry = tk.Entry(root)
    wt_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Pipe Material:").grid(row=row, column=0, sticky='e')
    material_var = tk.StringVar(value="2: Rusted/Corroded Welded Steel")
    material_options = [f"{k}: {v['material']}" for k, v in roughness_data.items()]
    ttk.Combobox(root, textvariable=material_var, values=material_options).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Fluid Type:").grid(row=row, column=0, sticky='e')
    fluid_var = tk.StringVar(value="1: Diesel")
    fluid_options = [f"{k}: {v['name']}" for k, v in fluid_data.items()]
    ttk.Combobox(root, textvariable=fluid_var, values=fluid_options).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="API Gravity (if Crude Oil):").grid(row=row, column=0, sticky='e')
    api_entry = tk.Entry(root)
    api_entry.insert(0, "40")
    api_entry.grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Max Drive Pressure (psi):").grid(row=row, column=0, sticky='e')
    max_drive_entry = tk.Entry(root)
    max_drive_entry.insert(0, "400")
    max_drive_entry.grid(row=row, column=1)
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
    elev_format_var = tk.StringVar(value="Excel")
    ttk.Combobox(root, textvariable=elev_format_var, values=["KMZ/KML", "Excel", "TXT"]).grid(row=row, column=1)
    row += 1
    
    tk.Label(root, text="Elevation Units:").grid(row=row, column=0, sticky='e')
    elev_units_var = tk.StringVar(value="Feet")
    ttk.Combobox(root, textvariable=elev_units_var, values=["Feet", "Meters"]).grid(row=row, column=1)
    row += 1
    
    tk.Button(root, text="Submit", command=validate_and_submit).grid(row=row, column=0, columnspan=2)
    tk.Button(root, text="Help", command=show_help).grid(row=row, column=1, sticky='e')
    
    logging.info("Starting Tkinter wait for input GUI")
    root.grab_set()
    start_time = time.time()
    dialog_root.update()
    dialog_root.wait_window(root)
    dialog_root.update()
    if time.time() - start_time > 120:
        logging.warning("Input GUI timeout after 120 seconds")
        raise ValueError("Input GUI timed out. Please try again.")
    logging.info("Exiting get_user_inputs")
    return inputs

def calculate_friction_loss(v, L, D, fluid_density, viscosity, roughness):
    try:
        Re = fluid_density * v * D / viscosity
        haaland = -1.8 * np.log10((roughness / D / 3.7)**1.11 + 6.9 / Re)
        f = (1 / haaland)**2
        return f * L / D * fluid_density * v**2 / (2 * 32.174) / 144
    except Exception as e:
        logging.error(f"Friction loss calculation failed: {str(e)}")
        raise ValueError(f"Error in friction loss calculation: {str(e)}")

def smart_purge_strategy(inputs, position, slope, cumulative_n2, target_volume):
    logging.debug(f"Slope at MP {position:.2f}: {slope:.2f} ft/mile")
    if cumulative_n2 >= target_volume:
        logging.info(f"Coasting at MP {position:.2f}: Nitrogen cutoff reached")
        return 0
    if position >= inputs['purge_end_mp'] - inputs['throttle_down_miles']:
        taper_factor = (inputs['purge_end_mp'] - position) / inputs['throttle_down_miles']
        pressure = inputs['max_drive_pressure'] * max(0.7, taper_factor)
        logging.info(f"Throttling down at MP {position:.2f}: Pressure {pressure:.0f} psi")
        return pressure
    if slope > 50:
        logging.info(f"Steep uphill at MP {position:.2f}: Full pressure {inputs['max_drive_pressure']:.0f} psi")
        return inputs['max_drive_pressure']
    logging.info(f"Max pressure at MP {position:.2f}: {inputs['max_drive_pressure']:.0f} psi")
    return inputs['max_drive_pressure']

def run_simulation(inputs, purge_mileposts, elevations, system_mileposts, system_elevations):
    logging.info("Entering run_simulation")
    start_time = time.time()
    pipe_od = nps_data[inputs['nps']]["OD_in"]
    pipe_id = pipe_od - 2 * inputs['pipe_wt']
    pipe_diameter = pipe_id / 12
    area = np.pi * (pipe_diameter / 2) ** 2
    atmospheric_pressure = 14.7
    roughness = roughness_data[inputs['roughness_num']]['roughness_ft']
    
    if inputs['fluid_num'] == 3:
        specific_gravity = 141.5 / (131.5 + inputs['api_gravity'])
        viscosity_cst = 10 ** (10 - 0.25 * inputs['api_gravity'])
    else:
        specific_gravity = fluid_data[inputs['fluid_num']]['sg']
        viscosity_cst = fluid_data[inputs['fluid_num']]['viscosity_cst']
    fluid_density = specific_gravity * 62.4
    viscosity = fluid_density * viscosity_cst * 1.076e-5
    
    purge_length = (inputs['purge_end_mp'] - inputs['purge_start_mp']) * 5280
    total_volume_scf = area * purge_length * (inputs['n2_end_pressure'] + atmospheric_pressure) / atmospheric_pressure
    cutoff_volume = inputs.get('cutoff_volume', total_volume_scf)
    
    n_points = len(purge_mileposts)
    distances = purge_mileposts * 5280
    switch_milepost = inputs['purge_end_mp'] - inputs['throttle_down_miles']
    
    cs = CubicSpline(system_mileposts, system_elevations)
    h_exit = cs(inputs['system_end_mp'])
    head_losses = fluid_density * (h_exit - elevations) / 144
    
    elapsed_times = np.zeros(n_points, dtype=np.float64)
    drive_pressures = np.zeros(n_points, dtype=np.float64)
    friction_losses = np.zeros(n_points, dtype=np.float64)
    exit_pressures = np.zeros(n_points, dtype=np.float64)
    injection_rates = np.zeros(n_points, dtype=np.float64)
    cumulative_n2 = np.zeros(n_points, dtype=np.float64)
    pig_speeds = np.zeros(n_points, dtype=np.float64)
    
    drive_pressures[0] = inputs['max_drive_pressure']
    exit_pressures[0] = inputs['min_pump_suction_pressure'] if inputs['has_ips'] else inputs['exit_pressure_run']
    elapsed_times[0] = 0.0
    cumulative_n2[0] = 0.0
    v_initial = min(inputs['max_pig_speed'] * 1.46667, inputs['max_pig_speed'] * 1.25 * 1.46667)
    logging.info(f"Initial velocity: {v_initial / 1.46667:.2f} mph")
    injection_active = True
    x_cutoff = 0
    last_valid_i = n_points - 1
    exceed_points = []
    
    ips_active = inputs['has_ips']
    current_system_end = inputs['ips_mp'] if inputs['has_ips'] else inputs['system_end_mp']
    current_exit_pressure = inputs['min_pump_suction_pressure'] if inputs['has_ips'] else inputs['exit_pressure_run']
    
    slope = calculate_trendline_slope(np.array(system_mileposts), np.array(system_elevations), 
                                     inputs['purge_start_mp'], inputs['system_end_mp'])
    
    def target_exit_pressure(mp):
        return inputs['exit_pressure_end'] if mp >= switch_milepost else current_exit_pressure
    
    for i in range(n_points - 1):
        if time.time() - start_time > 120:
            if not messagebox.askyesno("Time Warning", "Simulation time exceeded 120 seconds. Continue?", parent=dialog_root):
                last_valid_i = i
                logging.info(f"Simulation stopped at MP {purge_mileposts[i]:.2f} due to time limit")
                messagebox.showinfo("Simulation Stopped", "Simulation stopped due to time limit exceeding 120 seconds.", parent=dialog_root)
                break
        
        if ips_active and purge_mileposts[i] >= inputs['ips_mp'] - inputs['ips_shutdown_dist']:
            ips_active = False
            current_system_end = inputs['system_end_mp']
            current_exit_pressure = inputs['exit_pressure_run']
            logging.info(f"IPS shutdown at MP {purge_mileposts[i]:.2f}, switching system end to {current_system_end:.2f}")
        
        L_remaining = current_system_end * 5280 - distances[i]
        v_max = inputs['max_pig_speed'] * 1.46667
        target_exit = target_exit_pressure(purge_mileposts[i])
        
        if injection_active:
            P_drive = inputs['max_drive_pressure'] if ips_active else smart_purge_strategy(inputs, purge_mileposts[i], slope, cumulative_n2[i], cutoff_volume)
            if P_drive == 0:
                injection_active = False
                x_cutoff = purge_mileposts[i]
                P_drive = inputs['max_drive_pressure'] * (x_cutoff - inputs['purge_start_mp']) / max(purge_mileposts[i] - inputs['purge_start_mp'], 1e-6)
                P_drive = max(P_drive, inputs['exit_pressure_end'])
            else:
                friction_loss_max = calculate_friction_loss(v_max, L_remaining, pipe_diameter, fluid_density, viscosity, roughness)
                P_exit_max_drive = P_drive - friction_loss_max - head_losses[i]
                
                if inputs['taper_down_enabled'] and P_exit_max_drive >= target_exit:
                    P_drive_required = target_exit + friction_loss_max + head_losses[i]
                    P_drive = min(P_drive_required, P_drive)
                    v = v_max
                    P_exit = P_drive - friction_loss_max - head_losses[i]
                    friction_losses[i] = friction_loss_max
                elif inputs['hard_cap']:
                    if P_exit_max_drive >= target_exit:
                        v = v_max
                        P_exit = P_exit_max_drive
                        friction_losses[i] = friction_loss_max
                    else:
                        P_exit = target_exit
                        v = v_initial
                        for _ in range(30):
                            friction_loss = calculate_friction_loss(v, L_remaining, pipe_diameter, fluid_density, viscosity, roughness)
                            P_exit_calc = P_drive - friction_loss - head_losses[i]
                            if abs(P_exit_calc - P_exit) < 0.01:
                                break
                            if P_exit_calc < P_exit:
                                v *= 0.95
                            elif v < v_max:
                                v *= 1.05
                        friction_losses[i] = friction_loss
                        if v > v_max:
                            v = v_max
                            P_exit = P_drive - friction_loss_max - head_losses[i]
                else:
                    P_exit = target_exit
                    v = v_initial
                    for _ in range(30):
                        friction_loss = calculate_friction_loss(v, L_remaining, pipe_diameter, fluid_density, viscosity, roughness)
                        P_exit_calc = P_drive - friction_loss - head_losses[i]
                        if abs(P_exit_calc - P_exit) < 0.01:
                            break
                        if P_exit_calc < P_exit:
                            v *= 0.95
                        else:
                            v *= 1.05
                    friction_losses[i] = friction_loss
        
        else:
            P_exit = target_exit
            v = v_initial
            for _ in range(30):
                friction_loss = calculate_friction_loss(v, L_remaining, pipe_diameter, fluid_density, viscosity, roughness)
                P_exit_calc = P_drive - friction_loss - head_losses[i]
                if abs(P_exit_calc - P_exit) < 0.01:
                    break
                if P_exit_calc < P_exit:
                    v *= 0.95
                elif v < v_max:
                    v *= 1.05
            friction_losses[i] = friction_loss
        
        pig_speeds[i] = v / 1.46667
        logging.debug(f"MP {purge_mileposts[i]:.2f}: Speed {pig_speeds[i]:.2f} mph, P_drive {P_drive:.0f} psi")
        if pig_speeds[i] < inputs['min_pig_speed']:
            logging.info(f"Simulation stopped: Pig speed below {inputs['min_pig_speed']} mph at MP {purge_mileposts[i]:.2f}")
            print(f"Simulation stopped: Pig speed below {inputs['min_pig_speed']} mph at MP {purge_mileposts[i]:.2f}")
            last_valid_i = i
            messagebox.showinfo("Simulation Stopped", f"Simulation stopped at MP {purge_mileposts[i]:.2f} due to pig speed below {inputs['min_pig_speed']} mph.", parent=dialog_root)
            break
        if pig_speeds[i] > inputs['max_pig_speed'] * 1.25:
            logging.warning(f"Pig speed excursion at MP {purge_mileposts[i]:.2f}: {pig_speeds[i]:.2f} mph exceeds limit {inputs['max_pig_speed'] * 1.25:.2f} mph")
            if not messagebox.askyesno(
                "Speed Warning",
                f"Pig speed exceeded {pig_speeds[i]:.2f} mph (max × 1.25 = {inputs['max_pig_speed'] * 1.25:.2f} mph) at MP {purge_mileposts[i]:.2f}. Proceed?",
                parent=dialog_root
            ):
                pig_speeds[i] = inputs['max_pig_speed']
                v = pig_speeds[i] * 1.46667
                friction_losses[i] = calculate_friction_loss(v, L_remaining, pipe_diameter, fluid_density, viscosity, roughness)
                P_drive = target_exit + friction_losses[i] + head_losses[i]
                last_valid_i = i
                logging.info(f"Simulation stopped at MP {purge_mileposts[i]:.2f} due to speed excursion")
                messagebox.showinfo(
                    "Simulation Stopped",
                    f"Simulation stopped at MP {purge_mileposts[i]:.2f} due to pig speed excursion. Try setting Max Pig Speed to 4 mph, lowering Max Drive Pressure to 400 psi, or enabling Hard Cap.",
                    parent=dialog_root
                )
                break
        
        drive_pressures[i] = P_drive
        exit_pressures[i + 1] = P_exit
        
        if inputs['hard_cap'] and purge_mileposts[i] < switch_milepost and P_exit > target_exit + 0.01:
            exceed_points.append((purge_mileposts[i], P_exit))
        
        segment_length = distances[i + 1] - distances[i] if i < n_points - 2 else distances[i] - distances[i-1]
        time_step = segment_length / v / 3600
        elapsed_times[i + 1] = elapsed_times[i] + time_step
        if injection_active:
            volume_per_sec = area * v
            injection_rate_scf_per_sec = volume_per_sec * (P_drive + atmospheric_pressure) / atmospheric_pressure
            injection_rate = injection_rate_scf_per_sec * 60
            injection_rates[i] = injection_rate
            n2_volume = injection_rate * time_step * 60
            cumulative_n2[i + 1] = cumulative_n2[i] + n2_volume
        else:
            injection_rates[i] = 0
            cumulative_n2[i + 1] = cumulative_n2[i]
        v_initial = v
    
    if last_valid_i == n_points - 1:
        i = n_points - 1
        L_remaining = current_system_end * 5280 - distances[i]
        head_losses[i] = fluid_density * (h_exit - elevations[i]) / 144
        target_exit = target_exit_pressure(purge_mileposts[i])
        P_drive = smart_purge_strategy(inputs, purge_mileposts[i], slope, cumulative_n2[i], cutoff_volume)
        if P_drive == 0:
            x = max(purge_mileposts[i] - inputs['purge_start_mp'], 1e-6)
            P_drive = inputs['max_drive_pressure'] * (x_cutoff - inputs['purge_start_mp']) / x
            P_drive = max(P_drive, inputs['exit_pressure_end'])
        P_exit = target_exit
        v = v_initial
        for _ in range(30):
            friction_loss = calculate_friction_loss(v, L_remaining, pipe_diameter, fluid_density, viscosity, roughness)
            P_exit_calc = P_drive - friction_loss - head_losses[i]
            if abs(P_exit_calc - P_exit) < 0.01:
                break
            if P_exit_calc < P_exit:
                v *= 0.95
            elif v < v_max:
                v *= 1.05
        friction_losses[i] = friction_loss
        pig_speeds[i] = v / 1.46667
        logging.debug(f"Final MP {purge_mileposts[i]:.2f}: Speed {pig_speeds[i]:.2f} mph, P_drive {P_drive:.0f} psi")
        if pig_speeds[i] > inputs['max_pig_speed'] * 1.25:
            logging.warning(f"Pig speed excursion at final MP {purge_mileposts[i]:.2f}: {pig_speeds[i]:.2f} mph exceeds limit {inputs['max_pig_speed'] * 1.25:.2f} mph")
            if not messagebox.askyesno(
                "Speed Warning",
                f"Pig speed exceeded {pig_speeds[i]:.2f} mph (max × 1.25 = {inputs['max_pig_speed'] * 1.25:.2f} mph) at MP {purge_mileposts[i]:.2f}. Proceed?",
                parent=dialog_root
            ):
                pig_speeds[i] = inputs['max_pig_speed']
                v = pig_speeds[i] * 1.46667
                friction_losses[i] = calculate_friction_loss(v, L_remaining, pipe_diameter, fluid_density, viscosity, roughness)
                P_drive = target_exit + friction_losses[i] + head_losses[i]
                last_valid_i = i
                logging.info(f"Simulation stopped at final MP {purge_mileposts[i]:.2f} due to speed excursion")
                messagebox.showinfo(
                    "Simulation Stopped",
                    f"Simulation stopped at MP {purge_mileposts[i]:.2f} due to pig speed excursion. Try setting Max Pig Speed to 4 mph, lowering Max Drive Pressure to 400 psi, or enabling Hard Cap.",
                    parent=dialog_root
                )
        drive_pressures[i] = P_drive
        exit_pressures[i] = P_exit
        if injection_active and cumulative_n2[i-1] < cutoff_volume:
            volume_per_sec = area * v
            injection_rate_scf_per_sec = volume_per_sec * (P_drive + atmospheric_pressure) / atmospheric_pressure
            injection_rate = injection_rate_scf_per_sec * 60
            injection_rates[i] = injection_rate
            segment_length = distances[i] - distances[i-1]
            time_step = segment_length / v / 3600
            n2_volume = injection_rate * time_step * 60
            cumulative_n2[i] = cumulative_n2[i-1] + n2_volume
        else:
            injection_rates[i] = 0
            cumulative_n2[i] = cumulative_n2[i-1]
    
    logging.info("Exiting run_simulation")
    return {
        'purge_mileposts': purge_mileposts[:last_valid_i + 1],
        'elevations': elevations[:last_valid_i + 1],
        'elapsed_times': elapsed_times[:last_valid_i + 1],
        'drive_pressures': drive_pressures[:last_valid_i + 1],
        'friction_losses': friction_losses[:last_valid_i + 1],
        'head_losses': head_losses[:last_valid_i + 1],
        'exit_pressures': exit_pressures[:last_valid_i + 1],
        'injection_rates': injection_rates[:last_valid_i + 1],
        'cumulative_n2': cumulative_n2[:last_valid_i + 1],
        'pig_speeds': pig_speeds[:last_valid_i + 1],
        'exceed_points': exceed_points,
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
    ax2.axhline(y=inputs['max_pig_speed'] * 1.25, color='k', linestyle='--', label='Max Speed × 1.25')
    ax2.set_title('Pig Speed vs. Milepost')
    ax2.set_xlabel('Miles')
    ax2.set_ylabel('Speed (mph)')
    ax2.legend()
    
    ax3.plot(results['purge_mileposts'] - inputs['purge_start_mp'], results['drive_pressures'], 'g-', label='Drive')
    ax3.plot(results['purge_mileposts'] - inputs['purge_start_mp'], results['exit_pressures'], 'm-', label='Exit')
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
    print("\nCondensed Table (50 evenly spaced points):")
    print("Miles\tElevation (ft)\tElapsed Time (hours)\tDrive Pressure (psi)\tFriction Loss (psi)\tHead Pressure (psi)\tExit Pressure (psi)\tInjection Rate (SCFM)\tCumulative N2 (SCF)\tPig Speed (mph)")
    
    n_points = len(results['purge_mileposts'])
    indices = np.linspace(0, results['last_valid_i'], 50, dtype=int)
    
    for i in indices:
        miles = results['purge_mileposts'][i] - inputs['purge_start_mp']
        elevation = results['elevations'][i]
        time = results['elapsed_times'][i]
        drive_p = results['drive_pressures'][i]
        friction = results['friction_losses'][i]
        head = results['head_losses'][i]
        exit_p = results['exit_pressures'][i]
        inj_rate = results['injection_rates'][i]
        n2_vol = results['cumulative_n2'][i]
        speed = results['pig_speeds'][i]
        print(f"{miles:.2f}\t{elevation:.2f}\t{time:.2f}\t\t{drive_p:.0f}\t\t{friction:.0f}\t\t{head:.0f}\t\t{exit_p:.0f}\t\t{int(inj_rate):,}\t\t{int(n2_vol):,}\t\t{speed:.2f}")

def export_results(dialog_root, inputs, results):
    logging.info("Entering export_results")
    inputs_data = {
        "Parameter": [
            "Nominal Pipe Size (NPS)", "Outside Diameter (in)", "Wall Thickness (in)", "Pipe Material", "Fluid Type",
            "Max Drive Pressure (psi)", "Min Exit Pressure (Run) (psi)", "Exit Pressure (End) (psi)",
            "Max Pig Speed (mph)", "Min Pig Speed (mph)", "Purge Start Milepost", "Purge End Milepost",
            "System Endpoint Milepost", "Throttle Down Point (miles from end)", "Estimated N2 Pressure at End (psi)",
            "Has Intermediate Pump Station", "IPS Milepost", "IPS Shutdown Distance (miles)", "Minimum Pump Suction Pressure (psi)"
        ],
        "Value": [
            inputs['nps'], nps_data[inputs['nps']]["OD_in"], inputs['pipe_wt'],
            roughness_data[inputs['roughness_num']]['material'], fluid_data[inputs['fluid_num']]['name'],
            inputs['max_drive_pressure'], inputs['exit_pressure_run'], inputs['exit_pressure_end'],
            inputs['max_pig_speed'], inputs['min_pig_speed'], inputs['purge_start_mp'], inputs['purge_end_mp'],
            inputs['system_end_mp'], inputs['throttle_down_miles'], inputs['n2_end_pressure'],
            inputs['has_ips'], inputs['ips_mp'] or "N/A", inputs['ips_shutdown_dist'] or "N/A",
            inputs['min_pump_suction_pressure'] or "N/A"
        ]
    }
    inputs_df = pd.DataFrame(inputs_data)
    
    full_results_df = pd.DataFrame({
        "Miles": results['purge_mileposts'] - inputs['purge_start_mp'],
        "Elevation (ft)": results['elevations'],
        "Elapsed Time (hours)": results['elapsed_times'],
        "Drive Pressure (psi)": results['drive_pressures'],
        "Friction Loss (psi)": results['friction_losses'],
        "Head Pressure (psi)": results['head_losses'],
        "Exit Pressure (psi)": results['exit_pressures'],
        "Injection Rate (SCFM)": results['injection_rates'],
        "Cumulative N2 (SCF)": results['cumulative_n2'],
        "Pig Speed (mph)": results['pig_speeds']
    })
    
    n_points = len(results['purge_mileposts'])
    condensed_indices = np.linspace(0, results['last_valid_i'], 50, dtype=int)
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
        if time.time() - start_time > 120:
            logging.warning("Export file dialog timeout after 120 seconds")
            raise ValueError("Export file dialog timed out. Please try again.")
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

def main():
    logging.info("Starting Pipeline Pigging Simulation")
    print("Starting Pipeline Pigging Simulation")
    try:
        dialog_root = tk.Tk()
        dialog_root.withdraw()
        initial_inputs = {
            'elevation_format': 'Excel',
            'elevation_units': 'Feet',
            'purge_start_mp': 0
        }
        logging.info("Calling process_elevation_profile")
        purge_mileposts, elevations, system_mileposts, system_elevations, profile_start_mp, profile_end_mp = process_elevation_profile(dialog_root, initial_inputs)
        logging.info("Calling get_user_inputs")
        inputs = get_user_inputs(dialog_root, profile_start_mp, profile_end_mp)
        inputs['elevation_format'] = initial_inputs['elevation_format']
        inputs['elevation_units'] = initial_inputs['elevation_units']
        
        while True:
            logging.info("Running simulation")
            results = run_simulation(inputs, purge_mileposts, elevations, system_mileposts, system_elevations)
            visualize_results(results, inputs)
            print_condensed_table(results, inputs)
            
            if results['exceed_points'] and inputs['hard_cap']:
                first_exceed_mp, first_exceed_p_exit = results['exceed_points'][0]
                messagebox.showwarning(
                    "Pressure Warning",
                    f"Exit pressure exceeded target at MP {first_exceed_mp:.2f} with P_exit = {first_exceed_p_exit:.0f} psi.",
                    parent=dialog_root
                )
                if messagebox.askyesno("Revise Pressures", "Revise pressures?", parent=dialog_root):
                    try:
                        dialog_root.update()
                        new_drive = tk.simpledialog.askfloat(
                            "Input", f"New Max Drive Pressure (Last={inputs['max_drive_pressure']:.1f}):",
                            parent=dialog_root, minvalue=0.1
                        )
                        inputs['max_drive_pressure'] = new_drive if new_drive else inputs['max_drive_pressure']
                        new_exit_run = tk.simpledialog.askfloat(
                            "Input", f"New Min Exit Pressure (Run) (Last={inputs['exit_pressure_run']:.1f}):",
                            parent=dialog_root, minvalue=0
                        )
                        inputs['exit_pressure_run'] = new_exit_run if new_exit_run else inputs['exit_pressure_run']
                        new_exit_end = tk.simpledialog.askfloat(
                            "Input", f"New Exit Pressure (End) (Last={inputs['exit_pressure_end']:.1f}):",
                            parent=dialog_root, minvalue=0
                        )
                        inputs['exit_pressure_end'] = new_exit_end if new_exit_end else inputs['exit_pressure_end']
                        dialog_root.update()
                        continue
                    except ValueError as e:
                        logging.error(f"Input revision error: {str(e)}")
                        messagebox.showerror("Input Error", f"Invalid input: {str(e)}", parent=dialog_root)
            
            if messagebox.askyesno("Output Satisfactory", "Is the output satisfactory?", parent=dialog_root):
                export_results(dialog_root, inputs, results)
                break
            else:
                try:
                    dialog_root.update()
                    new_drive = tk.simpledialog.askfloat(
                        "Input", f"New Max Drive Pressure (Last={inputs['max_drive_pressure']:.1f}):",
                        parent=dialog_root, minvalue=0.1
                    )
                    inputs['max_drive_pressure'] = new_drive if new_drive else inputs['max_drive_pressure']
                    new_exit_run = tk.simpledialog.askfloat(
                        "Input", f"New Min Exit Pressure (Run) (Last={inputs['exit_pressure_run']:.1f}):",
                        parent=dialog_root, minvalue=0
                    )
                    inputs['exit_pressure_run'] = new_exit_run if new_exit_run else inputs['exit_pressure_run']
                    new_exit_end = tk.simpledialog.askfloat(
                        "Input", f"New Exit Pressure (End) (Last={inputs['exit_pressure_end']:.1f}):",
                        parent=dialog_root, minvalue=0
                    )
                    inputs['exit_pressure_end'] = new_exit_end if new_exit_end else inputs['exit_pressure_end']
                    new_cutoff = tk.simpledialog.askfloat(
                        "Input", f"New nitrogen cutoff volume in SCF (Last={cutoff_volume:.0f}):",
                        parent=dialog_root, minvalue=0.1
                    )
                    if new_cutoff:
                        inputs['cutoff_volume'] = new_cutoff
                    dialog_root.update()
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

# Unit Tests
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
