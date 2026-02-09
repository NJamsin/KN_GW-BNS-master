import pandas as pd
import numpy as np
import os
import subprocess
import sys
import matplotlib.pyplot as plt
from utils import generate_synth_lc
from utils import generate_synth_lc_v2
from utils import wind_ej, dyn_ej
from scipy.stats import qmc

BASE_DIR = "/home/stu_jamsin/jamsin/grid_test"  # change as needed
if len(BASE_DIR) > 60:
    print("Warning: BASE_DIR path is quite long, which may cause issues with some software. Consider using a shorter path if you encounter errors related to file paths.")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

num_lc = 20 # please put a multiple of 5 here for the plot

# define the range lower/upper bounds for each parameter
param_bounds = {
    "KNphi": (15, 75),
    "log10_mej_dyn": (-3, -1.5),
    "log10_mej_wind": (-3, -0.5),
    "KNtheta": (0, 90),
    "luminosity_distance": (10, 200)
}
lower_bounds = [bounds[0] for bounds in param_bounds.values()]
upper_bounds = [bounds[1] for bounds in param_bounds.values()]

# sample with latin hypercube sampling
sampler = qmc.LatinHypercube(d=len(param_bounds))
sample = sampler.random(n=num_lc)
scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)

# 2nd: generate synthetic lightcurves for each parameter combination in the sample and create a huge plot with all the lightcurves for visual check
filters_band = ['ps1__g', 'ps1__r', 'ps1__i', 'ps1__z'] # filters used for "observation"
fig, axs = plt.subplots(5, num_lc // 5, figsize=(10*(num_lc // 5), 6*(num_lc // 5)), sharex=True, sharey=True)
for i in range(num_lc):
    witness = num_lc // 5
    KNphi, log10_mej_dyn, log10_mej_wind, KNtheta, luminosity_distance = scaled_sample[i]
    model_param = {
        "KNphi": KNphi,
        "log10_mej_dyn": log10_mej_dyn,
        "log10_mej_wind": log10_mej_wind,
        "KNtheta": KNtheta,
        "luminosity_distance": luminosity_distance,
        "timeshift": 0
    }
    # save model param to csv for reference
    OUT_DIR = f"{BASE_DIR}/{i}"
    os.makedirs(OUT_DIR, exist_ok=True)
    param_df = pd.DataFrame([model_param])
    param_df.to_csv(f"{OUT_DIR}/true{i}.csv", index=False)   
    print(f"Generating synthetic lightcurve {i+1}/{num_lc}...")
    data_nmma_svd, trig = generate_synth_lc_v2(
            model_name='Bu2019lm',
            model_param=model_param,
            filters_band=filters_band,
            noise_level=0.,
            min_error_level=0.03,
            max_error_level=0.4,
            trigger_iso='2025-01-01T00:00:00',
            pts_per_day=2,
            obs_duration=15,
            jitter=0.,
            save=True,
            filename=f"{OUT_DIR}/data{i}.dat",
            detection_limit_dict={'ps1__g':26, 'ps1__r':26, 'ps1__i':26, 'ps1__z':26}
    )
    # plot part 
    row = i // (num_lc // 5)
    col = i % (num_lc // 5)
    ax = axs[row, col]
    for band in data_nmma_svd[1].unique():
        band_df = data_nmma_svd[data_nmma_svd[1]==band]
        times = pd.to_datetime(band_df[0].values)
        ax.errorbar(times, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
    if col == 0:
        ax.set_ylabel('Magnitude')  
    ax.legend(loc='upper right')
    if row == num_lc // 5: # only set x label for the bottom row
        ax.set_xlabel('Time [days]')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
ax.invert_yaxis() # invert y axis for magnitude
plt.tight_layout()
plt.savefig(f"{BASE_DIR}/all_lightcurves.png")


    