import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from utils import generate_synth_lc_v2

# BASE_DIR
BASE_DIR = f"/home/stu_jamsin/jamsin/grid_LSST_only"  # change as needed

OUT_DIR = f"/home/stu_jamsin/jamsin/grid_LSST_ZTF_aligned"  # change as needed
if len(OUT_DIR) > 60:
    print("Warning: OUT_DIR path is quite long, which may cause issues with some software. Consider using a shorter path if you encounter errors related to file paths.")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# ADD filter 
filt = ["ztfr", "ztfg", "ztfi"] # change as needed 

# loop over the synth lc to load and rewrites the file
num_lc = 25 # change as needed (up to the number of injections)

fig, axs = plt.subplots(5, num_lc // 5, figsize=(10*(num_lc // 5), 6*(num_lc // 5)), sharex=True, sharey=True)

for idx in range(num_lc):
    # load the model parameters and the lightcurve data
    lc = pd.read_csv(f"{BASE_DIR}/{idx}/data{idx}.dat", delim_whitespace=True, header=None)
    param = pd.read_csv(f"{BASE_DIR}/{idx}/true{idx}.csv")
    # extract model param and generate a new lc 
    model_param = {
                "KNphi": param["KNphi"].values[0],
                "log10_mej_dyn": param["log10_mej_dyn"].values[0],
                "log10_mej_wind": param["log10_mej_wind"].values[0],
                "KNtheta": param["KNtheta"].values[0],
                "luminosity_distance": param["luminosity_distance"].values[0],
                "timeshift": 0 # 0 ts to be compatible with ts_loop (same ts as the og grid)
    }
    df, trig = generate_synth_lc_v2(model_param=model_param, # /!\ make sure that the function parameters are the same as the ones used for the original grid
                    noise_level=0.2,
                    max_error_level=0.4,
                    trigger_iso='2025-01-01T00:00:00',
                    pts_per_day=0.5,
                    obs_duration=15,
                    filters_band=filt,
                    jitter=0.,
                    delay=0,
                    save=False,
                    detection_limit_dict={'ztfr': 20.6, 'ztfg': 20.8, 'ztfi': 19.9}) # change as needed based on the detection limit for the filter you want to add)
    # merge the new lc with the original one and save the new file
    new_lc = pd.concat([lc, df], ignore_index=True)
    OUT_FILE = f"{OUT_DIR}/{idx}/data{idx}.dat"
    if not os.path.exists(OUT_FILE):
        os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    new_lc.to_csv(OUT_FILE, index=False, header=False, sep=' ')
    OUT_TRUE_FILE = f"{OUT_DIR}/{idx}/true{idx}.csv"
    if not os.path.exists(OUT_TRUE_FILE):
        os.makedirs(os.path.dirname(OUT_TRUE_FILE), exist_ok=True)
    param.to_csv(OUT_TRUE_FILE, index=False)

    # plot the lc grid
    row = idx // (num_lc // 5)
    col = idx % (num_lc // 5)
    ax = axs[row, col]
    for band in new_lc[1].unique():
        band_df = new_lc[new_lc[1]==band]
        times = pd.to_datetime(band_df[0].values)
        ax.errorbar(times, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
    if col == 0:
        ax.set_ylabel('Magnitude')  
    ax.legend(loc='upper right')
    txt = f"$\\phi$: {model_param['KNphi']:.1f}\n$log_{{10}} M_{{dyn}}$: {model_param['log10_mej_dyn']:.2f}\n$log_{{10}} M_{{wind}}$: {model_param['log10_mej_wind']:.2f}\n$\\theta$: {model_param['KNtheta']:.1f}\n$D_L$: {model_param['luminosity_distance']:.1f} Mpc\n $M_1$: {param['mass_1'].values[0]:.2f} $M_\\odot$\n$M_2$: {param['mass_2'].values[0]:.2f} $M_\\odot$"
    ax.text(0.7, 0.99, txt, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.005, 0.99, f"LC {idx}", transform=ax.transAxes, fontsize=20, verticalalignment='top')
    if row == num_lc // 5: # only set x label for the bottom row
        ax.set_xlabel('Time [days]')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
ax.invert_yaxis() # invert y axis for magnitude
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_lightcurves.png")