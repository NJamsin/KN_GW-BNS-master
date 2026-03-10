import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import corner
import gc
import sys

idx = int(sys.argv[1]) # change as needed (can be adapted for condor if needed) int(sys.argv[1])
#idx = 10  # example injection index
ADD_UL = False # whether to add UL instead of the removed points
minus_pts = 8 # max number of removed points (update as needed, up to the number of time points - 1)

MODEL = 'Bu2019lm' # change as needed

BASE_DIR = f"/home/stu_jamsin/jamsin/grid_testFiesta/{idx}"  # change as needed
if len(BASE_DIR) > 60:
    print("Warning: BASE_DIR path is quite long, which may cause issues with some software. Consider using a shorter path if you encounter errors related to file paths.")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

DTYPE_FLOAT = np.float32 # attempt to reduce memory usage

def save_corner_plot(samples, truth_row, ts, out_path, title, model=MODEL):
    if model == 'Bu2019lm':
        labels = ['$D_L$', '$\\phi$', '$\\iota$', '$log_{10} M_{dyn}$', '$log_{10} M_{wind}$', 'timeshift']
        cols = ['luminosity_distance', 'KNphi', 'inclination_EM', 'log10_mej_dyn', 'log10_mej_wind', 'timeshift']
        truth_val = [truth_row['luminosity_distance'].values[0], truth_row['KNphi'].values[0], 
                    truth_row['inclination_EM'].values[0], truth_row['log10_mej_dyn'].values[0], 
                    truth_row['log10_mej_wind'].values[0], -1*ts],
    elif model == 'Ka2017':
        labels = ['$D_L$', '$\\iota$', '$\\log_{10} M_{ej}$', '$\\log_{10} v_{ej}$', '$log_{10} X_{lan}$', 'timeshift']
        cols = ['luminosity_distance', 'inclination_EM', 'log10_mej', 'log10_vej', 'log10_Xlan', 'timeshift']
        truth_val = [truth_row['luminosity_distance'].values[0], truth_row['inclination_EM'].values[0], 
                    truth_row['log10_mej'].values[0], truth_row['log10_vej'].values[0], truth_row['log10_Xlan'].values[0], -1*ts]

    # limit to 32 bit float to save memory
    plot_data = samples[cols].astype(np.float32)

    fig = corner.corner(
        plot_data,
        truths=truth_val,
        truth_color='red',
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        label_kwargs={'fontsize': 14},
        smooth=1.0,
        bins=30,
        color='steelblue',
        hist_kwargs={'density': True},
        max_n_ticks=4,
        figsize=(10, 10),
        labelpad=0.03, 
    )

    # get quantiles for annotations
    axes = np.array(fig.axes).reshape((len(cols), len(cols)))
    for i, col in enumerate(cols):
        ax = axes[i, i]
        q16, q50, q84 = plot_data[col].quantile([0.16, 0.5, 0.84])
        inf_txt = rf"${q50:.3f}^{{+{q84-q50:.3f}}}_{{-{q50-q16:.3f}}}$"
        truth_val = -1*ts if col == 'timeshift' else truth_row[col].values[0]
        ax.text(0.3, 1.03, inf_txt, transform=ax.transAxes, ha='center', fontsize=10)
        ax.text(0.8, 1.03, rf"{truth_val:.3f}", transform=ax.transAxes, ha='center', fontsize=10, color='red')

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=150) # [OPTIM] DPI fixe pour contrôler la taille du fichier
    plt.close(fig)
    del plot_data
    gc.collect()

# 1st load the data 
data = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delimiter=' ', header=None, dtype={0: str, 1: str, 2: DTYPE_FLOAT, 3: DTYPE_FLOAT})
truth = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")

# sort by time
data = data.sort_values(by=0, ascending=True).reset_index(drop=True)
# stock the time list 
times = data[0].unique()

# do the first analysis with the full data (ts should be close to 0)
cmd_lc = ["/home/stu_jamsin/.conda/envs/nmma_env/bin/lightcurve-analysis",
        "--model", MODEL,
        "--svd-path", "/home/stu_jamsin/jamsin/NMMA/svdmodels",
        "--outdir", f"{BASE_DIR}/minus0",
        "--label", f"minus0_{idx}",
        "--prior", f"/home/stu_jamsin/jamsin/NMMA/priors/{MODEL}200.prior",
        "--nlive", "512", 
        "--Ebv-max", "0",
        "--data", f"{BASE_DIR}/data{idx}.dat",
        "--error-budget", "0.5",
        "--plot", 
        "--ylim", "26,17",
        "--xlim=-4,14",
    ]
subprocess.run(cmd_lc, check=True, cwd=BASE_DIR) 
# plot the corner plot for the analysis with the full data
samples = pd.read_csv(f"{BASE_DIR}/minus0/minus0_{idx}_posterior_samples.dat", delimiter=' ', dtype=DTYPE_FLOAT)
ts = pd.to_datetime(times[0]) - pd.to_datetime('2025-01-01T00:00:00.000') # keep the same trigger time as for the original data to see how the timeshift evolves
ts = ts.total_seconds() / (3600*24) # convert to days
try:
    save_corner_plot(samples, truth, ts, f"{BASE_DIR}/minus0/corner_minus0_{idx}.png", "Corner plot for analysis with full data")
except Exception as e:
    print(f"Error occurred while saving corner plot: {e}")
del samples, cmd_lc
gc.collect()

# launch lc analysis after repeatedly taking out the first point to see how the timeshift evolves and how it affects the parameter estimation
for j in range(minus_pts): # change the range as needed (up to the number of time points - 1) /!\ update prior bounds if needed
    if ADD_UL:
        filt_list = [data[1][i] for i in range(len(data)) if data[0][i] == times[j]]
        mag_per_filter = {band: data[data[1]==band][2].values for band in data[1].unique()}
        temp_df = pd.DataFrame()
        for f in filt_list:
            dm = mag_per_filter[f][0] - mag_per_filter[f][1]
            if dm > 0:
                ul = mag_per_filter[f][0] - 0.75 * dm
            else:
                ul = mag_per_filter[f][0] + 0.75 * dm
            UL = pd.DataFrame([[times[j], f, ul, np.inf]], columns=[0,1,2,3])
            temp_df = pd.concat([temp_df, UL], ignore_index=True)
    # drop the first time point
    dupl = [True if data[0][i] == times[j] else False for i in range(0, len(data))]
    data = data[~pd.Series(dupl)].reset_index(drop=True) # modify the original data for the next iteration  
    if ADD_UL:
        # add an UL
        data = pd.concat([data, temp_df], ignore_index=True)
    data.to_csv(f"{BASE_DIR}/data_minus{j+1}.dat", sep=' ', index=False, header=False)
    # compute the timeshift
    ts = pd.to_datetime(data[0][0]) - pd.to_datetime('2025-01-01T00:00:00.000') # keep the same trigger time as for the original data to see how the timeshift evolves
    ts = ts.total_seconds() / (3600*24) # convert to days
    # launch lc analysis with the modified data
    cmd_lc_ts = ["/home/stu_jamsin/.conda/envs/nmma_env/bin/lightcurve-analysis",
        "--model", MODEL,
        "--svd-path", "/home/stu_jamsin/jamsin/NMMA/svdmodels",
        "--outdir", f"{BASE_DIR}/minus{j+1}",
        "--label", f"minus{j+1}_{idx}",
        "--prior", f"/home/stu_jamsin/jamsin/NMMA/priors/{MODEL}200.prior",
        "--nlive", "512", 
        "--Ebv-max", "0",
        "--data", f"{BASE_DIR}/data_minus{j+1}.dat",
        "--error-budget", "0.5",
        "--plot", 
        "--ylim", "26,17",
        "--xlim=-2,14"
    ]
    subprocess.run(cmd_lc_ts, check=True, cwd=BASE_DIR)
    # corner plot for the analysis with modified data
    samples = pd.read_csv(f"{BASE_DIR}/minus{j+1}/minus{j+1}_{idx}_posterior_samples.dat", delimiter=' ', dtype=DTYPE_FLOAT)
    try:
        save_corner_plot(samples, truth, ts, f"{BASE_DIR}/minus{j+1}/corner_minus{j+1}_{idx}.png", f"Corner plot for analysis with data minus {j+1} point(s)")
    except Exception as e:
        print(f"Error occurred while saving corner plot: {e}")
    del samples, cmd_lc_ts
    gc.collect()
