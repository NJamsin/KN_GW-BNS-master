import pandas as pd
import numpy as np
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import corner

idx = int(sys.argv[1]) # change as needed (can be adapted for condor if needed) int(sys.argv[1])
#idx = 12  # example injection index
filters = "ps1__g,ps1__r,ps1__i,ps1__z" # change as needed (keep the same as for the original analysis)
ADD_UL = False # whether to add UL instead of the removed points

BASE_DIR = f"/home/stu_jamsin/jamsin/grid_4_perday/{idx}"  # change as needed
if len(BASE_DIR) > 60:
    print("Warning: BASE_DIR path is quite long, which may cause issues with some software. Consider using a shorter path if you encounter errors related to file paths.")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

# 1st load the data 
data = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delim_whitespace=True, header=None)
truth = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")

# sort by time
data = data.sort_values(by=0, ascending=True).reset_index(drop=True)
# stock the time list 
times = data[0].unique()

# do the first analysis with the full data (ts should be close to 0)
cmd_lc = ["/home/stu_jamsin/.conda/envs/nmma_env/bin/lightcurve-analysis",
        "--model", "Bu2019lm",
        "--svd-path", "/home/stu_jamsin/jamsin/NMMA/svdmodels",
        "--outdir", f"{BASE_DIR}/minus0",
        "--label", f"minus0_{idx}",
        "--prior", "/home/stu_jamsin/jamsin/NMMA/priors/Bu2019lm200.prior",
        "--nlive", "256", 
        "--Ebv-max", "0",
        "--filters", filters,
        "--data", f"{BASE_DIR}/data{idx}.dat",
        "--error-budget", "0.5",
        "--plot", 
        "--ylim", "26,17",
        "--xlim=-2,14"
    ]
subprocess.run(cmd_lc, check=True, cwd=BASE_DIR) 
# plot the corner plot for the analysis with the full data
samples = pd.read_csv(f"{BASE_DIR}/minus0/minus0_{idx}_posterior_samples.dat", delim_whitespace=True)
ts = pd.to_datetime(times[0]) - pd.to_datetime('2025-01-01T00:00:00.000') # keep the same trigger time as for the original data to see how the timeshift evolves
ts = ts.total_seconds() / (3600*24) # convert to days
labels = ['$D_L$', '$\\phi$', '$\\theta$', '$log_{10} M_{dyn}$', '$log_{10} M_{wind}$', 'timeshift']
fig = corner.corner(samples[['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind', 'timeshift']],
                truths=[truth['luminosity_distance'].values[0], truth['KNphi'].values[0], truth['KNtheta'].values[0], truth['log10_mej_dyn'].values[0], truth['log10_mej_wind'].values[0], -1*ts],
                truth_color='red',
                labels=labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt='.3f',
                title_kwargs={
                    'fontsize': 14,
                    'pad': 12},
                label_kwargs={
                    'fontsize': 14},
                smooth=1.0,
                bins=30,
                color='steelblue',
                hist_kwargs={'density': True},
                max_n_ticks=4,
                figsize=(12, 12),
                labelpad=0.03, 
    )
# get quantiles for annotations
quantiles = {}
for i, col in enumerate(['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind', 'timeshift']):
    q16, q50, q84 = samples[col].quantile([0.16, 0.5, 0.84])
    quantiles[col] = (q16, q50, q84)
axes = np.array(fig.axes).reshape((6, 6))
for i in range(6):
    ax = axes[i, i]
    col = ['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind', 'timeshift'][i]
    minus = quantiles[col][1] - quantiles[col][0]
    plus = quantiles[col][2] - quantiles[col][1]
    #txt 
    inf_txt = rf"${quantiles[col][1]:.3f}^{{+{plus:.3f}}}_{{-{minus:.3f}}}$"
    if col == 'timeshift':
        truth_txt = rf"{-1*ts:.3f}"
    else:
        truth_txt = rf"{truth[col].values[0]:.3f}"
    #clear auto title
    ax.set_title('')
    #add txt
    ax.text(0.3, 1.03, inf_txt, transform=ax.transAxes, ha='center', fontsize=12, va='bottom', color='black')
    ax.text(0.8, 1.03, truth_txt, transform=ax.transAxes, ha='center', fontsize=12, va='bottom', color='red')

fig.suptitle(f"Corner plot for full lc (timeshift = {ts:.2f} days)", fontsize=16)
plt.savefig(f"{BASE_DIR}/minus0/corner_minus0_{idx}.png")
plt.close(fig)

# launch lc analysis after repeatedly taking out the first point to see how the timeshift evolves and how it affects the parameter estimation
for j in range(8): # change the range as needed (up to the number of time points - 1) /!\ update prior bounds if needed
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
    dupl
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
        "--model", "Bu2019lm",
        "--svd-path", "/home/stu_jamsin/jamsin/NMMA/svdmodels",
        "--outdir", f"{BASE_DIR}/minus{j+1}",
        "--label", f"minus{j+1}_{idx}",
        "--prior", "/home/stu_jamsin/jamsin/NMMA/priors/Bu2019lm200.prior",
        "--nlive", "256", 
        "--Ebv-max", "0",
        "--filters", filters,
        "--data", f"{BASE_DIR}/data_minus{j+1}.dat",
        "--error-budget", "0.5",
        "--plot", 
        "--ylim", "26,17",
        "--xlim=-2,14"
    ]
    subprocess.run(cmd_lc_ts, check=True, cwd=BASE_DIR)
    # corner plot for the analysis with modified data
    samples = pd.read_csv(f"{BASE_DIR}/minus{j+1}/minus{j+1}_{idx}_posterior_samples.dat", delim_whitespace=True)
    fig = corner.corner(samples[['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind', 'timeshift']],
                truths=[truth['luminosity_distance'].values[0], truth['KNphi'].values[0], truth['KNtheta'].values[0], truth['log10_mej_dyn'].values[0], truth['log10_mej_wind'].values[0], -1*ts],
                truth_color='red',
                labels=labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt='.3f',
                title_kwargs={
                    'fontsize': 14,
                    'pad': 12},
                label_kwargs={
                    'fontsize': 14},
                smooth=1.0,
                bins=30,
                color='steelblue',
                hist_kwargs={'density': True},
                max_n_ticks=4,
                figsize=(12, 12),
                labelpad=0.03, 
    )
    # get quantiles for annotations
    quantiles = {}
    for i, col in enumerate(['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind', 'timeshift']):
        q16, q50, q84 = samples[col].quantile([0.16, 0.5, 0.84])
        quantiles[col] = (q16, q50, q84)
    axes = np.array(fig.axes).reshape((6, 6))
    for i in range(6):
        ax = axes[i, i]
        col = ['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind', 'timeshift'][i]
        minus = quantiles[col][1] - quantiles[col][0]
        plus = quantiles[col][2] - quantiles[col][1]
        #txt 
        inf_txt = rf"${quantiles[col][1]:.3f}^{{+{plus:.3f}}}_{{-{minus:.3f}}}$"
        if col == 'timeshift':
            truth_txt = rf"{-1*ts:.3f}"
        else:
            truth_txt = rf"{truth[col].values[0]:.3f}"
        #clear auto title
        ax.set_title('')
        #add txt
        ax.text(0.3, 1.03, inf_txt, transform=ax.transAxes, ha='center', fontsize=12, va='bottom', color='black')
        ax.text(0.8, 1.03, truth_txt, transform=ax.transAxes, ha='center', fontsize=12, va='bottom', color='red')
    fig.suptitle(f"Corner plot for analysis with first {j+1} points removed (timeshift = {ts:.2f} days)", fontsize=16)
    plt.savefig(f"{BASE_DIR}/minus{j+1}/corner_minus{j+1}_{idx}.png")
    plt.close(fig)
