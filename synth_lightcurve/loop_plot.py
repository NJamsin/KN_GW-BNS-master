import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

######################################
# DO NOT FORGET TO UPDATE THE DIFF VAR
######################################

# !!!!! peut être prbl avec le ts pour les pp plots si la detection limit crée d'office un ts > 0, à vérifier et à corriger si besoin

DIR = f"/home/stu_jamsin/jamsin/grid_4_perday"  # change as needed
# control shape of plot
col_num = 5
row_num = 5 

lc_num = 25 # change as needed (up to the number of injections)
if lc_num > col_num * row_num:
    print(f"Warning: lc_num ({lc_num}) is greater than the number of subplots ({col_num * row_num}). Consider increasing col_num and/or row_num to accommodate all lightcurves in the plot.")
if lc_num < col_num * row_num:
    print(f"Note: lc_num ({lc_num}) is less than the number of subplots ({col_num * row_num}). Some subplots will be empty. Consider adjusting col_num and/or row_num to better fit the number of lightcurves.")

ts_max = -2
loop_size = 8
ts_range = np.arange(ts_max, 0, -ts_max/loop_size)
ts_range = list(ts_range) 
ts_range.append(0) # add the original data with no timeshift 
ts_range = ts_range[::-1]  
print(f"Timeshift values used for the analysis: {ts_range} days")
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(0, len(ts_range)-1)
fig, axs = plt.subplots(ncols=2*col_num, nrows=row_num, figsize=(15*col_num,5*row_num), gridspec_kw={'width_ratios': [0.333, 0.666]*col_num})
for idx in range(lc_num):
    BASE_DIR = f"{DIR}/{idx}"
    data = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delim_whitespace=True, header=None)
    data = data.sort_values(by=0, ascending=True).reset_index(drop=True)
    # stock the time list 
    times2 = data[0].unique()
    # attribute the left column to timeshift evolution and the right column to lightcurve and set up correctly the axes
    ax = axs[idx // col_num, (idx % col_num) * 2]
    axx = axs[idx // col_num, (idx % col_num) * 2 + 1]

    lc = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delim_whitespace=True, header=None)
    param = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")
    model_param = {
                "KNphi": param["KNphi"].values[0],
                "log10_mej_dyn": param["log10_mej_dyn"].values[0],
                "log10_mej_wind": param["log10_mej_wind"].values[0],
                "KNtheta": param["KNtheta"].values[0],
                "luminosity_distance": param["luminosity_distance"].values[0]
    }
    for band in lc[1].unique():
        band_df = lc[lc[1]==band]
        times = pd.to_datetime(band_df[0].values)
        axx.errorbar(times, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
    txt = f"$\\phi$: {model_param['KNphi']:.1f}\n$log_{{10}} M_{{dyn}}$: {model_param['log10_mej_dyn']:.2f}\n$log_{{10}} M_{{wind}}$: {model_param['log10_mej_wind']:.2f}\n$\\theta$: {model_param['KNtheta']:.1f}\n$D_L$: {model_param['luminosity_distance']:.1f} Mpc"
    axx.text(0.7, 0.99, txt, transform=axx.transAxes, fontsize=12, verticalalignment='top')
    axx.text(0.001, 0.99, f"LC {idx}", transform=axx.transAxes, fontsize=20, verticalalignment='top')
    axx.legend()
    axx.invert_yaxis()
    axx.set_xlabel('Time [days]')
    axx.set_ylabel('Magnitude')
    for i in range(8):
        samples = pd.read_csv(f"{BASE_DIR}/minus{i}/minus{i}_{idx}_posterior_samples.dat", delim_whitespace=True)
        # compute the timeshift
        ts = pd.to_datetime(times2[i]) - pd.to_datetime('2025-01-01T00:00:00.000') # keep the same trigger time as for the original data to see how the timeshift evolves
        ts = -1 *ts.total_seconds() / (3600*24) # convert to days
        lower = samples['timeshift'].quantile(0.16)
        upper = samples['timeshift'].quantile(0.84)
        median = samples['timeshift'].median()
        ax.errorbar(ts, median, yerr=[[median - lower], [upper - median]], fmt='o', color=cmap(norm(i)), label=f'true ts: {ts} days')
    ax.plot([ts_max-0.25, 0], [ts_max-0.25, 0], ls='--', color='red', label='perfect recovery')
    ax.set_xlabel('Timeshift [days]')
    ax.set_ylabel('Inferred timeshift [days]')
OUT_DIR = f"{DIR}/plots"
os.makedirs(OUT_DIR, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/timeshift_evolution.png")

# loop for the other parameters as well (corner plots for each analysis with modified data)
param_range = [(10,200),(15,75),(0,90),(-3,-1),(-3,-0.5)] # change as needed based on the prior bounds used for the analysis

for ii, param_name, param_label in zip(range(len(param_range)), ['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind'], ['D_L [Mpc]', '$\\phi$ [deg]', '$\\theta$ [deg]', '$log_{10} M_{dyn}$ [$M_\\odot$]', '$log_{10} M_{wind}$ [$M_\\odot$]']):
    fig, axs = plt.subplots(ncols=2*col_num, nrows=row_num, figsize=(15*col_num,5*row_num), gridspec_kw={'width_ratios': [0.333, 0.666]*col_num})
    # add fig, ax to do pp plots
    figg, axis = plt.subplots(figsize=(10,10))
    for idx in range(lc_num):
        BASE_DIR = f"{DIR}/{idx}"
        # attribute the left column to timeshift evolution and the right column to lightcurve and set up correctly the axes
        ax = axs[idx // col_num, (idx % col_num) * 2]
        axx = axs[idx // col_num, (idx % col_num) * 2 + 1]

        lc = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delim_whitespace=True, header=None)
        param = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")
        model_param = {
                    "KNphi": param["KNphi"].values[0],
                    "log10_mej_dyn": param["log10_mej_dyn"].values[0],
                    "log10_mej_wind": param["log10_mej_wind"].values[0],
                    "KNtheta": param["KNtheta"].values[0],
                    "luminosity_distance": param["luminosity_distance"].values[0]
        }
        for band in lc[1].unique():
            band_df = lc[lc[1]==band]
            times = pd.to_datetime(band_df[0].values)
            axx.errorbar(times, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
        txt = f"$\\phi$: {model_param['KNphi']:.1f}\n$log_{{10}} M_{{dyn}}$: {model_param['log10_mej_dyn']:.2f}\n$log_{{10}} M_{{wind}}$: {model_param['log10_mej_wind']:.2f}\n$\\theta$: {model_param['KNtheta']:.1f}\n$D_L$: {model_param['luminosity_distance']:.1f} Mpc"
        axx.text(0.7, 0.99, txt, transform=axx.transAxes, fontsize=12, verticalalignment='top')
        axx.text(0.001, 0.99, f"LC {idx}", transform=axx.transAxes, fontsize=20, verticalalignment='top')
        axx.legend()
        axx.invert_yaxis()
        axx.set_xlabel('Time [days]')
        axx.set_ylabel('Magnitude')
        for i, ts in enumerate(ts_range):
            samples = pd.read_csv(f"{BASE_DIR}/minus{i}/minus{i}_{idx}_posterior_samples.dat", delim_whitespace=True)
            truth = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")
            lower = samples[param_name].quantile(0.16)
            upper = samples[param_name].quantile(0.84)
            median = samples[param_name].median()
            ax.errorbar(truth[param_name].values[0], median, yerr=[[median - lower], [upper - median]], fmt='o', color=cmap(norm(i)))
            axis.errorbar(truth[param_name].values[0], median, yerr=[[median - lower], [upper - median]], fmt='o', color=cmap(norm(i)))
        ax.plot(param_range[ii], param_range[ii], ls='--', color='red', label='perfect recovery')
        ax.set_xlabel(param_label)
        ax.set_ylabel(f'Inferred {param_label}')
    axis.plot(param_range[ii], param_range[ii], ls='--', color='red', label='perfect recovery')
    axis.set_xlabel(f'Injected {param_label}')
    axis.set_ylabel(f'Inferred {param_label}')
    axis.set_title(f'Injection-recovery plot for {param_label}')
    OUT_DIR = f"{DIR}/plots"
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{param_name}_evolution.png")
    figg.tight_layout()
    figg.savefig(f"{OUT_DIR}/{param_name}_pp.png")