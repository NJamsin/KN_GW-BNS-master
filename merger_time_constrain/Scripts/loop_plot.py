import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

######################################
# DO NOT FORGET TO UPDATE THE DIFF VAR
######################################
gpaenvi = False

DIR = f"/home/stu_jamsin/jamsin/grid_Ka_4perday_Buinfer"  # change as needed
# control shape of plot
col_num = 5
row_num = 5 

UL = False # if UL are present in the data

minus_num = 9 # max number of removed points + 1 (to include the full data analysis as well)

MODEL = 'Bu2019lm' # change as needed

lc_num = 25 # change as needed (up to the number of injections)
if lc_num > col_num * row_num:
    print(f"Warning: lc_num ({lc_num}) is greater than the number of subplots ({col_num * row_num}). Consider increasing col_num and/or row_num to accommodate all lightcurves in the plot.")
if lc_num < col_num * row_num:
    print(f"Note: lc_num ({lc_num}) is less than the number of subplots ({col_num * row_num}). Some subplots will be empty. Consider adjusting col_num and/or row_num to better fit the number of lightcurves.")

cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(0, minus_num-1) # from 0 to the max number of points removed -1
ts_max = -4
#df_range = pd.DataFrame(columns=['parameter', 'ts', 'range']) # to store the range of the confidence interval for each parameter and each analysis (for the presentation)
fig, axs = plt.subplots(ncols=2*col_num, nrows=row_num, figsize=(15*col_num,5*row_num), gridspec_kw={'width_ratios': [0.333, 0.666]*col_num})
for idx in range(lc_num):
    BASE_DIR = f"{DIR}/{idx}"
    if os.path.getsize(f"{BASE_DIR}/data{idx}.dat") > 0:
        data = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delimiter=' ', header=None)
    else:
        print(f"Warning: Lightcurve data file {BASE_DIR}/data{idx}.dat is empty. Skipping this injection.")
        continue
    data = data.sort_values(by=0, ascending=True).reset_index(drop=True)
    # stock the time list 
    times2 = data[0].unique()
    # attribute the left column to timeshift evolution and the right column to lightcurve and set up correctly the axes
    ax = axs[idx // col_num, (idx % col_num) * 2]
    axx = axs[idx // col_num, (idx % col_num) * 2 + 1]

    lc = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delimiter=' ', header=None)
    param = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")
    if MODEL == 'Bu2019lm':
        model_param = {
                    "KNphi": param["KNphi"].values[0],
                    "log10_mej_dyn": param["log10_mej_dyn"].values[0],
                    "log10_mej_wind": param["log10_mej_wind"].values[0],
                    "inclination_EM": param["inclination_EM"].values[0],
                    "luminosity_distance": param["luminosity_distance"].values[0]
        }
    elif MODEL == 'Ka2017':
        model_param = {
            "inclination_EM": param["inclination_EM"].values[0],
            "log10_mej": param["log10_mej"].values[0],
            "log10_vej": param["log10_vej"].values[0],
            "log10_Xlan": param["log10_Xlan"].values[0],
            "luminosity_distance": param["luminosity_distance"].values[0]
        }
    for band in lc[1].unique():
        band_df = lc[lc[1]==band]
        times = pd.to_datetime(band_df[0].values)
        axx.errorbar(times, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
    if MODEL == 'Bu2019lm':
        txt = f"$\\phi$: {model_param['KNphi']:.1f}\n$log_{{10}} M_{{dyn}}$: {model_param['log10_mej_dyn']:.2f}\n$log_{{10}} M_{{wind}}$: {model_param['log10_mej_wind']:.2f}\n$\\iota$: {model_param['inclination_EM']:.1f}\n$D_L$: {model_param['luminosity_distance']:.1f} Mpc"
    elif MODEL == 'Ka2017':
        txt = f"$\\iota$: {model_param['inclination_EM']:.1f}\n$log_{{10}} M_{{ej}}$: {model_param['log10_mej']:.2f}\n$log_{{10}} v_{{ej}}$: {model_param['log10_vej']:.2f}\n$log_{{10}} X_{{lan}}$: {model_param['log10_Xlan']:.2f}\n$D_L$: {model_param['luminosity_distance']:.1f} Mpc"
    axx.text(0.7, 0.99, txt, transform=axx.transAxes, fontsize=12, verticalalignment='top')
    axx.text(0.001, 0.99, f"LC {idx}", transform=axx.transAxes, fontsize=20, verticalalignment='top')
    axx.legend()
    axx.invert_yaxis()
    axx.set_xlabel('Time [days]')
    axx.set_ylabel('Magnitude')
    for i in range(minus_num):
        SAMPLE_PATH = f"{BASE_DIR}/minus{i}/minus{i}_{idx}_posterior_samples.dat"
        if not os.path.exists(SAMPLE_PATH):
            print(f"Warning: Posterior samples file {SAMPLE_PATH} does not exist. Skipping this point.")
            continue
        samples = pd.read_csv(SAMPLE_PATH, delimiter=' ')
        if samples is None or samples.empty:
            print(f"Warning: Posterior samples for LC {idx} with minus {i} are missing or empty. Skipping this point.")
            continue
        # compute the timeshift
        if UL:
            ts = pd.to_datetime(times2[i]) - pd.to_datetime('2025-01-01T00:00:00.000') # keep the same trigger time as for the original data to see how the timeshift evolves
            ts = -1 *ts.total_seconds() / (3600*24) # convert to days
            adjust = pd.to_datetime(times2[i]) - pd.to_datetime(times2[0]) 
            adjust = -1 * adjust.total_seconds() / (3600*24)
            lower = samples['timeshift'].quantile(0.16) + adjust
            upper = samples['timeshift'].quantile(0.84) + adjust
            median = samples['timeshift'].median() + adjust
            ax.errorbar(ts, median, yerr=[[median - lower], [upper - median]], fmt='v', color=cmap(norm(i)), label=f'true ts: {ts} days')
        else:
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
if MODEL == 'Bu2019lm':
    param_range = [(10,200),(15,75),(0,np.pi/2),(-3,-1),(-3,-0.5)] # change as needed based on the prior bounds used for the analysis
    param_names = ['luminosity_distance', 'KNphi', 'inclination_EM', 'log10_mej_dyn', 'log10_mej_wind'] # change as needed based on the parameters used in the analysis
    param_labels = ['D_L [Mpc]', '$\\phi$ [deg]', '$\\iota$ [rad]', '$log_{10} M_{dyn}$ [$M_\\odot$]', '$log_{10} M_{wind}$ [$M_\\odot$]'] # change as needed for the plot labels
elif MODEL == 'Ka2017':
    param_range = [(10,200),(0, np.pi/2),(-3,-0.5),(-1.5,-0.5),(-9,-1)] # change as needed based on the prior bounds used for the analysis
    param_names = ['luminosity_distance', 'inclination_EM', 'log10_mej', 'log10_vej', 'log10_Xlan'] # change as needed based on the parameters used in the analysis
    param_labels = ['D_L [Mpc]', '$\\iota$ [rad]', '$log_{10} M_{ej}$ [$M_\\odot$]', '$log_{10} v_{ej}$ [c]', '$log_{10} X_{lan}$'] # change as needed for the plot labels

for ii, param_name, param_label in zip(range(len(param_range)), param_names, param_labels):
    fig, axs = plt.subplots(ncols=2*col_num, nrows=row_num, figsize=(15*col_num,5*row_num), gridspec_kw={'width_ratios': [0.333, 0.666]*col_num})
    # add fig, ax to do pp plots
    figg, axis = plt.subplots(figsize=(10,10))
    for idx in range(lc_num):
        BASE_DIR = f"{DIR}/{idx}"
        # attribute the left column to timeshift evolution and the right column to lightcurve and set up correctly the axes
        ax = axs[idx // col_num, (idx % col_num) * 2]
        axx = axs[idx // col_num, (idx % col_num) * 2 + 1]
        if os.path.getsize(f"{BASE_DIR}/data{idx}.dat") > 0:
            lc = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delimiter=' ', header=None)
        else:
            print(f"Warning: Lightcurve data file {BASE_DIR}/data{idx}.dat is empty. Skipping this injection.")
            continue
        param = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")
        if MODEL == 'Bu2019lm':
            model_param = {
                        "KNphi": param["KNphi"].values[0],
                        "log10_mej_dyn": param["log10_mej_dyn"].values[0],
                        "log10_mej_wind": param["log10_mej_wind"].values[0],
                        "inclination_EM": param["inclination_EM"].values[0],
                        "luminosity_distance": param["luminosity_distance"].values[0]
            }
        elif MODEL == 'Ka2017':
            model_param = {
                "inclination_EM": param["inclination_EM"].values[0],
                "log10_mej": param["log10_mej"].values[0],
                "log10_vej": param["log10_vej"].values[0],
                "log10_Xlan": param["log10_Xlan"].values[0],
                "luminosity_distance": param["luminosity_distance"].values[0]
            }
        for band in lc[1].unique():
            band_df = lc[lc[1]==band]
            times = pd.to_datetime(band_df[0].values)
            axx.errorbar(times, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
        if MODEL == 'Bu2019lm':
            txt = f"$\\phi$: {model_param['KNphi']:.1f}\n$log_{{10}} M_{{dyn}}$: {model_param['log10_mej_dyn']:.2f}\n$log_{{10}} M_{{wind}}$: {model_param['log10_mej_wind']:.2f}\n$\\iota$: {model_param['inclination_EM']:.2f}\n$D_L$: {model_param['luminosity_distance']:.1f} Mpc"
        elif MODEL == 'Ka2017':
            txt = f"$\\iota$: {model_param['inclination_EM']:.1f}\n$log_{{10}} M_{{ej}}$: {model_param['log10_mej']:.2f}\n$log_{{10}} v_{{ej}}$: {model_param['log10_vej']:.2f}\n$log_{{10}} X_{{lan}}$: {model_param['log10_Xlan']:.2f}\n$D_L$: {model_param['luminosity_distance']:.1f} Mpc"
        axx.text(0.7, 0.99, txt, transform=axx.transAxes, fontsize=12, verticalalignment='top')
        axx.text(0.001, 0.99, f"LC {idx}", transform=axx.transAxes, fontsize=20, verticalalignment='top')
        axx.legend()
        axx.invert_yaxis()
        axx.set_xlabel('Time [days]')
        axx.set_ylabel('Magnitude')
        for i in range(minus_num): 
            SAMPLE_PATH = f"{BASE_DIR}/minus{i}/minus{i}_{idx}_posterior_samples.dat"
            if not os.path.exists(SAMPLE_PATH):
                print(f"Warning: Posterior samples file {SAMPLE_PATH} does not exist. Skipping this point.")
                continue
            samples = pd.read_csv(SAMPLE_PATH, delimiter=' ')
            if samples is None or samples.empty:
                print(f"Warning: Posterior samples for LC {idx} with minus {i} are missing or empty. Skipping this point.")
                continue
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

# "inverted" loop to to pp plot per ts 
if gpaenvi:
    for ii, param_name, param_label in zip(range(len(param_range)), ['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind'], ['D_L [Mpc]', '$\\phi$ [deg]', '$\\theta$ [deg]', '$log_{10} M_{dyn}$ [$M_\\odot$]', '$log_{10} M_{wind}$ [$M_\\odot$]']):
        for idx in range(minus_num):
            figg, axis = plt.subplots(figsize=(10,10))
            for i in range(lc_num):
                BASE_DIR = f"{DIR}/{i}" 
                SAMPLE_PATH = f"{BASE_DIR}/minus{idx}/minus{idx}_{i}_posterior_samples.dat"
                if not os.path.exists(SAMPLE_PATH):
                    print(f"Warning: Posterior samples file {SAMPLE_PATH} does not exist. Skipping this point.")
                    continue
                samples = pd.read_csv(SAMPLE_PATH, delimiter=' ')
                if samples is None or samples.empty:
                    print(f"Warning: Posterior samples for LC {idx} with minus {i} are missing or empty. Skipping this point.")
                    continue
                truth = pd.read_csv(f"{BASE_DIR}/true{i}.csv")
                lower = samples[param_name].quantile(0.16)
                upper = samples[param_name].quantile(0.84)
                median = samples[param_name].median()
                axis.errorbar(truth[param_name].values[0], median, yerr=[[median - lower], [upper - median]], fmt='o', c='blue')
            axis.plot(param_range[ii], param_range[ii], ls='--', color='red', label='perfect recovery')
            axis.set_xlabel(f'Injected {param_label}')
            axis.set_ylabel(f'Inferred {param_label}')
            axis.set_title(f'Injection-recovery plot for {param_label}')
            OUT_DIR = f"{DIR}/plots/minus{idx}"
            os.makedirs(OUT_DIR, exist_ok=True)
            figg.tight_layout()
            figg.savefig(f"{OUT_DIR}/{param_name}_pp.png")