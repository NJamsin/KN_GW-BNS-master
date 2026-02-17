import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

DIR = f"/home/stu_jamsin/jamsin/grid_1_perday_noise"  # change as needed
# control shape of plot
col_num = 5
row_num = 5 

UL = False # if UL are present in the data

minus_num = 5 # max number of removed points + 1 (to include the full data analysis as well)

lc_num = 25 # change as needed (up to the number of injections)
if lc_num > col_num * row_num:
    print(f"Warning: lc_num ({lc_num}) is greater than the number of subplots ({col_num * row_num}). Consider increasing col_num and/or row_num to accommodate all lightcurves in the plot.")
if lc_num < col_num * row_num:
    print(f"Note: lc_num ({lc_num}) is less than the number of subplots ({col_num * row_num}). Some subplots will be empty. Consider adjusting col_num and/or row_num to better fit the number of lightcurves.")

cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(0, minus_num-1) # from 0 to the max number of points removed -1

# loop for the other parameters as well (corner plots for each analysis with modified data)
param_range = [(0.75, 2.25),(0,1),(0,4818),(1,2.3),(1,2.3)] # change as needed based on the prior bounds used for the analysis

for ii, param_name, param_label in zip(range(len(param_range)), ['chirp_mass', 'mass_ratio', 'EOS', 'mass_1', 'mass_2'], ['$\mathcal{M}$', '$q$', '$\mathrm{EOS}$', '$m_1$', '$m_2$']):
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
                    "luminosity_distance": param["luminosity_distance"].values[0],
                    "mass 1": param["mass_1"].values[0],
                    "mass 2": param["mass_2"].values[0]
        }
        for band in lc[1].unique():
            band_df = lc[lc[1]==band]
            times = pd.to_datetime(band_df[0].values)
            axx.errorbar(times, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
        txt = f"$\\phi$: {model_param['KNphi']:.1f}\n$log_{{10}} M_{{dyn}}$: {model_param['log10_mej_dyn']:.2f}\n$log_{{10}} M_{{wind}}$: {model_param['log10_mej_wind']:.2f}\n$\\theta$: {model_param['KNtheta']:.1f}\n$D_L$: {model_param['luminosity_distance']:.1f} Mpc\n$m_1$: {model_param['mass 1']:.2f} $M_\\odot$\n$m_2$: {model_param['mass 2']:.2f} $M_\\odot$"
        axx.text(0.7, 0.99, txt, transform=axx.transAxes, fontsize=12, verticalalignment='top')
        axx.text(0.001, 0.99, f"LC {idx}", transform=axx.transAxes, fontsize=20, verticalalignment='top')
        axx.legend()
        axx.invert_yaxis()
        axx.set_xlabel('Time [days]')
        axx.set_ylabel('Magnitude')
        for i in range(minus_num): 
            samples = pd.read_csv(f"{BASE_DIR}/minus{i}/resamp/posterior_samples.dat", delim_whitespace=True)
            truth = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")
            m1 = truth['mass_1'].values[0]
            m2 = truth['mass_2'].values[0]
            samples['mass_1'] = samples['chirp_mass'] * (samples['mass_ratio']**(-3/5)) * ((1 + samples['mass_ratio'])**(1/5))
            samples['mass_2'] = samples['chirp_mass'] * (samples['mass_ratio']**(2/5)) * ((1 + samples['mass_ratio'])**(1/5))
            true_q = m2 / m1
            true_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            truths_list = [true_chirp, true_q, 4818, m1, m2] # adjust as needed for the true EOS index
            truth_val = pd.DataFrame([truths_list], columns=['chirp_mass', 'mass_ratio', 'EOS', 'mass_1', 'mass_2'])
            lower = samples[param_name].quantile(0.16)
            upper = samples[param_name].quantile(0.84)
            median = samples[param_name].median()
            ax.errorbar(truth_val[param_name].values[0], median, yerr=[[median - lower], [upper - median]], fmt='o', color=cmap(norm(i)))
            axis.errorbar(truth_val[param_name].values[0], median, yerr=[[median - lower], [upper - median]], fmt='o', color=cmap(norm(i)))
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