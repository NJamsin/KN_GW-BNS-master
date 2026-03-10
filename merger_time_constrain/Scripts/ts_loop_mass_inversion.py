import pandas as pd
import numpy as np
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import corner
from utils import wind_ej, dyn_ej
from utils import ejecta_to_mass
from utils import ejecta_plot_v4
from utils import compare_mass_sets
import bilby

# replace the gwem resampling by the semi-analytical reversal of the fitting formula (EOS dependent)

idx = int(sys.argv[1]) # change as needed (can be adapted for condor if needed) int(sys.argv[1])
#idx = 0 # for test

BASE_DIR = f"/home/stu_jamsin/jamsin/grid_LSST_like_nlive512"  # change as needed
if len(BASE_DIR) > 60:
    print("Warning: BASE_DIR path is quite long, which may cause issues with some software. Consider using a shorter path if you encounter errors related to file paths.")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

num_points = 5 # max number of removed points + 1 (to include the full data analysis as well)

# resamp
BASE_DIR_i = f"{BASE_DIR}/{idx}"  # change as needed

for i in range(num_points):
    print(f"Starting resampling for lc {idx} with minus {i}")
    # set up output directory and EM post file
    POST_FILE = f"{BASE_DIR_i}/minus{i}/minus{i}_{idx}_posterior_samples.dat"
    if not os.path.exists(POST_FILE):
        print(f"Posterior file {POST_FILE} not found. Please run the lightcurve analysis for lc {idx} for minus{i} before running the resampling.")
        continue
    OUT_DIR = f"{BASE_DIR_i}/minus{i}/"
    EM_samples = pd.read_csv(POST_FILE, delim_whitespace=True)
    D_L = np.median(EM_samples["luminosity_distance"].values)
    KNphi = np.median(EM_samples["KNphi"].values)
    KNtheta = np.median(EM_samples["KNtheta"].values)
    log10_mej_dyn = EM_samples["log10_mej_dyn"].values
    log10_mej_wind = EM_samples["log10_mej_wind"].values
    dyn_range = (np.quantile(log10_mej_dyn, 0.16), np.quantile(log10_mej_dyn, 0.84))
    wind_range = (np.quantile(log10_mej_wind, 0.16), np.quantile(log10_mej_wind, 0.84))
    ejecta_param = {
        "luminosity_distance": D_L,
        "KNphi": KNphi,
        "KNtheta": KNtheta,
    }
    dict_ejecta = ejecta_plot_v4(4818, model_name='Bu2019lm', model_param=ejecta_param, plot=False, get_fig=False, filters=['ps1__g'])
    # reverse ff
    fig_mass, ax_mass, mass_pairs_dyn, mass_pairs_wind = ejecta_to_mass(dict_ejecta, plot=True, title=f"Synth lightcurve {idx} mass pairs from ejecta", get_fig=True, dyn_range=dyn_range, wind_range=wind_range)
    fig_mass.savefig(f"{OUT_DIR}/inj_{idx}_mass_pairs_from_ejecta_minus{i}.png", bbox_inches='tight')
    common_masses = compare_mass_sets(mass_pairs_dyn, mass_pairs_wind, tol=0.05)
    for mass_pair in common_masses:
        print(f"({mass_pair[0]:.3f}, {mass_pair[1]:.3f})")
    # plot
    plt.figure(figsize=(6,6))
    plt.scatter(mass_pairs_dyn[:,0], mass_pairs_dyn[:,1], label='Dynamical Ejecta Mass Pairs', alpha=0.5)
    plt.scatter(mass_pairs_wind[:,0], mass_pairs_wind[:,1], label='Wind Ejecta Mass Pairs', alpha=0.5)
    if common_masses.shape[0] > 0:
        plt.scatter(common_masses[:,0], common_masses[:,1], color='red', label='Common Mass Pairs', s=100, edgecolors='k')
    plt.xlabel('$M_1$ [$M_\\odot$]')
    plt.ylabel('$M_2$ [$M_\\odot$]')
    plt.title(f'Coherent mass pairs for lightcurve {idx}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUT_DIR}/inj_{idx}_coherent_mass_pair_minus{i}.png", bbox_inches='tight')
    # save recovered mass pairs to csv for reference
    m1 = np.array(common_masses[:,0])
    m2 = np.array(common_masses[:,1])
    inj_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    inj_q = m2 / m1
    out_dict = {
        "mass_1": m1,
        "mass_2": m2,
        "chirp_mass": inj_chirp,
        "mass_ratio": inj_q
    }
    # save to csv for reference
    pd.DataFrame(out_dict).to_csv(f"{BASE_DIR}/{idx}/minus{i}/recovered_masses{idx}.csv", index=False)

    # do the plot now
    truth = pd.read_csv(f"{BASE_DIR_i}/true{idx}.csv")
    true_q = truth['mass_2'].values[0] / truth['mass_1'].values[0]
    true_chirp = bilby.gw.conversion.component_masses_to_chirp_mass(truth['mass_1'].values[0], truth['mass_2'].values[0])
    truths_list2 = [true_chirp, true_q, truth['mass_1'].values[0], truth['mass_2'].values[0]] # adjust as needed for the true EOS index
    # convert out_dict to dataframe for corner plot
    out_dict = pd.DataFrame(out_dict)
    fig = corner.corner(
    out_dict[['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']],
    labels=['$\mathcal{M}$', '$q$', '$M_1$', '$M_2$'],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.3f',
    title_kwargs={'fontsize': 14, 'pad': 12},
    label_kwargs={'fontsize': 14},
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
    for j, col in enumerate(out_dict[['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']].columns):
        q16, q50, q84 = np.percentile(out_dict[col], [16, 50, 84])
        quantiles[col] = (q16, q50, q84)

    axes = np.array(fig.axes).reshape(4, 4)
    for j, col in enumerate(out_dict[['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']].columns):
        ax = axes[j, j]   # Distribution marginale (diagonale)

        q16, q50, q84 = quantiles[col]
        minus = q50 - q16
        plus  = q84 - q50

        # Texte inféré (ligne 1)
        inferred_text = rf"${q50:.3f}^{{+{plus:.3f}}}_{{-{minus:.3f}}}$"

        # Injection (ligne 2)
        truth_val = truths_list2[j]
        truth_text = rf"{truth_val:.3f}" if truth_val is not None else "N/A"

        # Clear the automatic title
        ax.set_title("")

        # Add manual 2 lines: one black, one red
        ax.text(
            0.3, 1.03,
            inferred_text,
            ha='center', va='bottom',
            fontsize=13,
            transform=ax.transAxes,
            color='black'
        )
        ax.text(
            0.8, 1.03,
            truth_text,
            ha='center', va='bottom',
            fontsize=13,
            transform=ax.transAxes,
            color='red'
        )
    fig.suptitle(f"Lc {idx} component mass posterior samples (minus {i})", y=1.01, fontsize=20)
    fig.savefig(f"{BASE_DIR_i}/minus{i}/{idx}_mass_corner.png", bbox_inches='tight')
    print(f"Resampling and plotting for minus {i} completed for lc {idx}\Plot saved to {BASE_DIR_i}/minus{i}/{idx}_mass_corner_minus{i}.png")