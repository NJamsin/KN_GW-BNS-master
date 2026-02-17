import pandas as pd
import numpy as np
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import corner
import bilby

idx = int(sys.argv[1]) # change as needed (can be adapted for condor if needed) int(sys.argv[1])
#idx = 0 # for test

BASE_DIR = f"/home/stu_jamsin/jamsin/grid_2_perday_noise"  # change as needed
if len(BASE_DIR) > 60:
    print("Warning: BASE_DIR path is quite long, which may cause issues with some software. Consider using a shorter path if you encounter errors related to file paths.")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

num_points = 9 # max number of removed points + 1 (to include the full data analysis as well)


# 1st create GW samples
gw_samples_file = f"{BASE_DIR}/GWsamples.dat"
if not os.path.exists(gw_samples_file):
    # Create GWsamples.dat if it does not exist (reuse code from gwsamples_generation.py)

    # load posterior file
    eos_post = np.loadtxt('/home/stu_jamsin/jamsin/add_files/posterior_probability.txt')

    npts = 150000 
    Neos = 5000
    nparams = 3

    ############# [mass1,    mass2,   DL] adjust as needed
    params_low =  [1., 1., 1.]
    params_high = [2.25,      2.25,     200.]

    # 1) create dummy EOS samples with eos_post from nature paper
    EOS_raw = np.arange(0, Neos)  # the gwem_resampling will add one to this
    EOS_samples = np.random.choice(EOS_raw, p=eos_post, size=npts, replace=True)

    # 2) generate samples for masses and distance
    mass_1 = np.random.uniform(params_low[0], params_high[0], size=npts)
    mass_2 = np.random.uniform(params_low[1], params_high[1], size=npts)
    mass_1, mass_2 = np.maximum(mass_1, mass_2), np.minimum(mass_1, mass_2)  
    mass_ratio = mass_2 / mass_1  # mass ratio q < 1 convention is used
    chirp_mass = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)
    lum_distance = np.random.uniform(params_low[2], params_high[2], size=npts)

    # 3) create pandas dataframe
    dataset = pd.DataFrame({'mass_1': mass_1, 'mass_2': mass_2, 'chirp_mass': chirp_mass, 'mass_ratio': mass_ratio, 'luminosity_distance': lum_distance, 'EOS': EOS_samples})

    # 4) save GWsamples.dat file
    dataset.to_csv(gw_samples_file, index=False, sep=' ')
    # ensure gw samples file is well formatted
    with open(gw_samples_file, 'r') as f:
        lines = f.readlines()
    with open(gw_samples_file, 'w') as f:
        for line in lines:
            if len(line.split()) == 6:
                f.write(line)

    print("GWsamples.dat created.")

# resamp
BASE_DIR_i = f"{BASE_DIR}/{idx}"  # change as needed

for i in range(num_points):
    print(f"Starting resampling for lc {idx} with minus {i}")
    # set up output directory and EM post file
    OUT_DIR = f"{BASE_DIR_i}/minus{i}/resamp"
    POST_FILE = f"{BASE_DIR_i}/minus{i}/minus{i}_{idx}_posterior_samples.dat"
    if not os.path.exists(POST_FILE):
        print(f"Posterior file {POST_FILE} not found. Please run the lightcurve analysis for lc {idx} for minus{i} before running the resampling.")
        continue
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    # run the resampling
    cmd_resamp = ["/home/stu_jamsin/.conda/envs/nmma_env/bin/gwem-resampling",
            "--outdir", OUT_DIR,
            "--GWsamples", gw_samples_file,
            "--GWprior", "/home/stu_jamsin/jamsin/NMMA/priors/GWBNS.prior",
            "--EMsamples", POST_FILE,
            "--EOSpath", "/home/stu_jamsin/jamsin/NMMA/EOS/15nsat_cse_uniform_R14/macro/",
            "--Neos", "5000",
            "--EMprior", "/home/stu_jamsin/jamsin/NMMA/priors/Bu2019lm_GW_200.prior",
            "--nlive", "2048"
        ]
    subprocess.run(cmd_resamp, check=True, cwd=BASE_DIR_i)

    # do the plot now
    samples = pd.read_csv(f"{OUT_DIR}/posterior_samples.dat", delim_whitespace=True)
    truth = pd.read_csv(f"{BASE_DIR_i}/true{idx}.csv")
    true_q = truth['mass_2'].values[0] / truth['mass_1'].values[0]
    true_chirp = bilby.gw.conversion.component_masses_to_chirp_mass(truth['mass_1'].values[0], truth['mass_2'].values[0])
    truths_list = [true_chirp, true_q, 4818] # adjust as needed for the true EOS index
    labels = ['$\mathcal{M}$', '$q$', 'EOS']
    samples['mass_1'] = samples['chirp_mass'] * (samples['mass_ratio']**(-3/5)) * ((1 + samples['mass_ratio'])**(1/5))
    samples['mass_2'] = samples['chirp_mass'] * (samples['mass_ratio']**(2/5)) * ((1 + samples['mass_ratio'])**(1/5))
    truths_list2 = [true_chirp, true_q, truth['mass_1'].values[0], truth['mass_2'].values[0]] # adjust as needed for the true EOS index

    fig = corner.corner(
    samples[['chirp_mass', 'mass_ratio', 'EOS', 'alpha', 'zeta']],
    labels=['$\mathcal{M}$', '$q$', 'EOS', r'$\alpha$', r'$\zeta$'],
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
    for j, col in enumerate(samples[['chirp_mass', 'mass_ratio', 'EOS', 'alpha', 'zeta']].columns):
        q16, q50, q84 = np.percentile(samples[col], [16, 50, 84])
        quantiles[col] = (q16, q50, q84)

    axes = np.array(fig.axes).reshape(5, 5)
    for j, col in enumerate(samples[['chirp_mass', 'mass_ratio', 'EOS']].columns):
        ax = axes[j, j]   # Distribution marginale (diagonale)

        q16, q50, q84 = quantiles[col]
        minus = q50 - q16
        plus  = q84 - q50

        # Texte inféré (ligne 1)
        if col == 'EOS':
            inferred_text = rf"${int(q50)}^{{+{int(plus)}}}_{{-{int(minus)}}}$"
        else:
            inferred_text = rf"${q50:.3f}^{{+{plus:.3f}}}_{{-{minus:.3f}}}$"

        # Injection (ligne 2)
        truth_val = truths_list[j]
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
    fig.suptitle(f"Lc {idx} resampling posterior samples (minus {i})", y=1.01, fontsize=20)
    fig.savefig(f"{BASE_DIR_i}/minus{i}/{idx}_resampling_corner.png", bbox_inches='tight')
    print(f"Resampling and plotting for minus {i} completed for lc {idx}\Plot saved to {BASE_DIR}/{idx}_resampling_corner_minus{i}.png")

    fig = corner.corner(
    samples[['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']],
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
    for j, col in enumerate(samples[['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']].columns):
        q16, q50, q84 = np.percentile(samples[col], [16, 50, 84])
        quantiles[col] = (q16, q50, q84)

    axes = np.array(fig.axes).reshape(4, 4)
    for j, col in enumerate(samples[['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']].columns):
        ax = axes[j, j]   # Distribution marginale (diagonale)

        q16, q50, q84 = quantiles[col]
        minus = q50 - q16
        plus  = q84 - q50

        # Texte inféré (ligne 1)
        if col == 'EOS':
            inferred_text = rf"${int(q50)}^{{+{int(plus)}}}_{{-{int(minus)}}}$"
        else:
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
    fig.suptitle(f"Lc {idx} resampling posterior samples (minus {i})", y=1.01, fontsize=20)
    fig.savefig(f"{BASE_DIR_i}/minus{i}/{idx}_mass_corner.png", bbox_inches='tight')
    print(f"Resampling and plotting for minus {i} completed for lc {idx}\Plot saved to {BASE_DIR}/{idx}_mass_corner_minus{i}.png")