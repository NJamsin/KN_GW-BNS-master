# do the full analysis for 1 injection
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import generate_synth_lc
from utils import wind_ej, dyn_ej

'''
!!! DO NOT run this several times for the same BASE_DIR as the true model parameters will be overwritten but not the lightcurve analysis!!!
'''

# define injection index
idx = int(sys.argv[1]) # change as needed (can be adapted for condor if needed) int(sys.argv[1]) 
#idx = 5  # example injection index
# 0th step: base directory
BASE_DIR = "/home/stu_jamsin/jamsin/testezu"  # change as needed

# 1st synth lc 
synth_data = f"{BASE_DIR}/{idx}/data{idx}.dat"
al_done = False

os.makedirs(f"{BASE_DIR}/{idx}", exist_ok=True) # create directory if necessary
# Create injections
DATA_path = f"{BASE_DIR}/{idx}/true{idx}.csv"
if os.path.exists(DATA_path):
    print(f"Synthetic data for injection {idx} already exists. Skipping generation.")
    al_done = True
else:
    print("Generating parameters...")

    # import best fit eos to get radius for fitting formulae
    eos = np.loadtxt('/home/stu_jamsin/jamsin/NMMA/EOS/15nsat_cse_uniform_R14/macro/4818.dat')
    r_eos = eos[:,0]  # radius in km
    M_eos = eos[:,1]  # mass in solar masses
    # interpolate to get radius at 1.4 solar masses
    R_16 = np.interp(1.6, M_eos, r_eos) # radius of 1.6 solar masses

    # fix a seed for reproducibility
    #np.random.seed(42 + idx)
    mej_dyn = -1
    # draw random parameters from prior
    while mej_dyn <= 0:
        mass_1 = np.random.uniform(1.2, np.max(M_eos))  # solar masses
        mass_2 = np.random.uniform(1.19, mass_1)  # solar masses
        r1 = np.interp(mass_1, M_eos, r_eos)
        r2 = np.interp(mass_2, M_eos, r_eos)
        mej_dyn = dyn_ej(M1=mass_1, M2=mass_2, R1=r1, R2=r2)
    log10_mej_dyn = np.log10(mej_dyn)
    log10_mej_wind = wind_ej(M1=mass_1, M2=mass_2, Mtov=np.max(M_eos), R16=R_16) + np.log10(0.3) # assume 30% of disk mass goes into wind

    model_param = {
                        "KNphi": np.random.uniform(15,75),
                        "log10_mej_dyn": log10_mej_dyn,
                        "log10_mej_wind": log10_mej_wind,
                        "KNtheta": np.random.uniform(0, 90),
                        "luminosity_distance": np.random.uniform(10,200),
                        "timeshift": np.random.uniform(-1,0)
                        }
    # save model param to csv for reference
    param_df = pd.DataFrame([{"mass_1": mass_1, "mass_2": mass_2}, model_param])
    param_df.to_csv(f"{BASE_DIR}/{idx}/true{idx}.csv", index=False)   

    # generate sample times
    sample_times = np.arange(0.1, 10, 0.5) # 10 days, 2 per day cadence
    for i in range(len(sample_times)):
        sample_times[i]+= np.random.uniform(-0.2, 0.2)  # add some jitter to sample times
    filters_band = ['ps1__g', 'ps1__r', 'ps1__i', 'ps1__z'] # filters used for "observation"
    print("Generating synthetic lightcurve...")
    data_nmma_svd, trig = generate_synth_lc(
            model_name='Bu2019lm',
            model_param=model_param,
            filters_band=filters_band,
            sample_times=sample_times,
            noise_level=0.2,
            min_error_level=0.03,
            max_error_level=0.4,
            trigger_iso='2025-01-01T00:00:00',
            save=True,
            filename=f"{BASE_DIR}/{idx}/data{idx}.dat",
            detection_limit_dict={'ps1__g':26, 'ps1__r':26, 'ps1__i':26, 'ps1__z':26}
    )

# 2nd step: run lightcurve analysis
inj_posterior_file = f"{BASE_DIR}/{idx}/{idx}_posterior_samples.dat"
OUT_DIR = f"{BASE_DIR}/{idx}"
if os.path.exists(inj_posterior_file):
    print(f"Posterior file for {idx} already exists. Skipping analysis.")
else:
    DATA_FILE = f"{BASE_DIR}/{idx}/data{idx}.dat"
    import json
    os.makedirs(OUT_DIR, exist_ok=True)

    # Lightcurve Analysis (adjust parameters as needed)
    detection_limit_dict = {'ps1__g':26, 'ps1__r':26, 'ps1__i':26, 'ps1__z':26}
    print(f"Starting lightcurve-analysis for {idx}...")
    cmd_lc = ["/home/stu_jamsin/.conda/envs/nmma_env/bin/lightcurve-analysis",
        "--model", "Bu2019lm",
        "--svd-path", "/home/stu_jamsin/jamsin/NMMA/svdmodels",
        "--outdir", OUT_DIR,
        "--label", f"{idx}",
        "--prior", "/home/stu_jamsin/jamsin/NMMA/priors/Bu2019lm500.prior",
        "--nlive", "2048", 
        "--Ebv-max", "0",
        "--filters", "ps1__g,ps1__r,ps1__i,ps1__z",
        "--remove-nondetections",
        "--data", DATA_FILE,
        "--error-budget", "0.5",
        "--plot", 
        "--ylim", "26,17",
        #"--detection-limit", json.dumps(detection_limit_dict)
       ]
    subprocess.run(cmd_lc, check=True, cwd=BASE_DIR) 

# 3rd step: analytical reversal
from utils import ejecta_to_mass
from utils import ejecta_plot_v4
from utils import compare_mass_sets

EM_samples = pd.read_csv(f"{BASE_DIR}/{idx}/{idx}_posterior_samples.dat", delim_whitespace=True)
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
fig_mass.savefig(f"{OUT_DIR}/inj_{idx}_mass_pairs_from_ejecta.png", bbox_inches='tight')
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
plt.title('Mass pairs corresponding to original Bulla dataset ejecta ranges')
plt.legend()
plt.grid(True)
plt.savefig(f"{OUT_DIR}/inj_{idx}_coherent_mass_pair.png", bbox_inches='tight')

# 4th step: plotting (based on reversingposterior.ipynb)

# mass chirp
def chirp_mass(m1, m2):
    return (m1*m2)**(3/5) / (m1 + m2)**(1/5)

# mass ratio
def mass_ratio(m1, m2):
    return m2 / m1

# Load resampled posterior

# import synth lc parameters 
if al_done:
    print(f"Loading existing parameters for injection {idx}...")
    param_df = pd.read_csv(f"{BASE_DIR}/{idx}/true{idx}.csv")
    m1 = param_df["mass_1"].values[0] # 0 for masses and 1 model params
    m2 = param_df["mass_2"].values[0]
    model_param = {
                        "KNphi": param_df["KNphi"].values[1],
                        "log10_mej_dyn": param_df["log10_mej_dyn"].values[1],
                        "log10_mej_wind": param_df["log10_mej_wind"].values[1],
                        "KNtheta": param_df["KNtheta"].values[1],
                        "luminosity_distance": param_df["luminosity_distance"].values[1],
                        "timeshift": param_df["timeshift"].values[1]
                        }
else:
    m1 = mass_1
    m2 = mass_2

print(f"True masses: m1 = {m1}, m2 = {m2}")

inj_chirp = m1**(3/5) * m2**(3/5) / (m1 + m2)**(1/5)
inj_q = m2 / m1

truth1_params = [m1, m2, inj_chirp, inj_q]
truth2_params = [model_param["log10_mej_dyn"], model_param["log10_mej_wind"]]
print(f"True ejecta: log10_mej_dyn = {truth2_params[0]}, log10_mej_wind = {truth2_params[1]} (via fitting formulae)")
print(f'True chirp mass = {inj_chirp}, mass ratio = {inj_q}')

samples_m1 = common_masses[:,0] 
samples_m2 = common_masses[:,1]

Mc = chirp_mass(samples_m1, samples_m2)
q  = samples_m2 / samples_m1

import corner 

figure = corner.corner(
    np.array([samples_m1, samples_m2, Mc, q]).T,
    labels=["$m_1$", "$m_2$", "$\mathcal{M}$", "$q$"],
    truths=truth1_params,
    truth_color='red',
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
for i, col in enumerate(["m1", "m2", "chirp_mass", "mass_ratio"]):
    q16, q50, q84 = np.percentile([samples_m1, samples_m2, Mc, q][i], [16, 50, 84])
    quantiles[col] = (q16, q50, q84)

axes = np.array(figure.axes).reshape(len(["m1", "m2", "chirp_mass", "mass_ratio"]), len(["m1", "m2", "chirp_mass", "mass_ratio"]))

for i, col in enumerate(["m1", "m2", "chirp_mass", "mass_ratio"]):
    ax = axes[i, i]   # Distribution marginale (diagonale)

    q16, q50, q84 = quantiles[col]
    minus = q50 - q16
    plus  = q84 - q50

    # Texte inféré (ligne 1)
    inferred_text = rf"${q50:.3f}^{{+{plus:.3f}}}_{{-{minus:.3f}}}$"

    # Injection (ligne 2)
    truth_val = truth1_params[i]
    truth_text = rf"{truth_val:.3f}"

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

# set the figure suptitle and then show the plot
figure.suptitle(f"Injection {idx} posterior samples", y=1.05, fontsize=20)
figure.savefig(f"{OUT_DIR}/inj_{idx}_qchirp_to_masses.png", bbox_inches='tight')

# EM corner plot
col_labels = {
    'luminosity_distance': '$D_L$ [Mpc]',
    'KNphi': r'$\phi$ [deg]',
    'KNtheta': r'$\theta$ [deg]',
    'log10_mej_dyn': r'$\log_{10} M_{ej,dyn}$ [$M_\odot$]',
    'log10_mej_wind': r'$\log_{10} M_{ej,wind}$ [$M_\odot$]',
    'timeshift': r'$t_{0}$ [days]'
}
true_inj = {
    'luminosity_distance': model_param["luminosity_distance"], # adjust for 0-based index
    'KNphi': model_param["KNphi"],
    'KNtheta': model_param["KNtheta"],
    'log10_mej_dyn': model_param["log10_mej_dyn"],
    'log10_mej_wind': model_param["log10_mej_wind"],
    'timeshift': model_param["timeshift"]
}
truths_list = [
    true_inj['luminosity_distance'],
    true_inj['KNphi'],
    true_inj['KNtheta'],
    true_inj['log10_mej_dyn'],
    true_inj['log10_mej_wind'],
    true_inj['timeshift']
]

cols_to_plot = ['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind', 'timeshift']

# ensure columns exist in EM_samples
cols_available = [c for c in cols_to_plot if c in EM_samples.columns]
if len(cols_available) < len(cols_to_plot):
    missing = set(cols_to_plot) - set(cols_available)
    print(f"Warning: missing columns in EM_samples: {missing}")

# build labels list in same order
labels_list = [col_labels.get(c, c) for c in cols_available]

# build truths list (handle missing keys in true_inj)
truths_list = []
for c in cols_available:
    v = true_inj.get(c)
    if v is None:
        truths_list.append(None)
    elif isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
        truths_list.append(float(v[0]))
    else:
        truths_list.append(float(v))

fig = corner.corner(
    EM_samples[cols_available],  # double brackets
    truths=truths_list,
    labels=labels_list,
    truth_color='red',
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
for i, col in enumerate(cols_available):
    q16, q50, q84 = np.percentile(EM_samples[col], [16, 50, 84])
    quantiles[col] = (q16, q50, q84)

axes = np.array(fig.axes).reshape(len(cols_available), len(cols_available))

for i, col in enumerate(cols_available):
    ax = axes[i, i]   # Distribution marginale (diagonale)

    q16, q50, q84 = quantiles[col]
    minus = q50 - q16
    plus  = q84 - q50

    # Texte inféré (ligne 1)
    inferred_text = rf"${q50:.3f}^{{+{plus:.3f}}}_{{-{minus:.3f}}}$"

    # Injection (ligne 2)
    truth_val = truths_list[i]
    truth_text = rf"{truth_val:.3f}"

    # Clear the automatic title
    ax.set_title("")

    # Add manual 2 lines: one black, one red
    ax.text(
        0.25, 1.03,
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

fig.suptitle(f"Injection {idx} EM posterior samples", y=0.99, fontsize=20)
fig.savefig(f"{OUT_DIR}/inj_{idx}_EM_corner.png", bbox_inches='tight')