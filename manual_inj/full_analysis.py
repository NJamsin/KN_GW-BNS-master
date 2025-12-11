# do the full analysis for 1 injection
import pandas as pd
import numpy as np
import os
import subprocess
import sys

# define injection index
idx = 7 # change as needed (can be adapted for condor if needed) int(sys.argv[1]) 

# 0th step: base directory
BASE_DIR = "/home/liteul/memoir_code/manual_inj_test_nlive2048"

# 1st injection file
inj_file = f"{BASE_DIR}/injection.json"
if not os.path.exists(inj_file):
    # Create injection file if it does not exist (reuse code from create_injection.py)
    INJ_NUMBER = 50  # inj number
    OS_INJECTIONS = "/home/liteul/memoir_code/bns_O4_injections.dat" # NMMA injections sample file to get parameter ranges

    os.makedirs(BASE_DIR, exist_ok=True) # create directory if necessary

    # Create injections
    print("Generating parameters...")
    samples = pd.read_csv(OS_INJECTIONS, delim_whitespace=True)
    injection_df = pd.DataFrame(columns=['simulation_id', 'longitude', 'latitude', 'inclination', 'distance', 'mass1', 'mass2', 'spin1z', 'spin2z'])

    np.random.seed(42) # reproducibility

    for i in range(INJ_NUMBER):
        injection = [ # Adjust bounds as necessary
            int(i),  # simulation_id (IMPORTANT: start at 0 to match Condor Process)
            np.random.uniform(samples['longitude'].min(), samples['longitude'].max()),
            np.random.uniform(samples['latitude'].min(), samples['latitude'].max()),
            np.random.uniform(0, np.pi/2),
            np.random.uniform(0, 500),
            np.random.uniform(1.0, 3.0),
            np.random.uniform(1.0, 3.0),
            np.random.uniform(np.min(samples['spin1z']), np.max(samples['spin1z'])),
            np.random.uniform(np.min(samples['spin2z']), np.max(samples['spin2z']))
        ]
        injection_df.loc[i] = injection

    # Save temporary .dat file
    dat_file = os.path.join(BASE_DIR, "manual_inj.dat")
    injection_df.to_csv(dat_file, sep='\t', index=False)

    # 2. Convert to JSON with NMMA 
    print("Converting to JSON via nmma-create-injection...")
    cmd = [
        "nmma-create-injection",
        "--injection-file", dat_file,
        "--prior-file", "NMMA/priors/Bu2019lm500.prior", # change prior as needed
        "--eos-file", "NMMA/EOS/15nsat_cse_uniform_R14/macro/2098", # change EOS as needed
        "--binary-type", "BNS",
        "--n-injection", str(INJ_NUMBER),
        "--original-parameters",
        "--extension", "json",
        "--aligned-spin",
        "-f", os.path.join(BASE_DIR, "injection.json")
    ]

    subprocess.run(cmd, check=True, cwd="/home/liteul/memoir_code")
    print("injection.json created.")

# 1.5th step: ensure GWsamples.dat exists
gw_samples_file = f"{BASE_DIR}/GWsamples.dat"
if not os.path.exists(gw_samples_file):
    # Create GWsamples.dat if it does not exist (reuse code from gwsamples_generation.py)
    import bilby

    # load posterior file
    eos_post = np.loadtxt('/home/liteul/memoir_code/posterior_probability.txt')

    npts = 150000 
    Neos = 5000
    nparams = 3

    ############# [mass1,    mass2,   DL] adjust as needed
    params_low =  [1., 1., 0.]
    params_high = [3.,      3.,     500.]

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
    print("GWsamples.dat created.")

# 2nd step: run lightcurve analysis
inj_posterior_file = f"{BASE_DIR}/inj_{idx}/inj_{idx}_posterior_samples.dat"
OUT_DIR = f"{BASE_DIR}/inj_{idx}"
if os.path.exists(inj_posterior_file):
    print(f"Posterior file for injection {idx} already exists. Skipping analysis.")
else:
    INJ_FILE = f"{BASE_DIR}/injection.json"
    GW_SAMPLES = f"{BASE_DIR}/GWsamples.dat"

    os.makedirs(OUT_DIR, exist_ok=True)

    # Lightcurve Analysis (adjust parameters as needed)
    print(f"Starting lightcurve-analysis for {idx}...")
    cmd_lc = [
        "lightcurve-analysis",
        "--model", "Bu2019lm",
        "--svd-path", "NMMA/svdmodels",
        "--outdir", OUT_DIR,
        "--label", f"inj_{idx}",
        "--prior", "NMMA/priors/Bu2019lm500.prior",
        "--tmin", "0.1", "--tmax", "20", "--dt", "0.1",
        "--nlive", "2048", 
        "--Ebv-max", "0",
        "--injection", INJ_FILE,
        "--injection-num", str(idx), 
        "--injection-outfile", f"{OUT_DIR}/injection_out.csv",
        "--generation-seed", "42",
        "--filters", "ps1__g,ps1__r,ps1__i,ps1__z,sdssu,2massh",
        "--remove-nondetections",
        "--error-budget", "0.5"
    ]
    subprocess.run(cmd_lc, check=True, cwd=BASE_DIR) 

# 3rd step: run gwem-resampling
resampling_out = f"{BASE_DIR}/inj_{idx}/resampling/posterior_samples.dat"
if os.path.exists(resampling_out):
    print(f"Resampling output for injection {idx} already exists. Skipping resampling.")
else:
    print(f"Starting gwem-resampling for {idx}...")
    resamp_out = f"{OUT_DIR}/resampling"
    os.makedirs(resamp_out, exist_ok=True)

    # The file produced by lightcurve-analysis
    posterior_file = f"{OUT_DIR}/inj_{idx}_posterior_samples.dat"

    cmd_resamp = [
        "gwem-resampling",
        "--outdir", resamp_out,
        "--GWsamples", GW_SAMPLES,
        "--GWprior", "NMMA/priors/GWBNS2.prior",
        "--EMsamples", posterior_file,
        "--EOSpath", "NMMA/EOS/15nsat_cse_uniform_R14/macro/",
        "--Neos", "5000",
        "--EMprior", "NMMA/priors/Bu2019lm_GW_500.prior",
        "--nlive", "2048"
    ]
    subprocess.run(cmd_resamp, check=True, cwd=BASE_DIR)

    print(f"--- Job {idx} completed successfully ---")

print("Full analysis completed.")

# 4th step: plotting (based on reversingposterior.ipynb)
import matplotlib.pyplot as plt

# import necessary functions
from lal import MRSUN_SI
def dyn_ej(a = -9.3335, b = 114.17, d = -337.56, n = 1.5465, M1 = 1.4, R1 = 10, M2 = 1.4, R2 = 10):
    C1 = M1 / (R1 * 1e3 / MRSUN_SI)
    C2 = M2 / (R2 * 1e3 / MRSUN_SI)
    x = (a/C1 + b*(M2**n/M1**n) + d*C1)*M1 + (a/C2 + b*(M1**n/M2**n) + d*C2)*M2
    if x < 0:
        return 0
    else:
        return x/1000
    
def wind_ej(M1, M2, a0=-1.581, deltaa=-2.439, b0=-0.538, deltab=-0.406, c=0.953, d=0.0417, beta=3.91, qtrans=0.9, Mtov=1.97, R16=11.137): 
    r16 = R16 * 1e3 / MRSUN_SI
    Mtresh = (2.38 - 3.606 * (Mtov/r16))*Mtov
    q = M2/M1
    xsi = 0.5 * np.tanh(beta * (q - qtrans))
    a = a0 + deltaa * xsi
    b = b0 + deltab * xsi
    mwind = a * (1 + b * np.tanh( (c - (M1+M2)/Mtresh)/d ))
    mwind = np.maximum(-3.0, mwind)
    return mwind
    

# mass chirp
def chirp_mass(m1, m2):
    return (m1*m2)**(3/5) / (m1 + m2)**(1/5)

# mass ratio
def mass_ratio(m1, m2):
    return m2 / m1

# Load resampled posterior
samples = pd.read_csv(f"{BASE_DIR}/inj_{idx}/resampling/posterior_samples.dat", delim_whitespace=True)
EM_samples = pd.read_csv(f"{BASE_DIR}/inj_{idx}/inj_{idx}_posterior_samples.dat", delim_whitespace=True)

# import injection parameters and 
import json

inj_path = f"{BASE_DIR}/injection.json"

with open(inj_path) as f:
    inj = json.load(f)

inj = inj['injections']
inj = inj['content']

m1 = inj['mass_1'][idx-1] # idx-1 because injections start at 1 and python at 0 (only for test_nlive2048)
m2 = inj['mass_2'][idx-1]
print(f"Injection masses: m1 = {m1}, m2 = {m2}")

inj_chirp = m1**(3/5) * m2**(3/5) / (m1 + m2)**(1/5)
inj_q = m2 / m1

truth1_params = [m1, m2, inj_chirp, inj_q]
truth2_params = [inj['log10_mej_dyn'][0], inj['log10_mej_wind'][0]]
print(f"Injection ejecta: log10_mej_dyn = {truth2_params[0]}, log10_mej_wind = {truth2_params[1]}")
print(f'Injection chirp mass = {inj_chirp}, mass ratio = {inj_q}')

Mc = samples["chirp_mass"]
q  = samples["mass_ratio"]

m1 = Mc * (1 + q)**(1/5) * q**(-3/5)
m2 = m1 * q

samples["m1"] = m1
samples["m2"] = m2


import corner 

figure = corner.corner(
    samples[["m1", "m2", "chirp_mass", "mass_ratio"]],
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
    'luminosity_distance': inj['luminosity_distance'][idx-1], # adjust for 0-based index
    'KNphi': inj['KNphi'][idx-1],
    'KNtheta': inj['KNtheta'][idx-1],
    'log10_mej_dyn': inj['log10_mej_dyn'][idx-1],
    'log10_mej_wind': inj['log10_mej_wind'][idx-1],
    'timeshift': inj['timeshift'][idx-1]
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