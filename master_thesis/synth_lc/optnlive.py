import pandas as pd
import numpy as np
import os
import subprocess
import sys
import time
from utils import generate_synth_lc

idx = int(sys.argv[1]) # condor idx
nlive = 32 # starts at 32, will be modified in loop
BASE_DIR = f"/home/stu_jamsin/jamsin/optnlive/{idx}"
os.makedirs(BASE_DIR, exist_ok=True)
OUT_DIR = f"{BASE_DIR}/lc_analysis_{nlive}"
os.makedirs(OUT_DIR, exist_ok=True)
# Synth lc creation command
np.random.seed(6942067+idx) # different seed for each condor job
model_param = {
                    "KNphi": np.random.uniform(15,75),
                    "log10_mej_dyn": np.random.uniform(-3.5,-1),
                    "log10_mej_wind": np.random.uniform(-3.,-0.5),
                    "KNtheta": np.random.uniform(0, 90),
                    "luminosity_distance": np.random.uniform(10,200),
                    "timeshift": np.random.uniform(-2,0)
                    }
sample_times = np.arange(0.1, 20, 0.5) # 20 days, 2 per day cadence
filters_band = ['ps1__g', 'ps1__r', 'ps1__i', 'ps1__z', 'sdssu', '2massh'] # filters used for "observation"
print("Generating synthetic lightcurve...")
data_nmma_svd, trig = generate_synth_lc(
        model_name='Bu2019lm',
        model_param=model_param,
        filters_band=filters_band,
        sample_times=sample_times,
        noise_level=0.2,
        min_error_level=0.1,
        max_error_level=0.6,
        trigger_iso='2025-01-01T00:00:00',
        save=True,
        filename=f"{OUT_DIR}/data{idx}.dat"
)
DATA_FILE = f"{OUT_DIR}/data{idx}.dat"
'''
cmd = [
        "nmma-create-injection",
        "--prior-file", "NMMA/priors/Bu2019lm_inj_500.prior", # change prior as needed
        "--eos-file", "NMMA/EOS/15nsat_cse_uniform_R14/macro/4818.dat", # change EOS as needed
        "--binary-type", "BNS",
        "--n-injection", "1",
        "--extension", "json",
        "--aligned-spin",
        "--original-parameters", # NO ejecta computations from NS masses (injection only based on prior used)
        "-f", f"{BASE_DIR}/injection_{idx}.json"
    ]

OUT_DIR = f"{BASE_DIR}/"
os.makedirs(OUT_DIR, exist_ok=True)

# check injection file createdµ
INJ_FILE = f"{BASE_DIR}/injection_{idx}.json"
if not os.path.exists(INJ_FILE):
    subprocess.run(cmd, check=True)
'''

lc_path = "/home/stu_jamsin/.conda/envs/nmma_env/bin/lightcurve-analysis" # light-curve analysis path
# Lightcurve Analysis (adjust parameters as needed)
cmd_lc = [lc_path,
        "--model", "Bu2019lm",
        "--svd-path", "/home/stu_jamsin/jamsin/NMMA/svdmodels",
        "--outdir", OUT_DIR,
        "--label", f"nlive_{nlive}",
        "--prior", "/home/stu_jamsin/jamsin/NMMA/priors/Bu2019lm500.prior",
        "--tmin", "0.1", "--tmax", "20", "--dt", "0.1",
        "--nlive", str(nlive), 
        "--Ebv-max", "0",
        "--filters", "ps1__g,ps1__r,ps1__i,ps1__z,sdssu,2massh",
        "--remove-nondetections",
        "--data", DATA_FILE,
        "--error-budget", "0.5",
        "--sampler-kwargs", f"{{\"resume\": False}}"
]


# full workflow to be automated in a loop over nlive values

n_lives_values = [2**(5+i) for i in range(8)] # nlive = 32, 64, 128, 256, 512, 1024, 2048, 4096


for nlive in n_lives_values:
    # Lightcurve Analysis command setup
    OUT_DIR = f"{BASE_DIR}/lc_analysis_nlive_{nlive}"
    os.makedirs(OUT_DIR, exist_ok=True)
    # modify the command with the current nlive and output directory
    cmd_lc[cmd_lc.index("--nlive") + 1] = str(nlive)
    cmd_lc[cmd_lc.index("--outdir") + 1] = OUT_DIR
    cmd_lc[cmd_lc.index("--label") + 1] = f"nlive_{nlive}"
    # run the lightcurve analysis and time it
    start_time = time.time()
    subprocess.run(cmd_lc, check=True, cwd=BASE_DIR) 
    end_time = time.time()
    inner_time = end_time - start_time                                                          # TIME TAKEN FOR THIS NLIVE
    # compute the deviation from the injected values
    out_csv = f"{OUT_DIR}/nlive_{nlive}_posterior_samples.dat" # posterior samples output file
    df = pd.read_csv(out_csv, delim_whitespace=True, comment='#')
    params = []
    stds = []
    for param in ['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind']:
        temp = float(np.abs(1 - df[param].median() / model_param[param])) # deviation in percent from injected value
        params.append(temp)                                                                  # RECOVERED DEVIATION VALUE 
        temp = float(df[param].std() / np.abs(model_param[param]))
        stds.append(temp)                                                                    # STD DEV VALUE
    inner_recovery = np.median(params)
    inner_std = np.mean(stds)
    # save results to a dataframe
    data_dict = {
        "nlive" : nlive,
        "times" : inner_time,
        "recoveries" : inner_recovery,
        "recoveries std" : inner_std 
    }
    df = pd.DataFrame([data_dict])
    DATA_DIR = "/home/stu_jamsin/jamsin/optnlive"
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = f"{DATA_DIR}/out.dat"
    # Only write the header if the file does not exist yet
    header = not os.path.exists(out_file)

    df.to_csv(out_file, mode='a', header=header, index_label=" ")                              # APPEND TO OUTPUT FILE