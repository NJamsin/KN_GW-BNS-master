import sys
import os
import subprocess

# Condor argument: injection index
idx = int(sys.argv[1]) 

print(f"--- Starting job for injection {idx} ---")

# Paths (Adjust if needed, use absolute paths to be sure)
BASE_DIR = "/home/liteul/memoir_code"
OUT_DIR = f"{BASE_DIR}/manual_inj/inj_{idx}"
INJ_FILE = f"{BASE_DIR}/manual_inj/injection.json"
GW_SAMPLES = f"{BASE_DIR}/manual_inj/GWsamples.dat"

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Lightcurve Analysis
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
# cwd is important for NMMA to find its relative folders
subprocess.run(cmd_lc, check=True, cwd=BASE_DIR) 

# 2. GWEM Resampling (Optional if you want to do it in the same job)
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