import sys
import os
import subprocess

# Condor argument: injection index
idx = int(sys.argv[1]) 

print(f"--- Starting job for injection {idx+1} ---")

# Paths (Adjust if needed, use absolute paths to be sure)
BASE_DIR = "/home/liteul/memoir_code"
OUT_DIR = f"{BASE_DIR}/manual_inj_test_nlive2048/inj_{idx+1}"
INJ_FILE = f"{BASE_DIR}/manual_inj_test_nlive2048/injection.json"
GW_SAMPLES = f"{BASE_DIR}/manual_inj_test_nlive2048/GWsamples.dat"

os.makedirs(OUT_DIR, exist_ok=True)

# The file produced by lightcurve-analysis
posterior_file = f"{OUT_DIR}/inj_{idx+1}_posterior_samples.dat"
resamp_out = f"{OUT_DIR}/resampling"
os.makedirs(resamp_out, exist_ok=True)

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

print(f"--- Job {idx+1} completed successfully ---")