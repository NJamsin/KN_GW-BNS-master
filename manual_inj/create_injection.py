import pandas as pd
import numpy as np
import os
import subprocess

# Configuration
INJ_NUMBER = 50  # inj number
BASE_DIR = "/home/liteul/memoir_code/manual_inj"
OS_INJECTIONS = "/home/liteul/memoir_code/bns_O4_injections.dat" # NMMA injections sample file to get parameter ranges

os.makedirs(BASE_DIR, exist_ok=True) # create directory if necessary

# Create injections
print("Generating parameters...")
samples = pd.read_csv(OS_INJECTIONS, delim_whitespace=True)
injection_df = pd.DataFrame(columns=['simulation_id', 'longitude', 'latitude', 'inclination', 'distance', 'mass1', 'mass2', 'spin1z', 'spin2z'])

np.random.seed(42) 

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