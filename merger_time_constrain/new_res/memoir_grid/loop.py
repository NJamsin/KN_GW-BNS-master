import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os
import glob

# list all directory starting with grids in the base dir
BASE_DIR = "/home/stu_jamsin/memoir_grid"
grids = glob.glob(f"{BASE_DIR}/*")
# suppress logs and prior and sub_files directories
grids = [g for g in grids if not any(x in g for x in ["logs", "prior", "sub_files"])]
# keep directories containong a SUFFIX var
SUFFIX = "_copy" # change as needed
grids = [g for g in grids if SUFFIX in g]
#just keep directories
grids = [g for g in grids if os.path.isdir(g)]
print(f"Found grids: {grids}")

# create a .submit file to apply ts loop on each lc of each grid
def create_submit_file(grid_dir, model, prior, minus_pts, em_prior, out_dir):
    abs_grid_dir = os.path.join(BASE_DIR, grid_dir)
    file = f"""
getenv = True

executable = /home/stu_jamsin/.conda/envs/fiesta/bin/kn-ts-loop
arguments = --idx $(Process) --grid-dir {abs_grid_dir} --model {model} --svd-path /home/stu_jamsin/jamsin/NMMA/svdmodels --prior-file {prior} --minus-pts {minus_pts} --nlive 512 --resampling --EM-prior {em_prior}

output = /home/stu_jamsin/memoir_grid/logs/{grid_dir}_$(Process).out
error = /home/stu_jamsin/memoir_grid/logs/{grid_dir}_$(Process).err
log = /home/stu_jamsin/memoir_grid/logs/{grid_dir}_$(Process).log

request_cpus = 1
request_gpus = 0
request_memory = 2GB

queue 6
"""
    with open(f'{out_dir}/{grid_dir}_{SUFFIX}_ts_loop.submit', 'w') as f:
        f.write(file)
    return 0

for grid in grids:
    grid_name = os.path.basename(grid)
    print(f"Processing grid: {grid_name}")
    # extract model name from grid name
    model0 = grid_name.split("_")[0]
    if model0 == "bu26":
        model = "Bu2026_MLP"
    elif model0 == "bu19":
        model = "Bu2019lm"
    elif model0 == "ka17":
        model = "Ka2017"
    print(f"Model: {model}")
    # def prior file path
    prior = f"/home/stu_jamsin/memoir_grid/prior/{model0}.prior"
    em_prior = f"/home/stu_jamsin/memoir_grid/prior/{model0}_GW.prior"
    out_dir = f"/home/stu_jamsin/memoir_grid/sub_files"
    # create submit file
    create_submit_file(grid_name, model, prior, 8, em_prior, out_dir)
    # submit the file
    subprocess.run(f"condor_submit {out_dir}/{grid_name}_{SUFFIX}_ts_loop.submit", shell=True)