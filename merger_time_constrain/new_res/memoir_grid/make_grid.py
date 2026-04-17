import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os
import glob

# define the base directories

dir = ['/home/stu_jamsin/memoir_grid/ka17_real', '/home/stu_jamsin/memoir_grid/bu19_real', '/home/stu_jamsin/memoir_grid/bu26_real'] # change as needed
models = ['Ka2017', 'Bu2019lm', 'Bu2026_MLP'] # change as needed (keep the same order as the dirs)
# define the config to use
delay = 0.3
noise = 0.2
jitter = 0
cadence = 0.5 
obs_duration = 10
#get kn-make-grid executable path
kn_make_grid_path = "/home/stu_jamsin/.conda/envs/fiesta/bin/kn-make-grid"
# loop over the directories and create the grids
for d, m in zip(dir, models):
    print(f"Creating grid for {m}...")
    # create the grid using kn-make-grid
    subprocess.run(f"{kn_make_grid_path} --model {m} --delay {delay} --noise {noise} --jitter {jitter} --cadence {cadence} --obs-duration {obs_duration} --out-dir {d} --filters ps1::g ps1::r ps1::i ps1::z ps1::y --detection-limit ps1::g=24.7 ps1::r=24.2 ps1::i=23.8 ps1::z=23.2 ps1::y=22.3 --eos-path /home/stu_jamsin/peket/example_file/KN_grid/eos.dat", shell=True, cwd="/home/stu_jamsin/memoir_grid")