import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = f"/home/stu_jamsin/jamsin/condor_synth_large/"
lower_bound = 0.16
higher_bound = 0.64 # for 1 sigma credible interval, but can be modified as needed

for (param, bounds) in zip(['luminosity_distance', 'KNphi', 'KNtheta', 'log10_mej_dyn', 'log10_mej_wind', 'timeshift'], [(0, 500), (0, 90), (0, 90), (-3, -1), (-3, -0.5), (-2, 1)]):
    fig = plt.figure(figsize=(8, 8))
    for i in range(25):
        out_file = f"{BASE_DIR}/{i}/{i}_posterior_samples.dat"
        if not os.path.exists(out_file): # skip if file does not exist
            continue
        df = pd.read_csv(out_file, delim_whitespace=True)
        D_L_median = df[param].median()
        D_L_16 = df[param].quantile(lower_bound)
        D_L_84 = df[param].quantile(higher_bound)
        # get inj val 
        inj_data = pd.read_csv(f"{BASE_DIR}/{i}/true{i}.csv")
        D_L_inj = inj_data[param].values[1]
        #print(f"Injection {i+1} for {param}: injected value = {D_L_inj}, recovered median = {D_L_median}")
        
        plt.errorbar(D_L_inj, D_L_median, yerr=[[D_L_median - D_L_16], [D_L_84 - D_L_median]], fmt='o', markersize=5, ecolor='red', color='blue', zorder=10)
    # plot identity line
    plt.plot([bounds[0], bounds[1]], [bounds[0], bounds[1]], 'k--', label='identity line', zorder=0, alpha=0.25)
    plt.xlabel(f'Injected {param}')
    plt.ylabel(f'Recovered {param}')
    plt.legend()
    plt.title(f'Injection-Recovery of {param} for Synthetic lightcurves data (1 $\sigma$ credible interval)')
    file_path = f'{BASE_DIR}/plots/{param}_injection_recovery_full.png'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)
    plt.show()

    # resampling 
for (param, bounds) in zip(['chirp_mass', 'mass_ratio'], [(1., 3.), (0.125, 1.)]):
    fig = plt.figure(figsize=(8, 8))
    for i in range(25):
        # Get M1 and M2 from chirp mass and mass ratio
        out_file = f"{BASE_DIR}/{i}/resampling/posterior_samples.dat"
        if not os.path.exists(out_file): # skip if file does not exist
            continue
        samples = pd.read_csv(out_file, delim_whitespace=True)

        D_L_median = samples[param].median()
        D_L_16 = samples[param].quantile(lower_bound)
        D_L_84 = samples[param].quantile(higher_bound)
        # get inj val 

        # get inj val 
        inj_data = pd.read_csv(f"{BASE_DIR}/{i}/true{i}.csv")
        m1 = inj_data["mass_1"].values[0]
        m2 = inj_data["mass_2"].values[0]
        inj_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        inj_q = m2 / m1
        D_L_inj = inj_chirp if param == 'chirp_mass' else inj_q
        #print(f"Injection {i+1} for {param}: injected value = {D_L_inj}, recovered median = {D_L_median}")
        plt.errorbar(D_L_inj, D_L_median, yerr=[[D_L_median - D_L_16], [D_L_84 - D_L_median]], fmt='o', markersize=5, color='blue', ecolor='red', zorder=10)
    # plot identity line
    plt.plot([bounds[0], bounds[1]], [bounds[0], bounds[1]], 'k--', label='identity line', zorder=0, alpha=0.25)
    plt.xlabel(f'Injected {param}')
    plt.ylabel(f'Recovered {param}')
    plt.legend()
    plt.title(f'Injection-Recovery of {param} for Synthetic lightcurves data (1 $\sigma$ credible interval)')
    file_path = f'{BASE_DIR}/plots/{param}_injection_recovery_full.png'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)
    plt.show()

# component masses
for (param, bounds) in zip(['mass_1', 'mass_2'], [(1., 3.), (1., 3.)]):
    fig = plt.figure(figsize=(8, 8))
    for i in range(25):
        # Get M1 and M2 from chirp mass and mass ratio
        out_file = f"{BASE_DIR}/{i}/resampling/posterior_samples.dat"
        if not os.path.exists(out_file): # skip if file does not exist
            continue
        samples = pd.read_csv(out_file, delim_whitespace=True)
        Mc = samples["chirp_mass"]
        q  = samples["mass_ratio"]

        m1 = Mc * (1 + q)**(1/5) * q**(-3/5)
        m2 = m1 * q

        samples["mass_1"] = m1
        samples["mass_2"] = m2

        D_L_median = samples[param].median()
        D_L_16 = samples[param].quantile(lower_bound)
        D_L_84 = samples[param].quantile(higher_bound)
        # get inj val (json file)
        inj_data = pd.read_csv(f"{BASE_DIR}/{i}/true{i}.csv")
        m1_inj = inj_data["mass_1"].values[0]
        m2_inj = inj_data["mass_2"].values[0]
        D_L_inj = m1_inj if param == 'mass_1' else m2_inj
        #print(f"Injection {i+1} for {param}: injected value = {D_L_inj}, recovered median = {D_L_median}")
        plt.errorbar(D_L_inj, D_L_median, yerr=[[D_L_median - D_L_16], [D_L_84 - D_L_median]], fmt='o', markersize=5, color='blue', ecolor='red', zorder=10)
    # plot identity line
    plt.plot([bounds[0], bounds[1]], [bounds[0], bounds[1]], 'k--', label='identity line', zorder=0, alpha=0.25)
    plt.xlabel(f'Injected {param}')
    plt.ylabel(f'Recovered {param}')
    plt.legend()
    plt.title(f'Injection-Recovery of {param} for Synthetic lightcurves data (1 $\sigma$ credible interval)')
    file_path = f'{BASE_DIR}/plots/{param}_injection_recovery_full.png'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)
    plt.show()