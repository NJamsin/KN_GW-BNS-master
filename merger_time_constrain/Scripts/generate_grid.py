import pandas as pd
import numpy as np
import os
import subprocess
import sys
import matplotlib.pyplot as plt
from utils import generate_synth_lc_v2
from utils import wind_ej, dyn_ej
from scipy.stats import qmc

MODEL = 'Ka2017' # change as needed 

BASE_DIR = "/home/stu_jamsin/jamsin/grid_Ka_1per2day_noise"  # change as needed
if len(BASE_DIR) > 60:
    print("Warning: BASE_DIR path is quite long, which may cause issues with some software. Consider using a shorter path if you encounter errors related to file paths.")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

num_lc = 25 # please put a multiple of 5 here for the plot

filters_band = ['ps1__g', 'ps1__r', 'ps1__i', 'ps1__z', 'ps1__y'] # filters used for "observation" # LSST filters

# setup EOS for the fitting formula 
eos = np.loadtxt('/home/stu_jamsin/jamsin/NMMA/EOS/15nsat_cse_uniform_R14/macro/4818.dat')
r_eos = eos[:,0]  # radius in km
M_eos = eos[:,1]  # mass in solar masses
# interpolate to get radius at 1.4 solar masses
R_16 = np.interp(1.6, M_eos, r_eos) # radius of 1.6 solar masses

param_bounds = {
        "KNphi": (15, 75),
        "mass_1": (1.0, 2.25),
        "mass_2": (1.0, 2.25),
        "inclination_EM": (0, np.pi/2),
        "luminosity_distance": (10, 200),
        "log10_vej": (-1.52, -0.53), # value from Ka17 prior
        "log10_Xlan": (-9, -1) # value from Ka17 prior
}

l_bounds = [bounds[0] for bounds in param_bounds.values()]
u_bounds = [bounds[1] for bounds in param_bounds.values()]

sampler = qmc.Halton(d=len(param_bounds), scramble=True)
valid_samples = []

while len(valid_samples) < num_lc:
    # On génère un petit lot
    points = sampler.random(n=20)
    points_scaled = qmc.scale(points, l_bounds, u_bounds)
    
    for i, p in enumerate(points_scaled):
        r1 = np.interp(p[1], M_eos, r_eos)
        r2 = np.interp(p[2], M_eos, r_eos)
        mej_dyn = dyn_ej(M1=p[1], M2=p[2], R1=r1, R2=r2)
        zeta = np.random.uniform(0.01, 1)
        log10_mej_wind = wind_ej(M1=p[1], M2=p[2], Mtov=np.max(M_eos), R16=R_16) + np.log10(zeta) # consider between 1 and 100% of the disk mass as wind ejecta
        if mej_dyn > 0 and p[1] > p[2]:
            M_ej_tot = mej_dyn + 10**log10_mej_wind
            dic = {
                "KNphi": p[0],
                "mass_1": p[1],
                "mass_2": p[2],
                "inclination_EM": p[3],
                "luminosity_distance": p[4],
                "log10_mej_dyn": np.log10(mej_dyn),
                "log10_mej_wind": log10_mej_wind,
                "log10_mej": np.log10(M_ej_tot),
                "log10_vej": p[5],
                "log10_Xlan": p[6],
                "zeta": zeta
            }
            valid_samples.append(dic)
            if len(valid_samples) >= num_lc:
                break

# 2nd: generate synthetic lightcurves for each parameter combination in the sample and create a huge plot with all the lightcurves for visual check
fig, axs = plt.subplots(5, num_lc // 5, figsize=(10*(num_lc // 5), 6*(num_lc // 5)), sharex=True, sharey=True)
for i, sample in enumerate(valid_samples):
    witness = num_lc // 5
    if MODEL == 'Bu2019lm':
        model_param = {
            "KNphi": sample["KNphi"],
            "log10_mej_dyn": sample["log10_mej_dyn"],
            "log10_mej_wind": sample["log10_mej_wind"],
            "inclination_EM": sample["inclination_EM"],
            "luminosity_distance": sample["luminosity_distance"],
            "timeshift": 0
        }
    elif MODEL == 'Ka2017':
        model_param = {
            "luminosity_distance": sample["luminosity_distance"],
            "log10_vej": sample["log10_vej"],
            "log10_Xlan": sample["log10_Xlan"],
            "timeshift": 0,
            "log10_mej": sample["log10_mej"],
            "inclination_EM": sample["inclination_EM"]
        }
    # save model param to csv for reference
    OUT_DIR = f"{BASE_DIR}/{i}"
    os.makedirs(OUT_DIR, exist_ok=True)
    true_dic = {
        "KNphi": sample["KNphi"],
        "inclination_EM": sample["inclination_EM"],
        "log10_mej_dyn": sample["log10_mej_dyn"],
        "log10_mej_wind": sample["log10_mej_wind"],
        "log10_mej": sample["log10_mej"],
        "log10_vej": sample["log10_vej"],
        "log10_Xlan": sample["log10_Xlan"],
        "luminosity_distance": sample["luminosity_distance"],
        "mass_1": sample["mass_1"],
        "mass_2": sample["mass_2"],
        "zeta": sample["zeta"]
    }
    param_df = pd.DataFrame([true_dic])
    param_df.to_csv(f"{OUT_DIR}/true{i}.csv", index=False)   
    print(f"Generating synthetic lightcurve {i+1}/{num_lc}...")
    data_nmma_svd, trig = generate_synth_lc_v2(
            model_name=MODEL,
            model_param=model_param,
            filters_band=filters_band,
            noise_level=0.2,
            max_error_level=0.4,
            trigger_iso='2025-01-01T00:00:00',
            pts_per_day=0.5,
            obs_duration=15,
            jitter=0.,
            save=True,
            filename=f"{OUT_DIR}/data{i}.dat",
            detection_limit_dict={'ps1__g':24.7, 'ps1__r':24.2, 'ps1__i':23.8, 'ps1__z':23.2, 'ps1__y':22.3}
    )
    # plot part 
    row = i // (num_lc // 5)
    col = i % (num_lc // 5)
    ax = axs[row, col]
    for band in data_nmma_svd[1].unique():
        band_df = data_nmma_svd[data_nmma_svd[1]==band]
        times = pd.to_datetime(band_df[0].values)
        ax.errorbar(times, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
    if col == 0:
        ax.set_ylabel('Magnitude')  
    ax.legend(loc='upper right')
    if MODEL == 'Bu2019lm':
        txt = f"$\\phi$: {model_param['KNphi']:.1f}\n$log_{{10}} M_{{dyn}}$: {model_param['log10_mej_dyn']:.2f}\n$log_{{10}} M_{{wind}}$: {model_param['log10_mej_wind']:.2f}\n$\\iota$: {model_param['inclination_EM']:.1f}\n$D_L$: {model_param['luminosity_distance']:.1f} Mpc\n $M_1$: {sample['mass_1']:.2f} $M_\\odot$\n$M_2$: {sample['mass_2']:.2f} $M_\\odot$"
    elif MODEL == 'Ka2017':
        txt = f"$\\log_{{10}} M_{{ej}}$: {model_param['log10_mej']:.2f}\n$\\log_{{10}} V_{{ej}}$: {model_param['log10_vej']:.2f}\n$log_{{10}} X_{{lan}}$: {model_param['log10_Xlan']:.2f}\n$D_L$: {model_param['luminosity_distance']:.1f} Mpc\n $M_1$: {sample['mass_1']:.2f} $M_\\odot$\n$M_2$: {sample['mass_2']:.2f} $M_\\odot$\n$\\iota$: {model_param['inclination_EM']:.1f}"
    ax.text(0.7, 0.99, txt, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.005, 0.99, f"LC {i}", transform=ax.transAxes, fontsize=20, verticalalignment='top')
    if row == num_lc // 5: # only set x label for the bottom row
        ax.set_xlabel('Time [days]')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
ax.invert_yaxis() # invert y axis for magnitude
plt.tight_layout()
plt.savefig(f"{BASE_DIR}/all_lightcurves.png")


    