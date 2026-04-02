#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
from astropy.time import Time
import os
import pandas as pd
import json
from .utils import dyn_ej, wind_ej
from .utils import generate_synth_lc_fiesta
from .utils import generate_synth_lc_v2
from nmma.em.model import FiestaKilonovaModel
from nmma.em.model import SVDLightCurveModel
import argparse

# adapt the make_grid.py to Bu2026 with NMMA/FIESTA 
def main():
    '''
    Step 0: Get the parser and parse the arguments
    '''
    parser = argparse.ArgumentParser(description="Generate a pseudo-randomly sampled grid of synthetic lightcurves using NMMA/FIESTA with Bu2026 model. And save them as photometic .dat files formatted for nmma's lightcurve-analysis.")
    parser.add_argument("--out-dir", type=str, help="Base directory for the output files.")
    parser.add_argument("--model", type=str, default='Bu2026_MLP', help="Model name to use for generating synthetic lightcurves. (default: Bu2026_MLP). Currently supports 'Bu2026_MLP', 'Bu2019lm' and 'Ka2017'.")
    parser.add_argument("--num-lc", type=int, default=25, help="Number of lightcurves to generate.")
    parser.add_argument("--filters", nargs="+", help="Filters used for observation.")
    parser.add_argument("--eos-path", type=str, help="Path to the EOS file for the fitting formulae.")
    parser.add_argument("--noise-level", type=float, default=0.2, help="Noise level for the synthetic lightcurves.")
    parser.add_argument("--max-error-level", type=float, default=0.4, help="Max error level for the synthetic lightcurves.")
    parser.add_argument("--trigger-isot", type=str, default='2020-01-07T00:00:00', help="Trigger time for the synthetic lightcurves.")
    parser.add_argument("--cadence", type=float, default=0.5, help="Cadence in days for the synthetic lightcurves.")
    parser.add_argument("--obs-duration", type=float, default=7, help="Observation duration in days for the synthetic lightcurves.")
    parser.add_argument("--jitter", type=float, default=0.0, help="Jitter in days for the synthetic lightcurves.")
    parser.add_argument("--detection-limit", nargs='+', help="Detection limits for the synthetic data in the format: filter1:limit1 filter2:limit2 ... (e.g., --detect_limit ps1::g=24.7 ps1::r=24.2)")
    parser.add_argument("--param-ranges", nargs="+", help="Parameter ranges for the synthetic lightcurves in the format: param1=(min,max) param2=(min,max) ...")
    parser.add_argument("--save-json", default=None, action='store_true', help="Whether to save the times and magnitudes for each sample in a json file. (json files will be saved in the same directory as the .dat files)")

    args = parser.parse_args()

    BASE_DIR = os.path.abspath(args.out_dir)
    MODEL = args.model
    num_lc = args.num_lc
    filters_band = []

    for f in args.filters:
        filters_band.append(f)
    eos_path = args.eos_path
    noise_level = args.noise_level
    max_error_level = args.max_error_level
    trigger_isot = args.trigger_isot
    cadence = args.cadence
    obs_duration = args.obs_duration
    jitter = args.jitter
    detect_limit_dict = {}

    if args.detection_limit:
        for limit in args.detection_limit:
            filter_name, limit_value = limit.split('=')
            detect_limit_dict[filter_name] = float(limit_value)

    if args.param_ranges:
        param_bounds = {}
        for param in args.param_ranges:
            name, bounds = param.split('=')
            min_val, max_val = bounds.strip('()').split(',')
            param_bounds[name] = (float(min_val), float(max_val))
    else:
        param_bounds = { 
            "mass_1": (1.0, 2.25),
            "mass_2": (1.0, 2.25),
            "inclination_EM": (0, np.pi/2),
            "luminosity_distance": (10, 200), # common parameters for Bu26 19 and Ka17
            "vej_dyn": (0.12, 0.28), 
            "vej_wind": (0.05, 0.15),
            "Ye_dyn": (0.15, 0.35),
            "Ye_wind": (0.2, 0.4),
            "KNphi": (15, 75),
            "log10_Xlan": (-9, 1),
            "log10_vej": (-1.52, -0.53)
    }
    if len(BASE_DIR) > 60:
        print("Warning: BASE_DIR path is quite long, which may cause issues with some software. Consider using a shorter path if you encounter errors related to file paths.")
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    '''
    Step 1: generate the grid of parameters 
    '''
    # setup EOS for the fitting formula 
    eos = np.loadtxt(eos_path)
    r_eos = eos[:,0]  # radius in km
    M_eos = eos[:,1]  # mass in solar masses
    # interpolate to get radius at 1.4 solar masses
    R_16 = np.interp(1.6, M_eos, r_eos) # radius of 1.6 solar masses


    l_bounds = [bounds[0] for bounds in param_bounds.values()]
    u_bounds = [bounds[1] for bounds in param_bounds.values()]

    sampler = qmc.Halton(d=len(param_bounds), scramble=True) # pseudo random sampler for better coverage of the parameter space
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
                    "mass_1": p[0],
                    "mass_2": p[1],
                    "inclination_EM": p[2],
                    "luminosity_distance": p[3],
                    "log10_mej_dyn": np.log10(mej_dyn),
                    "log10_mej_wind": log10_mej_wind,
                    "vej_dyn": p[4],
                    "vej_wind": p[5],
                    "Ye_dyn": p[6],
                    "Ye_wind": p[7],
                    "KNphi": p[8],
                    "log10_Xlan": p[9],
                    "log10_vej": p[10],
                    "zeta": zeta
                }
                valid_samples.append(dic)
                if len(valid_samples) >= num_lc:
                    break
    '''
    Step 2: generate synthetic lightcurves for each parameter combination in the sample and create a huge plot with all the lightcurves for visual check
    '''

    fig, axs = plt.subplots(5, num_lc // 5, figsize=(10*(num_lc // 5), 6*(num_lc // 5)), sharex=True, sharey=True)
    for i, sample in enumerate(valid_samples):
        witness = num_lc // 5
        if MODEL == 'Bu2026_MLP':
            model_param = {
                "log10_mej_dyn": np.array(sample["log10_mej_dyn"]),
                "log10_mej_wind": np.array(sample["log10_mej_wind"]),
                "luminosity_distance": np.array(sample["luminosity_distance"]),
                "inclination_EM": np.array(sample["inclination_EM"]),
                "v_ej_dyn": np.array(sample["vej_dyn"]),
                "v_ej_wind": np.array(sample["vej_wind"]),
                "Ye_dyn": np.array(sample["Ye_dyn"]),
                "Ye_wind": np.array(sample["Ye_wind"]),
                "redshift": np.array(z_at_value(Planck18.luminosity_distance, sample["luminosity_distance"]*u.Mpc)),
                "timeshift": 0
            }
            # save model param to csv for reference
            OUT_DIR = f"{BASE_DIR}/{i}"
            os.makedirs(OUT_DIR, exist_ok=True)
            true_dic = {
                "log10_mej_dyn": model_param["log10_mej_dyn"],
                "log10_mej_wind": model_param["log10_mej_wind"],
                "luminosity_distance": model_param["luminosity_distance"],
                "mass_1": sample["mass_1"],
                "mass_2": sample["mass_2"],
                "zeta": sample["zeta"],
                "inclination_EM" : model_param["inclination_EM"],
                "v_ej_dyn": model_param["v_ej_dyn"],
                "v_ej_wind": model_param["v_ej_wind"],
                "Ye_dyn": model_param["Ye_dyn"],
                "Ye_wind": model_param["Ye_wind"],
                "KNphi": sample["KNphi"],
                "log10_Xlan": sample["log10_Xlan"],
                "log10_vej": sample["log10_vej"]
            }
            param_df = pd.DataFrame([true_dic])
            param_df.to_csv(f"{OUT_DIR}/true{i}.csv", index=False)   
            print(f"Generating synthetic lightcurve {i+1}/{num_lc}...")
            data_nmma_svd, trig = generate_synth_lc_fiesta(
                    model_name='Bu2026_MLP',
                    model_param=model_param,
                    filters_band=filters_band,
                    noise_level=noise_level,
                    max_error_level=max_error_level,
                    trigger_iso=trigger_isot,
                    pts_per_day=cadence,
                    obs_duration=obs_duration,
                    jitter=jitter,
                    save=True,
                    filename=f"{OUT_DIR}/data{i}.dat",
                    detection_limit_dict=detect_limit_dict
            )
        elif MODEL == 'Bu2019lm':
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
                "log10_mej": sample["log10_mej_dyn"] + np.log10(1+10**(sample["log10_mej_wind"] - sample["log10_mej_dyn"])), # total ejecta mass for Ka2017
                "inclination_EM": sample["inclination_EM"]
            }
        if MODEL != 'Bu2026_MLP': # for the other two models, we can use the same function as before
            # save model param to csv for reference
            OUT_DIR = f"{BASE_DIR}/{i}"
            os.makedirs(OUT_DIR, exist_ok=True)
            true_dic = {
                "log10_mej_dyn": sample["log10_mej_dyn"],
                "log10_mej_wind": sample["log10_mej_wind"],
                "luminosity_distance": model_param["luminosity_distance"],
                "mass_1": sample["mass_1"],
                "mass_2": sample["mass_2"],
                "zeta": sample["zeta"],
                "inclination_EM" : model_param["inclination_EM"],
                "v_ej_dyn": sample["vej_dyn"],
                "v_ej_wind": sample["vej_wind"],
                "Ye_dyn": sample["Ye_dyn"],
                "Ye_wind": sample["Ye_wind"],
                "KNphi": sample["KNphi"],
                "log10_Xlan": sample["log10_Xlan"],
                "log10_vej": sample["log10_vej"]
            }
            param_df = pd.DataFrame([true_dic])
            param_df.to_csv(f"{OUT_DIR}/true{i}.csv", index=False)   
            print(f"Generating synthetic lightcurve {i+1}/{num_lc}...")
            data_nmma_svd, trig = generate_synth_lc_v2(
                    model_name=MODEL,
                    model_param=model_param,
                    filters_band=filters_band,
                    noise_level=noise_level,
                    max_error_level=max_error_level,
                    trigger_iso=trigger_isot,
                    pts_per_day=cadence,
                    obs_duration=obs_duration,
                    jitter=jitter,
                    save=True,
                    filename=f"{OUT_DIR}/data{i}.dat",
                    detection_limit_dict=detect_limit_dict
            )

        if args.save_json:
            print(f"Saving times and magnitudes for sample {i+1} in a json file...")
        # save the times and magnitudes for this sample in a json file"
            # transform data_nmma_svd to a dict with filter names as keys and list of magnitudes as values, and times as a list of iso strings
            tt = (str(t) for t in data_nmma_svd[0])
            times = [Time(t, format='isot', scale='utc').mjd for t in tt]
            # Create a JSON-safe version of model_param by converting ndarrays to standard lists/floats
            json_safe_params = {
                key: (val.tolist() if isinstance(val, np.ndarray) else val) 
                for key, val in model_param.items()
            }
            magnitudes = {filter_name: data_nmma_svd[data_nmma_svd[1]==filter_name][2].values.tolist() for filter_name in filters_band}
            with open(f"{BASE_DIR}/{i}/data{i+1}.json", "w") as f:
                json.dump({"times": times, "magnitudes": magnitudes, "parameters": json_safe_params}, f, indent=4)

        # plot part 
        print(f"Plotting lightcurve {i+1}/{num_lc}...")
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
        ax.text(0.005, 0.99, f"LC {i}", transform=ax.transAxes, fontsize=20, verticalalignment='top')
        if row == num_lc // 5: # only set x label for the bottom row
            ax.set_xlabel('Time [days]')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.invert_yaxis() # invert y axis for magnitude
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/all_lightcurves.png", dpi=150)
    plt.close(fig)

    print(f"All synthetic lightcurves generated and saved in {BASE_DIR}.")

    return 0

if __name__ == "__main__":
    sys.exit(main())