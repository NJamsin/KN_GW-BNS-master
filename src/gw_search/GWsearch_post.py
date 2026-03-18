#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import glob
import os
import yaml
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
import argparse


def main():
    # Load configuration
    parser = argparse.ArgumentParser(description="PyCBC Pipeline Step")
    parser.add_argument("config", help="Path to the config file")
    parser.add_argument("--expected-trigger-time", default=None, help="Expected trigger time to be searched, in gps format. Used only in the final trigger distribution plot.")
    parser.add_argument("--plot-spectrogram", default=False, action="store_true", help="If true, will generate a spectrogram plot for the top trigger in the post-processing step. This can be useful for visually inspecting the trigger.")
    parser.add_argument("--spectrogram-range", default="0,15", help="vmin and vmax for the spectrogram plot. Only used if --plot-spectrogram is set.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    '''
    Step 0: Load var from the config file
    '''
    BASE_DIR = os.path.abspath(config['Directory']['BASE_DIR'])
    SUFFIX = config['Directory']['run_name']
    KN_detection_date = config['KN_data']['first_detection']
    KN_EM_post = config['KN_data']['EM_post_file']

    '''
    Step 1: Scan the output directory for all trigger files and collect triggers with their ranking statistic, time, and other relevant info.
    '''
    trigger_dir = f'{BASE_DIR}/out'
    trigger_files = glob.glob(os.path.join(trigger_dir, f'{SUFFIX}triggers_*.hdf'))
    # Filter strictly to the parent directory files
    trigger_files = [f for f in trigger_files if os.path.dirname(f) == trigger_dir.rstrip('/')]
    trigger_files.sort()

    # List to store all triggers from all files
    all_triggers = []

    print(f"Scanning {len(trigger_files)} files in {trigger_dir}...")

    for tfile in trigger_files:
        try:
            with h5py.File(tfile, 'r') as f:
                if 'network' not in f:
                    continue
                
                # Use reweighted SNR as ranking statistic
                if 'reweighted_snr' in f['network']:
                    ranking_stat_name = 'reweighted_snr'
                    snrs = f['network']['reweighted_snr'][:]
                elif 'coherent_snr' in f['network']:
                    ranking_stat_name = 'coherent_snr'
                    snrs = f['network']['coherent_snr'][:]
                else:
                    continue
                
                if len(snrs) == 0:
                    continue
                
                # Get other datasets
                coh_snrs = f['network']['coherent_snr'][:] if 'coherent_snr' in f['network'] else np.zeros_like(snrs)
                
                # Time requires careful handling
                # If end_time_gc is available, use it (geocentric time) (normally it is always availiable with pycbc_multi_inspiral outputs)
                if 'end_time_gc' in f['network']:
                    times = f['network']['end_time_gc'][:]
                else:
                    # Fallback: recreate from H1 event ids
                    if 'H1_event_id' in f['network'] and 'H1' in f:
                        h1_indices = f['network']['H1_event_id'][:]
                        # We need to load H1 end times.
                        h1_times = f['H1']['end_time'][:]
                        times = h1_times[h1_indices]
                    else:
                        times = np.zeros_like(snrs)

                # Store triggers above some threshold to save memory, or just store all if not too many
                
                # Get indices of top 10
                if len(snrs) > 10:
                    top_indices = np.argsort(snrs)[-10:]
                else:
                    top_indices = np.arange(len(snrs))
                    
                for idx in top_indices:
                    all_triggers.append({
                        'rank_stat': snrs[idx],
                        'coherent_snr': coh_snrs[idx],
                        'time': times[idx],
                        'file': tfile,
                        'idx': idx # allows to find the trigger in the file later if needed
                    })

        except Exception as e:
            print(f"Error processing {tfile}: {e}")

    # Sort all collected triggers by ranking statistic
    all_triggers.sort(key=lambda x: x['rank_stat'], reverse=True)

    print(f"\nCollected {len(all_triggers)} candidate triggers.")
    print(f"Ranking Statistic used: {ranking_stat_name if 'ranking_stat_name' in locals() else 'Unknown'}")

    # Print top 5 candidates + save them to a text file
    OUT_file = os.path.join(f"{BASE_DIR}/out", f'{SUFFIX}_top_candidates.txt')
    print("\n--- Top 5 Candidate Signals ---")
    for i, trig in enumerate(all_triggers[:5]):
        print(f"{i+1}. Rank Stat: {trig['rank_stat']:.4f} | Coherent SNR: {trig['coherent_snr']:.4f} | Time: {trig['time']:.4f} | File: {os.path.basename(trig['file'])}")

    with open(OUT_file, 'w') as f:
        for i, trig in enumerate(all_triggers[:5]):
            f.write(f"{i+1}. Rank Stat: {trig['rank_stat']:.4f} | Coherent SNR: {trig['coherent_snr']:.4f} | Time: {trig['time']:.4f} | File: {os.path.basename(trig['file'])}\n")

    '''
    Step 2: Plot the distribution of the ranking statistic (e.g., reweighted SNR) across time, and mark the best candidate and the known GW170817 merger time.
    '''
    print("\nGenerating plot for trigger distribution")
    # Get data start time
    # Load the EM posterior samples
    EM_samp = pd.read_csv(KN_EM_post, delimiter=' ', dtype=np.float32)

    # 1. convert the first detection time to mjd time
    KN_t0 = Time(KN_detection_date, format='isot', scale='utc').mjd

    # 2. Calculate 1 sigma interval (16th and 84th percentiles) of the timeshift samples
    p16, p50, p84 = np.percentile(EM_samp['timeshift'], [15.865, 50, 84.135])

    # 3. Define the search window around the median timeshift, extending to the 1sigma interval
    t_start = KN_t0 + p50 + p16
    DATA_START_TIME = Time(t_start, format='mjd').gps - 16 # start of the window -16s for padding

    # Collect plot data 
    plot_times = [t['time'] for t in all_triggers]
    plot_ranks = [t['rank_stat'] for t in all_triggers]
    plot_coh = [t['coherent_snr'] for t in all_triggers]

    # Find best candidate
    best_cand = all_triggers[0]  # Since we sorted them
    print(f"Best Candidate Time: {best_cand['time']:.2f} (GPS)")

    plt.figure(figsize=(10, 6))
    plt.scatter(plot_times, plot_ranks, alpha=0.6, label='Reweighted SNR', color='blue', s=20)
    plt.scatter([best_cand['time']], [best_cand['rank_stat']], color='red', s=150, marker='*', label=f'Best Candidate (SNR={best_cand["rank_stat"]:.2f})')

    if args.expected_trigger_time:
        expected_time = float(args.expected_trigger_time)
        if min(plot_times) < expected_time < max(plot_times):
            plt.axvline(x=expected_time, color='green', linestyle='--', label=f'Expected Trigger Time (GPS={expected_time})', zorder=0)
    plt.xlabel('GPS Time')
    plt.ylabel('Reweighted SNR')
    plt.title(f'Trigger Distribution (Top {len(plot_times)})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(f"{BASE_DIR}/plots", f'{SUFFIX}_trigger_distribution.png'))

    '''
    Strain plot for the best candidate (if the flag is set)
    '''
    if args.plot_spectrogram:
        print("\nGenerating spectrogram for the best candidate")
        best_time = best_cand['time']
        window = (-0.75, 0.15) # seconds around the trigger time
        # load the strain data around the best candidate time
        H1_file = f"{BASE_DIR}/data/{SUFFIX}_H1_READY.gwf"
        L1_file = f"{BASE_DIR}/data/{SUFFIX}_L1_READY.gwf"
        from gwpy.timeseries import TimeSeries
        H1_strain = TimeSeries.read(H1_file, channel='H1:GWOSC-4KHZ_R1_STRAIN')
        L1_strain = TimeSeries.read(L1_file, channel='L1:GWOSC-4KHZ_R1_STRAIN')
        # prepare the data
        for ts, label in zip([H1_strain, L1_strain], ['H1', 'L1']):
            padded_data = ts.crop(best_time -30, best_time + 10) # get 30s-10s of data around the trigger time
            q_scan = padded_data.q_transform(outseg=(best_time - 7, best_time + 0.5), frange=(30, 512), qrange=(80, 120))
            # 3. Plot
            vmin = float(args.spectrogram_range.split(',')[0])
            vmax = float(args.spectrogram_range.split(',')[1])
            fig = q_scan.plot(figsize=(12, 5), vmin=vmin, vmax=vmax)
            ax = fig.gca()
            ax.set_epoch(best_time) # Sets "0" on the X-axis for display
            ax.set_xlim(best_time - 7, best_time + 0.5) # CORRECTED: Use absolute GPS times for limits!
            ax.set_title(f"{label} Spectrogram (Q-transform)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel(f"Time from {best_time} (s)")
            ax.colorbar(label="Normalized Energy")
            fig.savefig(os.path.join(f"{BASE_DIR}/plots", f'{SUFFIX}_{label}_best_candidate_spectrogram.png'))
            print(f"Saved spectrogram for {label} as '{SUFFIX}_{label}_best_candidate_spectrogram.png'")

    print("\n")
    print("Post-processing completed. Check the output and plots directory.")
    return 0

if __name__ == '__main__':
    import re
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())