#  Import necessary libraries and functions

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import os
from lal import MRSUN_SI
import glob
from nmma.em.model import FiestaKilonovaModel
from nmma.em.model import SVDLightCurveModel

'''
Fitting formulae (see https://arxiv.org/abs/2205.08513)
'''

def dyn_ej(a = -9.3335, b = 114.17, d = -337.56, n = 1.5465, M1 = 1.4, R1 = 10, M2 = 1.4, R2 = 10):
    C1 = M1 / (R1 * 1e3 / MRSUN_SI)
    C2 = M2 / (R2 * 1e3 / MRSUN_SI)
    x = (a/C1 + b*(M2**n/M1**n) + d*C1)*M1 + (a/C2 + b*(M1**n/M2**n) + d*C2)*M2
    if x < 0:
        return 0
    else:
        return x/1000
    
def wind_ej(M1, M2, a0=-1.725, deltaa=-2.337, b0=-0.564, deltab=-0.437, c=0.958, d=0.057, beta=5.879, qtrans=0.886, Mtov=1.97, R16=11.137): 
    r16 = R16 * 1e3 / MRSUN_SI # convert km to SI (like in NMMA) 
    Mtresh = (2.38 - 3.606 * (Mtov/r16))*Mtov
    q = M2/M1
    xsi = 0.5 * np.tanh(beta * (q - qtrans))
    a = a0 + deltaa * xsi
    b = b0 + deltab * xsi
    mwind = a * (1 + b * np.tanh( (c - (M1+M2)/Mtresh)/d ))
    mwind = np.maximum(-3.0, mwind)
    return mwind

# mass chirp
def chirp_mass(m1, m2):
    return (m1*m2)**(3/5) / (m1 + m2)**(1/5)

# mass ratio
def mass_ratio(m1, m2):
    return m2 / m1

'''
Class to load and explore the eos files from https://zenodo.org/records/6106130#.Y1pdM9JBxhG (15nsat_cse_uniform_R14)
'''
    
class EOSDataset:
    """
    Class to load and manage EOS files from NMMA/EOS directory.
    Similar to BullaDataset but for equation of state data.
    """
    
    def __init__(self, folder_path):
        """
        Initialize the EOSDataset.
        
        Parameters:
        -----------
        folder_path : str
            Path to the folder containing the EOS .dat files
        """
        self.folder_path = folder_path
        self.files = []
        self.eos_dict = {}
        
        # Load all files
        self._load_all_files()
        
        # Compute statistics
        self._compute_statistics()
    
    def _load_all_files(self):
        """Load all .dat files from the folder (excluding Zone.Identifier files)."""
        all_files = sorted([f for f in glob.glob(os.path.join(self.folder_path, '*.dat'))
                           if not f.endswith('Zone.Identifier')],
                          key=lambda x: int(os.path.basename(x).replace('.dat', '')))
        
        print(f"Loading {len(all_files)} EOS files...")
        
        for filepath in all_files:
            filename = os.path.basename(filepath)
            eos_id = int(filename.replace('.dat', ''))
            
            self.files.append(filepath)
            self.eos_dict[eos_id] = {
                'filepath': filepath,
                'filename': filename,
                'eos_id': eos_id,
                'data': None,  # Loaded on demand
                'radius_km': None,
                'mass_solar': None,
                'pressure': None,
                'max_mass': None,
                'radius_at_1_4': None,
                'radius_at_1_6': None
            }
        
        print(f"{len(self.eos_dict)} EOS files indexed")
    
    def _load_eos_data(self, eos_id):
        """Load data for a specific EOS if not already loaded."""
        if eos_id not in self.eos_dict:
            print(f"EOS {eos_id} not found in dataset")
            return None
        
        eos_info = self.eos_dict[eos_id]
        
        if eos_info['data'] is None:
            try:
                # Load data: columns are Radius[km], Mass[Solar Mass], Central_pressure
                data = np.loadtxt(eos_info['filepath'])
                eos_info['data'] = data
                eos_info['radius_km'] = data[:, 0]
                eos_info['mass_solar'] = data[:, 1]
                eos_info['pressure'] = data[:, 2]
                
                # Compute derived quantities
                eos_info['max_mass'] = np.max(data[:, 1])
                
                # Find radius at 1.4 M☉
                try:
                    idx_1_4 = np.argmin(np.abs(data[:, 1] - 1.4))
                    eos_info['radius_at_1_4'] = data[idx_1_4, 0]
                except:
                    eos_info['radius_at_1_4'] = None
                
                # Find radius at 1.6 M☉
                try:
                    idx_1_6 = np.argmin(np.abs(data[:, 1] - 1.6))
                    eos_info['radius_at_1_6'] = data[idx_1_6, 0]
                except:
                    eos_info['radius_at_1_6'] = None
                    
            except Exception as e:
                print(f"Error loading {eos_info['filepath']}: {e}")
                return None
        
        return eos_info
    
    def _compute_statistics(self):
        """Compute statistics across all EOS (lazy loading approach)."""
        print("\nComputing statistics (sampling 500 random EOS)...")
        
        # Sample 500 random EOS to compute statistics
        sample_ids = np.random.choice(list(self.eos_dict.keys()), 
                                     size=min(500, len(self.eos_dict)), 
                                     replace=False)
        
        max_masses = []
        r14_values = []
        r16_values = []
        
        for eos_id in sample_ids:
            info = self._load_eos_data(eos_id)
            if info is not None:
                if info['max_mass'] is not None:
                    max_masses.append(info['max_mass'])
                if info['radius_at_1_4'] is not None:
                    r14_values.append(info['radius_at_1_4'])
                if info['radius_at_1_6'] is not None:
                    r16_values.append(info['radius_at_1_6'])
        
        self.statistics = {
            'max_mass': {
                'min': np.min(max_masses) if max_masses else None,
                'max': np.max(max_masses) if max_masses else None,
                'mean': np.mean(max_masses) if max_masses else None,
                'std': np.std(max_masses) if max_masses else None
            },
            'radius_at_1_4': {
                'min': np.min(r14_values) if r14_values else None,
                'max': np.max(r14_values) if r14_values else None,
                'mean': np.mean(r14_values) if r14_values else None,
                'std': np.std(r14_values) if r14_values else None
            },
            'radius_at_1_6': {
                'min': np.min(r16_values) if r16_values else None,
                'max': np.max(r16_values) if r16_values else None,
                'mean': np.mean(r16_values) if r16_values else None,
                'std': np.std(r16_values) if r16_values else None
            }
        }
    
    def get_eos(self, eos_id, load_data=True):
        """
        Retrieve an EOS by its ID.
        
        Parameters:
        -----------
        eos_id : int
            ID of the EOS (filename without .dat)
        load_data : bool
            If True, load the data from the file
        
        Returns:
        --------
        dict : Information about the EOS (filepath, data, derived quantities)
        """
        if eos_id not in self.eos_dict:
            print(f"EOS {eos_id} not found")
            return None
        
        if load_data:
            return self._load_eos_data(eos_id)
        else:
            return self.eos_dict[eos_id]
    
    def get_random_eos(self, n=1, load_data=True):
        """
        Get n random EOS from the dataset.
        
        Parameters:
        -----------
        n : int
            Number of random EOS to retrieve
        load_data : bool
            If True, load the data
        
        Returns:
        --------
        list : List of EOS info dictionaries
        """
        random_ids = np.random.choice(list(self.eos_dict.keys()), size=n, replace=False)
        
        results = []
        for eos_id in random_ids:
            eos_info = self.get_eos(eos_id, load_data=load_data)
            if eos_info is not None:
                results.append(eos_info)
        
        return results
    
    def find_eos_by_criteria(self, max_mass_min=None, max_mass_max=None,
                            r14_min=None, r14_max=None,
                            r16_min=None, r16_max=None,
                            max_results=None):
        """
        Find EOS matching specified criteria.
        
        Parameters:
        -----------
        max_mass_min, max_mass_max : float
            Range for maximum mass [M☉]
        r14_min, r14_max : float
            Range for radius at 1.4 M☉ [km]
        r16_min, r16_max : float
            Range for radius at 1.6 M☉ [km]
        max_results : int
            Maximum number of results to return
        
        Returns:
        --------
        list : List of matching EOS IDs
        """
        matching_ids = []
        
        for eos_id in self.eos_dict.keys():
            info = self._load_eos_data(eos_id)
            
            if info is None:
                continue
            
            # Check criteria
            if max_mass_min is not None and (info['max_mass'] is None or info['max_mass'] < max_mass_min):
                continue
            if max_mass_max is not None and (info['max_mass'] is None or info['max_mass'] > max_mass_max):
                continue
            if r14_min is not None and (info['radius_at_1_4'] is None or info['radius_at_1_4'] < r14_min):
                continue
            if r14_max is not None and (info['radius_at_1_4'] is None or info['radius_at_1_4'] > r14_max):
                continue
            if r16_min is not None and (info['radius_at_1_6'] is None or info['radius_at_1_6'] < r16_min):
                continue
            if r16_max is not None and (info['radius_at_1_6'] is None or info['radius_at_1_6'] > r16_max):
                continue
            
            matching_ids.append(eos_id)
            
            if max_results is not None and len(matching_ids) >= max_results:
                break
        
        return matching_ids
    
    def print_statistics(self):
        """Display statistics about the dataset."""
        print("\n" + "="*80)
        print("EOS DATASET STATISTICS")
        print("="*80)
        print(f"\nTotal number of EOS: {len(self.eos_dict)}")
        
        if hasattr(self, 'statistics'):
            print("\nSampled statistics (from 100 random EOS):")
            
            for param_name, stats in self.statistics.items():
                print(f"\n{param_name.upper().replace('_', ' ')}:")
                if stats['mean'] is not None:
                    print(f"  • Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                    print(f"  • Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
                else:
                    print("  • No data available")
        
        print("\n" + "="*80)
    
    def plot_mr_curves(self, eos_ids=None, n_random=5, figsize=(10, 7), plot_all=False):
        """
        Plot Mass-Radius curves for selected EOS.
        
        Parameters:
        -----------
        eos_ids : list
            List of EOS IDs to plot (if None, plot random EOS)
        n_random : int
            Number of random EOS to plot if eos_ids is None
        figsize : tuple
            Figure size
        plot_all : bool
            If True, plot all EOS in the dataset
        """
        import matplotlib.pyplot as plt
        
        if eos_ids is None:
            if plot_all:
                eos_list = [self.get_eos(eos_id, load_data=True) for eos_id in self.eos_dict.keys()]
            else:
                eos_list = self.get_random_eos(n=n_random, load_data=True)
        else:
            eos_list = [self.get_eos(eos_id, load_data=True) for eos_id in eos_ids]
            eos_list = [eos for eos in eos_list if eos is not None]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for eos in eos_list:
            if plot_all == True:
                c = plt.cm.viridis(eos['eos_id'] % 256 / 256)
                ax.plot(eos['radius_km'], eos['mass_solar'], 
                       color=c, lw=1, alpha=0.3)
            else:
                ax.plot(eos['radius_km'], eos['mass_solar'], 
                   label=f"EOS {eos['eos_id']} (M_max={eos['max_mass']:.2f})", 
                   lw=2, alpha=0.7)
        
        ax.set_xlabel('Radius [km]', fontsize=14)
        ax.set_ylabel('Mass [$M_\\odot$]', fontsize=14)
        ax.set_title('Mass-Radius Relations', fontsize=16)
        if plot_all == False:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(7.8, 18)
        ax.set_ylim(0, 3.5)
        
        plt.tight_layout()
        plt.show()
    
    def __len__(self):
        """Number of EOS in the dataset."""
        return len(self.eos_dict)
    
    def __repr__(self):
        return f"EOSDataset({len(self)} EOS files from {self.folder_path})"
    
'''
Functions used to generate synthetic lc data
'''
def abs_to_app_mag(mag_abs, distance_mpc):
    distance_modulus = 5 * np.log10(distance_mpc) + 25 # distance in Mpc
    mag_app = {}
    for filter, mags in mag_abs.items():
        mag_app[filter] = mags + distance_modulus
    return mag_app

def add_noise_error(mag, noise_level=0.3, max_error_level=0.6):
    noisy = {}
    errors = {}
    for filter, mags in mag.items():
        print(f"Adding noise to filter {filter} with noise level {noise_level}")
        noise = np.random.normal(0, noise_level, size=mags.shape)
        noisy[filter] = mags + noise
        err = []
        for noisi in noise:
            # find the smallest error level that would still contains the true mag without the noise
            if abs(noisi) > max_error_level:
                err_i = abs(noisi) + np.random.uniform(0, 0.1) # if we have a lot of noise, we put an error level that is slightly bigger than the noise to have a difference between the error level and the noise
            else:
                err_i = np.random.uniform(abs(noisi), max_error_level) # min value of the error at the noise level to be sure that the true mag is within the error bars, max value is the max error level
            err.append(err_i)

        errors[filter] = np.array(err)
    return noisy, errors

def format_nmma_data_v2(times, mag, errors, filters, trigger_iso = '2025-01-01T00:00:00', timeshift=0):
    data_dict = {}
    # format NMMA : ISOTIME, BAND, MAG, MAG_ERR : 2017-08-18T00:00:00.000 ps1::g 17.41000 0.02000
    # convert svd time to NMMA time format
    iso_times = []
    trigger_dt = pd.to_datetime(trigger_iso)
    trigger_mjd  = trigger_dt.to_julian_date() - 2400000.5  # time of 1st detection -> use it as a filter to remove points before the trigger if timeshift is negative
    # convert trigger in MJD
    print("Trigger ISO:", trigger_iso)
    trigger_dt = pd.to_datetime(trigger_iso)
    for t in times:
        iso_dt = trigger_dt + pd.to_timedelta(t, unit='D')
        if iso_dt.to_julian_date() - 2400000.5 < trigger_mjd - timeshift:
            iso_times.append('NaN') # if the time is before the trigger, we put NaN to be able to remove it later
            continue
        iso_str = iso_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        iso_times.append(iso_str)
    print("ISO times sample:", iso_times)
    # construire le format NMMA
    data_rows = []
    for filter in filters:
        for i, t in enumerate(times):
            if iso_times[i] == 'NaN':
                continue
            data_rows.append([iso_times[i], filter, mag[filter][i], errors[filter][i]])
    df = pd.DataFrame(data_rows)
    return df, trigger_mjd

def generate_synth_lc_fiesta(model_name='Bu2026_MLP',
                    model_param={
                            "log10_mej_dyn": -2,
                            "log10_mej_wind": -1,
                            "inclination_EM": 0.5,
                            "luminosity_distance": 40,
                            "v_ej_dyn": 0.2, 
                            "v_ej_wind": 0.1,
                            "Ye_dyn": 0.2,
                            "Ye_wind": 0.35,
                            "timeshift": 0.0
                        },
                    filters_band=['ps1::g', 'ps1::r', 'ps1::i', 'ps1::z', 'ps1::y'],
                    noise_level=0.2,
                    max_error_level=0.4,
                    trigger_iso='2025-01-01T00:00:00',
                    pts_per_day=2,
                    obs_duration=15,
                    jitter=0.,
                    save=False,
                    filename='test_lc_fiesta.dat',
                    detection_limit_dict={'ps1::g':24.7, 'ps1::r':24.2, 'ps1::i':23.8, 'ps1::z':23.2, 'ps1::y':22.3}):
    """Generate synthetic lightcurves using SVDLightCurveModel.
    Parameters
    ----------
    model_name : str
        Name of the SVD model to be used.
    model_param : dict
        Dictionary of model parameters.
    filters_band : list
        List of filters to be used for the lightcurve generation.
    sample_times : np.ndarray
        Array of sample times.
    noise_level : float
        Standard deviation of the Gaussian noise to be added.
    max_error_level : float
        Maximum error level to be added to the magnitudes.
    trigger_iso : str
        ISO formatted trigger time.
    pts_per_day : int
        Number of observation points per day.
    obs_duration : int
        Duration of the observation in days.
    jitter : float
        Maximum jitter to be added to the sample times in days.
    save : bool
        If True, save the generated data to a file.
    filename : str
        Filename to save the generated data if save is True.
    detection_limit_dict : dict or None
        Dictionary specifying detection limits for each filter. If None, no detection limits are applied.
    
    Returns
    -------
    data_nmma_svd : pd.DataFrame
        DataFrame containing the synthetic lightcurve data in NMMA format.
    trigger : float
        Trigger time in MJD.
    
    """
    ts = -1 * model_param["timeshift"]
    print(f"Generating synthetic lightcurve with timeshift = {ts} days")
    model_param["timeshift"] = 0.0 # we set timeshift to 0 for the generation and we will apply it later to the sample time array 
    sample_times = np.arange(0, obs_duration, 1/pts_per_day)
    for t in range(len(sample_times)):
        if t == 0:
            continue # we don't want to add jitter to the first point to be sure that we have a point at the trigger time 
        sample_times[t] += np.random.uniform(-jitter, jitter)
    try:
        model = FiestaKilonovaModel("Bu2026_MLP", filters=filters_band)
    except Exception as e:
        print(f"  - {model_name}: error during SVD model creation ->", e)
        raise e
    del model_param["timeshift"] # we remove timeshift from the model parameters because it's not a parameter of the model, it's just a shift that we will apply to the sample times later
    mag_svd = model.generate_lightcurve(sample_times, model_param)
    mag_svd_noisy, mag_svd_errors = add_noise_error(mag_svd, noise_level=noise_level, max_error_level=max_error_level)
    mag_svd_noisy_app = abs_to_app_mag(mag_svd_noisy, distance_mpc=model_param["luminosity_distance"])
    model_param["timeshift"] = -1 * ts # we put back the original timeshift value for the formatting function
    print("Filters:", filters_band)
    data_nmma_svd, trigger = format_nmma_data_v2(sample_times, mag_svd_noisy_app, mag_svd_errors, filters_band, trigger_iso=trigger_iso, timeshift=ts)
    if detection_limit_dict is not None:
        # apply detection limits
        for filter, limit in detection_limit_dict.items():
            filter_mask = data_nmma_svd[1] == filter # all rows with this filter
            limit_mask = data_nmma_svd[2] > limit # all rows with mag > limit (mag is an inverted scale)
            to_remove = filter_mask & limit_mask # combine masks
            data_nmma_svd = data_nmma_svd[~to_remove]
    if save:
        data_nmma_svd.to_csv(filename, sep=' ', index=False, header=False)
    return data_nmma_svd, trigger

def abs_to_app_mag(mag_abs, distance_mpc):
    distance_modulus = 5 * np.log10(distance_mpc) + 25 # distance in Mpc
    mag_app = {}
    for filter, mags in mag_abs.items():
        mag_app[filter] = mags + distance_modulus
    return mag_app

def generate_synth_lc_v2(model_name='Bu2019lm',
                    model_param={
                            "KNphi": 30,
                            "log10_mej_dyn": -2,
                            "log10_mej_wind": -1,
                            "inclination_EM": 0.5,
                            "luminosity_distance": 40,
                            "timeshift": 0.0
                        },
                    filters_band=['ps1__g', 'ps1__r', 'ps1__i'],
                    noise_level=0.3,
                    max_error_level=0.6,
                    trigger_iso='2025-01-01T00:00:00',
                    pts_per_day=2,
                    obs_duration=15,
                    delay = 0,
                    jitter=0.1,
                    save=False,
                    filename='synthetic_kilonova_svd.dat',
                    detection_limit_dict=None):
    """Generate synthetic lightcurves using SVDLightCurveModel.
    Parameters
    ----------
    model_name : str
        Name of the SVD model to be used.
    model_param : dict
        Dictionary of model parameters.
    filters_band : list
        List of filters to be used for the lightcurve generation.
    sample_times : np.ndarray
        Array of sample times.
    noise_level : float
        Standard deviation of the Gaussian noise to be added.
    max_error_level : float
        Maximum error level to be added to the magnitudes.
    trigger_iso : str
        ISO formatted trigger time.
    pts_per_day : int
        Number of observation points per day.
    obs_duration : int
        Duration of the observation in days.
    delay : float
        Delay in days to be added to the sample times (positive delay means that the first point will be after the trigger time). (Additional delay to the timeshift parameter in model_param). (Here to combine LSST + ZTF sampling)
    jitter : float
        Maximum jitter to be added to the sample times in days.
    save : bool
        If True, save the generated data to a file.
    filename : str
        Filename to save the generated data if save is True.
    detection_limit_dict : dict or None
        Dictionary specifying detection limits for each filter. If None, no detection limits are applied.
    
    Returns
    -------
    data_nmma_svd : pd.DataFrame
        DataFrame containing the synthetic lightcurve data in NMMA format.
    trigger : float
        Trigger time in MJD.
    
    """
    ts = -1 * model_param["timeshift"]
    print(f"Generating synthetic lightcurve with timeshift = {ts} days")
    model_param["timeshift"] = 0.0 # we set timeshift to 0 for the generation and we will apply it later to the sample time array 
    sample_times = np.arange(delay, obs_duration, 1/pts_per_day)
    for t in range(len(sample_times)):
        if t == 0:
            continue # we don't want to add jitter to the first point to be sure that we have a point at the trigger time 
        sample_times[t] += np.random.uniform(-jitter, jitter)
    svd_path = "/home/stu_jamsin/jamsin/NMMA/svdmodels"
    try:
        svd_model = SVDLightCurveModel(
                model=model_name,
                sample_times=sample_times,
                svd_path=svd_path,
                interpolation_type='tensorflow',
                filters=filters_band
        )
    except Exception as e:
        print(f"  - {model_name}: error during SVD model creation ->", e)
    mag_svd = svd_model.generate_lightcurve(sample_times, model_param)
    mag_svd_noisy, mag_svd_errors = add_noise_error(mag_svd, noise_level=noise_level, max_error_level=max_error_level)
    mag_svd_noisy_app = abs_to_app_mag(mag_svd_noisy, distance_mpc=model_param["luminosity_distance"])
    model_param["timeshift"] = -1 * ts # we put back the original timeshift value for the formatting function
    print("Filters:", filters_band)
    data_nmma_svd, trigger = format_nmma_data_v2(sample_times, mag_svd_noisy_app, mag_svd_errors, filters_band, trigger_iso=trigger_iso, timeshift=ts)
    if detection_limit_dict is not None:
        # apply detection limits
        for filter, limit in detection_limit_dict.items():
            filter_mask = data_nmma_svd[1] == filter # all rows with this filter
            limit_mask = data_nmma_svd[2] > limit # all rows with mag > limit (mag is an inverted scale)
            to_remove = filter_mask & limit_mask # combine masks
            data_nmma_svd = data_nmma_svd[~to_remove]
    if save:
        data_nmma_svd.to_csv(filename, sep=' ', index=False, header=False)
    return data_nmma_svd, trigger