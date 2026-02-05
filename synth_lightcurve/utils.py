#  Import necessary libraries and functions

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
import os
import sys
from lal import MRSUN_SI
import nmma.em.io as io
from nmma.em.model import SVDLightCurveModel

SVD_PATH = '/home/stu_jamsin/jamsin/NMMA/svdmodels'

from lal import MRSUN_SI
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
    
    # V3 of the function : now returns fig, ax 
# v4 ejecta plot
def ejecta_plot_v4(eos_number, model_name='Bu2019lm', model_param={"luminosity_distance": 40.0, "KNphi": 30.0, "KNtheta": 30.0}, filters=['ztfr', 'ztfi', 'ztfg'], plot=True, title=None, get_fig=False, EOS_path='/home/liteul/memoir_code/NMMA/EOS/15nsat_cse_uniform_R14/macro', mass_limit=(1.0, 3.0)):
    '''Plot the ejecta properties for a given EOS model.
    Parameters
    ----------
    eos_number : int
        Number of the equation of state (EOS) to be used.
    title : str
        Title for the plots.
    model_name : str, optional
        Name of the SVD model to be used (default is 'Bu2019lm'). Attention: : the model must have parameters "log10_mej_dyn" and "log10_mej_wind".
    model_param : dict, optional
        Parameters for the SVD model (default is {"luminosity_distance": 40.0, "KNphi": 30.0, "KNtheta": 30.0}).
    filters : list, optional
        Filters to be used for the light curve.
    plot : bool, optional
        Whether to display the plots (default is True).
    get_fig : bool, optional
        Whether to return the figure and axes objects (default is False).
    EOS_path : str, optional
        Path to the EOS data files (default is '/home/liteul/memoir_code/NMMA/EOS/15nsat_cse_uniform_R14/macro/').
    mass_limit : tuple, optional
        Mass limits for the neutron stars (default is (1.0, 3.0) solar masses).
    Returns
    -----------
    Returns a dictionary with ejecta masses and magnitudes and optionally the figure and axes objects.
    -----------
    '''
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    import numpy as np

    # 1 build the M R relation from the EOS
    eos_file = f'{EOS_path}/{eos_number}.dat'
    d = np.loadtxt(eos_file)
    name = eos_number
    print(f"Processing EOS: {name}")
    Mr = d[:, 1]  # mass in solar masses
    Rr = d[:, 0]  # radius in km
    # Limit mass range
    mass_mask = (Mr >= mass_limit[0]) & (Mr <= mass_limit[1])
    Mr = Mr[mass_mask]
    Rr = Rr[mass_mask]

    pair_mass = []
    for i, m1 in enumerate(Mr):
        for j, m2 in enumerate(Mr):
            if m1 > m2:
                pair_mass.append((m1,m2))
    pair_array = np.array(pair_mass)
    n_pairs = pair_array.shape[0]

    # Prepare arrays to store results
    fit_masses = np.zeros(n_pairs)
    wind_log = np.zeros(n_pairs)

    # Retrieve the radius for 1.6 M_solar NS
    radius16 = np.interp(1.6, Mr, Rr)

    # Compute ejecta masses for each pair
    for idx, (m1, m2) in enumerate(pair_array):
        idx1 = np.argmin(np.abs(Mr - m1))
        idx2 = np.argmin(np.abs(Mr - m2))
        R1 = Rr[idx1]
        R2 = Rr[idx2]
        fit_masses[idx] = dyn_ej(M1=m1, R1=R1, M2=m2, R2=R2)
        wind_log[idx] = wind_ej(m1, m2, Mtov=Mr.max(), R16=radius16)
    wind_linear = 0.3 * 10 ** (wind_log)
    total_mass = fit_masses +  wind_linear

    # Setup SVD model
    svd_path = '/home/liteul/memoir_code/NMMA/svdmodels'
    tmin_svd, tmax_svd, dt_svd = 0.1, 5.0, 0.1
    sample_times_svd = np.arange(tmin_svd, tmax_svd + dt_svd, dt_svd)
    try:
        svd_model = SVDLightCurveModel(
            model=model_name,
            sample_times=sample_times_svd,
            svd_path=svd_path,
            mag_ncoeff=10,
            lbol_ncoeff=10,
            filters=filters
        )
    except Exception as e:
        print(f"  - {name}: erreur lors de la création du modèle SVD ->", e)
        return
    # Plotting

    # Convert to log scale for plotting and lightcurve calculation
    log_10wind = np.log10(0.3) + wind_log
    log_10dyn = np.zeros(n_pairs)
    for i, m in enumerate(fit_masses):
        if m > 0:
            log_10dyn[i] = np.log10(m)
        else:
            log_10dyn[i] = -5  
    log_10total = np.log10(total_mass)

    print('Starting to compute magnitudes for each ejecta pair...')
    
    # Compute magnitude using SVD model
    mags= np.zeros(n_pairs)
    peak_times = np.zeros(n_pairs)
    for idx, param in tqdm(enumerate(zip(log_10dyn, log_10wind)), total=len(log_10wind), desc='Computing magnitudes', unit='pair'):
        model_param.update({
            "log10_mej_dyn": param[0],              # log10(Masse éjectée dynamique en M☉)
            "log10_mej_wind": param[1],              # log10(Masse éjectée par vent en M☉)
        })
        _, mag_svd = svd_model.generate_lightcurve(sample_times_svd, model_param)
        mag = np.min(mag_svd[filters[0]])
        mags[idx] = mag
        peak_times[idx] = np.argmin(mag_svd[filters[0]])
    print('Magnitude computation completed.')
    if plot:
        if title is None:
            title = f'Ejecta properties for EOS: {eos_number}'

        fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharey=True)

        # Top left: ejecta mass
        norm_ejecta = colors.LogNorm(vmin=total_mass.min(), vmax=total_mass.max())
        sc0 = axes[0,0].scatter(pair_array[:, 0], pair_array[:, 1], c=total_mass, cmap='viridis', s=300, edgecolors=None, linewidths=0.2, norm=norm_ejecta)
        axes[0,0].set_xlabel('$M_1$ [$M_\\odot$]', fontsize=12)
        axes[0,0].set_ylabel('$M_2$ [$M_\\odot$]', fontsize=12)
        axes[0,0].set_title('Total mass')
        axes[0,0].grid(True, alpha=0.25)
        cbar0 = fig.colorbar(sc0, ax=axes[0,0], orientation='vertical', fraction=0.05)
        cbar0.set_label('$M_{total}$ [M$_\\odot$]')

        # text box for ejecta stats
        ejecta_stats = (f'Ejecta mass statistics:\n'
                    f'  - min: {total_mass.min():.1%} M$_\\odot$\n'
                    f'  - max: {total_mass.max():.1%} M$_\\odot$\n'
                    f'  - mean: {total_mass.mean():.1%} M$_\\odot$')
        axes[0,0].text(0.03, 0.97, ejecta_stats, transform=axes[0,0].transAxes,
                    fontsize=15, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.85, edgecolor='0'))

        # Top right: peak magnitude
        norm_mag = colors.Normalize(vmin=mags.min(), vmax=mags.max())
        sc1 = axes[0,1].scatter(pair_array[:, 0], pair_array[:, 1], c=mags, cmap='viridis_r', s=300, edgecolors=None, linewidths=0.2, norm=norm_mag)
        axes[0,1].set_xlabel('$M_1$ [$M_\\odot$]', fontsize=12)
        axes[0,1].set_ylabel('$M_2$ [$M_\\odot$]', fontsize=12)
        axes[0,1].set_title(f'Peak absolute magnitude (filter: {filters[0]})')
        axes[0,1].grid(True, alpha=0.25)
        cbar1 = fig.colorbar(sc1, ax=axes[0,1], orientation='vertical', fraction=0.05)
        cbar1.set_label(f'$M_{{peak}}$ [mag] using {model_name}')

        # text box for mag
        mag_stats = (f'Magnitude statistics:\n'
                    f'  - min: {mags.min():.2f}\n'
                    f'  - max: {mags.max():.2f}\n'
                    f'  - mean: {mags.mean():.2f}')
        axes[0,1].text(0.03, 0.97, mag_stats, transform=axes[0,1].transAxes,
                    fontsize=15, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.85, edgecolor='0'))

        # Bottom left: dyn ejecta mass
        norm_dyn = colors.SymLogNorm(vmin=0, vmax=fit_masses.max(), linthresh=1e-4, linscale=0.5)
        sc2 = axes[1,0].scatter(pair_array[:, 0], pair_array[:, 1], c=fit_masses, cmap='viridis', s=300, edgecolors=None, linewidths=0.2, norm=norm_dyn)
        axes[1,0].set_xlabel('$M_1$ [$M_\\odot$]', fontsize=12) 
        axes[1,0].set_ylabel('$M_2$ [$M_\\odot$]', fontsize=12)
        axes[1,0].set_title('Dynamical ejecta mass')
        axes[1,0].grid(True, alpha=0.25)
        cbar2 = fig.colorbar(sc2, ax=axes[1,0], orientation='vertical', fraction=0.05)
        cbar2.set_label('$M_{dyn}$ [M$_\\odot$]')

        # text box for dyn ejecta stats
        dyn_stats = (f'Dynamical ejecta mass statistics:\n'
                    f'  - min: {fit_masses.min():.1%} M$_\\odot$\n'
                    f'  - max: {fit_masses.max():.1%} M$_\\odot$\n'
                    f'  - mean: {fit_masses.mean():.1%} M$_\\odot$')
        axes[1,0].text(0.03, 0.97, dyn_stats, transform=axes[1,0].transAxes,
                    fontsize=15, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.85, edgecolor='0'))

        # Bottom right: wind ejecta mass
        norm_wind = colors.LogNorm(vmin=10e-5, vmax=wind_linear.max())
        sc3 = axes[1,1].scatter(pair_array[:, 0], pair_array[:, 1], c=wind_linear, cmap='viridis', s=300, edgecolors=None, linewidths=0.2, norm=norm_wind)
        axes[1,1].set_xlabel('$M_1$ [$M_\\odot$]', fontsize=12) 
        axes[1,1].set_ylabel('$M_2$ [$M_\\odot$]', fontsize=12)
        axes[1,1].set_title('Wind ejecta mass')
        axes[1,1].grid(True, alpha=0.25)
        cbar3 = fig.colorbar(sc3, ax=axes[1,1], orientation='vertical', fraction=0.05)
        cbar3.set_label('$M_{wind}$ [M$_\\odot$]')
        
        # text box for wind ejecta stats
        wind_stats = (f'Wind ejecta mass statistics:\n'
                    f'  - min: {wind_linear.min():.2%} M$_\\odot$\n'
                    f'  - max: {wind_linear.max():.1%} M$_\\odot$\n'
                    f'  - mean: {wind_linear.mean():.1%} M$_\\odot$')
        axes[1,1].text(0.03, 0.97, wind_stats, transform=axes[1,1].transAxes,
                    fontsize=15, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.85, edgecolor='0'))
        
        plt.suptitle(title, fontsize=16, y=1.0002)

        plt.tight_layout()
        plt.show()

    if get_fig:
        return fig, axes, {
        "model_param": model_param,
        "magnitude": mags,
        "fit_masses": fit_masses,
        "wind_linear": wind_linear,
        "pair_array": pair_array,
        "filter": filters,
        "times": peak_times
        }
    else:
        return {
        "model_param": model_param,
        "magnitude": mags,
        "fit_masses": fit_masses,
        "wind_linear": wind_linear,
        "pair_array": pair_array,
        "filter": filters,
        "times": peak_times
        }
    
    # Interactive plot to select points and get their parameters

def get_indices(dic):
    """Return the list of selected indices for a mag-mass plot.
    Parameters
    ----------
    dic : dict
        Dictionary containing 'pair_array' and 'magnitude' keys (returned by ejecta_plot_v2).
    ----------
    Returns
        selected_indices : list
            List of selected indices.
    """

    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend as fallback
    # Use widget backend for interactivity in Jupyter
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    def onclick(event):
        """Function to handle click events on the plot."""
        if event.inaxes is None:
            return

        # Get click coordinates
        m1_click = event.xdata
        m2_click = event.ydata

        # Find the closest point in the scatter plot
        distances = np.sqrt((dic['pair_array'][:, 0] - m1_click)**2 + (dic['pair_array'][:, 1] - m2_click)**2)
        closest_idx = np.argmin(distances)
        
        # Add the index to the list if not already selected
        if closest_idx in selected_indices:
            print(f"Point {closest_idx} already selected.")
            return
        else:
            selected_indices.append(closest_idx)

        # Mark the selected point
        event.inaxes.scatter(dic['pair_array'][closest_idx, 0], dic['pair_array'][closest_idx, 1], 
                            c='cyan', s=400, marker='*', linewidths=2,
                            zorder=10)
        event.inaxes.text(dic['pair_array'][closest_idx, 0], dic['pair_array'][closest_idx, 1], 
                        f'  {len(selected_indices)}', fontsize=12, fontweight='bold',
                        color='cyan', zorder=11)
        plt.draw()
        

    selected_indices = []

    # Create interactive plot
    fig, ax = plt.subplots(figsize=(12, 10))

    norm_mag = colors.Normalize(vmin=dic['magnitude'].min(), vmax=dic['magnitude'].max())
    sc = ax.scatter(dic['pair_array'][:, 0], dic['pair_array'][:, 1], c=dic['magnitude'], cmap='inferno_r', 
                    s=300, edgecolors=None, linewidths=0.2, norm=norm_mag)
    ax.set_xlabel('$M_1$ [$M_\\odot$]', fontsize=14)
    ax.set_ylabel('$M_2$ [$M_\\odot$]', fontsize=14)
    ax.set_title('Peak magnitude - Click to select points', fontsize=16)
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.05)
    cbar.set_label('$M_{{peak}}$ [mag]', fontsize=12)

    # Connect the click event to the handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout()
    plt.show()

    # Switch back to inline backend

    return selected_indices

class BullaDataset:
    """
    Class to load and manage the Bulla 2019 dataset files.
    """
    
    def __init__(self, folder_path):
        """
        Initialize the BullaDataset.
        
        Parameters:
        -----------
        folder_path : str
            Path to the folder containing the Bulla 2019 files
        """
        self.folder_path = folder_path
        self.files = []
        self.params_list = []
        self.data_dict = {}
        
        # Load all files 
        self._load_all_files()
        
        # Compute parameter ranges
        self._compute_param_ranges()
    
    def _load_all_files(self):
        """Load all .dat files from the folder."""
        import glob
        import os
        
        all_files = sorted([f for f in glob.glob(os.path.join(self.folder_path, '*.dat')) 
                           if not f.endswith('Zone.Identifier')])
        
        print(f"Loading {len(all_files)} Bulla 2019 files...")
        
        for filepath in all_files:
            filename = os.path.basename(filepath)
            params = self.Bulla_get_param(filename)

            # Create a unique key based on the parameters
            key = self._params_to_key(params)
            
            self.files.append(filepath)
            self.params_list.append(params)
            self.data_dict[key] = {
                'filepath': filepath,
                'filename': filename,
                'params': params,
                'data': None,  # Loaded on demand
                'filters': None    # Loaded on demand
            }

        print(f"✓ {len(self.data_dict)} files indexed")

    def Bulla_get_param(self, filename):
        '''
        Extract parameters from Bulla 2019 filename.
        '''

        # Get rid of the .dat extension
        name_without_ext = filename.replace('.dat', '')

        # cut the file by '_'
        parts = name_without_ext.split('_')

        params = {}
        for part in parts:
            if 'nph' in part:
                params['nph'] = part.replace('nph', '')
            elif 'mejdyn' in part:
                params['mejdyn'] = float(part.replace('mejdyn', ''))
            elif 'mejwind' in part:
                params['mejwind'] = float(part.replace('mejwind', ''))
            elif 'phi' in part:
                params['phi'] = float(part.replace('phi', ''))
            elif 'theta' in part:
                params['theta'] = float(part.replace('theta', ''))
            elif 'dMpc' in part:
                params['dMpc'] = float(part.replace('dMpc', ''))

        return params

    def _params_to_key(self, params):
        """Convert a parameter dictionary to a unique key."""
        return (
            params.get('mejdyn', 0),
            params.get('mejwind', 0),
            params.get('phi', 0),
            params.get('theta', 0),
            params.get('dMpc', 0)
        )
    
    def _compute_param_ranges(self):
        """Compute the ranges and unique values for each parameter."""
        import numpy as np
        
        self.param_ranges = {}

        # Collect all values for each parameter
        param_values = {
            'mejdyn': [],
            'mejwind': [],
            'phi': [],
            'theta': [],
            'dMpc': []
        }
        
        for params in self.params_list:
            for key in param_values.keys():
                if key in params:
                    param_values[key].append(params[key])
        
        # Compute statistics
        for param_name, values in param_values.items():
            unique_vals = sorted(set(values))
            self.param_ranges[param_name] = {
                'min': min(values) if values else None,
                'max': max(values) if values else None,
                'unique': unique_vals,
                'n_unique': len(unique_vals),
                'count': len(values)
            }

    def _extract_filters(self, filepath):
        """Extract filter names from file header."""
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#'):
                    # Parse header: # time filter1 filter2 ...
                    return first_line.split()[2:] # Skip '#' and 'time'
                else:
                    # No header, count columns from data
                    data = np.loadtxt(filepath, max_rows=1)
                    return ['time'] + [f'filter_{i}' for i in range(1, len(data))]
        except Exception as e:
            print(f"Warning: Could not extract filters from {filepath}: {e}")
            return None
    
    def get_file(self, mejdyn=None, mejwind=None, phi=None, theta=None, dMpc=None, load_data=True):
        """
        Retrieve a file based on its parameters.
        
        Parameters:
        -----------
        mejdyn, mejwind, phi, theta, dMpc : float
            Parameters of the requested file
        load_data : bool
            If True, load the data from the file

        Returns:
        --------
        dict : Information about the file (filepath, params, data)
        """
        key = (mejdyn or 0, mejwind or 0, phi or 0, theta or 0, dMpc or 0)
        
        if key not in self.data_dict:
            print(f"⚠ No file found for these parameters:")
            print(f"  mejdyn={mejdyn}, mejwind={mejwind}, phi={phi}, theta={theta}, dMpc={dMpc}")
            return None
        
        file_info = self.data_dict[key]
        
        # Load the data if requested and not yet loaded
        if load_data and file_info['data'] is None:
            file_info['data'] = np.loadtxt(file_info['filepath'])
            file_info['filters'] = self._extract_filters(file_info['filepath'])
        
        return file_info
    
    def find_files(self, mejdyn=None, mejwind=None, phi=None, theta=None, dMpc=None):
        """
        Find all files matching the criteria (None = all).
        
        Returns:
        --------
        list : List of matching files
        """
        results = []
        
        for key, file_info in self.data_dict.items():
            params = file_info['params']
            
            # Check each criterion
            if mejdyn is not None and params.get('mejdyn') != mejdyn:
                continue
            if mejwind is not None and params.get('mejwind') != mejwind:
                continue
            if phi is not None and params.get('phi') != phi:
                continue
            if theta is not None and params.get('theta') != theta:
                continue
            if dMpc is not None and params.get('dMpc') != dMpc:
                continue
            
            results.append(file_info)
        
        return results
    
    def print_param_ranges(self):
        """Display the ranges of all parameters."""
        print("\n" + "="*80)
        print("RANGES OF PARAMETERS IN THE BULLA 2019 DATASET")
        print("="*80)
        
        for param_name, stats in self.param_ranges.items():
            print(f"\n{param_name.upper()}:")
            print(f"  • Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  • Number of unique values: {stats['n_unique']}")
            print(f"  • Values: {stats['unique']}")
        
        print("\n" + "="*80)
        print(f"TOTAL: {len(self.data_dict)} files in the dataset")
        print("="*80)
    
    def get_param_grid(self):
        """
        Return a grid of all unique parameters.
        
        Returns:
        --------
        dict : Dictionary with unique values of each parameter
        """
        return {
            param: stats['unique'] 
            for param, stats in self.param_ranges.items()
        }
    
    def __len__(self):
        """Number of files in the dataset."""
        return len(self.data_dict)
    
    def __repr__(self):
        return f"BullaDataset({len(self)} files)"

# Modification of the plot_indices function

def plot_lc_for_indices_v3(selected_indices, dic, title, filters_band, model_name='Bu2019lm'):
    """Plot the lightcurves for the selected indices and print their parameters.
    Parameters
    ----------
    selected_indices : list
        List of selected indices.
    dic : dict
        Dictionary containing 'pair_array', 'magnitude', 'fit_masses', 'wind_linear', and 'model_param' keys (returned by ejecta_plot_v2).
    title : str
        Title for the plot.
    filters_band : list
        List of filters to be used for the plot.
    model_name : str, optional
        Name of the SVD model to be used (default is 'Bu2019lm') Ideally the same model used in ejecta_plot_v2.
    ----------
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    for idx in selected_indices:
        print(f"Selected point (index {idx}):")
        print(f"  M1: {dic['pair_array'][idx, 0]:.3f} M☉")
        print(f"  M2: {dic['pair_array'][idx, 1]:.3f} M☉")
        print(f"  Magnitude: {dic['magnitude'][idx]:.2f}")
        print(f"  M_dyn: {dic['fit_masses'][idx]:.4f} M☉")
        print(f"  M_wind: {dic['wind_linear'][idx]:.4f} M☉")

        print()

    # Setup SVD model
    svd_path = SVD_PATH
    tmin_svd, tmax_svd, dt_svd = 0.1, 19.9, 0.1
    sample_times_svd = np.arange(tmin_svd, tmax_svd + dt_svd, dt_svd)
    svd_model = SVDLightCurveModel(
        model=model_name,
        sample_times=sample_times_svd,
        svd_path=svd_path,
        mag_ncoeff=10,
        lbol_ncoeff=10,
        filters=filters_band
        )

    # Extract data from dictionary
    log_10dyn = np.zeros(len(dic['fit_masses']))
    for i, m in enumerate(dic['fit_masses']):
        if m > 0:
            log_10dyn[i] = np.log10(m)
        else:
            log_10dyn[i] = -5  
    log_10wind = np.log10(0.3) + np.log10(dic['wind_linear'])
    pair_array = dic['pair_array']
    mags = dic['magnitude']
    n_points = len(selected_indices)
    model_param = dic['model_param']
    filter = dic['filter']

    # Plot lc in g r i bands
    # 2 col per point: lc + position on the grid

    fig, axes = plt.subplots(n_points, 2, figsize=(18, 6*n_points), gridspec_kw={'width_ratios': [2.25, 1]})
    # Ensure axes is always 2D
    if n_points == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(selected_indices):

        model_param.update({
            "log10_mej_dyn": log_10dyn[idx],              # log10(Masse éjectée dynamique en M☉)
            "log10_mej_wind": log_10wind[idx],              # log10(Masse éjectée par vent en M☉)
            })
        _, mag_svd = svd_model.generate_lightcurve(sample_times_svd, model_param)

        # Get corresponding Bulla dataset file (closest match if exact not found)
        Bulla_parameters = {
            'mejdyn': 10**log_10dyn[idx],
            'mejwind': 10**log_10wind[idx],
            'phi': model_param['KNphi'],
            'theta': model_param['KNtheta'],
            'dMpc': 0
        }
        low_params_file = bulla_dataset.get_file(**Bulla_parameters, load_data=True)
        if low_params_file is None:
            closest_file = None
            closest_distance = float('inf') # Initialize with infinity
            for file in bulla_dataset.data_dict.values():
                params = file['params']
                distance = sum(np.sqrt((params.get(key, 0) - Bulla_parameters[key])**2) for key in ['mejdyn', 'mejwind', 'phi', 'theta', 'dMpc']) #  Euclidean distance 
                if distance < closest_distance:
                    closest_distance = distance
                    closest_file = file
            low_params_file = closest_file
            print(f"Using closest match: {low_params_file['filename']} with params {low_params_file['params']}")
            low_params_file['data'] = np.loadtxt(low_params_file['filepath'])
            low_params_file['filters'] = bulla_dataset._extract_filters(low_params_file['filepath'])

        # Plot
        cols = [['red', 'darkred'], ['orange', '#d95f14'], ['cyan', 'darkcyan']]

        for (filter_name, color) in zip(filters_band, cols):
            axes[i,0].plot(sample_times_svd, mag_svd[filter_name], label=f'Model : {filter_name}', linewidth=2, color=color[0])
            if filter_name in low_params_file['filters']:
                filter_idx = low_params_file['filters'].index(filter_name)
                data = low_params_file['data']
                times = data[:, 0]
                axes[i,0].plot(times, data[:, filter_idx + 1], ls='--', label=f'Bulla dataset : {filter_name}', markersize=6, color=color[1])
            else: # Horrible hack to match ps1__g r i filters due to naming inconsistency between NMMA and Bulla dataset
                if filter_name == 'ps1__g' :
                    filter_idx = low_params_file['filters'].index('ps1::g')
                    data = low_params_file['data']
                    times = data[:, 0]
                    axes[i,0].plot(times, data[:, filter_idx + 1], ls='--', label=f'Bulla dataset : {filter_name}', markersize=6, color=color[1])
                if filter_name == 'ps1__r' :
                    filter_idx = low_params_file['filters'].index('ps1::r')
                    data = low_params_file['data']
                    times = data[:, 0]
                    axes[i,0].plot(times, data[:, filter_idx + 1], ls='--', label=f'Bulla dataset : {filter_name}', markersize=6, color=color[1])
                if filter_name == 'ps1__i' :
                    filter_idx = low_params_file['filters'].index('ps1::i')
                    data = low_params_file['data']
                    times = data[:, 0]
                    axes[i,0].plot(times, data[:, filter_idx + 1], ls='--', label=f'Bulla dataset : {filter_name}', markersize=6, color=color[1])

        axes[i,0].invert_yaxis()
        axes[i,0].set_xlabel('Time [days]', fontsize=10)
        axes[i,0].set_ylabel('Absolute Magnitude', fontsize=10)
        axes[i,0].set_title(f'Model parameter : $M_{{dyn}}$={10**log_10dyn[idx]:.4f} M☉, $M_{{wind}}$={10**log_10wind[idx]:.4f} M☉, $\\phi$={model_param["KNphi"]:.2f}, $\\theta$={model_param["KNtheta"]:.2f}\nDataset parameter : $M_{{dyn}}$={low_params_file["params"]["mejdyn"]:.4f} M☉, $M_{{wind}}$={low_params_file["params"]["mejwind"]:.4f} M☉, $\\phi$={low_params_file["params"]["phi"]:.2f}, $\\theta$={low_params_file["params"]["theta"]:.2f}', fontsize=11)
        axes[i,0].legend(fontsize=8)
        axes[i,0].grid(True, alpha=0.3)

        # Zoom on peak
        axins = inset_axes(axes[i,0], width="20%", height="50%", loc='lower left')
        for (filter_name, color) in zip(filters_band, cols):
            axins.plot(sample_times_svd, mag_svd[filter_name], color=color[0])
            if filter_name in low_params_file['filters']:
                filter_idx = low_params_file['filters'].index(filter_name)
                data = low_params_file['data']
                times = data[:, 0]
                axins.plot(times, data[:, filter_idx + 1], ls='--', color=color[1])
            else:
                if filter_name == 'ps1__g' :
                    filter_idx = low_params_file['filters'].index('ps1::g')
                    data = low_params_file['data']
                    times = data[:, 0]
                    axins.plot(times, data[:, filter_idx + 1], ls='--', color=color[1])
                if filter_name == 'ps1__r' :
                    filter_idx = low_params_file['filters'].index('ps1::r')
                    data = low_params_file['data']
                    times = data[:, 0]
                    axins.plot(times, data[:, filter_idx + 1], ls='--', color=color[1])
                if filter_name == 'ps1__i' :
                    filter_idx = low_params_file['filters'].index('ps1::i')
                    data = low_params_file['data']
                    times = data[:, 0]
                    axins.plot(times, data[:, filter_idx + 1], ls='--', color=color[1])

        # Set zoom limits
        lowest_mag = float('inf')
        for filter_name in filters_band:
            mag_filter = mag_svd[filter_name]
            min_mag = np.min(mag_filter)
            if min_mag < lowest_mag:
                lowest_mag = min_mag
                sametimes_min = sample_times_svd[np.argmin(mag_filter)]
        x1 = sametimes_min - 1
        x2 = sametimes_min +1
        y1 = lowest_mag - 0.5
        y2 = lowest_mag + 3.5
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.invert_yaxis()
        axins.set_title('Zoom on Peak', fontsize=10, pad=2, y=0.9)

        mark_inset(axes[i,0], axins, loc1=2, loc2=1, ec='none', lw=1)

        # Show position on the grid
        norm_mag = colors.Normalize(vmin=mags.min(), vmax=mags.max())
        sc = axes[i,1].scatter(pair_array[:, 0], pair_array[:, 1], c=mags, cmap='viridis_r', 
                            s=100, edgecolors=None, linewidths=0.2, norm=norm_mag)
        cbar = fig.colorbar(sc, ax=axes[i,1], orientation='vertical', fraction=0.05)
        cbar.set_label('peak $M_{AB}$', fontsize=10)
        axes[i,1].scatter(pair_array[idx, 0], pair_array[idx, 1], 
                        c='cyan', s=200, marker='*', linewidths=2,
                        zorder=10)
        axes[i,1].text(pair_array[idx, 0], pair_array[idx, 1], f'  {i+1}', fontsize=12, fontweight='bold', color='cyan')
        axes[i,1].set_xlabel('$M_1$ [$M_\\odot$]', fontsize=10)
        axes[i,1].set_ylabel('$M_2$ [$M_\\odot$]', fontsize=10)
        axes[i,1].set_title(f'Position on the grid (Peak Magnitude in {filters_band[0]} band)', fontsize=11)
        axes[i,1].grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=16, y=1.0001)
    plt.tight_layout()
    plt.show()
    
import os
import glob
import numpy as np

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
    
def add_noise(mag, noise_level=0.3):
    noisy = {}
    for filter, mags in mag.items():
        noise = np.random.normal(0, noise_level, size=mags.shape)
        #print(max(noise), min(noise), np.mean(noise), np.std(noise))
        noisy[filter] = mags + noise
    return noisy

def add_errors(mag, max_error_level=0.6, min_error_level=0.1):
    errors = {}
    for filter, mags in mag.items():
        errors[filter] = np.random.uniform(min_error_level, max_error_level, size=mags.shape)
    return errors

def abs_to_app_mag(mag_abs, distance_mpc):
    distance_modulus = 5 * np.log10(distance_mpc) + 25 # distance in Mpc
    mag_app = {}
    for filter, mags in mag_abs.items():
        mag_app[filter] = mags + distance_modulus
    return mag_app

def format_nmma_data(times, mag, errors, filters, trigger_iso = '2025-01-01T00:00:00', timeshift=0.0):
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
        iso_dt = trigger_dt + pd.to_timedelta(t+timeshift, unit='D')
        # check if iso_dt is after the trigger time, if not, skip it
        if iso_dt.to_julian_date() - 2400000.5 < trigger_mjd:
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

def generate_synth_lc(sample_times,
                    model_name='Bu2019lm',
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
                    min_error_level=0.1,
                    max_error_level=0.6,
                    trigger_iso='2025-01-01T00:00:00',
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
    min_error_level : float
        Minimum error level to be added to the magnitudes.
    max_error_level : float
        Maximum error level to be added to the magnitudes.
    trigger_iso : str
        ISO formatted trigger time.
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
    svd_path = SVD_PATH
    try:
        svd_model = SVDLightCurveModel(
                model=model_name,
                sample_times=sample_times,
                svd_path=svd_path,
                mag_ncoeff=10,
                lbol_ncoeff=10,
                filters=filters_band
        )
    except Exception as e:
        print(f"  - {model_name}: error during SVD model creation ->", e)
    _, mag_svd = svd_model.generate_lightcurve(sample_times, model_param)
    mag_svd_noisy = add_noise(mag_svd, noise_level=noise_level)
    mag_svd_errors = add_errors(mag_svd_noisy, max_error_level=max_error_level, min_error_level=min_error_level)
    mag_svd_noisy_app = abs_to_app_mag(mag_svd_noisy, distance_mpc=model_param["luminosity_distance"])
    print("Timeshift:", model_param["timeshift"])
    data_nmma_svd, trigger = format_nmma_data(sample_times, mag_svd_noisy_app, mag_svd_errors, filters_band, trigger_iso=trigger_iso, timeshift=model_param["timeshift"])
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

