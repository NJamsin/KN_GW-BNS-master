import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_true_ts(first_det, true_merger):
    ts = pd.to_datetime(first_det) - pd.to_datetime(true_merger)
    ts = -1 * ts.total_seconds() / (3600*24) # convert to days
    return ts

def violin_plot_ts(grid_dir, true_merger='2020-01-07T00:00:00', num_lc=25, minus=5):
    print(f"Processing grid: {grid_dir}")
    quantiles_levels = [0.00135, 0.02275, 0.15865, 0.84135, 0.97725, 0.99865]

    # Couleurs et épaisseurs correspondantes (symétriques)
    # Ordre : 3-sigma (bas), 2-sigma (bas), 1-sigma (bas), 1-sigma (haut), 2-sigma (haut), 3-sigma (haut)
    q_colors = ['purple', 'blue', 'red', 'red', 'blue', 'purple']
    q_linewidths = [1, 1, 1, 1, 1, 1]
    save_path = f"{grid_dir}/plots/violin/"
    grid_dir = grid_dir + "/"

    # --- INITIALISATION DU PLOT ---
    # On crée une figure avec 1 ligne et 'minus' colonnes. sharey=True permet de partager l'axe des ordonnées (i)
    fig, axes = plt.subplots(nrows=1, ncols=minus, figsize=(minus*2.22, num_lc/2 -2.5), sharey=True)

    # --- AJOUT: Initialisation de la liste des stats ---
    global_stats = []

    # Boucles
    for j in range(minus): 
        # Listes pour stocker les données d'une colonne (pour un j donné) avant de plotter
        data_for_j = []
        positions_i = []
        
        for i in range(num_lc): 
            DIR = glob.glob(grid_dir + f"{i}/minus{j}/")
            if len(DIR) == 0:
                print(f"No directory found for index {i}, minus {j}")
                continue
                
            post_file = glob.glob(DIR[0] + "*posterior_samples.dat") 
            if len(post_file) == 0:
                print(f"No posterior file found in directory {DIR[0]}")
                continue
                
            post_df = pd.read_csv(post_file[0], delimiter=" ", dtype=np.float32)
            
            if j == 0:
                lc_file = glob.glob(grid_dir + f"{i}/data{i}.dat")
            else:
                lc_file = glob.glob(grid_dir + f"{i}/data_minus{j}.dat")
                
            if lc_file is None or len(lc_file) == 0:
                print(f"No LC file found for index {i} and minus {j}")
                continue
                
            lc_df = pd.read_csv(lc_file[0], delimiter=" ", header=None)
            first_det = lc_df[0].min() 
            true_ts = calculate_true_ts(first_det, true_merger)
            
            inferred_ts = post_df["timeshift"].values
            normalized_ts = inferred_ts - true_ts
            
            # Au lieu de plotter tout de suite, on ajoute les données à nos listes
            data_for_j.append(normalized_ts)
            positions_i.append(i)
            
        # --- CRÉATION DU VIOLIN PLOT POUR LA COLONNE j ---
        ax = axes[j]
        if data_for_j:
            # Création des violons avec les multiples quantiles
            parts = ax.violinplot(
                data_for_j, 
                positions=positions_i, 
                vert=False, 
                showmeans=False, 
                showmedians=False,
                showextrema=False,
                quantiles=[quantiles_levels] * len(data_for_j) 
            )
            
            # Application des couleurs et épaisseurs sur les barres de quantiles
            if 'cquantiles' in parts:
                parts['cquantiles'].set_color(q_colors * len(data_for_j))
                parts['cquantiles'].set_linewidth(q_linewidths * len(data_for_j))

            medians = [np.median(np.asarray(x, dtype=float)) for x in data_for_j]
            ax.scatter(medians, positions_i, color='orange', marker='o', s=20, zorder=3)

            # try to have a median distrib
            all_samples_j = np.concatenate(data_for_j)
            
            # --- AJOUT: Calcul des statistiques globales pour ce 'minus j' ---
            # On calcule les valeurs exactes des quantiles pour la distribution globale
            q_vals = np.quantile(all_samples_j, quantiles_levels)
            
            # Indices dans q_vals (basés sur tes quantiles_levels):
            # 0: 3-sigma bas | 1: 2-sigma bas | 2: 1-sigma bas
            # 3: 1-sigma haut| 4: 2-sigma haut| 5: 3-sigma haut
            
            width_1sigma = q_vals[3] - q_vals[2]
            width_2sigma = q_vals[4] - q_vals[1]
            width_3sigma = q_vals[5] - q_vals[0]
            
            # Médiane de la distribution globale (tu l'avais déjà définie plus bas)
            global_median = np.median(all_samples_j)
            
            # On stocke les résultats dans un dictionnaire
            global_stats.append({
                'minus_j': j,
                'median': global_median,
                'width_1sigma': width_1sigma,
                'width_2sigma': width_2sigma,
                'width_3sigma': width_3sigma
            })
            # ------------------------------------------------------------------

            # Violon global à la position -2
            global_pos = -2
            parts_global = ax.violinplot(
                [all_samples_j], 
                positions=[global_pos], 
                vert=False, 
                showmeans=False, 
                showmedians=False,
                showextrema=False,
                quantiles=[quantiles_levels]
            )
            
            # Personnalisation du violon global
            for pc in parts_global['bodies']:
                pc.set_facecolor('green')
                pc.set_edgecolor('green')
                pc.set_alpha(0.3) # Un peu plus transparent pour mieux voir les barres colorées
                
            # Couleurs des quantiles pour le violon global
            if 'cquantiles' in parts_global:
                parts_global['cquantiles'].set_color(q_colors)
                parts_global['cquantiles'].set_linewidth(q_linewidths)
                
            # Médiane de la distribution globale
            global_median = np.median(all_samples_j)
            ax.scatter([global_median], [global_pos], color='orange', marker='*', s=150, zorder=4)
            
        # Mise en forme du subplot courant
        ax.set_title(f"Minus {j}")
        
        # Ligne rouge pointillée à 0 pour bien voir le centrage de l'erreur
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # --- FINITIONS DU PLOT ---
    axes[0].set_ylabel("Index (i) - Lightcurves")
    y_ticks_positions = [-2] + list(range(num_lc))
    y_ticks_labels = ['Global'] + [str(i) for i in range(num_lc)]
    fig.supxlabel(r"$\Delta$ $t_0$ (Inferred - True) [days]", fontsize=14)

    axes[0].set_yticks(y_ticks_positions)
    axes[0].set_yticklabels(y_ticks_labels)

    axes[-1].tick_params(axis='y', labelright=True, right=True)
    axes[-1].set_ylabel("Index (i) - Lightcurves", rotation=270, labelpad=17)
    axes[-1].yaxis.set_label_position("right")

    plt.ylim(-4, num_lc)

    plt.tight_layout() 
    os.makedirs(save_path, exist_ok=True) 
    plt.savefig(f"{save_path}ts_violin_plot.png", dpi=300)
    plt.close()
    
    # --- AJOUT: Sauvegarde et affichage des statistiques ---
    stats_df = pd.DataFrame(global_stats)
    stats_csv_path = f"{save_path}global_timeshift_stats.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    
    print("\n--- Statistiques Globales par 'Minus' ---")
    print(stats_df.to_string(index=False))
    print(f"Statistiques sauvegardées dans : {stats_csv_path}\n")
    # -------------------------------------------------------

    return 0
    



def violin_plot_mej(grid_dir, num_lc=25, minus=5, type="dyn"):
    print(f"Processing grid: {grid_dir}")
    quantiles_levels = [0.00135, 0.02275, 0.15865, 0.84135, 0.97725, 0.99865]

    # Couleurs et épaisseurs correspondantes (symétriques)
    # Ordre : 3-sigma (bas), 2-sigma (bas), 1-sigma (bas), 1-sigma (haut), 2-sigma (haut), 3-sigma (haut)
    q_colors = ['purple', 'blue', 'red', 'red', 'blue', 'purple']
    q_linewidths = [1, 1, 1, 1, 1, 1]
    if "ka" in grid_dir:
        return 1 # Skip this grid as it doesn't have the required data structure for this plot
    # Paramètres
    save_path = f'{grid_dir}/plots/violin/'
    grid_dir = grid_dir + "/"

    # --- INITIALISATION DU PLOT ---
    # On crée une figure avec 1 ligne et 'minus' colonnes. sharey=True permet de partager l'axe des ordonnées (i)
    fig, axes = plt.subplots(nrows=1, ncols=minus, figsize=(minus*2.22, num_lc/2 -2.5), sharey=True)

    # Boucles
    for j in range(minus): 
        # Listes pour stocker les données d'une colonne (pour un j donné) avant de plotter
        data_for_j = []
        positions_i = []
        
        for i in range(num_lc): 
            DIR = glob.glob(grid_dir + f"{i}/minus{j}/")
            if len(DIR) == 0:
                print(f"No directory found for index {i}, minus {j}")
                continue
                
            post_file = glob.glob(DIR[0] + "*posterior_samples.dat") 
            if len(post_file) == 0:
                print(f"No posterior file found in directory {DIR[0]}")
                continue
                
            post_df = pd.read_csv(post_file[0], delimiter=" ", dtype=np.float32)
            
            truth_file = glob.glob(grid_dir + f"{i}/true{i}.csv")
                
            if truth_file is None or len(truth_file) == 0:
                print(f"No truth file found for index {i} and minus {j}")
                continue
            truth_df = pd.read_csv(truth_file[0])
            true_m = truth_df[f"log10_mej_{type}"].values[0] # works fine
            
            inferred_m = post_df[f"log10_mej_{type}"].values
            normalized_m = inferred_m - true_m
            
            # Au lieu de plotter tout de suite, on ajoute les données à nos listes
            data_for_j.append(normalized_m)
            positions_i.append(i)
            
        # --- CRÉATION DU VIOLIN PLOT POUR LA COLONNE j ---
        ax = axes[j]
        if data_for_j:
            # vert=False met les violons à l'horizontale
            # quantiles=[[0.16, 0.84]] dessine les barres pour l'intervalle de confiance à 1-sigma
            parts = ax.violinplot(
                data_for_j, 
                positions=positions_i, 
                vert=False, 
                showmeans=False, 
                showmedians=False,
                showextrema=False,
                quantiles=[quantiles_levels] * len(data_for_j) 
            )

            if 'cquantiles' in parts:
                parts['cquantiles'].set_color(q_colors * len(data_for_j))
                parts['cquantiles'].set_linewidth(q_linewidths * len(data_for_j))
            medians = [np.median(np.asarray(x, dtype=float)) for x in data_for_j]
            ax.scatter(medians, positions_i, color='orange', marker='o', s=20)

            # try to have a median distrib
            all_samples_j = np.concatenate(data_for_j)
            
            # On trace un violon spécial à la position -2
            global_pos = -2
            parts_global = ax.violinplot(
                [all_samples_j], 
                positions=[global_pos], 
                vert=False, 
                showmeans=False, 
                showmedians=False,
                showextrema=False,
                quantiles=[quantiles_levels]
            )
            
            # On change la couleur du violon global pour qu'il ressorte (ex: en vert)
            for pc in parts_global['bodies']:
                pc.set_facecolor('green')
                pc.set_edgecolor('green')
                pc.set_alpha(0.3) # Un peu plus transparent pour mieux voir les barres colorées
                
            # Couleurs des quantiles pour le violon global
            if 'cquantiles' in parts_global:
                parts_global['cquantiles'].set_color(q_colors)
                parts_global['cquantiles'].set_linewidth(q_linewidths)
                
            # Médiane de la distribution globale (étoile orange pour faire le lien avec les ronds)
            global_median = np.median(all_samples_j)
            ax.scatter([global_median], [global_pos], color='orange', marker='*', s=150, zorder=4)
            
        # Mise en forme du subplot courant
        ax.set_title(f"Minus {j}")
        #ax.set_xlabel(r"$\Delta$ TS (Inferred - True) [days]")
        
        # Ligne rouge pointillée à 0 pour bien voir le centrage de l'erreur
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # --- FINITIONS DU PLOT ---
    # On met le label Y uniquement sur la première colonne pour alléger le graphique
    axes[0].set_ylabel("Index (i) - Lightcurves")
    y_ticks_positions = [-2] + list(range(num_lc))
    y_ticks_labels = ['Global'] + [str(i) for i in range(num_lc)]
    if type == "dyn":
        fig.supxlabel(r"$\Delta$ $log_{10}(M_{ej}^{dyn})$ (Inferred - True)", fontsize=14)
    else:
        fig.supxlabel(r"$\Delta$ $log_{10}(M_{ej}^{wind})$ (Inferred - True)", fontsize=14)

    axes[0].set_yticks(y_ticks_positions)
    axes[0].set_yticklabels(y_ticks_labels)

    axes[-1].tick_params(axis='y', labelright=True, right=True)  # Affiche les ticks à droite
    # On ajoute le titre de l'axe Y à droite, en le tournant pour qu'il soit lisible de haut en bas
    axes[-1].set_ylabel("Index (i) - Lightcurves", rotation=270, labelpad=17)
    axes[-1].yaxis.set_label_position("right")

    plt.ylim(-4, num_lc)

    #plt.suptitle("Diagnostic Plot: Normalized Timeshift evolution over 'minus' columns", fontsize=16)
    plt.tight_layout() # Ajuste automatiquement les marges pour que rien ne se chevauche
    os.makedirs(save_path, exist_ok=True) # Assure que le dossier de sauvegarde existe
    plt.savefig(f"{save_path}mej{type}_violin_plot.png", dpi=300)
    plt.close()
    return 0
    

import json
# 1. AJOUT DE BoundaryNorm ICI :
from matplotlib.colors import ListedColormap, BoundaryNorm 

def plot_chi2_vs_ts(grid_dir, true_merger='2020-01-07T00:00:00', num_lc=25, minus=5, n_sig=3):
    print(f"Processing grid: {grid_dir}")
    # Paramètres
    save_path = f'{grid_dir}/plots/chi2/'
    grid_dir = grid_dir + "/"

    os.makedirs(save_path, exist_ok=True)

    filters = ["ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y", "total"]

    # --- CRÉATION DE LA PALETTE DE 25 COULEURS DISTINCTES ---
    distinct_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors[:5]
    cmap_25 = ListedColormap(distinct_colors)

    # 2. CRÉATION DU BOUNDARY NORM POUR FORCER L'ALIGNEMENT :
    # On crée des frontières exactes à -0.5, 0.5, 1.5... jusqu'à 24.5
    bounds = np.arange(-0.5, num_lc + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap_25.N)

    # --- BOUCLE PRINCIPALE SUR LES MINUS ---
    for j in range(minus): 
        
        plot_data = {f: {'chi2': [], 'y_val': [], 'lc_idx': []} for f in filters}
        
        for i in range(num_lc): 
            DIR = glob.glob(grid_dir + f"{i}/minus{j}/")
            if len(DIR) == 0: continue
                
            post_file = glob.glob(DIR[0] + "*posterior_samples.dat") 
            best_fit_file = glob.glob(DIR[0] + "*bestfit_params.json")
            if len(post_file) == 0 or len(best_fit_file) == 0: continue
                
            post_df = pd.read_csv(post_file[0], delimiter=" ", dtype=np.float32)
            best_fit = json.load(open(best_fit_file[0]))
            
            chi_dic = best_fit.get("chi2_dict_raw", best_fit.get("chi2_dict", {}))
            
            if j == 0:
                lc_file = glob.glob(grid_dir + f"{i}/data{i}.dat")
            else:
                lc_file = glob.glob(grid_dir + f"{i}/data_minus{j}.dat")
                
            if lc_file is None or len(lc_file) == 0: continue

            lc_df = pd.read_csv(lc_file[0], delimiter=" ", header=None)
            first_det = lc_df[0].min() 
            true_ts = calculate_true_ts(first_det, true_merger)
            
            inferred_ts = post_df["timeshift"].values
            normalized_ts = inferred_ts - true_ts
            
            if n_sig == 3:
                p16 = np.percentile(normalized_ts, 0.135)
                p84 = np.percentile(normalized_ts, 99.865) # 3 sigmas
            elif n_sig == 2:
                p16 = np.percentile(normalized_ts, 2.275)
                p84 = np.percentile(normalized_ts, 97.725) # 2 sigmas
            else:
                p16 = np.percentile(normalized_ts, 16)
                p84 = np.percentile(normalized_ts, 84) # 1 sigmas
            
            if p16 <= 0 <= p84:
                y_val = 0.0
            elif p16 > 0:
                y_val = p16
            else: 
                y_val = abs(p84)

            for f in filters:
                val = chi_dic.get(f, np.nan)
                if not np.isnan(val) and val > 0:
                    plot_data[f]['chi2'].append(val)
                    plot_data[f]['y_val'].append(y_val)
                    plot_data[f]['lc_idx'].append(i)

        has_data = any(len(plot_data[f]['chi2']) > 0 for f in filters)
        if not has_data: continue

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), sharey=True)
        axes = axes.flatten()
        sc = None 
        
        for idx, filt in enumerate(filters):
            ax = axes[idx]
            
            chi2 = np.array(plot_data[filt]['chi2'])
            y_val = np.array(plot_data[filt]['y_val'])
            lc_idx = np.array(plot_data[filt]['lc_idx'])
            
            if len(chi2) > 0:
                # 3. ON UTILISE norm=norm À LA PLACE DE vmin ET vmax :
                sc = ax.scatter(
                    chi2, y_val, 
                    c=lc_idx, cmap=cmap_25, norm=norm,
                    edgecolors='k', alpha=0.9, s=70
                )

            ax.set_title(f"{filt}")
            ax.grid(True, linestyle=':', alpha=0.6, zorder=0)
            ax.set_xscale('log')
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=1)
            
            if idx >= 3: 
                ax.set_xlabel(r"$\chi^2$ (Log Scale)")
            if idx % 3 == 0: 
                ax.set_ylabel(rf"Min ${n_sig}\sigma$ Deviation from Truth [days]")
                
        plt.suptitle(f"Minus {j}", fontsize=16, y=0.95)
        plt.subplots_adjust(left=0.06, bottom=0.08, right=0.90, top=0.90, wspace=0.1, hspace=0.15)
        
        if sc is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
            cbar = fig.colorbar(sc, cax=cbar_ax, ticks=range(num_lc))
            cbar.set_label('Lightcurve Index', fontsize=12)
        
        plt.savefig(save_path + f"coverage_chi2_vs_ts_minus{j}_{n_sig}sig.png", dpi=300)
        plt.close()
    return 0
        


def plot_detections_vs_ts(grid_dir, true_merger='2020-01-07T00:00:00', num_lc=25, minus=5, n_sig=3):
    print(f"Processing grid: {grid_dir}")
    save_path = f'{grid_dir}/plots/detections/'
    grid_dir = grid_dir + "/"

    os.makedirs(save_path, exist_ok=True)

    filters = ["ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y", "total"]

    # --- CRÉATION DE LA PALETTE DE 25 COULEURS DISTINCTES ---
    distinct_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors[:5]
    cmap_25 = ListedColormap(distinct_colors)

    # CRÉATION DU BOUNDARY NORM POUR FORCER L'ALIGNEMENT :
    bounds = np.arange(-0.5, num_lc + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap_25.N)

    # --- BOUCLE PRINCIPALE SUR LES MINUS ---
    for j in range(minus): 
        
        # Remplacement de 'chi2' par 'num_det' (number of detections)
        plot_data = {f: {'num_det': [], 'y_val': [], 'lc_idx': []} for f in filters}
        
        for i in range(num_lc): 
            DIR = glob.glob(grid_dir + f"{i}/minus{j}/")
            if len(DIR) == 0: continue
                
            post_file = glob.glob(DIR[0] + "*posterior_samples.dat") 
            if len(post_file) == 0: continue
                
            post_df = pd.read_csv(post_file[0], delimiter=" ", dtype=np.float32)
            
            if j == 0:
                lc_file = glob.glob(grid_dir + f"{i}/data{i}.dat")
            else:
                lc_file = glob.glob(grid_dir + f"{i}/data_minus{j}.dat")
                
            if lc_file is None or len(lc_file) == 0: continue

            # Lecture du fichier de la courbe de lumière
            lc_df = pd.read_csv(lc_file[0], delimiter=" ", header=None)
            
            # --- COMPTAGE DES DÉTECTIONS ---
            det_counts = {"total": len(lc_df)} # Total = nombre de lignes
            # On compte les occurrences de chaque filtre (colonne 1)
            filter_counts = lc_df[1].value_counts().to_dict()
            for f in filters:
                if f != "total":
                    det_counts[f] = filter_counts.get(f, 0)
            
            first_det = lc_df[0].min() 
            true_ts = calculate_true_ts(first_det, true_merger)
            
            inferred_ts = post_df["timeshift"].values
            normalized_ts = inferred_ts - true_ts
            
            if n_sig == 3:
                p16 = np.percentile(normalized_ts, 0.135)
                p84 = np.percentile(normalized_ts, 99.865) # 3 sigmas
            elif n_sig == 2:
                p16 = np.percentile(normalized_ts, 2.275)
                p84 = np.percentile(normalized_ts, 97.725) # 2 sigmas
            else:
                p16 = np.percentile(normalized_ts, 16)
                p84 = np.percentile(normalized_ts, 84) # 1 sigma

            if p16 <= 0 <= p84:
                y_val = 0.0
            elif p16 > 0:
                y_val = p16
            else: 
                y_val = abs(p84)

            for f in filters:
                val = det_counts.get(f, 0)
                if val > 0: # On n'ajoute le point que s'il y a au moins 1 détection
                    plot_data[f]['num_det'].append(val)
                    plot_data[f]['y_val'].append(y_val)
                    plot_data[f]['lc_idx'].append(i)

        has_data = any(len(plot_data[f]['num_det']) > 0 for f in filters)
        if not has_data: continue

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), sharey=True)
        axes = axes.flatten()
        sc = None 
        
        for idx, filt in enumerate(filters):
            ax = axes[idx]
            
            num_det = np.array(plot_data[filt]['num_det'])
            y_val = np.array(plot_data[filt]['y_val'])
            lc_idx = np.array(plot_data[filt]['lc_idx'])
            
            if len(num_det) > 0:
                # AJOUT DU JITTER : On décale très légèrement les points sur l'axe X 
                # pour éviter qu'ils ne se cachent si plusieurs LC ont le même nombre de détections.
                jitter = np.random.uniform(-0.15, 0.15, size=len(num_det))
                
                sc = ax.scatter(
                    num_det + jitter, y_val, 
                    c=lc_idx, cmap=cmap_25, norm=norm,
                    edgecolors='k', alpha=0.8, s=70
                )

            ax.set_title(f"{filt}")
            ax.grid(True, linestyle=':', alpha=0.6, zorder=0)
            
            # On supprime l'échelle log sur X, car ce sont des entiers (1, 2, 3...)
            # On force matplotlib à n'afficher que des entiers sur l'axe X
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=1)
            
            if idx >= 3: 
                ax.set_xlabel("Number of Detections")
            if idx % 3 == 0: 
                ax.set_ylabel(rf"Min ${n_sig}\sigma$ Deviation from Truth [days]")
                
        plt.suptitle(f"Minus {j}", fontsize=16, y=0.95)
        plt.subplots_adjust(left=0.06, bottom=0.08, right=0.90, top=0.90, wspace=0.1, hspace=0.15)
        
        if sc is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
            cbar = fig.colorbar(sc, cax=cbar_ax, ticks=range(num_lc))
            cbar.set_label('Lightcurve Index (i)', fontsize=12)
        
        plt.savefig(save_path + f"coverage_det_vs_ts_minus{j}_{n_sig}sig.png", dpi=300)
        plt.close()
    return 0


from matplotlib.colors import ListedColormap, BoundaryNorm 

def plot_kldiv_vs_shrinkage(grid_dir, num_lc=25, minus=5, prior_width_days=8):
    print(f"Processing grid: {grid_dir}")
# --- PARAMÈTRES ---
    save_path = f'{grid_dir}/plots/comparison/'
    grid_dir = grid_dir + "/"

    # /!\ À MODIFIER SELON TON PRIOR DANS NMMA /!\
    PRIOR_WIDTH_DAYS = prior_width_days
    SIGMA_PRIOR = PRIOR_WIDTH_DAYS / np.sqrt(12) 

    os.makedirs(save_path, exist_ok=True)

    # --- CRÉATION DE LA PALETTE ---
    distinct_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors[:5]
    cmap_25 = ListedColormap(distinct_colors)
    bounds = np.arange(-0.5, num_lc + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap_25.N)

    # --- STRUCTURE DE DONNÉES PAR LIGHTCURVE ---
    plot_data = {i: {'minus_j': [], 'kl_div': [], 'shrinkage': []} for i in range(num_lc)}

    # --- RÉCOLTE DES DONNÉES ---
    for j in range(minus): 
        for i in range(num_lc): 
            DIR = glob.glob(grid_dir + f"{i}/minus{j}/")
            if len(DIR) == 0: continue
                
            post_file = glob.glob(DIR[0] + "*posterior_samples.dat") 
            if len(post_file) == 0: continue
                
            post_df = pd.read_csv(post_file[0], delimiter=" ", dtype=np.float32)
            sigma_post = np.std(post_df["timeshift"].values)
                
            # 1. Calcul de la KL Divergence (en bits)
            entropy_prior = np.log2(PRIOR_WIDTH_DAYS)
            entropy_post = 0.5 * np.log2(2 * np.pi * np.e * sigma_post**2)
            kl_div = max(0.0, entropy_prior - entropy_post)
                
            # 2. Calcul du Shrinkage (linéaire)
            shrinkage = max(0.0, 1.0 - (sigma_post / SIGMA_PRIOR))

            plot_data[i]['minus_j'].append(j)
            plot_data[i]['kl_div'].append(kl_div)
            plot_data[i]['shrinkage'].append(shrinkage)

    # --- CRÉATION DE LA FIGURE COMPARATIVE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for i in range(num_lc):
        if len(plot_data[i]['minus_j']) > 0:
            color = cmap_25(norm(i))
            
            # Panneau 1 : KL Divergence (Gain d'info log)
            ax1.plot(plot_data[i]['minus_j'], plot_data[i]['kl_div'], 
                    marker='o', markersize=8, color=color, linewidth=2.5, alpha=0.8)
            
            # Panneau 2 : Shrinkage (Gain d'info linéaire)
            ax2.plot(plot_data[i]['minus_j'], plot_data[i]['shrinkage'], 
                    marker='o', markersize=8, color=color, linewidth=2.5, alpha=0.8)

    # --- FINITIONS KL DIV ---
    ax1.set_title("Information Gain ($D_{KL}$)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Minus $j$", fontsize=12)
    ax1.set_ylabel("KL Divergence [bits]", fontsize=12)
    ax1.set_xticks(range(minus))
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_ylim(bottom=-0.1)

    # --- FINITIONS SHRINKAGE ---
    ax2.set_title("Posterior Shrinkage ($S$)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Minus $j$", fontsize=12)
    ax2.set_ylabel("Shrinkage (normalized)", fontsize=12)
    ax2.set_xticks(range(minus))
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.axhline(1, color='green', linestyle=':', alpha=0.5)
    ax2.axhline(0, color='red', linestyle=':', alpha=0.5)

    # --- BARRE DE COULEUR ---
    sm = plt.cm.ScalarMappable(cmap=cmap_25, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=range(num_lc))
    cbar.set_label('Lightcurve Index (i)', fontsize=12)

    #plt.suptitle("Comparison of Information Gain Metrics for Timeshift Inference", fontsize=16, y=0.96)
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.90, top=0.90, wspace=0.2)

    plt.savefig(save_path + "kldiv_vs_shrinkage.png", dpi=300)
    plt.close()
    return 0

def violin_plot_mchirp(grid_dir, num_lc=25, minus=5):
    print(f"Processing grid: {grid_dir}")
    if "ka" in grid_dir:
        return 1 # Skip this grid as it doesn't have the required data structure for this plot
    quantiles_levels = [0.00135, 0.02275, 0.15865, 0.84135, 0.97725, 0.99865]

    # Couleurs et épaisseurs correspondantes (symétriques)
    # Ordre : 3-sigma (bas), 2-sigma (bas), 1-sigma (bas), 1-sigma (haut), 2-sigma (haut), 3-sigma (haut)
    q_colors = ['purple', 'blue', 'red', 'red', 'blue', 'purple']
    q_linewidths = [1, 1, 1, 1, 1, 1]
    save_path = f"{grid_dir}/plots/violin/"
    grid_dir = grid_dir + "/"

    # --- INITIALISATION DU PLOT ---
    # On crée une figure avec 1 ligne et 'minus' colonnes. sharey=True permet de partager l'axe des ordonnées (i)
    fig, axes = plt.subplots(nrows=1, ncols=minus, figsize=(minus*2.22, num_lc/2 -2.5), sharey=True)

    # Boucles
    for j in range(minus): 
        # Listes pour stocker les données d'une colonne (pour un j donné) avant de plotter
        data_for_j = []
        positions_i = []
        
        for i in range(num_lc): 
            DIR = glob.glob(grid_dir + f"{i}/minus{j}/")
            if len(DIR) == 0:
                print(f"No directory found for index {i}, minus {j}")
                continue
                
            post_file = glob.glob(DIR[0] + "resamp/posterior_samples.dat") 
            if len(post_file) == 0:
                print(f"No posterior file found in directory {DIR[0]}")
                continue
                
            post_df = pd.read_csv(post_file[0], delimiter=" ", dtype=np.float32)
            
            truth_file = glob.glob(grid_dir + f"{i}/true{i}.csv")

            if truth_file is None or len(truth_file) == 0:
                print(f"No truth file found for index {i} and minus {j}")
                continue
            truth_df = pd.read_csv(truth_file[0])

            m1 = truth_df["mass_1"].values[0]
            m2 = truth_df["mass_2"].values[0]
            true_mchirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            inferred_mchirp = post_df["chirp_mass"].values
            normalized_mchirp = inferred_mchirp - true_mchirp
            # Au lieu de plotter tout de suite, on ajoute les données à nos listes
            data_for_j.append(normalized_mchirp)
            positions_i.append(i)

        # --- CRÉATION DU VIOLIN PLOT POUR LA COLONNE j ---
        ax = axes[j]
        if data_for_j:
            # vert=False met les violons à l'horizontale
            # quantiles=[[0.16, 0.84]] dessine les barres pour l'intervalle de confiance à 1-sigma
            parts = ax.violinplot(
                data_for_j, 
                positions=positions_i, 
                vert=False, 
                showmeans=False, 
                showmedians=False,
                showextrema=False,
                quantiles=[quantiles_levels] * len(data_for_j) 
            )

            if 'cquantiles' in parts:
                parts['cquantiles'].set_color(q_colors * len(data_for_j))
                parts['cquantiles'].set_linewidth(q_linewidths * len(data_for_j))
            medians = [np.median(np.asarray(x, dtype=float)) for x in data_for_j]
            ax.scatter(medians, positions_i, color='orange', marker='o', s=20)

            # try to have a median distrib
            all_samples_j = np.concatenate(data_for_j)
            
            # On trace un violon spécial à la position -2
            global_pos = -2
            parts_global = ax.violinplot(
                [all_samples_j], 
                positions=[global_pos], 
                vert=False, 
                showmeans=False, 
                showmedians=False,
                showextrema=False,
                quantiles=[quantiles_levels]
            )
            
            # On change la couleur du violon global pour qu'il ressorte (ex: en vert)
            for pc in parts_global['bodies']:
                pc.set_facecolor('green')
                pc.set_edgecolor('green')
                pc.set_alpha(0.3) # Un peu plus transparent pour mieux voir les barres colorées
                
            # Couleurs des quantiles pour le violon global
            if 'cquantiles' in parts_global:
                parts_global['cquantiles'].set_color(q_colors)
                parts_global['cquantiles'].set_linewidth(q_linewidths)
                
            # Médiane de la distribution globale (étoile orange pour faire le lien avec les ronds)
            global_median = np.median(all_samples_j)
            ax.scatter([global_median], [global_pos], color='orange', marker='*', s=150, zorder=4)
            
        # Mise en forme du subplot courant
        ax.set_title(f"Minus {j}")
        #ax.set_xlabel(r"$\Delta$ TS (Inferred - True) [days]")
        
        # Ligne rouge pointillée à 0 pour bien voir le centrage de l'erreur
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # --- FINITIONS DU PLOT ---
    # On met le label Y uniquement sur la première colonne pour alléger le graphique
    axes[0].set_ylabel("Index (i) - Lightcurves")
    y_ticks_positions = [-2] + list(range(num_lc))
    y_ticks_labels = ['Global'] + [str(i) for i in range(num_lc)]
    fig.supxlabel(r"$\Delta$ $\mathcal{M}$ (Inferred - True)", fontsize=14)

    axes[0].set_yticks(y_ticks_positions)
    axes[0].set_yticklabels(y_ticks_labels)

    axes[-1].tick_params(axis='y', labelright=True, right=True)  # Affiche les ticks à droite
    # On ajoute le titre de l'axe Y à droite, en le tournant pour qu'il soit lisible de haut en bas
    axes[-1].set_ylabel("Index (i) - Lightcurves", rotation=270, labelpad=17)
    axes[-1].yaxis.set_label_position("right")

    plt.ylim(-4, num_lc)

    #plt.suptitle("Diagnostic Plot: Normalized Timeshift evolution over 'minus' columns", fontsize=16)
    plt.tight_layout() # Ajuste automatiquement les marges pour que rien ne se chevauche
    os.makedirs(save_path, exist_ok=True) # Assure que le dossier de sauvegarde existe
    plt.savefig(f"{save_path}mchirp_violin_plot.png", dpi=300)
    plt.close()
    return 0