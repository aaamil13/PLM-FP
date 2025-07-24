# Файл: mcmc_analysis/runners/plot_hubble_diagram.py

import numpy as np
import sys
import os
import logging
import matplotlib.pyplot as plt
import emcee

# Добавяме директориите на проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Импортираме моделите и данните за SN
from mcmc_analysis.models.plm_model_fp import PLM
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def get_best_fit_from_mcmc(hdf5_file, n_burnin=1000):
    """Extracts the best-fit parameters (median) from the HDF5 file."""
    try:
        backend = emcee.backends.HDFBackend(hdf5_file, read_only=True)
        flat_samples = backend.get_chain(discard=1000, flat=True)
        return np.percentile(flat_samples, 50, axis=0)
    except Exception as e:
        logging.error(f"Could not load HDF5 file '{hdf5_file}': {e}")
        return None

def main():
    logging.info("Generating Hubble Diagram (μ vs z)...")

    # --- 1. Load Data and Models ---
    try:
        sn_likelihood = SupernovaeLikelihood()
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # PLM Parameters from best-fit (CMB-constrained)
    plm_hdf5_file = os.path.join(os.path.dirname(__file__), '../results/PLM_CMB_constrained_optimized_checkpoint.h5')
    plm_best_params_7 = get_best_fit_from_mcmc(plm_hdf5_file, n_burnin=1000)
    if plm_best_params_7 is None: return

    # Separating cosmological and nuisance parameters for PLM
    plm_cosmo_params = [plm_best_params_7[0], plm_best_params_7[1], plm_best_params_7[2], plm_best_params_7[3], plm_best_params_7[4], 0.01]
    delta_M_plm = plm_best_params_7[5]
    z_local_plm = plm_best_params_7[6]
    plm_model = PLM(*plm_cosmo_params)

    # ΛCDM Parameters
    lcdm_params = [67.36, 0.1428, 0.02237, 0.9649, 2.1e-9, 0.0544]
    lcdm_model = LCDM(*lcdm_params)
    # For fair comparison, we will adjust ΛCDM with the same delta_M found for PLM
    delta_M_adjusted_for_lcdm = delta_M_plm

    # --- 2. Calculating theoretical curves ---
    z_grid = np.geomspace(min(sn_likelihood.z), max(sn_likelihood.z), 200) # Smooth curve
    
    # Adjust z for PLM model
    z_theory_plm = (1.0 + z_grid) / (1.0 + z_local_plm) - 1.0
    
    mu_plm = np.array([plm_model.distance_modulus(z) for z in z_theory_plm]) + delta_M_plm
    mu_lcdm = np.array([lcdm_model.distance_modulus(z) for z in z_grid]) + delta_M_adjusted_for_lcdm

    # --- 3. Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # --- Top part: Hubble Diagram ---
    # Data points. We will subtract ΔM to be consistent with μ_model, which does not include it.
    # For clarity, we will plot μ_data against μ_model + ΔM.
    # For comparability, we will also subtract μ_ΛCDM(z=0) from the data.
    # It's simpler to plot μ_data - (5*log10(cz) + const)
    
    # Let's just plot μ_data and theoretical curves.
    ax1.errorbar(sn_likelihood.z, sn_likelihood.mu_obs, 
                 yerr=np.sqrt(np.diag(sn_likelihood.cov_matrix)), # Diagonal errors only for visualization
                 fmt='.', color='gray', alpha=0.2, label='Pantheon+ Data')
                 
    ax1.plot(z_grid, mu_plm, color='royalblue', linewidth=2, label='PLM Model (best fit)')
    ax1.plot(z_grid, mu_lcdm, color='green', linestyle='--', linewidth=2, label='ΛCDM Model (Planck 2018)')
    
    ax1.set_ylabel('Distance Modulus (μ)')
    ax1.set_title('Hubble Diagram: PLM vs ΛCDM')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")
    ax1.set_xscale('log')

    # --- Bottom part: Residuals relative to PLM model ---
    # Calculate μ_model for PLM for each data point
    z_theory_plm_data = (1.0 + sn_likelihood.z) / (1.0 + z_local_plm) - 1.0
    mu_plm_data = np.array([plm_model.distance_modulus(z) for z in z_theory_plm_data]) + delta_M_plm
    
    residuals = sn_likelihood.mu_obs - mu_plm_data
    
    ax2.scatter(sn_likelihood.z, residuals, s=10, alpha=0.4, color='royalblue')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    
    ax2.set_xlabel('Redshift (z)')
    ax2.set_ylabel('Residuals (Δμ)')
    ax2.set_ylim(-1.5, 1.5) # Zoom around zero
    ax2.grid(True, which="both", ls="--")

    # --- Final settings and saving ---
    plt.tight_layout()
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "hubble_diagram_comparison.png")
    
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Hubble diagram saved to: {plot_path}")

if __name__ == "__main__":
    main()
