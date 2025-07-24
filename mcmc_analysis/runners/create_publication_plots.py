# Файл: mcmc_analysis/runners/create_publication_plots.py

import numpy as np
import sys
import os
import logging
import matplotlib.pyplot as plt
import emcee
from matplotlib import rcParams

# Добавяме директориите на проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.models.plm_model_fp import PLM
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# --- Настройки за publication-ready plots ---
rcParams['font.size'] = 16
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.major.size'] = 8
rcParams['ytick.major.size'] = 8
rcParams['xtick.minor.size'] = 4
rcParams['ytick.minor.size'] = 4
rcParams['legend.frameon'] = False


def get_best_fit_from_mcmc(hdf5_file, n_burnin=1000):
    try:
        backend = emcee.backends.HDFBackend(hdf5_file, read_only=True)
        flat_samples = backend.get_chain(discard=1000, flat=True)
        return np.percentile(flat_samples, 50, axis=0)
    except Exception as e:
        logging.error(f"Грешка при зареждане на HDF5 файл '{hdf5_file}': {e}")
        return None

def main():
    # --- 1. Зареждане на данни и най-добрите модели ---
    sn_likelihood = SupernovaeLikelihood()
    
    plm_hdf5_file = os.path.join(os.path.dirname(__file__), '../results/PLM_CMB_constrained_optimized_checkpoint.h5')
    plm_params_7 = get_best_fit_from_mcmc(plm_hdf5_file, n_burnin=1000)
    if plm_params_7 is None: return

    plm_cosmo_params = [plm_params_7[0], plm_params_7[1], plm_params_7[2], plm_params_7[3], plm_params_7[4], 0.01]
    delta_M_plm = plm_params_7[5]
    z_local_plm = plm_params_7[6]
    plm_model = PLM(*plm_cosmo_params)

    lcdm_params = [67.36, 0.1428, 0.02237, 0.9649, 2.1e-9, 0.0544]
    lcdm_model = LCDM(*lcdm_params)

    # --- 2. Генериране на Фигура 1: Физика на PLM модела ---
    logging.info("Генериране на Фигура 1: Физика на модела...")
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('Physical Components of the PLM-FP Model', fontsize=18, y=0.95)

    z_range_wide = np.logspace(-3, 3, 500)
    
    # Plot 1: H(z) evolution
    h_plm = np.array([plm_model.H_of_z(z) for z in z_range_wide])
    h_lcdm = np.array([lcdm_model.H_of_z(z) for z in z_range_wide])
    ax1.plot(z_range_wide, h_plm, color='royalblue', linewidth=2.5, label='PLM Model')
    ax1.plot(z_range_wide, h_lcdm, color='green', linestyle='--', linewidth=2.5, label='ΛCDM Model')
    ax1.set_xlabel('Redshift (z)')
    ax1.set_ylabel('H(z) [km s⁻¹ Mpc⁻¹]')
    ax1.set_title('Hubble Parameter Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    ax1.set_xscale('log')

    # Plot 2: Time Dilation dτ/dt
    dtau_dt = np.array([plm_model._time_dilation(z) for z in z_range_wide])
    ax2.plot(z_range_wide, dtau_dt, color='red', linewidth=2.5)
    ax2.set_xlabel('Redshift (z)')
    ax2.set_ylabel('dτ/dt')
    ax2.set_title('Time Dilation Evolution')
    ax2.grid(True, alpha=0.5)
    ax2.set_xscale('log')

    # Plot 3: Bound fraction evolution
    f_bound = np.array([plm_model._f_bound(z) for z in z_range_wide])
    ax3.plot(z_range_wide, f_bound, color='purple', linewidth=2.5)
    ax3.set_xlabel('Redshift (z)')
    ax3.set_ylabel('Bound Matter Fraction f_bound(z)')
    ax3.set_title('Structure Formation Evolution')
    ax3.grid(True, alpha=0.5)
    ax3.set_xscale('log')

    # Plot 4: Effective w(z)
    H0_plm, Omega_m_plm, Omega_r_plm = plm_model.params['H0'], plm_model.Omega_m, plm_model.Omega_r
    h_sq_plm = (h_plm / H0_plm)**2
    omega_de = h_sq_plm - Omega_m_plm * (1 + z_range_wide)**3 - Omega_r_plm * (1 + z_range_wide)**4
    omega_de[omega_de < 0] = 1e-9
    log_rho_de = np.log(omega_de)
    log_a = -np.log(1 + z_range_wide)
    d_log_rho_d_log_a = np.gradient(log_rho_de, log_a)
    w_eff = -1.0 - (1.0 / 3.0) * d_log_rho_d_log_a
    ax4.plot(z_range_wide, w_eff, color='darkorange', linewidth=2.5)
    ax4.axhline(y=-1, color='black', linestyle='--', alpha=0.5, label='ΛCDM (w=-1)')
    ax4.set_xlabel('Redshift (z)')
    ax4.set_ylabel('w_eff(z)')
    ax4.set_title('Effective Equation of State')
    ax4.legend()
    ax4.grid(True, alpha=0.5)
    ax4.set_xscale('log')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    plot_path1 = os.path.join(results_dir, "figure1_model_physics.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    logging.info(f"Figure 1 (model physics) saved to: {plot_path1}")


    # --- 3. Generate Figure 2: Hubble Diagram and Residuals ---
    logging.info("Generating Figure 2: Hubble Diagram...")
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # Горна част - Hubble Диаграма
    z_grid = np.geomspace(min(sn_likelihood.z), max(sn_likelihood.z), 200)
    z_theory_plm = (1.0 + z_grid) / (1.0 + z_local_plm) - 1.0
    mu_plm = np.array([plm_model.distance_modulus(z) for z in z_theory_plm]) + delta_M_plm
    mu_lcdm = np.array([lcdm_model.distance_modulus(z) for z in z_grid]) + delta_M_plm # Adjusted LCDM with PLM's delta_M
    
    ax1.errorbar(sn_likelihood.z, sn_likelihood.mu_obs, 
                 yerr=np.sqrt(np.diag(sn_likelihood.cov_matrix)),
                 fmt='o', color='gray', ecolor='lightgray', alpha=0.4, markersize=3,
                 label='Pantheon+ Data', zorder=1)

    ax1.plot(z_grid, mu_plm, color='royalblue', linewidth=2.5, 
             label='PLM-FP Model (Best Fit)', zorder=3)
    ax1.plot(z_grid, mu_lcdm, color='crimson', linestyle='--', linewidth=2.5, 
             label='ΛCDM Model (Planck 2018)', zorder=2)
    
    ax1.set_ylabel('Distance Modulus $\\mu$', fontsize=16)
    ax1.set_title('Hubble Diagram: PLM-FP vs. $\\Lambda$CDM', fontsize=18)
    ax1.legend(loc='lower right', fontsize=14)
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax1.set_xscale('log')
    ax1.tick_params(labelbottom=False)

    textstr = (f'PLM-FP: $\\chi^2 = 676,298$\n'
               f'$\\Lambda$CDM: $\\chi^2 = 7,814,451$\n'
               f'$\\Delta$BIC = -7,138,146') # Hardcoded values from previous analysis
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.03, 0.97, textstr, transform=ax1.transAxes, fontsize=13,
             verticalalignment='top', bbox=props)

    # Долна част - Остатъци
    z_theory_plm_data = (1.0 + sn_likelihood.z) / (1.0 + z_local_plm) - 1.0
    mu_plm_data = np.array([plm_model.distance_modulus(z) for z in z_theory_plm_data]) + delta_M_plm
    mu_lcdm_data = np.array([lcdm_model.distance_modulus(z) for z in sn_likelihood.z]) + delta_M_plm # Adjusted LCDM with PLM's delta_M
    
    residuals_plm = sn_likelihood.mu_obs - mu_plm_data
    residuals_lcdm = sn_likelihood.mu_obs - mu_lcdm_data # Calculate LCDM residuals too

    ax2.scatter(sn_likelihood.z, residuals_plm, s=10, alpha=0.5, 
                edgecolor='royalblue', facecolor='lightblue', label='PLM Residuals')
    ax2.scatter(sn_likelihood.z, residuals_lcdm, s=10, alpha=0.5, 
                marker='x', color='crimson', label='ΛCDM Residuals') # Plot LCDM residuals

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.2)
    ax2.set_xscale('log')
    ax2.set_xlim(0.01, 3) # По-добър зуум за SN данни
    ax2.set_ylim(-1, 1) # Зуумваме около нулата
    ax2.set_xlabel('Redshift $z$', fontsize=16)
    ax2.set_ylabel('Residuals $\\Delta\\mu$', fontsize=16)
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.legend(loc='lower right', fontsize=14) # Add legend for residuals

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    plot_path2 = os.path.join(results_dir, "Hubble_Diagram_Publication.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    logging.info(f"Figure 2 (Hubble diagram) saved to: {plot_path2}")

if __name__ == "__main__":
    main()
