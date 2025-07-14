import numpy as np
import sys
import os
import logging
import matplotlib.pyplot as plt
import emcee

# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Импортираме моделите и likelihood функциите
from mcmc_analysis.models.plm_model_fp import PLM
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood

# --- Конфигурация на логването ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Глобални обекти за данни (зареждат се веднъж) ---
try:
    logging.info("Зареждане на данни за свръхнови...")
    sn_likelihood = SupernovaeLikelihood()
    logging.info("Данните за свръхнови са заредени успешно.")
except Exception as e:
    logging.error(f"Критична грешка при зареждане на данни за свръхнови: {e}")
    sys.exit(1)

def get_best_fit_from_mcmc(hdf5_file, n_burnin=1000):
    """
    Извлича най-добрите параметри от HDF5 файла на emcee.
    Връща 50-тия персентил (медианата) като най-добър представител.
    """
    try:
        backend = emcee.backends.HDFBackend(hdf5_file, read_only=True)
        flat_samples = backend.get_chain(discard=n_burnin, flat=True)
        best_params = np.percentile(flat_samples, 50, axis=0)
        return best_params
    except Exception as e:
        logging.error(f"Не може да се зареди HDF5 файл '{hdf5_file}': {e}")
        return None

def main():
    logging.info("Стартиране на анализ на остатъци на Хъбъл диаграма...")

    # --- 1. Вземане на най-добрите параметри за PLM модела ---
    plm_hdf5_file = os.path.join(os.path.dirname(__file__), '../results/PLM_CMB_constrained_optimized_checkpoint.h5')
    plm_best_params_7 = get_best_fit_from_mcmc(plm_hdf5_file, n_burnin=1000)
    
    if plm_best_params_7 is None:
        logging.error("Не може да се продължи без PLM параметри.")
        return

    # Извличаме космологичните параметри, delta_M и z_local
    cosmo_params_plm = plm_best_params_7[:-2] # Всички без delta_M и z_local
    delta_M_plm = plm_best_params_7[-2]       # delta_M
    z_local_plm = plm_best_params_7[-1]       # z_local
    
    # "Инжектираме" фиксираната стойност на k
    FIXED_K_FOR_MODEL = 0.01 # Същата стойност като в run_mcmc.py
    
    # Създаваме пълния списък с параметри за PLM модела: H0, omega_m_h2, z_crit, w_crit, f_max, FIXED_K
    plm_model_params = np.append(cosmo_params_plm[:-1], FIXED_K_FOR_MODEL) # Remove original k and add fixed k
    
    plm_model = PLM(*plm_model_params)
    logging.info(f"PLM (k={FIXED_K_FOR_MODEL}) параметри: {[f'{p:.4f}' for p in plm_model_params]}")
    logging.info(f"PLM delta_M: {delta_M_plm:.4f}")
    logging.info(f"PLM z_local: {z_local_plm:.4f}")

    # --- 2. Вземане на параметрите за ΛCDM модела ---
    lcdm_best_params = [
        67.36,      # H0
        0.1428,     # omega_m_h2
        0.02237,    # omega_b_h2
        0.9649,     # n_s
        2.100e-9,   # A_s
        0.0544      # tau_reio
    ]
    lcdm_model = LCDM(*lcdm_best_params)
    logging.info(f"ΛCDM (Planck 2018) параметри: {[f'{p:.4f}' for p in lcdm_best_params]}")

    # --- 3. Изчисляване на остатъци за SN данни ---
    sn_z = sn_likelihood.z
    sn_mu_data = sn_likelihood.mu_obs

    plm_residuals = []
    lcdm_residuals = []
    
    logging.info("Изчисляване на остатъци...")
    # Apply z_local correction to sn_z before calculating mu_model_theory
    sn_z_corrected = (1 + sn_z) / (1 + z_local_plm) - 1
    
    for i, mu_data_val in enumerate(sn_mu_data):
        z_val_corrected = sn_z_corrected[i]
        
        plm_mu_model_theory = plm_model.distance_modulus(z_val_corrected)
        lcdm_mu_model_theory = lcdm_model.distance_modulus(sn_z[i]) # LCDM uses original z
        
        # Прилагаме отместването delta_M за PLM
        plm_residuals.append(mu_data_val - (plm_mu_model_theory + delta_M_plm))
        
        # За LCDM, ако искаме да центрираме, можем да използваме намереното delta_M от PLM
        # или да приемем, че за LCDM също има някакво отместване.
        # За целите на сравнението, ще използваме същото delta_M за LCDM, за да центрираме остатъците.
        lcdm_residuals.append(mu_data_val - (lcdm_mu_model_theory + delta_M_plm)) 

    plm_residuals = np.array(plm_residuals)
    lcdm_residuals = np.array(lcdm_residuals)

    # --- 4. Визуализация на остатъците ---
    plt.figure(figsize=(10, 6))
    plt.scatter(sn_z, plm_residuals, s=10, alpha=0.6, label=f'PLM (k={FIXED_K_FOR_MODEL}, $\\Delta M$={delta_M_plm:.3f}, $z_{{local}}$={z_local_plm:.3f}) Residuals')
    plt.scatter(sn_z, lcdm_residuals, s=10, alpha=0.6, label='ΛCDM Residuals (adjusted by PLM $\\Delta M$)')
    
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8, label='Zero Residuals')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Distance Modulus Residuals (μ_data - μ_model - $\\Delta M$)')
    plt.title('Hubble Diagram Residuals: PLM vs ΛCDM (SN Data) - $\\Delta M$ & $z_{local}$ adjusted (CMB Constrained)')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "hubble_residuals_plm_CMB_constrained.png")
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Графика на остатъци запазена в: {plot_path}")
    logging.info("Анализ на остатъци завършен.")

if __name__ == "__main__":
    main()
