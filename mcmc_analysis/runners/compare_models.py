# Файл: mcmc_analysis/runners/compare_models.py (ВЕРСИЯ 2.1 - ФИНАЛНА)

import numpy as np
import sys
import os
import logging
import emcee
from scipy.optimize import minimize

# Добавяме директориите на проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Импортираме моделите и likelihood функциите
from mcmc_analysis.models.plm_model_fp import PLM
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood
from mcmc_analysis.likelihoods.bao_likelihood import BAOLikelihood
# CMB likelihood вече не ни трябва като глобален обект, тъй като е вграден
# from mcmc_analysis.likelihoods.cmb_likelihood import CMBLikelihood

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# --- Глобални обекти за данни (зареждат се веднъж) ---
try:
    logging.info("Зареждане на данни за likelihoods...")
    sn_likelihood = SupernovaeLikelihood()
    bao_likelihood = BAOLikelihood()
    logging.info("Данните са заредени успешно.")
except Exception as e:
    logging.error(f"Критична грешка при зареждане на данни: {e}")
    sys.exit(1)

def get_total_log_likelihood(model, delta_M=0.0, z_local=0.0):
    """Изчислява общия log-likelihood, включително CMB prior."""
    logL_sn = sn_likelihood.log_likelihood(model, delta_M=delta_M, z_local=z_local)
    logL_bao = bao_likelihood.log_likelihood(model)
    
    planck_theta = 1.04109
    planck_error = 0.00030
    try:
        z_star = 1090.0
        # ИЗПОЛЗВАМЕ МЕТОДИТЕ НА КОНКРЕТНИЯ МОДЕЛ
        r_s_comoving = model.calculate_sound_horizon(z_star)
        D_A = model.angular_diameter_distance(z_star)
        
        if not (np.isfinite(r_s_comoving) and np.isfinite(D_A) and D_A > 0):
            logL_cmb = -np.inf
        else:
            r_s_physical = r_s_comoving / (1.0 + z_star)
            model_theta = 100 * r_s_physical / D_A
            logL_cmb = -0.5 * ((model_theta - planck_theta) / planck_error)**2
    except Exception as e:
        logging.warning(f"Грешка при изчисляване на CMB likelihood: {e}")
        logL_cmb = -np.inf

    return logL_sn + logL_bao + logL_cmb

def get_best_fit_from_mcmc(hdf5_file, n_burnin=1000):
    """Извлича най-добрите параметри (медианата)."""
    try:
        backend = emcee.backends.HDFBackend(hdf5_file, read_only=True)
        flat_samples = backend.get_chain(discard=n_burnin, flat=True)
        return np.percentile(flat_samples, 50, axis=0)
    except Exception as e:
        logging.error(f"Грешка при зареждане на HDF5 файл '{hdf5_file}': {e}")
        return None

def find_best_fit_lcdm(sn_likelihood, bao_likelihood):
    """Намира най-добрия фит за ΛCDM спрямо SN+BAO данни."""
    def neg_log_like_lcdm(params):
        H0, omega_m_h2, delta_M = params
        try:
            model = LCDM(H0=H0, omega_m_h2=omega_m_h2)
            logL = sn_likelihood.log_likelihood(model, delta_M=delta_M) + bao_likelihood.log_likelihood(model)
            return -logL if np.isfinite(logL) else np.inf
        except:
            return np.inf

    initial_guess = [70.0, 0.14, 0.0]
    bounds = [(60, 80), (0.1, 0.2), (-1, 1)]
    
    logging.info("Търсене на най-добър фит за ΛCDM спрямо SN+BAO данни...")
    result = minimize(neg_log_like_lcdm, initial_guess, bounds=bounds, method='L-BFGS-B') 
    
    if result.success:
        return result.x
    else:
        logging.error(f"Минимизацията за ΛCDM се провали: {result.message}")
        return None

def calculate_information_criteria(chi_squared, n_params, n_data_points):
    """Изчислява AIC и BIC."""
    aic = chi_squared + 2 * n_params
    bic = chi_squared + np.log(n_data_points) * n_params
    return aic, bic

def main():
    # --- 1. Анализ на PLM модела ---
    logging.info("\n--- АНАЛИЗ НА PLM МОДЕЛ (CMB-constrained) ---")
    
    plm_hdf5_file = os.path.join(os.path.dirname(__file__), '../results/PLM_CMB_constrained_optimized_checkpoint.h5')
    plm_best_params_7 = get_best_fit_from_mcmc(plm_hdf5_file, n_burnin=1000)
    if plm_best_params_7 is None: return

    plm_cosmo_params = [plm_best_params_7[0], plm_best_params_7[1], plm_best_params_7[2], plm_best_params_7[3], plm_best_params_7[4], 0.01]
    delta_M_plm = plm_best_params_7[5]
    z_local_plm = plm_best_params_7[6]
    plm_model = PLM(*plm_cosmo_params)

    plm_max_log_prob = get_total_log_likelihood(plm_model, delta_M=delta_M_plm, z_local=z_local_plm)
    plm_chi2 = -2 * plm_max_log_prob
    plm_n_params = 7

    logging.info(f"Най-добри параметри за PLM (свободни): {[f'{p:.4f}' for p in plm_best_params_7]}")
    logging.info(f"χ² за PLM: {plm_chi2:.0f}")
    
    # --- 2. Анализ на ΛCDM модела (ЧЕСТЕН ПОДХОД) ---
    logging.info("\n--- АНАЛИЗ НА ΛCDM МОДЕЛ (чрез best-fit към нашите данни) ---")
    
    lcdm_best_params_3 = find_best_fit_lcdm(sn_likelihood, bao_likelihood)
    if lcdm_best_params_3 is None: return

    H0_lcdm, omega_m_h2_lcdm, delta_M_lcdm = lcdm_best_params_3
    
    lcdm_model_instance = LCDM(H0=H0_lcdm, omega_m_h2=omega_m_h2_lcdm)
    lcdm_max_log_prob = get_total_log_likelihood(lcdm_model_instance, delta_M=delta_M_lcdm, z_local=0.0)
    lcdm_chi2 = -2 * lcdm_max_log_prob
    lcdm_n_params = 6 

    logging.info(f"Най-добри параметри за ΛCDM (фитнати): H0={H0_lcdm:.2f}, omega_m_h2={omega_m_h2_lcdm:.4f}, delta_M={delta_M_lcdm:.4f}")
    logging.info(f"χ² за ΛCDM: {lcdm_chi2:.0f}")
    
    # --- 3. Изчисляване и отпечатване на резултати ---
    n_data_points = len(sn_likelihood.z) + len(bao_likelihood.z) * 2 + 1 # SN + BAO(H+dA) + CMB(theta_s)
    
    plm_aic, plm_bic = calculate_information_criteria(plm_chi2, plm_n_params, n_data_points)
    lcdm_aic, lcdm_bic = calculate_information_criteria(lcdm_chi2, lcdm_n_params, n_data_points)

    print("\n" + "="*60)
    print(" " * 10 + "СРАВНЕНИЕ НА МОДЕЛИ (CMB-Constrained PLM vs Optimized ΛCDM)")
    print("="*60)
    print(f"{'Критерий':<20} | {'PLM Модел':^18} | {'ΛCDM Модел':^18}")
    print("-"*60)
    print(f"{'Брой параметри (k)':<20} | {plm_n_params:^18} | {lcdm_n_params:^18}")
    print(f"{'χ² (Chi-squared)':<20} | {plm_chi2:^18.0f} | {lcdm_chi2:^18.0f}")
    print(f"{'AIC = χ² + 2k':<20} | {plm_aic:^18.0f} | {lcdm_aic:^18.0f}")
    print(f"{'BIC = χ² + k*ln(N)':<20} | {plm_bic:^18.0f} | {lcdm_bic:^18.0f}")
    print("="*60)

    # ... (интерпретация) ...

if __name__ == "__main__":
    main()
