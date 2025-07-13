# Файл: mcmc_analysis/runners/predict_cmb_angle.py

import numpy as np
import sys
import os
import logging
import emcee

# Добавяме директориите на проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.models.plm_model_fp import PLM
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.likelihoods.cmb_likelihood import CMBLikelihood # Import CMBLikelihood

# --- Конфигурация на логването ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Global CMBLikelihood instance
cmb_likelihood_instance = CMBLikelihood()

def get_best_fit_from_mcmc(hdf5_file, n_burnin=1000):
    """Извлича най-добрите параметри (медианата) от HDF5 файла."""
    try:
        backend = emcee.backends.HDFBackend(hdf5_file, read_only=True)
        flat_samples = backend.get_chain(discard=n_burnin, flat=True)
        best_params = np.percentile(flat_samples, 50, axis=0)
        return best_params
    except Exception as e:
        logging.error(f"Не може да се зареди HDF5 файл '{hdf5_file}': {e}")
        return None

def calculate_cmb_angle(model, z_star=1090.0):
    """Изчислява r_s, D_A и 100*theta_s за даден модел."""
    logging.info(f"Изчисляване на CMB параметри за z_star = {z_star}...")
    
    # Изчисляване на КОМОЛОГИЧНИЯ звуков хоризонт
    r_s_comoving = cmb_likelihood_instance.calculate_sound_horizon(model, z_star)
    if not np.isfinite(r_s_comoving):
        logging.error("Изчислението на r_s върна невалидна стойност.")
        return None, None, None
    
    # Изчисляване на ъгловото разстояние
    D_A = model.angular_diameter_distance(z_star)
    if not np.isfinite(D_A):
        logging.error("Изчислението на D_A върна невалидна стойност.")
        return None, None, None
        
    # Изчисляване на ФИЗИЧЕСКИЯ звуков хоризонт
    r_s_physical = r_s_comoving / (1.0 + z_star)

    # Изчисляване на ъгъла theta_s в радиани
    theta_s = r_s_physical / D_A
    hundred_theta_s = 100 * theta_s
    
    # Връщаме комологичния r_s за сравнение, но използваме правилната формула
    return r_s_comoving, D_A, hundred_theta_s

def main():
    logging.info("--- Предсказания за ъгъла на CMB (100 * θ_s) ---")
    
    # Референтна стойност от Planck 2018
    planck_value = 1.04109
    planck_error = 0.00030

    # --- 1. Предсказание от PLM модела ---
    logging.info("\n--- АНАЛИЗ НА PLM МОДЕЛ ---")
    plm_hdf5_file = os.path.join(os.path.dirname(__file__), '../results/PLM_z_local_optimized_checkpoint.h5')
    
    # Взимаме 7-те свободни параметъра
    plm_best_params_7 = get_best_fit_from_mcmc(plm_hdf5_file, n_burnin=1000)
    if plm_best_params_7 is None: return

    # PLM моделът очаква 6 параметъра (без delta_M и z_local)
    # Трябва да създадем инстанция само с космологичните параметри
    # H0, omega_m_h2, z_crit, w_crit, f_max, k
    # ВАЖНО: Трябва да знаем кой параметър къде е в масива!
    # От run_mcmc знаем, че редът е: H0, omega_m_h2, z_crit, w_crit, f_max, delta_M, z_local
    
    plm_cosmo_params = [
        plm_best_params_7[0], # H0
        plm_best_params_7[1], # omega_m_h2
        plm_best_params_7[2], # z_crit
        plm_best_params_7[3], # w_crit
        plm_best_params_7[4], # f_max
        0.01                  # Фиксираният k
    ]
    plm_model = PLM(*plm_cosmo_params)
    
    r_s_plm, D_A_plm, hundred_theta_s_plm = calculate_cmb_angle(plm_model)

    # --- 2. Предсказание от ΛCDM модела ---
    logging.info("\n--- АНАЛИЗ НА ΛCDM МОДЕЛ ---")
    lcdm_params = [67.36, 0.1428, 0.02237, 0.9649, 2.1e-9, 0.0544]
    lcdm_model = LCDM(*lcdm_params)
    r_s_lcdm, D_A_lcdm, hundred_theta_s_lcdm = calculate_cmb_angle(lcdm_model)
    
    # --- 3. Сравнителна таблица ---
    print("\n" + "="*70)
    print(" " * 15 + "СРАВНЕНИЕ НА CMB ПРЕДСКАЗАНИЯ (z_star=1090)")
    print("="*70)
    print(f"{'Параметър':<20} | {'PLM Модел':^20} | {'ΛCDM Модел':^20}")
    print("-"*70)
    if r_s_plm is not None:
        print(f"{'Звуков хоризонт r_s [Mpc]':<20} | {r_s_plm:^20.2f} | {r_s_lcdm:^20.2f}")
        print(f"{'Ъглово разстояние D_A [Mpc]':<20} | {D_A_plm:^20.2f} | {D_A_lcdm:^20.2f}")
        print(f"{'Ъгъл 100*θ_s':<20} | {hundred_theta_s_plm:^20.5f} | {hundred_theta_s_lcdm:^20.5f}")
    print("-"*70)
    print(f"{'Измерено от Planck':<20} | {planck_value:.5f} ± {planck_error:.5f}")
    print("="*70)

    # --- 4. Заключение ---
    if hundred_theta_s_plm is not None:
        diff = abs(hundred_theta_s_plm - planck_value)
        sigma_diff = diff / planck_error
        logging.info(f"\nРазлика между PLM и Planck: {diff:.5f} ({sigma_diff:.1f}σ)")
        if sigma_diff < 3:
            logging.info(">>> ЗАКЛЮЧЕНИЕ: PLM моделът е в отлично съгласие с измерената стойност на ъгъла на CMB!")
        elif sigma_diff < 5:
            logging.info(">>> ЗАКЛЮЧЕНИЕ: PLM моделът е в умерено съгласие с измерената стойност на ъгъла на CMB.")
        else:
            logging.info(">>> ЗАКЛЮЧЕНИЕ: Има значително напрежение между предсказанието на PLM и данните от CMB за този параметър.")

if __name__ == "__main__":
    main()
