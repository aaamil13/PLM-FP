# Файл: mcmc_analysis/runners/compare_models.py

import numpy as np
import sys
import os
import logging
import emcee
from scipy.optimize import minimize

# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Импортираме моделите и likelihood функциите
from mcmc_analysis.models.plm_model_fp import PLM
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood
from mcmc_analysis.likelihoods.bao_likelihood import BAOLikelihood
from mcmc_analysis.likelihoods.cmb_likelihood import CMBLikelihood

# --- Конфигурация на логването ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Глобални обекти за данни (зареждат се веднъж) ---
try:
    logging.info("Зареждане на данни за likelihoods...")
    sn_likelihood = SupernovaeLikelihood()
    bao_likelihood = BAOLikelihood()
    cmb_likelihood = CMBLikelihood()
    logging.info("Данните са заредени успешно.")
except Exception as e:
    logging.error(f"Критична грешка при зареждане на данни: {e}")
    sys.exit(1)

def get_total_log_likelihood(model, delta_M=0.0, z_local=0.0):
    """Изчислява общия log-likelihood за даден модел."""
    logL_sn = sn_likelihood.log_likelihood(model, delta_M=delta_M, z_local=z_local) # Pass delta_M and z_local here
    logL_bao = bao_likelihood.log_likelihood(model)
    logL_cmb = cmb_likelihood.log_likelihood(model) 
    return logL_sn + logL_bao + logL_cmb 

def get_best_fit_from_mcmc(hdf5_file, n_burnin=1000): # Добавяме n_burnin
    """
    Извлича най-добрите параметри от HDF5 файла на emcee.
    Връща 50-тия персентил (медианата) като най-добър представител.
    """
    try:
        backend = emcee.backends.HDFBackend(hdf5_file, read_only=True)
        # Взимаме "сплесканата" верига след burn-in
        flat_samples = backend.get_chain(discard=n_burnin, flat=True)
        
        # 50-тият персентил е по-стабилна мярка за "най-добър" параметър
        best_params = np.percentile(flat_samples, 50, axis=0)
        
        return best_params
    except Exception as e:
        logging.error(f"Не може да се зареди HDF5 файл '{hdf5_file}': {e}")
        return None


def calculate_information_criteria(chi_squared, n_params, n_data_points):
    """Изчислява AIC и BIC."""
    aic = chi_squared + 2 * n_params
    bic = chi_squared + np.log(n_data_points) * n_params
    return aic, bic

def main():
    """Основна функция за сравнение на моделите."""
    
    # --- 1. Анализ на PLM модела (свободен H0, фиксирано k, свободен delta_M, свободен z_local) ---
    logging.info("\n--- АНАЛИЗ НА PLM МОДЕЛ (свободен H0, фиксирано k, свободен delta_M, свободен z_local) ---")
    
    # ПРОМЕНЕТЕ ИМЕТО НА ФАЙЛА, ЗА ДА СОЧИ КЪМ РЕЗУЛТАТА ОТ НОВАТА СИМУЛАЦИЯ
    plm_hdf5_file = os.path.join(os.path.dirname(__file__), '../results/PLM_z_local_optimized_checkpoint.h5')
    
    # ВАЖНО: Уверете се, че n_burnin съвпада с този от симулацията!
    plm_best_params_7 = get_best_fit_from_mcmc(plm_hdf5_file, n_burnin=1000)
    
    if plm_best_params_7 is None: return

    # Параметрите, които ще подадем на модела PLM са 6: H0, omega_m_h2, z_crit, w_crit, f_max, k
    # delta_M е предпоследният параметър, z_local е последният
    cosmo_params_plm = plm_best_params_7[:-2] # Всички без delta_M и z_local
    delta_M_plm = plm_best_params_7[-2]       # delta_M
    z_local_plm = plm_best_params_7[-1]       # z_local
    
    # "Инжектираме" фиксираната стойност на k
    FIXED_K_FOR_MODEL = 0.01 # Същата стойност като в run_mcmc.py
    
    # Създаваме пълния списък с параметри за PLM модела: H0, omega_m_h2, z_crit, w_crit, f_max, FIXED_K
    plm_model_params = np.append(cosmo_params_plm[:-1], FIXED_K_FOR_MODEL) # Remove original k and add fixed k

    plm_n_params = len(plm_best_params_7) # Броят на СВОБОДНИТЕ параметри е 7!

    plm_model_instance = PLM(*plm_model_params)
    plm_max_log_prob = get_total_log_likelihood(plm_model_instance, delta_M=delta_M_plm, z_local=z_local_plm)

    plm_chi2 = -2 * plm_max_log_prob
    
    logging.info(f"Най-добри параметри за PLM (свободни): {[f'{p:.4f}' for p in plm_best_params_7]}")
    logging.info(f"Фиксиран k (за модела) = {FIXED_K_FOR_MODEL}")
    logging.info(f"Максимален log-likelihood за PLM: {plm_max_log_prob:.2f}")
    logging.info(f"χ² за PLM: {plm_chi2:.0f}") # Changed to .0f
    
    # --- 2. Анализ на ΛCDM модела (остава същият) ---
    logging.info("\n--- АНАЛИЗ НА ΛCDM МОДЕЛ ---")
    planck_2018_params = [67.36, 0.1428, 0.02237, 0.9649, 2.100e-9, 0.0544]
    lcdm_best_params = planck_2018_params
    
    lcdm_model_instance = LCDM(*lcdm_best_params)
    lcdm_max_log_prob = get_total_log_likelihood(lcdm_model_instance)

    lcdm_chi2 = -2 * lcdm_max_log_prob
    lcdm_n_params = len(lcdm_best_params)
    
    logging.info(f"Параметри за ΛCDM (от Planck 2018): {[f'{p:.4f}' for p in lcdm_best_params]}")
    logging.info(f"Максимален log-likelihood за ΛCDM: {lcdm_max_log_prob:.2f}")
    logging.info(f"χ² за ΛCDM: {lcdm_chi2:.0f}") # Changed to .0f

    # --- 3. Изчисляване на информационни критерии ---
    n_data_points = len(sn_likelihood.z) + len(bao_likelihood.z) * 2 + 2
    
    # ВАЖНО: Използваме броя на СВОБОДНИТЕ параметри (5 за PLM, 6 за LCDM)
    plm_aic, plm_bic = calculate_information_criteria(plm_chi2, plm_n_params, n_data_points)
    lcdm_aic, lcdm_bic = calculate_information_criteria(lcdm_chi2, lcdm_n_params, n_data_points)

    # --- 4. Отпечатване на сравнителна таблица ---
    print("\n" + "="*60)
    print(" " * 10 + "СРАВНЕНИЕ НА МОДЕЛИ (PLM с фиксирано k)")
    print("="*60)
    print(f"{'Критерий':<20} | {'PLM (k≈0) Модел':^18} | {'ΛCDM Модел':^18}")
    print("-"*60)
    print(f"{'Брой параметри (k)':<20} | {plm_n_params:^18} | {lcdm_n_params:^18}")
    print(f"{'χ² (Chi-squared)':<20} | {plm_chi2:^18.0f} | {lcdm_chi2:^18.0f}") # Changed to .0f
    print(f"{'AIC = χ² + 2k':<20} | {plm_aic:^18.0f} | {lcdm_aic:^18.0f}") # Changed to .0f
    print(f"{'BIC = χ² + k*ln(N)':<20} | {plm_bic:^18.0f} | {lcdm_bic:^18.0f}") # Changed to .0f
    print("="*60)

    # --- 5. Интерпретация на резултатите ---
    print("\nИнтерпретация:")
    delta_aic = plm_aic - lcdm_aic
    delta_bic = plm_bic - lcdm_bic

    print(f"ΔAIC (PLM - ΛCDM) = {delta_aic:.0f}") # Changed to .0f
    if delta_aic < -10:
        print("-> PLM моделът е **значително по-предпочитан** според AIC.")
    elif delta_aic < -2:
        print("-> Има съществени доказателства в полза на PLM модела според AIC.")
    elif delta_aic < 2:
        print("-> Двата модела са статистически неразличими според AIC.")
    elif delta_aic < 10:
        print("-> Има съществени доказателства в полза на ΛCDM модела според AIC.")
    else:
        print("-> ΛCDM моделът е **значително по-предпочитан** според AIC.")

    print(f"\nΔBIC (PLM - ΛCDM) = {delta_bic:.0f}") # Changed to .0f
    if delta_bic < -10:
        print("-> PLM моделът е **значително по-предпочитан** според BIC.")
    elif delta_bic < -6:
        print("-> Има силни доказателства в полза на PLM модела според BIC.")
    elif delta_bic < -2:
        print("-> Има доказателства в полза на PLM модела според BIC.")
    elif delta_bic < 2:
        print("-> Двата модела са статистически неразличими според BIC.")
    elif delta_bic < 6:
        print("-> Има доказателства в полза на ΛCDM модела според BIC.")
    else:
        print("-> ΛCDM моделът е **значително по-предпочитан** според BIC.")
    print("\n(Забележка: По-ниските стойности на AIC/BIC са по-добри)")


if __name__ == "__main__":
    main()
