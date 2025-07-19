# Файл: mcmc_analysis/runners/test_sn_likelihood.py (ДЕБЪГ ВЕРСИЯ)

import numpy as np
import sys
import os
import logging

# --- DEBUGGING PATH ---
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

# Ensure the project root is in sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- NEW DEBUGGING PRINT ---
print(f"sys.path before imports: {sys.path}")
print(f"Project root being added: {project_root}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# Импортираме моделите и likelihood функциите
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood

def main():
    logging.info("Starting SN Likelihood DEEP DEBUG test...")

    # 1. Създайте инстанция на sn_likelihood
    try:
        sn_likelihood = SupernovaeLikelihood()
        logging.info("SupernovaeLikelihood instance created successfully.")
    except Exception as e:
        logging.error(f"Error creating SupernovaeLikelihood instance: {e}")
        return

    # 2. Създайте инстанция на стандартен ΛCDM модел с параметрите от Planck
    H0_planck = 67.4
    Om0_planck = 0.315
    omega_m_h2_planck = Om0_planck * (H0_planck / 100.0)**2
    omega_b_h2_planck = 0.02238
    
    try:
        lcdm_model = LCDM(H0=H0_planck, omega_m_h2=omega_m_h2_planck, omega_b_h2=omega_b_h2_planck)
        logging.info(f"LCDM model instance created with H0={H0_planck}, omega_m_h2={omega_m_h2_planck:.5f}")
    except Exception as e:
        logging.error(f"Error creating LCDM model instance: {e}")
        return

    # --- ДЕБЪГ: Проверка на входните данни ---
    # Нека използваме директния път до файла за сигурност.
    data_file_path = os.path.join(os.path.dirname(__file__), '../../mcmc_analysis/data/pantheon_plus_data.txt')
    
    # Проверете дали файлът съществува
    if not os.path.exists(data_file_path):
        logging.error(f"Data file not found: {data_file_path}")
        return

    # Колоните според официалното описание на Pantheon+:
    # z_cmb (индекс 3), d_mb (индекс 5), MU_SH0ES (индекс 10)
    # Зареждаме само тези колони, за да избегнем грешка с нечислови данни (напр. 'name')
    data_full = np.loadtxt(data_file_path, skiprows=1, usecols=(3, 5, 10))
    
    z_cmb = data_full[:, 0]  # Първата заредена колона (индекс 3 от оригиналния файл)
    d_mb = data_full[:, 1]   # Втората заредена колона (индекс 5 от оригиналния файл)
    mu_shoes = data_full[:, 2] # Третата заредена колона (индекс 10 от оригиналния файл)

    mu_model = np.array([lcdm_model.distance_modulus(z) for z in z_cmb])
    
    delta_mu = mu_shoes - mu_model
    mu_obs_err = d_mb # Приемаме, че грешката в μ е същата като в m_b

    # Уверете се, че няма нулеви грешки, за да избегнете деление на нула
    mu_obs_err[mu_obs_err == 0] = np.finfo(float).eps # Малка стойност вместо 0

    # --- Отпечатване на първите 5 стойности за проверка ---
    print("\n--- DEBUGGING DATA (first 5 rows) ---")
    print(f"{'z_cmb':<10} {'mu_obs':<10} {'mu_model':<10} {'delta_mu':<10} {'mu_err':<10}")
    for i in range(5):
        print(f"{z_cmb[i]:<10.4f} {mu_shoes[i]:<10.4f} {mu_model[i]:<10.4f} {delta_mu[i]:<10.4f} {mu_obs_err[i]:<10.4f}")
    
    # --- Изчисляване на χ² ---
    chi2_simple = np.sum((delta_mu / mu_obs_err)**2)

    print("\n--- RESULTS ---")
    print(f"Simple Chi-squared (diagonal errors): {chi2_simple}")
    if len(delta_mu) > 0 and mu_obs_err[0] != 0: # Avoid division by zero if array is empty or first error is zero
        print(f"Value of first term in sum: {(delta_mu[0]/mu_obs_err[0])**2:.2f}")

    if chi2_simple > 1e10:
        print("\n[!!!] CRITICAL ERROR: Simple Chi-squared is astronomically large.")
        problem_terms_indices = np.where((delta_mu / mu_obs_err)**2 > 1e10)[0]
        if len(problem_terms_indices) > 0:
            idx = problem_terms_indices[0]
            print(f"The problem likely originates from terms like index {idx}:")
            print(f"  delta_mu[{idx}] = {delta_mu[idx]}")
            print(f"  mu_obs_err[{idx}] = {mu_obs_err[idx]}")
            print(f"  (delta_mu / mu_err)^2 = {(delta_mu[idx]/mu_obs_err[idx])**2}")
        else:
            print("No single term found to be astronomically large, but the sum is.")

if __name__ == "__main__":
    main()
