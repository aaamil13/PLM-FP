# Файл: mcmc_analysis/runners/analyze_lensing_effect.py

import numpy as np
import sys
import os
import logging
import matplotlib.pyplot as plt
import emcee

# Добавяме пътищата и импортираме нужните класове
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mcmc_analysis.models.plm_model_fp import PLM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood
from mcmc_analysis.models.lcdm_model import LCDM # Import LCDM model

# --- Конфигурация ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
plt.style.use('seaborn-v0_8-whitegrid')

# --- Физически и астрофизични константи ---
SIGMA_SB = 5.670374e-8  # Константа на Стефан-Болцман (W m^-2 K^-4)
T_EFF_SNIA = 10000.0    # Приблизителна ефективна температура на SN Ia (K)
R_SNIA_METERS = 1.5e12    # Приблизителен радиус на фотосферата на SN Ia (m) ~ 100 AU

def get_best_fit_plm():
    """Зарежда най-добрите параметри за PLM от последния MCMC фит."""
    hdf5_file = os.path.join(os.path.dirname(__file__), '../results/PLM_CMB_constrained_optimized_checkpoint.h5')
    try:
        backend = emcee.backends.HDFBackend(hdf5_file, read_only=True)
        flat_samples = backend.get_chain(discard=1000, flat=True)
        return np.percentile(flat_samples, 50, axis=0)
    except Exception as e:
        logging.error(f"Не може да се зареди HDF5 файл '{hdf5_file}': {e}")
        return None

def main():
    logging.info("Стартиране на анализ на ефекта на гравитационно-времева леща...")

    # 1. Зареждане на данните и моделите
    try:
        sn_data = SupernovaeLikelihood()
    except Exception as e:
        logging.error(f"Не може да се заредят данните за SN: {e}")
        return

    plm_params = get_best_fit_plm()
    if plm_params is None: return
    
    # Създаваме инстанция на PLM с най-добрите параметри от CMB-ограничения тест
    # Ред: H0, omega_m_h2, z_crit, w_crit, f_max, delta_M, z_local
    plm_cosmo_params = [plm_params[0], plm_params[1], plm_params[2], plm_params[3], plm_params[4], 0.01] # k fixed at 0.01
    plm_model = PLM(*plm_cosmo_params)
    
    # 2. Дефиниране на хипотезата и целта
    # Нашата хипотеза: d_L_observed = d_L_true / sqrt(A_mag)
    # Ние напасваме d_L_PLM към d_L_observed.
    # Истинската космология може би е по-близка до ΛCDM с H0=73.
    # A_mag = (d_L_true / d_L_observed)^2
    
    # Създаваме референтен модел (ΛCDM с H0=73), който да представлява "истинското" разстояние
    h0_true_model = LCDM(H0=73.0, omega_m_h2=0.147) # Типични стойности от SH0ES
    
    # 3. Изчисляване на фактора на увеличение за всяка свръхнова
    redshifts = sn_data.z
    
    # Изчисляваме предсказаните разстояния от двата модела
    d_l_plm = np.array([plm_model.luminosity_distance(z) for z in redshifts])
    d_l_true_hypothetical = np.array([h0_true_model.luminosity_distance(z) for z in redshifts])
    
    # Факторът на увеличение на потока A_mag
    # Защита от деление на нула
    d_l_plm[d_l_plm < 1e-9] = 1e-9
    A_mag = (d_l_true_hypothetical / d_l_plm)**2

    # 4. Анализ и визуализация на резултатите
    logging.info("\n--- РЕЗУЛТАТИ ОТ АНАЛИЗА НА ЛЕЩАТА ---")
    
    mean_A_mag = np.mean(A_mag)
    median_A_mag = np.median(A_mag)
    std_A_mag = np.std(A_mag)
    
    logging.info(f"Предсказан среден фактор на увеличение на потока (A_mag): {mean_A_mag:.2f}")
    logging.info(f"Предсказана медиана на увеличението на потока: {median_A_mag:.2f}")
    logging.info(f"Стандартно отклонение: {std_A_mag:.2f}")
    
    # Проверка на хипотезата: H_eff = H_true / sqrt(A_mag)
    # Забележка: връзката е по-скоро с A_lens за ъгловия размер, не A_mag за потока.
    # d_L ~ 1/H0. d_L_obs = d_L_true / sqrt(A_mag). 1/H0_obs = 1/H0_true / sqrt(A_mag) => H0_obs = H0_true * sqrt(A_mag)
    # H0_PLM = H0_SH0ES * sqrt(A_mag) => 47.3 = 73 * sqrt(A_mag) => A_mag = (47.3/73)^2 = 0.42
    # Това е в противоречие. Нека се придържаме към по-простата интерпретация.
    
    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Графика 1: A_mag като функция на z
    ax = axes[0]
    scatter = ax.scatter(redshifts, A_mag, c=redshifts, cmap='viridis', s=10, alpha=0.7)
    ax.axhline(median_A_mag, color='crimson', linestyle='--', label=f'Медиана A_mag = {median_A_mag:.2f}')
    ax.set_xlabel("Червено отместване (z)")
    ax.set_ylabel("Предсказан фактор на увеличение (A_mag)")
    ax.set_title("Ефект на 'Времева Леща' върху Свръхнови")
    ax.set_xscale('log')
    ax.legend()
    fig.colorbar(scatter, ax=ax, label='Червено отместване (z)')

    # Графика 2: Хистограма на A_mag
    ax = axes[1]
    ax.hist(A_mag, bins=100, density=True, color='royalblue', alpha=0.8)
    ax.axvline(median_A_mag, color='crimson', linestyle='--', label=f'Медиана A_mag = {median_A_mag:.2f}')
    ax.set_xlabel("Предсказан фактор на увеличение (A_mag)")
    ax.set_ylabel("Плътност на вероятността")
    ax.set_title("Разпределение на фактора на увеличение")
    ax.legend()
    
    plt.tight_layout()
    
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    plot_path = os.path.join(results_dir, "lensing_effect_analysis.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logging.info(f"Графики на анализа запазени в: {plot_path}")

if __name__ == "__main__":
    main()
