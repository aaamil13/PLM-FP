# Файл: mcmc_analysis/runners/test_h0_evolution.py

import numpy as np
import sys
import os
import logging
import matplotlib.pyplot as plt
import emcee
from scipy.integrate import quad

# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Импортираме PLM модела
from mcmc_analysis.models.plm_model_fp import PLM

# --- Конфигурация на логването ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

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

def main():
    logging.info("Стартиране на тест за локална еволюция на H(z) в PLM модела...")

    # --- 1. Вземане на най-добрите параметри от симулацията с фиксирано k ---
    plm_hdf5_file = os.path.join(os.path.dirname(__file__), '../results/PLM_k_fixed_optimized_checkpoint.h5')
    plm_best_params_5 = get_best_fit_from_mcmc(plm_hdf5_file, n_burnin=1000)
    
    if plm_best_params_5 is None:
        logging.error("Не може да се продължи без PLM параметри.")
        return

    FIXED_K = 0.01 
    plm_full_params = np.append(plm_best_params_5, FIXED_K)
    plm_model = PLM(*plm_full_params)
    
    logging.info(f"Най-добри параметри от MCMC (PLM_k_fixed):")
    param_names = ["H0_obs(z=0)", "Omega_m h^2", "z_crit", "w_crit", "f_max"]
    for name, val in zip(param_names, plm_best_params_5):
        logging.info(f"  {name} = {val:.4f}")
    logging.info(f"  Фиксиран k = {FIXED_K}")

    # --- 2. Изчисляване на H(z) при ниски z ---
    z_range = np.linspace(0, 0.05, 500) # Диапазон, по-голям от този на SH0ES (z~0.01)
    h_values = np.array([plm_model.H_of_z(z) for z in z_range])

    # --- 3. Изчисляване на ефективната локална H₀ ---
    z_max_sh0es = 0.01 # Типичен горен край за калибрацията на SH0ES

    # Интегрираме H_obs(z) от 0 до z_max_sh0es
    integral_H, err = quad(plm_model.H_of_z, 0, z_max_sh0es)
    
    # Изчисляваме средната стойност
    H0_effective_local = integral_H / z_max_sh0es

    logging.info("\n--- РЕЗУЛТАТИ ОТ АНАЛИЗА ---")
    logging.info(f"Стойност на H(z=0) от MCMC фита (глобална): {plm_model.params['H0']:.2f} km/s/Mpc")
    logging.info(f"Изчислена ефективна локална H₀ (осреднена до z={z_max_sh0es}): {H0_effective_local:.2f} km/s/Mpc")
    
    H0_SH0ES = 73.0
    diff_percent = ((H0_effective_local - H0_SH0ES) / H0_SH0ES) * 100
    logging.info(f"Стойност от SH0ES (измерена локална): ~{H0_SH0ES:.2f} km/s/Mpc")
    logging.info(f"Разлика: {abs(H0_effective_local - H0_SH0ES):.2f} km/s/Mpc ({diff_percent:.1f}%)")

    if abs(diff_percent) < 15: # Ако разликата е под 15%
        logging.info(">>> ЗАКЛЮЧЕНИЕ: Резултатът е забележително съвместим! Моделът обяснява напрежението с H₀.")
    else:
        logging.info(">>> ЗАКЛЮЧЕНИЕ: Все още има значителна разлика. Моделът не обяснява напълно напрежението с H₀.")


    # --- 4. Визуализация ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Графика на H(z)
    ax.plot(z_range, h_values, label='H_obs(z) от PLM Модела', color='royalblue', linewidth=2)
    
    # Хоризонтални линии, показващи различните стойности
    ax.axhline(plm_model.params['H0'], color='crimson', linestyle='--', label=f'Глобална H₀ (от фита) = {plm_model.params["H0"]:.2f}')
    ax.axhline(H0_SH0ES, color='green', linestyle=':', label=f'Локална H₀ (SH0ES) = {H0_SH0ES:.2f}')
    ax.axhline(H0_effective_local, color='darkorange', linestyle='-.', label=f'Ефективна локална H₀ (изчислена) = {H0_effective_local:.2f}')
    
    # Оцветяване на региона на интегриране
    z_fill = np.linspace(0, z_max_sh0es, 100)
    h_fill = np.array([plm_model.H_of_z(z) for z in z_fill])
    ax.fill_between(z_fill, h_fill, color='orange', alpha=0.2, label=f'Регион на осредняване (z < {z_max_sh0es})')
    
    ax.set_xlabel('Червено отместване (z)')
    ax.set_ylabel('Хъбъл параметър H(z) [km/s/Mpc]')
    ax.set_title('Локална еволюция на H(z) според PLM модела')
    ax.legend()
    ax.set_xlim(0, z_range[-1])
    ax.set_ylim(bottom=35)

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "h0_local_evolution.png")
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"\nГрафика на локалната еволюция на H(z) е запазена в: {plot_path}")

if __name__ == "__main__":
    main()
