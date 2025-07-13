"""
Диагностичен скрипт за тестване на PLM модела и Numba компилацията.
"""

import numpy as np
import sys
import os
import time
import logging
import multiprocessing # Добавяме multiprocessing
import traceback # Добавяме traceback

# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.models.plm_model_fp import PLM # Използваме новия модел PLM-FP

def setup_test_logging():
    """Конфигурира системата за логване за тестовия скрипт."""
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"test_plm_module_{int(time.time())}.log")
    
    logging.basicConfig(
        level=logging.INFO, # Започваме с INFO, може да се смени на DEBUG при нужда
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout) # Извежда лога и на конзолата
        ]
    )
    logging.info(f"Лог файлът се записва в: {log_filename}")

def run_plm_test():
    setup_test_logging()
    logging.info("Стартиране на диагностичен тест за PLM модела...")

    # Примерни параметри за PLM
    H0 = 70.0
    omega_m_h2 = 0.14
    z_crit = 1100.0
    alpha = 2.0
    epsilon = 0.0
    beta = 1.0

    try:
        logging.info(f"Инициализация на PLM модел с H0={H0}, omega_m_h2={omega_m_h2}, z_crit={z_crit}, alpha={alpha}, epsilon={epsilon}, beta={beta}")
        # Използваме параметрите на новия PLM-FP модел
        model = PLM(H0=H0, omega_m_h2=omega_m_h2, z_crit=2.5, w_crit=0.5, f_max=0.85, k=2.0)
        logging.info("PLM модел инициализиран успешно.")

        test_zs = np.array([0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0, 1089.8, 2000.0])

        logging.info("\nТестване на H(z) за различни червени отмествания:")
        for z_val in test_zs:
            start_time_h_z = time.time()
            H_val = model.H_of_z(z_val)
            end_time_h_z = time.time()
            logging.info(f"  H(z={z_val:.1f}) = {H_val:.4f} km/s/Mpc (Time: {(end_time_h_z - start_time_h_z):.4f}s)")
            if not np.isfinite(H_val):
                logging.warning(f"  H(z={z_val:.1f}) върна невалидна стойност: {H_val}")

        logging.info("\nТестване на distance_modulus(z) за различни червени отмествания:")
        for z_val in test_zs:
            start_time_dm = time.time()
            dm_val = model.distance_modulus(z_val)
            end_time_dm = time.time()
            logging.info(f"  DM(z={z_val:.1f}) = {dm_val:.4f} mag (Time: {(end_time_dm - start_time_dm):.4f}s)")
            if not np.isfinite(dm_val):
                logging.warning(f"  DM(z={z_val:.1f}) върна невалидна стойност: {dm_val}")
        
        logging.info("\nТестване на calculate_sound_horizon(z_star=1089.8):")
        start_time_rs = time.time()
        rs_val = model.calculate_sound_horizon(1089.8)
        end_time_rs = time.time()
        logging.info(f"  r_s(z_recomb=1089.8) = {rs_val:.4f} Mpc (Time: {(end_time_rs - start_time_rs):.4f}s)")
        if not np.isfinite(rs_val):
            logging.warning(f"  r_s върна невалидна стойност: {rs_val}")

        logging.info("\nДиагностичен тест за PLM модела завърши успешно.")

    except Exception as e:
        logging.error(f"Грешка по време на PLM диагностичен тест: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # За Windows съвместимост
    run_plm_test()
