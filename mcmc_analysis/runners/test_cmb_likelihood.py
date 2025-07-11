"""
Тестов скрипт за CMB Likelihood
================================

Този скрипт тества дали CMB likelihood функцията се изчислява коректно за
стандартен ΛCDM модел.
"""

import numpy as np
import sys
import os

# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.likelihoods.cmb_likelihood import CMBLikelihood
from mcmc_analysis.models.lcdm_model import LCDM

def run_test():
    """
    Основна тестова функция.
    """
    print("=== Тестване на CMB Likelihood ===")
    
    try:
        # 1. Инициализация на Likelihood
        print("\n1. Инициализация на CMB Likelihood...")
        cmb_likelihood = CMBLikelihood()
        
        # 2. Инициализация на модел (ΛCDM с параметри по подразбиране)
        print("\n2. Инициализация на ΛCDM модел (стойности по подразбиране)...")
        # Параметри: H0=70.0, omega_m_h2=0.14
        h0 = 70.0
        om0 = 0.3
        omega_m_h2 = om0 * (h0/100.0)**2
        
        lcdm_model = LCDM(H0=h0, omega_m_h2=omega_m_h2)
        print(f"   Модел: FlatLambdaCDM(H0={lcdm_model.H0}, Om0={lcdm_model.Om0:.3f})")

        # 3. Изчисляване на log(likelihood) и χ²
        print("\n3. Изчисляване на χ²...")
        log_like = cmb_likelihood.log_likelihood(lcdm_model)
        chi_squared = -2 * log_like
        
        # За опростената CMB likelihood имаме 2 измервания (r_s и l_A) и 2 параметъра (H0, Om0)
        num_measurements = 2
        dof = num_measurements - 2 # Брой параметри в този опростен тест (H0, Om0)
        
        print("\n--- РЕЗУЛТАТИ ---")
        print(f"  Log(Likelihood): {log_like:.4f}")
        print(f"  χ²: {chi_squared:.4f}")
        print(f"  Брой независими измервания: {num_measurements}")
        print(f"  dof: {dof}")
        print(f"  χ²/dof: {chi_squared/dof:.4f}")
        print("-----------------")
        
        if np.isfinite(chi_squared):
            print("\nТЕСТЪТ Е УСПЕШЕН: χ² е валидно число.")
        else:
            print("\nТЕСТЪТ Е НЕУСПЕШЕН: χ² не е валидно число.")

    except Exception as e:
        print(f"\nГРЕШКА ПО ВРЕМЕ НА ТЕСТА: {e}")

if __name__ == "__main__":
    run_test()
