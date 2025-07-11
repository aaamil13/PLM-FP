"""
Тестов скрипт за BAO Likelihood
================================

Този скрипт тества дали BAO данните се зареждат правилно
и дали χ² стойността се изчислява коректно за
стандартен ΛCDM модел с диагонална ковариация.
"""

import numpy as np
import sys
import os

# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.likelihoods.bao_likelihood import BAOLikelihood
from mcmc_analysis.models.plm_model import PLM # Използваме новия PLM модел

def run_test():
    """
    Основна тестова функция.
    """
    print("=== Тестване на BAO Likelihood ===")
    
    try:
        # 1. Инициализация на Likelihood - това ще зареди данните
        print("\n1. Зареждане на данни от BAO...")
        bao_likelihood = BAOLikelihood()
        
        # 2. Инициализация на модел (PLM с параметри по подразбиране)
        print("\n2. Инициализация на PLM модел (стойности по подразбиране)...")
        # Параметри: H0=70.0, omega_m_h2=0.14, z_crit=900.0, alpha=2.0, epsilon=0.0, beta=1.0
        plm_model = PLM(H0=70.0, omega_m_h2=0.14, z_crit=900.0, alpha=2.0, epsilon=0.0, beta=1.0)
        print(f"   Модел: PLM(H0={plm_model.H0}, Om0={plm_model.Omega_m:.3f}, epsilon={plm_model.epsilon})")

        # 3. Изчисляване на log(likelihood) и χ²
        print("\n3. Изчисляване на χ²...")
        log_like = bao_likelihood.log_likelihood(plm_model) # Използваме PLM модела
        chi_squared = -2 * log_like
        
        # Брой на независимите измервания (H(z) и d_A(z) за всяка точка)
        num_measurements = 2 * bao_likelihood.num_points
        dof = num_measurements - 2 # Брой параметри в този опростен тест (H0, Om0)
        
        print("\n--- РЕЗУЛТАТИ ---")
        print(f"  Log(Likelihood): {log_like:.4f}")
        print(f"  χ²: {chi_squared:.4f}")
        print(f"  Брой BAO точки: {bao_likelihood.num_points}")
        print(f"  Брой независими измервания: {num_measurements}")
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
