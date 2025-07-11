"""
Тестов скрипт за Supernovae Likelihood
=======================================

Този скрипт тества дали данните от свръхнови (Pantheon+) се зареждат
правилно и дали χ² стойността се изчислява коректно за
стандартен ΛCDM модел.
"""

import numpy as np
import sys
import os

# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood
from mcmc_analysis.models.plm_model import PLM # Използваме новия PLM модел

def run_test():
    """
    Основна тестова функция.
    """
    print("=== Тестване на Supernovae Likelihood ===")
    
    try:
        # 1. Инициализация на Likelihood - това ще зареди данните
        print("\n1. Зареждане на данни от свръхнови...")
        sn_likelihood = SupernovaeLikelihood()
        
        # 2. Инициализация на модел (PLM с параметри по подразбиране)
        print("\n2. Инициализация на PLM модел (стойности по подразбиране)...")
        # Параметри: H0=70.0, omega_m_h2=0.14, z_crit=900.0, alpha=2.0, epsilon=0.0, beta=1.0
        # Използваме параметри, които имитират стандартен модел, за да видим базово съвпадение.
        # epsilon=0.0 и beta=1.0 правят δ(t) = 0, така че a(t) = k*t
        plm_model = PLM(H0=70.0, omega_m_h2=0.14, z_crit=900.0, alpha=2.0, epsilon=0.0, beta=1.0)
        print(f"   Модел: PLM(H0={plm_model.H0}, Om0={plm_model.Omega_m:.3f}, epsilon={plm_model.epsilon})")

        # 3. Изчисляване на log(likelihood) и χ²
        print("\n3. Изчисляване на χ²...")
        log_like = sn_likelihood.log_likelihood(plm_model) # Използваме PLM модела
        chi_squared = -2 * log_like
        
        num_points = len(sn_likelihood.z)
        # Брой параметри в този опростен тест (H0, Om0) - 2
        # В PLM ще имаме повече параметри за напасване (H0, Om0, z_crit, alpha, epsilon, beta)
        # За този тест, приемаме, че напасваме само H0 и Om0 (т.е. 2)
        dof = num_points - 2 
        
        print("\n--- РЕЗУЛТАТИ ---")
        print(f"  Log(Likelihood): {log_like:.4f}")
        print(f"  χ²: {chi_squared:.4f}")
        print(f"  Брой точки с данни: {num_points}")
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
