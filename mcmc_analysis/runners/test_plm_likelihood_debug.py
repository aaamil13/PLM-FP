"""
Тестов скрипт за дебъгване на PLM log_likelihood
=================================================

Този скрипт тества директно PLM log_likelihood функцията с конкретни
параметри, за да диагностицира проблеми с NaN/-inf или други грешки.
"""

import numpy as np
import sys
import os

# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.models.plm_model import PLM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood
from mcmc_analysis.likelihoods.bao_likelihood import BAOLikelihood
from mcmc_analysis.likelihoods.cmb_likelihood import CMBLikelihood

def run_debug_test():
    print("=== Дебъгване на PLM log_likelihood ===")

    # Инициализация на Likelihood обекти (данните се зареждат тук)
    try:
        sn_likelihood = SupernovaeLikelihood()
        bao_likelihood = BAOLikelihood()
        cmb_likelihood = CMBLikelihood()
        print("Данни за Likelihood заредени успешно.")
    except Exception as e:
        print(f"Грешка при зареждане на Likelihood данни: {e}")
        return

    # Параметри за тест (същите като началните за PLM в run_mcmc.py)
    test_params = [70.0, 0.14, 900.0, 2.0, 0.0, 1.0] # H0, omega_m_h2, z_crit, alpha, epsilon, beta
    
    print(f"\nТестване на log_likelihood_plm с параметри: {test_params}")

    try:
        H0, omega_m_h2, z_crit, alpha, epsilon, beta = test_params
        model = PLM(H0=H0, omega_m_h2=omega_m_h2, z_crit=z_crit, alpha=alpha, epsilon=epsilon, beta=beta)
        print("PLM модел инициализиран успешно.")

        lp = 0.0
        sn_lp = sn_likelihood.log_likelihood(model)
        bao_lp = bao_likelihood.log_likelihood(model)
        cmb_lp = cmb_likelihood.log_likelihood(model)

        lp += sn_lp
        lp += bao_lp
        lp += cmb_lp
        
        print(f"\nРезултати от log_likelihood:")
        print(f"  SN log_likelihood: {sn_lp}")
        print(f"  BAO log_likelihood: {bao_lp}")
        print(f"  CMB log_likelihood: {cmb_lp}")
        print(f"  Общ log_likelihood: {lp}")

        if np.isinf(lp) or np.isnan(lp):
            print("\nВНИМАНИЕ: Общият log_likelihood е INF или NaN. Това е проблем!")
            # Допълнителни проверки за индивидуалните компоненти
            if np.isinf(sn_lp) or np.isnan(sn_lp): print("  - SN log_likelihood е INF/NaN")
            if np.isinf(bao_lp) or np.isnan(bao_lp): print("  - BAO log_likelihood е INF/NaN")
            if np.isinf(cmb_lp) or np.isnan(cmb_lp): print("  - CMB log_likelihood е INF/NaN")
            
            # Проверка на модела за специфични грешки, които водят до INF/NaN
            print("\nПроверка на модела за грешки при изчисленията:")
            test_z_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0, 2000.0])
            for z_val in test_z_values:
                try:
                    dL = model.luminosity_distance(z_val)
                    Hz = model.H_of_z(z_val)
                    rs = model.calculate_sound_horizon(z_val)
                    print(f"  z={z_val:.1f}: dL={dL:.2e}, Hz={Hz:.2e}, rs={rs:.2e}")
                except Exception as ex_model:
                    print(f"  Грешка при изчисление на модела за z={z_val:.1f}: {ex_model}")

        else:
            print("\nОбщият log_likelihood е валиден и краен.")

    except Exception as e:
        print(f"\nГрешка по време на тестване на log_likelihood: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug_test()
