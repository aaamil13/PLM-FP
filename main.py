#!/usr/bin/env python3
"""
Главен изпълнителен файл за линейния космологичен модел

Този файл демонстрира основните възможности на модела и 
създава примерни анализи и визуализации.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Добавяме пътя към модулите
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'analysis'))

from analysis.cosmological_model.scale_factor import ScaleFactorAnalyzer, ScaleFactorParameters
from analysis.time_evolution.temporal_dynamics import TemporalAnalyzer, TemporalParameters
from analysis.density_analysis.density_evolution import DensityAnalyzer, DensityParameters


def main():
    """Главна функция за демонстрация на модела"""
    
    print("=" * 60)
    print("ЛИНЕЕН КОСМОЛОГИЧЕН МОДЕЛ")
    print("Тест на времеви пространствен модел без тъмна енергия")
    print("=" * 60)
    print()
    
    # Параметри на модела
    k = 1.0  # Константа на разширение
    a0 = 1.0  # Начален мащабен фактор
    t0 = 1.0  # Начално време
    rho0 = 1.0  # Начална плътност
    tau0 = 1.0  # Начално темпо на времето
    
    print("ПАРАМЕТРИ НА МОДЕЛА:")
    print(f"Константа на разширение: k = {k}")
    print(f"Начален мащабен фактор: a₀ = {a0}")
    print(f"Начално време: t₀ = {t0}")
    print(f"Начална плътност: ρ₀ = {rho0}")
    print(f"Начално темпо на времето: τ₀ = {tau0}")
    print()
    
    # 1. АНАЛИЗ НА МАЩАБНИЯ ФАКТОР
    print("1. АНАЛИЗ НА МАЩАБНИЯ ФАКТОР")
    print("-" * 40)
    
    scale_params = ScaleFactorParameters(k=k, a0=a0, t0=t0)
    scale_analyzer = ScaleFactorAnalyzer(scale_params)
    
    # Тестови времена
    test_times = [0.1, 1.0, 5.0, 10.0]
    
    for t in test_times:
        a = scale_analyzer.scale_factor(t)
        H = scale_analyzer.hubble_parameter(t)
        print(f"t = {t:4.1f}: a(t) = {a:6.2f}, H(t) = {H:6.2f}")
    
    print()
    
    # 2. АНАЛИЗ НА ВРЕМЕВАТА ЕВОЛЮЦИЯ
    print("2. АНАЛИЗ НА ВРЕМЕВАТА ЕВОЛЮЦИЯ")
    print("-" * 40)
    
    temporal_params = TemporalParameters(tau0=tau0, rho0=rho0, a0=a0, k=k)
    temporal_analyzer = TemporalAnalyzer(temporal_params)
    
    for t in test_times:
        tempo = temporal_analyzer.time_tempo(t)
        dilation = temporal_analyzer.time_dilation_factor(t)
        print(f"t = {t:4.1f}: τ(t) = {tempo:8.3f}, Забавяне = {dilation:8.3f}")
    
    print()
    
    # 3. АНАЛИЗ НА ПЛЪТНОСТТА
    print("3. АНАЛИЗ НА ПЛЪТНОСТТА")
    print("-" * 40)
    
    density_params = DensityParameters(rho0=rho0, a0=a0, k=k)
    density_analyzer = DensityAnalyzer(density_params)
    
    for t in test_times:
        rho = density_analyzer.density(t)
        energy = density_analyzer.energy_density(t)
        print(f"t = {t:4.1f}: ρ(t) = {rho:8.3f}, E(t) = {energy:8.3f}")
    
    print()
    
    # 4. СПЕЦИАЛНИ ИЗЧИСЛЕНИЯ
    print("4. СПЕЦИАЛНИ ИЗЧИСЛЕНИЯ")
    print("-" * 40)
    
    # Време за половин темпо
    t_half_tempo = temporal_analyzer.find_half_tempo_time()
    print(f"Време за половин темпо: t = {t_half_tempo:.3f}")
    
    # Време за половин плътност
    t_half_density = density_analyzer.half_density_time()
    print(f"Време за половин плътност: t = {t_half_density:.3f}")
    
    # Анализ на времевата асиметрия
    asymmetry = temporal_analyzer.analyze_time_asymmetry(t_max=10.0)
    print(f"Съотношение ранно/късно време: {asymmetry['съотношение_ранно_късно']:.1f}")
    
    # Анализ на стабилността
    stability = density_analyzer.stability_analysis(t_eval=1.0)
    print(f"Стабилност на системата: {stability['стабилност']}")
    
    print()
    
    # 5. СРАВНЕНИЕ С ДРУГИ МОДЕЛИ
    print("5. СРАВНЕНИЕ С ДРУГИ МОДЕЛИ")
    print("-" * 40)
    
    t_comp = 5.0
    
    # Линеен модел
    a_linear = k * t_comp
    rho_linear = rho0 * (a0 / (k * t_comp))**3
    
    # Стандартни модели (за сравнение)
    a_radiation = np.sqrt(t_comp)  # a ∝ t^(1/2)
    a_matter = t_comp**(2/3)       # a ∝ t^(2/3)
    
    print(f"При t = {t_comp}:")
    print(f"Линеен модел:     a = {a_linear:.2f}, ρ = {rho_linear:.3f}")
    print(f"Лъчение модел:    a = {a_radiation:.2f}")
    print(f"Материя модел:    a = {a_matter:.2f}")
    
    print()
    
    # 6. ГРАФИКИ
    print("6. СЪЗДАВАНЕ НА ГРАФИКИ")
    print("-" * 40)
    
    try:
        # Мащабен фактор
        print("Създаване на графики за мащабния фактор...")
        fig1, axes1 = scale_analyzer.plot_evolution(t_max=10.0)
        plt.savefig('scale_factor_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Времева еволюция
        print("Създаване на графики за времевата еволюция...")
        fig2, axes2 = temporal_analyzer.plot_temporal_evolution(t_max=10.0)
        plt.savefig('temporal_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Плътност
        print("Създаване на графики за плътността...")
        fig3, axes3 = density_analyzer.plot_density_evolution(t_max=10.0)
        plt.savefig('density_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Сравнение на модели
        print("Създаване на сравнителни графики...")
        fig4, ax4 = scale_analyzer.compare_with_standard_model(t_max=10.0)
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Всички графики са запазени успешно!")
        
    except Exception as e:
        print(f"Грешка при създаването на графики: {e}")
        print("Моля, уверете се, че matplotlib е инсталиран правилно.")
    
    print()
    
    # 7. РЕЗУЛТАТИ И ЗАКЛЮЧЕНИЯ
    print("7. РЕЗУЛТАТИ И ЗАКЛЮЧЕНИЯ")
    print("-" * 40)
    
    print("КЛЮЧОВИ РЕЗУЛТАТИ:")
    print("• Мащабният фактор расте линейно с времето")
    print("• Плътността намалява като 1/t³")
    print("• Темпото на времето се ускорява като 1/t³")
    print("• Параметърът на Хъбъл намалява като 1/t")
    print("• Няма забавяне или ускорение на разширението")
    print()
    
    print("ФИЗИЧЕСКА ИНТЕРПРЕТАЦИЯ:")
    print("• Времето в ранната Вселена е текло много бавно")
    print("• Съвременното време тече значително по-бързо")
    print("• Плътността намалява по-бързо от стандартните модели")
    print("• Моделът не изисква тъмна енергия")
    print()
    
    print("ОГРАНИЧЕНИЯ:")
    print("• Моделът е опростен и не включва всички физически ефекти")
    print("• Не обяснява ранните фази като инфлация")
    print("• Изисква валидация с наблюдения")
    print("• Не съвпада с предсказанията на ОТО за материална Вселена")
    
    print()
    print("=" * 60)
    print("АНАЛИЗЪТ ЗАВЪРШИ УСПЕШНО!")
    print("=" * 60)


if __name__ == "__main__":
    main() 