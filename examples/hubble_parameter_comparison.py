#!/usr/bin/env python3
"""
Сравнение на Хъбъл параметъра в различни космологични модели

Този пример демонстрира точните математически връзки и разликите
между линейния модел и стандартния материален модел.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Добавяме пътя към модулите
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.core.cosmology import (
    hubble_parameter, 
    hubble_parameter_matter_model,
    compare_hubble_evolution,
    cosmic_age_from_hubble
)


def main():
    """Главна функция за демонстрация"""
    
    print("=" * 70)
    print("СРАВНЕНИЕ НА ХЪБЪЛ ПАРАМЕТЪРА В РАЗЛИЧНИ МОДЕЛИ")
    print("=" * 70)
    print()
    
    # Параметри
    H0 = 70.0  # km/s/Mpc (примерна стойност)
    
    print("МАТЕМАТИЧЕСКИ ОСНОВИ:")
    print("-" * 40)
    print("Линеен модел:")
    print("  a(t) = kt")
    print("  da/dt = k")
    print("  H(t) = (da/dt)/a = k/(kt) = 1/t")
    print()
    print("Материален модел:")
    print("  a(t) ∝ t^(2/3)")
    print("  da/dt ∝ (2/3)t^(-1/3)")
    print("  H(t) = (da/dt)/a = (2/3)/t")
    print()
    
    # Възраст на Вселената
    print("ВЪЗРАСТ НА ВСЕЛЕНАТА ПРИ H₀ = 70 km/s/Mpc:")
    print("-" * 50)
    
    age_linear = cosmic_age_from_hubble(H0, 'linear')
    age_matter = cosmic_age_from_hubble(H0, 'matter')
    
    # Преобразуваме в години (приблизително)
    # 1/H0 ≈ 14 млрд години за H0 = 70 km/s/Mpc
    # По-точно: 1/H0 = 1/(70 km/s/Mpc) ≈ 14 млрд години
    hubble_time_gyr = 14.0 / H0 * 70.0  # милиарди години
    
    print(f"Линеен модел:   t₀ = 1/H₀ = {age_linear:.3f} × H₀⁻¹ ≈ {age_linear * hubble_time_gyr:.1f} млрд години")
    print(f"Материален модел: t₀ = (2/3)/H₀ = {age_matter:.3f} × H₀⁻¹ ≈ {age_matter * hubble_time_gyr:.1f} млрд години")
    print(f"ΛCDM модел:     t₀ ≈ 13.8 млрд години (наблюдения)")
    print()
    
    # Съотношение
    ratio = age_linear / age_matter
    print(f"Съотношение възрасти: Линеен/Материален = {ratio:.3f} = 3/2")
    print()
    
    # Еволюция на H(t)
    print("ЕВОЛЮЦИЯ НА H(t) С ВРЕМЕТО:")
    print("-" * 40)
    
    # Времева решетка
    t_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    print("Време t   |  H_linear(t)  |  H_matter(t)  |  Съотношение")
    print("-" * 55)
    
    for t in t_values:
        H_lin = hubble_parameter(t)
        H_mat = hubble_parameter_matter_model(t)
        ratio_h = H_lin / H_mat
        print(f"{t:6.1f}   |   {H_lin:8.3f}     |   {H_mat:8.3f}     |    {ratio_h:.3f}")
    
    print()
    print("Забележка: Съотношението H_linear/H_matter = 3/2 = 1.500 (константно)")
    print()
    
    # Графики
    print("СЪЗДАВАНЕ НА ГРАФИКИ:")
    print("-" * 30)
    
    # Времева решетка за графики
    t_plot = np.linspace(0.1, 10, 1000)
    
    # Сравнение на H(t)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Сравнение на Хъбъл параметъра в различни модели', fontsize=14)
    
    # H(t) еволюция
    H_linear = hubble_parameter(t_plot)
    H_matter = hubble_parameter_matter_model(t_plot)
    
    axes[0, 0].plot(t_plot, H_linear, 'b-', linewidth=2, label='Линеен: H = 1/t')
    axes[0, 0].plot(t_plot, H_matter, 'r--', linewidth=2, label='Материя: H = (2/3)/t')
    axes[0, 0].set_xlabel('Време t')
    axes[0, 0].set_ylabel('Хъбъл параметър H(t)')
    axes[0, 0].set_title('Еволюция на H(t)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xscale('log')
    
    # Съотношение
    ratio_plot = H_linear / H_matter
    axes[0, 1].plot(t_plot, ratio_plot, 'g-', linewidth=2)
    axes[0, 1].axhline(y=1.5, color='k', linestyle=':', alpha=0.7, label='Теоретично: 3/2')
    axes[0, 1].set_xlabel('Време t')
    axes[0, 1].set_ylabel('H_linear/H_matter')
    axes[0, 1].set_title('Съотношение на Хъбъл параметрите')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Мащабни фактори
    a_linear = t_plot  # a ∝ t
    a_matter = t_plot**(2/3)  # a ∝ t^(2/3)
    
    axes[1, 0].plot(t_plot, a_linear, 'b-', linewidth=2, label='Линеен: a ∝ t')
    axes[1, 0].plot(t_plot, a_matter, 'r--', linewidth=2, label='Материя: a ∝ t^(2/3)')
    axes[1, 0].set_xlabel('Време t')
    axes[1, 0].set_ylabel('Мащабен фактор a(t)')
    axes[1, 0].set_title('Еволюция на a(t)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Производни da/dt
    dadt_linear = np.ones_like(t_plot)  # da/dt = const
    dadt_matter = (2/3) * t_plot**(-1/3)  # da/dt ∝ t^(-1/3)
    
    axes[1, 1].plot(t_plot, dadt_linear, 'b-', linewidth=2, label='Линеен: da/dt = const')
    axes[1, 1].plot(t_plot, dadt_matter, 'r--', linewidth=2, label='Материя: da/dt ∝ t^(-1/3)')
    axes[1, 1].set_xlabel('Време t')
    axes[1, 1].set_ylabel('da/dt')
    axes[1, 1].set_title('Скорост на разширение')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('hubble_parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Графиката е запазена като 'hubble_parameter_comparison.png'")
    print()
    
    # Наблюдателни последици
    print("НАБЛЮДАТЕЛНИ ПОСЛЕДИЦИ:")
    print("-" * 30)
    print("1. Различна възраст на Вселената:")
    print(f"   - Линеен модел: {age_linear * hubble_time_gyr:.1f} млрд години")
    print(f"   - Материален модел: {age_matter * hubble_time_gyr:.1f} млрд години")
    print(f"   - Разлика: {(age_linear - age_matter) * hubble_time_gyr:.1f} млрд години")
    print()
    print("2. Различна еволюция на разстоянията:")
    print("   - Интегралите ∫dt/a(t) дават различни резултати")
    print("   - Различни прогнози за светимост на отдалечени обекти")
    print()
    print("3. Измерими разлики:")
    print("   - Възрастта на най-старите звезди")
    print("   - Прецизни измервания на H(t) в различни епохи")
    print("   - Калибриране на стандартни свещи")
    print()
    
    # Количествени тестове
    print("КОЛИЧЕСТВЕНИ СРАВНЕНИЯ:")
    print("-" * 30)
    
    # При определено време
    t_test = 1.0
    comparison = compare_hubble_evolution(t_test)
    
    print(f"При t = {t_test}:")
    print(f"  H_linear = {comparison['linear']:.3f}")
    print(f"  H_matter = {comparison['matter']:.3f}")
    print(f"  Съотношение = {comparison['ratio_linear_to_matter']:.3f}")
    print()
    
    # При различни времена
    print("Развитие на различието във времето:")
    test_times = [0.1, 1.0, 10.0]
    for t in test_times:
        comp = compare_hubble_evolution(t)
        diff_percent = (comp['linear'] - comp['matter']) / comp['matter'] * 100
        print(f"  t = {t:4.1f}: Разлика = {diff_percent:+5.1f}%")
    
    print()
    print("=" * 70)
    print("ЗАКЛЮЧЕНИЕ: Разликата е постоянна (50%) но измерима!")
    print("=" * 70)


if __name__ == "__main__":
    main() 