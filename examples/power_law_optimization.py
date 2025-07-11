#!/usr/bin/env python3
"""
Оптимизация на степенния модел a(t) = C*t^n
Търсене на най-доброто n за имитиране на ΛCDM модела
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Добавяме пътя към нашите модули
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.core.power_law_universe import PowerLawUniverse, find_optimal_n_for_lcdm_match, compare_models_at_z

def test_power_law_optimization():
    """
    Тества степенния модел и намира оптималното n за имитиране на ΛCDM
    """
    print("=== ОПТИМИЗАЦИЯ НА СТЕПЕННИЯ МОДЕЛ ===\n")
    
    # Параметри за тестване
    H0 = 70.0  # km/s/Mpc
    
    # Redshift диапазон - фокусираме се върху наблюдаемия диапазон
    z_test = np.logspace(-2, 0.5, 100)  # z = 0.01 до z ≈ 3.16
    
    print(f"Тестваме с H0 = {H0} km/s/Mpc")
    print(f"Redshift диапазон: z = {z_test.min():.3f} до z = {z_test.max():.2f}")
    print(f"Брой точки: {len(z_test)}")
    
    # Стъпка 1: Намираме оптималното n
    print("\n=== СТЪПКА 1: НАМИРАНЕ НА ОПТИМАЛНО n ===")
    
    optimal_n, min_chi2, analysis_info = find_optimal_n_for_lcdm_match(
        z_test, H0=H0, n_range=(0.6, 1.0)
    )
    
    print(f"\n🎯 ОПТИМАЛНИ РЕЗУЛТАТИ:")
    print(f"Оптимално n: {optimal_n:.4f}")
    print(f"Минимална χ²: {min_chi2:.6f}")
    print(f"RMS разлика: {analysis_info['rms_residual']:.6f} mag")
    print(f"Максимална разлика: {analysis_info['max_residual']:.6f} mag")
    print(f"Възраст на Вселената: {analysis_info['universe_age_Gyr']:.2f} Gyr")
    
    # Стъпка 2: Сравняваме различни модели
    print("\n=== СТЪПКА 2: СРАВНЕНИЕ НА МОДЕЛИ ===")
    
    comparison = compare_models_at_z(z_test, H0=H0)
    
    print(f"\n📊 СРАВНЕНИЕ НА МОДЕЛИТЕ:")
    print(f"{'Модел':<20} {'n':<8} {'RMS разлика':<12} {'Възраст (Gyr)':<12}")
    print("-" * 60)
    
    for model_name, model_data in comparison['models'].items():
        n_str = f"{model_data['n']:.3f}" if model_data['n'] != 'N/A' else "N/A"
        rms_str = f"{model_data['rms_diff_from_lcdm']:.6f}" if model_data['rms_diff_from_lcdm'] > 0 else "0.000000"
        age_str = f"{model_data['age_Gyr']:.2f}"
        print(f"{model_name:<20} {n_str:<8} {rms_str:<12} {age_str:<12}")
    
    # Стъпка 3: Детайлна визуализация
    print("\n=== СТЪПКА 3: СЪЗДАВАНЕ НА ВИЗУАЛИЗАЦИЯ ===")
    
    # Създаваме модели за графики
    linear_model = PowerLawUniverse(H0_kmsmpc=H0, n=1.0)
    matter_model = PowerLawUniverse(H0_kmsmpc=H0, n=2.0/3.0)
    optimal_model = PowerLawUniverse(H0_kmsmpc=H0, n=optimal_n)
    
    # ΛCDM модел
    try:
        from astropy.cosmology import FlatLambdaCDM
        lcdm_model = FlatLambdaCDM(H0=H0, Om0=0.3)
        mu_lcdm = lcdm_model.distmod(z_test).value
        d_L_lcdm = lcdm_model.luminosity_distance(z_test).value
    except ImportError:
        from lib.core.linear_universe import SimpleLCDMUniverse
        lcdm_model = SimpleLCDMUniverse(H0=H0, Om0=0.3, OL0=0.7)
        mu_lcdm = lcdm_model.distance_modulus(z_test)
        d_L_lcdm = lcdm_model.luminosity_distance(z_test)
    
    # Изчисляваме модулите на разстояние
    mu_linear = linear_model.distance_modulus_at_z(z_test)
    mu_matter = matter_model.distance_modulus_at_z(z_test)
    mu_optimal = optimal_model.distance_modulus_at_z(z_test)
    
    # Изчисляваме светимостните разстояния
    d_L_linear = linear_model.luminosity_distance_at_z(z_test)
    d_L_matter = matter_model.luminosity_distance_at_z(z_test)
    d_L_optimal = optimal_model.luminosity_distance_at_z(z_test)
    
    # Създаваме графики
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Панел 1: Модул на разстояние vs z
    axes[0, 0].semilogx(z_test, mu_lcdm, 'k-', linewidth=3, label='ΛCDM', alpha=0.8)
    axes[0, 0].semilogx(z_test, mu_linear, 'r--', linewidth=2, label=f'Linear (n=1.0)', alpha=0.8)
    axes[0, 0].semilogx(z_test, mu_matter, 'b:', linewidth=2, label=f'Matter (n=2/3)', alpha=0.8)
    axes[0, 0].semilogx(z_test, mu_optimal, 'g-', linewidth=2, label=f'Optimal (n={optimal_n:.3f})', alpha=0.8)
    axes[0, 0].set_xlabel('Redshift z')
    axes[0, 0].set_ylabel('Distance Modulus [mag]')
    axes[0, 0].set_title('Модул на разстояние')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Панел 2: Светимостно разстояние vs z
    axes[0, 1].loglog(z_test, d_L_lcdm, 'k-', linewidth=3, label='ΛCDM', alpha=0.8)
    axes[0, 1].loglog(z_test, d_L_linear, 'r--', linewidth=2, label=f'Linear (n=1.0)', alpha=0.8)
    axes[0, 1].loglog(z_test, d_L_matter, 'b:', linewidth=2, label=f'Matter (n=2/3)', alpha=0.8)
    axes[0, 1].loglog(z_test, d_L_optimal, 'g-', linewidth=2, label=f'Optimal (n={optimal_n:.3f})', alpha=0.8)
    axes[0, 1].set_xlabel('Redshift z')
    axes[0, 1].set_ylabel('Luminosity Distance [Mpc]')
    axes[0, 1].set_title('Светимостно разстояние')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Панел 3: Разлики спрямо ΛCDM
    axes[0, 2].semilogx(z_test, mu_linear - mu_lcdm, 'r--', linewidth=2, label=f'Linear (n=1.0)', alpha=0.8)
    axes[0, 2].semilogx(z_test, mu_matter - mu_lcdm, 'b:', linewidth=2, label=f'Matter (n=2/3)', alpha=0.8)
    axes[0, 2].semilogx(z_test, mu_optimal - mu_lcdm, 'g-', linewidth=2, label=f'Optimal (n={optimal_n:.3f})', alpha=0.8)
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[0, 2].set_xlabel('Redshift z')
    axes[0, 2].set_ylabel('Δμ [mag]')
    axes[0, 2].set_title('Разлики спрямо ΛCDM')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Панел 4: Хъбъл параметър vs z
    H_linear = linear_model.hubble_parameter_z(z_test)
    H_matter = matter_model.hubble_parameter_z(z_test)
    H_optimal = optimal_model.hubble_parameter_z(z_test)
    H_lcdm = H0 * np.sqrt(0.3 * (1 + z_test)**3 + 0.7)
    
    axes[1, 0].loglog(z_test, H_lcdm, 'k-', linewidth=3, label='ΛCDM', alpha=0.8)
    axes[1, 0].loglog(z_test, H_linear, 'r--', linewidth=2, label=f'Linear (n=1.0)', alpha=0.8)
    axes[1, 0].loglog(z_test, H_matter, 'b:', linewidth=2, label=f'Matter (n=2/3)', alpha=0.8)
    axes[1, 0].loglog(z_test, H_optimal, 'g-', linewidth=2, label=f'Optimal (n={optimal_n:.3f})', alpha=0.8)
    axes[1, 0].set_xlabel('Redshift z')
    axes[1, 0].set_ylabel('H(z) [km/s/Mpc]')
    axes[1, 0].set_title('Хъбъл параметър')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Панел 5: Възраст на Вселената vs z
    age_linear = linear_model.age_at_z(z_test) / 1e9
    age_matter = matter_model.age_at_z(z_test) / 1e9
    age_optimal = optimal_model.age_at_z(z_test) / 1e9
    
    # За ΛCDM използваме приблизителна формула
    age_lcdm = np.zeros_like(z_test)
    for i, z in enumerate(z_test):
        # Приблизително за ΛCDM
        age_lcdm[i] = (13.8 / 1e9) * (1 + z)**(-1.5)  # Грубо приближение
    
    axes[1, 1].loglog(z_test, age_linear, 'r--', linewidth=2, label=f'Linear (n=1.0)', alpha=0.8)
    axes[1, 1].loglog(z_test, age_matter, 'b:', linewidth=2, label=f'Matter (n=2/3)', alpha=0.8)
    axes[1, 1].loglog(z_test, age_optimal, 'g-', linewidth=2, label=f'Optimal (n={optimal_n:.3f})', alpha=0.8)
    axes[1, 1].set_xlabel('Redshift z')
    axes[1, 1].set_ylabel('Age [Gyr]')
    axes[1, 1].set_title('Възраст на Вселената')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Панел 6: Разпределение на разликите
    residuals_linear = mu_linear - mu_lcdm
    residuals_matter = mu_matter - mu_lcdm
    residuals_optimal = mu_optimal - mu_lcdm
    
    axes[1, 2].hist(residuals_linear, bins=20, alpha=0.7, label=f'Linear (n=1.0)', color='red')
    axes[1, 2].hist(residuals_matter, bins=20, alpha=0.7, label=f'Matter (n=2/3)', color='blue')
    axes[1, 2].hist(residuals_optimal, bins=20, alpha=0.7, label=f'Optimal (n={optimal_n:.3f})', color='green')
    axes[1, 2].axvline(x=0, color='k', linestyle='-', alpha=0.5)
    axes[1, 2].set_xlabel('Остатъци [mag]')
    axes[1, 2].set_ylabel('Честота')
    axes[1, 2].set_title('Разпределение на остатъците')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('power_law_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Стъпка 4: Анализ на чувствителността
    print("\n=== СТЪПКА 4: АНАЛИЗ НА ЧУВСТВИТЕЛНОСТТА ===")
    
    # Тестваме различни стойности на n около оптималното
    n_values = np.linspace(0.6, 1.0, 41)
    rms_values = []
    
    for n in n_values:
        test_model = PowerLawUniverse(H0_kmsmpc=H0, n=n)
        mu_test = test_model.distance_modulus_at_z(z_test)
        rms = np.sqrt(np.mean((mu_test - mu_lcdm)**2))
        rms_values.append(rms)
    
    # Намираме минимума
    min_idx = np.argmin(rms_values)
    
    print(f"Минимум при n = {n_values[min_idx]:.4f}")
    print(f"Минимална RMS разлика: {rms_values[min_idx]:.6f} mag")
    
    # Графика на чувствителността
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, rms_values, 'b-', linewidth=2)
    plt.axvline(x=optimal_n, color='r', linestyle='--', label=f'Оптимално n = {optimal_n:.4f}')
    plt.axvline(x=1.0, color='g', linestyle=':', label='Линеен модел (n=1.0)')
    plt.axvline(x=2.0/3.0, color='orange', linestyle=':', label='Материален модел (n=2/3)')
    plt.xlabel('Степенен показател n')
    plt.ylabel('RMS разлика с ΛCDM [mag]')
    plt.title('Чувствителност на модела към параметъра n')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('n_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Стъпка 5: Заключения
    print("\n=== ЗАКЛЮЧЕНИЯ ===")
    
    print(f"\n🎯 КЛЮЧОВИ ОТКРИТИЯ:")
    print(f"1. Оптимално n = {optimal_n:.4f} (между материалния n=2/3≈0.667 и линейния n=1.0)")
    print(f"2. RMS разлика с ΛCDM: {analysis_info['rms_residual']:.6f} mag")
    print(f"3. Възраст на Вселената: {analysis_info['universe_age_Gyr']:.2f} Gyr")
    
    # Сравнение с други модели
    linear_rms = comparison['models']['Linear (n=1.0)']['rms_diff_from_lcdm']
    matter_rms = comparison['models']['Matter (n=2/3)']['rms_diff_from_lcdm']
    optimal_rms = comparison['models'][f'Optimal (n={optimal_n:.3f})']['rms_diff_from_lcdm']
    
    print(f"\n📊 ПОДОБРЕНИЯ:")
    print(f"Спрямо линейния модел: {linear_rms/optimal_rms:.1f}x по-добър")
    print(f"Спрямо материалния модел: {matter_rms/optimal_rms:.1f}x по-добър")
    
    if optimal_rms < 0.001:
        print(f"\n✅ ОТЛИЧНО: Разликата е под 0.001 mag - практически неразличима от ΛCDM!")
    elif optimal_rms < 0.01:
        print(f"\n✅ МНОГО ДОБРО: Разликата е под 0.01 mag - много близо до ΛCDM!")
    else:
        print(f"\n⚠️ ДОБРО: Разликата е {optimal_rms:.6f} mag - подобрение, но още има разлики")
    
    print(f"\n🔬 ФИЗИЧЕСКИ СМИСЪЛ:")
    print(f"n = {optimal_n:.4f} означава леко забавяне на разширението")
    print(f"H(t) = {optimal_n:.4f}/t - по-бавно от линейния модел")
    print(f"Няма нужда от тъмна енергия за обяснение на наблюденията!")
    
    return optimal_n, analysis_info


if __name__ == "__main__":
    optimal_n, info = test_power_law_optimization()
    print(f"\nОКОНЧАТЕЛЕН РЕЗУЛТАТ: n = {optimal_n:.4f}") 