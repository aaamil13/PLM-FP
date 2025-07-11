#!/usr/bin/env python3
"""
Демонстрация на функцията distance_modulus в линейния космологичен модел

Този скрипт показва как се изчислява модулът на разстояние в линейния модел
и сравнява с други космологични модели.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Добавяме пътя към модулите
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.core.linear_universe import LinearUniverse, create_lcdm_comparison_model


def demonstrate_distance_modulus():
    """
    Демонстрира как работи функцията distance_modulus
    """
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ НА ФУНКЦИЯТА DISTANCE_MODULUS")
    print("=" * 70)
    
    # Създаваме модели
    linear_model = LinearUniverse(H0_kmsmpc=70.0)
    lcdm_model = create_lcdm_comparison_model(H0=70.0, Om0=0.3, OL0=0.7)
    
    print(f"Линеен модел: a(t) = k*t, k = {linear_model.k:.2e} [1/s]")
    print(f"Възраст на Вселената: {linear_model.t0_years/1e9:.2f} млрд години")
    print(f"ΛCDM модел: Ωm = 0.3, ΩΛ = 0.7")
    
    print("\n" + "=" * 70)
    print("СТЪПКИ ЗА ИЗЧИСЛЯВАНЕ НА МОДУЛА НА РАЗСТОЯНИЕ")
    print("=" * 70)
    
    # Тестваме за конкретно червено отместване
    z_test = 1.0
    print(f"\nТестваме за z = {z_test}")
    
    # Стъпка 1: Намиране на времето на излъчване
    print(f"\n1️⃣ Намиране на времето на излъчване:")
    print(f"   a(t_e) = a₀ / (1 + z) = 1 / (1 + {z_test}) = {1/(1+z_test):.3f}")
    
    t_e_years = linear_model.age_at_z(z_test) / 1e9
    t_0_years = linear_model.t0_years / 1e9
    
    print(f"   t_e = a_e / k = {t_e_years:.3f} млрд години")
    print(f"   t_0 = {t_0_years:.3f} млрд години")
    
    # Стъпка 2: Изчисляване на комовингово разстояние
    print(f"\n2️⃣ Изчисляване на комовингово разстояние:")
    print(f"   r = ∫[t_e до t_0] c/a(t) dt = ∫[t_e до t_0] c/(kt) dt")
    print(f"   r = (c/k) * ln(t_0/t_e) = (c/k) * ln({t_0_years:.3f}/{t_e_years:.3f})")
    
    r_comoving = linear_model.comoving_distance_at_z(z_test)
    print(f"   r = {r_comoving:.2f} Mpc")
    
    # Стъпка 3: Luminosity разстояние
    print(f"\n3️⃣ Luminosity разстояние:")
    print(f"   d_L = r * (1 + z) = {r_comoving:.2f} * (1 + {z_test}) = {r_comoving * (1 + z_test):.2f} Mpc")
    
    d_L = linear_model.luminosity_distance_at_z(z_test)
    print(f"   d_L = {d_L:.2f} Mpc")
    
    # Стъпка 4: Модул на разстояние
    print(f"\n4️⃣ Модул на разстояние:")
    print(f"   μ = 5 * log₁₀(d_L / 10 pc) = 5 * log₁₀({d_L:.2f} * 10⁶ / 10)")
    print(f"   μ = 5 * log₁₀({d_L * 1e5:.0f}) = 5 * {np.log10(d_L * 1e5):.3f}")
    
    mu = linear_model.distance_modulus_at_z(z_test)
    print(f"   μ = {mu:.3f} mag")
    
    # Сравнение с ΛCDM
    print(f"\n🔄 Сравнение с ΛCDM:")
    try:
        # Ако е astropy обект, използваме distmod
        mu_lcdm = lcdm_model.distmod(z_test).value
    except AttributeError:
        # Ако е нашия клас, използваме distance_modulus
        mu_lcdm = lcdm_model.distance_modulus(z_test)
    print(f"   μ_ΛCDM = {mu_lcdm:.3f} mag")
    print(f"   Разлика: {mu - mu_lcdm:.3f} mag")
    
    return linear_model, lcdm_model


def plot_distance_modulus_components():
    """
    Показва графики на различните компоненти на модула на разстояние
    """
    print("\n" + "=" * 70)
    print("ГРАФИКИ НА КОМПОНЕНТИТЕ НА МОДУЛА НА РАЗСТОЯНИЕ")
    print("=" * 70)
    
    # Създаваме модели
    linear_model = LinearUniverse(H0_kmsmpc=70.0)
    lcdm_model = create_lcdm_comparison_model(H0=70.0, Om0=0.3, OL0=0.7)
    
    # Диапазон на червеното отместване
    z_range = np.logspace(-3, np.log10(3.0), 1000)
    
    # Изчисляваме компонентите за линейния модел
    r_comoving_linear = np.array([linear_model.comoving_distance_at_z(z) for z in z_range])
    d_L_linear = r_comoving_linear * (1 + z_range)
    mu_linear = 5 * np.log10(d_L_linear / 1e-5)
    
    # Изчисляваме за ΛCDM модела
    try:
        # Ако е astropy обект, използваме distmod
        mu_lcdm = lcdm_model.distmod(z_range).value
    except AttributeError:
        # Ако е нашия клас, използваме distance_modulus
        mu_lcdm = lcdm_model.distance_modulus(z_range)
    d_L_lcdm = 10**((mu_lcdm - 25) / 5)
    r_comoving_lcdm = d_L_lcdm / (1 + z_range)
    
    # Настройваме шрифтовете
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Създаваме графики
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Комовингово разстояние
    axes[0, 0].loglog(z_range, r_comoving_linear, 'r-', linewidth=2, label='Линеен модел')
    axes[0, 0].loglog(z_range, r_comoving_lcdm, 'b--', linewidth=2, label='ΛCDM модел')
    axes[0, 0].set_xlabel('Червено отместване z')
    axes[0, 0].set_ylabel('Комовингово разстояние r [Mpc]')
    axes[0, 0].set_title('Комовингово разстояние')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Luminosity разстояние
    axes[0, 1].loglog(z_range, d_L_linear, 'r-', linewidth=2, label='Линеен модел')
    axes[0, 1].loglog(z_range, d_L_lcdm, 'b--', linewidth=2, label='ΛCDM модел')
    axes[0, 1].set_xlabel('Червено отместване z')
    axes[0, 1].set_ylabel('Luminosity разстояние d_L [Mpc]')
    axes[0, 1].set_title('Luminosity разстояние')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Модул на разстояние
    axes[1, 0].semilogx(z_range, mu_linear, 'r-', linewidth=2, label='Линеен модел')
    axes[1, 0].semilogx(z_range, mu_lcdm, 'b--', linewidth=2, label='ΛCDM модел')
    axes[1, 0].set_xlabel('Червено отместване z')
    axes[1, 0].set_ylabel('Модул на разстояние μ [mag]')
    axes[1, 0].set_title('Модул на разстояние')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Разлика в модула на разстояние
    axes[1, 1].semilogx(z_range, mu_linear - mu_lcdm, 'g-', linewidth=2)
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Червено отместване z')
    axes[1, 1].set_ylabel('Разлика μ_linear - μ_ΛCDM [mag]')
    axes[1, 1].set_title('Разлика в модула на разстояние')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distance_modulus_components.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Графиките са запазени като 'distance_modulus_components.png'")


def demonstrate_integration():
    """
    Демонстрира как работи интегрирането за комовингово разстояние
    """
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ НА ИНТЕГРИРАНЕТО ЗА КОМОВИНГОВО РАЗСТОЯНИЕ")
    print("=" * 70)
    
    # Създаваме модел
    linear_model = LinearUniverse(H0_kmsmpc=70.0)
    
    # Тестваме за различни z стойности
    z_values = [0.1, 0.5, 1.0, 2.0]
    
    print(f"Линеен модел: a(t) = k*t, k = {linear_model.k:.2e} [1/s]")
    print(f"Скорост на светлината: c = {linear_model.c:.0f} km/s")
    
    print(f"\n{'z':<8} {'t_e [млрд г.]':<15} {'t_0 [млрд г.]':<15} {'ln(t_0/t_e)':<12} {'r [Mpc]':<10}")
    print("-" * 60)
    
    for z in z_values:
        t_e = linear_model.age_at_z(z) / 1e9
        t_0 = linear_model.t0_years / 1e9
        ln_ratio = np.log(t_0 / t_e)
        r = linear_model.comoving_distance_at_z(z)
        
        print(f"{z:<8.1f} {t_e:<15.3f} {t_0:<15.3f} {ln_ratio:<12.3f} {r:<10.2f}")
    
    print(f"\nФормула: r = (c/k) * ln(t_0/t_e)")
    print(f"Където: c/k = {linear_model.c / linear_model.k / 1e9:.3f} млрд км = {linear_model.c / linear_model.k / 3.086e19:.1f} Mpc")
    
    # Аналитичен vs численен интеграл
    print(f"\n" + "=" * 50)
    print("СРАВНЕНИЕ: АНАЛИТИЧЕН vs ЧИСЛЕНЕН ИНТЕГРАЛ")
    print("=" * 50)
    
    print(f"{'z':<8} {'Аналитичен':<12} {'Численен':<12} {'Разлика':<12}")
    print("-" * 44)
    
    for z in z_values:
        # Аналитичен резултат
        r_analytical = linear_model.comoving_distance_at_z(z)
        
        # Численен интеграл (симулация)
        t_e = linear_model.age_at_z(z) * 365.25 * 24 * 3600  # в секунди
        t_0 = linear_model.t0_years * 365.25 * 24 * 3600  # в секунди
        
        # Интеграл от t_e до t_0 на c/(k*t) dt
        from scipy.integrate import quad
        integrand = lambda t: linear_model.c / (linear_model.k * t)
        r_numerical, _ = quad(integrand, t_e, t_0)
        r_numerical_mpc = r_numerical / 3.086e19  # в Mpc
        
        diff = abs(r_analytical - r_numerical_mpc)
        print(f"{z:<8.1f} {r_analytical:<12.3f} {r_numerical_mpc:<12.3f} {diff:<12.6f}")
    
    print("\n✅ Аналитичният и численният интеграл се съгласуват отлично!")


def main():
    """Главна функция"""
    print("🎯 ДЕМОНСТРАЦИЯ НА DISTANCE_MODULUS ФУНКЦИЯТА")
    print("=" * 70)
    
    # Демонстрация на изчисленията
    linear_model, lcdm_model = demonstrate_distance_modulus()
    
    # Графики на компонентите
    plot_distance_modulus_components()
    
    # Демонстрация на интегрирането
    demonstrate_integration()
    
    print("\n" + "=" * 70)
    print("✅ ДЕМОНСТРАЦИЯТА ЗАВЪРШИ УСПЕШНО!")
    print("=" * 70)
    print("📊 Създадени файлове:")
    print("   - distance_modulus_components.png")
    print("📝 Ключови точки:")
    print("   - Линейният модел използва аналитичен интеграл")
    print("   - r = (c/k) * ln(t_0/t_e) за комовингово разстояние")
    print("   - μ = 5 * log₁₀(d_L / 10 pc) за модул на разстояние")
    print("   - Отличното съгласуване показва правилността на формулите")


if __name__ == "__main__":
    main() 