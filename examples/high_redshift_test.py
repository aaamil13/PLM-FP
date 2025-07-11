#!/usr/bin/env python3
"""
Тестване на линейния модел при високи redshift-ове (z < 1000)
Сравнение с ΛCDM модела за различни космологични параметри
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Добавяме пътя към нашите модули
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.core.linear_universe import LinearUniverse, create_lcdm_comparison_model

def test_high_redshift_comparison():
    """
    Тества и сравнява модели при високи redshift-ове
    """
    print("=== ТЕСТВАНЕ НА МОДЕЛИ ПРИ ВИСОКИ REDSHIFT-ОВЕ ===\n")
    
    # Създаваме модели за тестване
    H0_values = [67.4, 70.0, 73.0]  # различни стойности на H0
    
    # Redshift диапазон от 0 до 1000
    z_low = np.logspace(-1, 1, 50)      # z = 0.1 до 10
    z_medium = np.logspace(1, 2, 50)    # z = 10 до 100  
    z_high = np.logspace(2, 3, 50)      # z = 100 до 1000
    z_all = np.concatenate([z_low, z_medium, z_high])
    z_all = np.sort(z_all)
    
    # Ключови redshift-ове за анализ
    z_key = np.array([1, 3, 10, 100, 300, 1000])
    
    plt.figure(figsize=(15, 12))
    
    # Панел 1: Светимостно разстояние
    plt.subplot(2, 3, 1)
    colors = ['red', 'blue', 'green']
    
    for i, H0 in enumerate(H0_values):
        # Линеен модел
        linear_model = LinearUniverse(H0)
        d_L_linear = linear_model.luminosity_distance_at_z(z_all)
        
        # ΛCDM модел
        lcdm_model = create_lcdm_comparison_model(H0=H0, Om0=0.3, OL0=0.7)
        try:
            # Пробваме astropy версия
            d_L_lcdm = lcdm_model.luminosity_distance(z_all).value
        except:
            # Използваме нашата версия
            d_L_lcdm = lcdm_model.luminosity_distance(z_all)
        
        plt.loglog(z_all, d_L_linear, '--', color=colors[i], 
                  label=f'Linear H0={H0}', linewidth=2)
        plt.loglog(z_all, d_L_lcdm, '-', color=colors[i], 
                  label=f'ΛCDM H0={H0}', linewidth=2)
    
    plt.xlabel('Redshift z')
    plt.ylabel('Luminosity Distance [Mpc]')
    plt.title('Светимостно разстояние vs Redshift')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Панел 2: Отношение на разстоянията
    plt.subplot(2, 3, 2)
    
    for i, H0 in enumerate(H0_values):
        linear_model = LinearUniverse(H0)
        d_L_linear = linear_model.luminosity_distance_at_z(z_all)
        
        lcdm_model = create_lcdm_comparison_model(H0=H0, Om0=0.3, OL0=0.7)
        try:
            d_L_lcdm = lcdm_model.luminosity_distance(z_all).value
        except:
            d_L_lcdm = lcdm_model.luminosity_distance(z_all)
        
        ratio = d_L_linear / d_L_lcdm
        plt.semilogx(z_all, ratio, color=colors[i], 
                    label=f'H0={H0}', linewidth=2)
    
    plt.xlabel('Redshift z')
    plt.ylabel('d_L(Linear) / d_L(ΛCDM)')
    plt.title('Отношение на разстоянията')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='black', linestyle=':', alpha=0.5)
    
    # Панел 3: Възраст на Вселената
    plt.subplot(2, 3, 3)
    
    for i, H0 in enumerate(H0_values):
        linear_model = LinearUniverse(H0)
        age_linear = linear_model.age_at_z(z_all)
        
        lcdm_model = create_lcdm_comparison_model(H0=H0, Om0=0.3, OL0=0.7)
        try:
            age_lcdm = lcdm_model.age(z_all).value
        except:
            age_lcdm = lcdm_model.age(z_all)
        
        plt.loglog(z_all, age_linear/1e9, '--', color=colors[i], 
                  label=f'Linear H0={H0}', linewidth=2)
        plt.loglog(z_all, age_lcdm/1e9, '-', color=colors[i], 
                  label=f'ΛCDM H0={H0}', linewidth=2)
    
    plt.xlabel('Redshift z')
    plt.ylabel('Age [Gyr]')
    plt.title('Възраст на Вселената')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Панел 4: Hubble параметър
    plt.subplot(2, 3, 4)
    
    for i, H0 in enumerate(H0_values):
        linear_model = LinearUniverse(H0)
        # За линейния модел: H(z) = H0 * (1 + z)
        H_linear = H0 * (1 + z_all)
        
        # За ΛCDM: H(z) = H0 * E(z)
        lcdm_model = create_lcdm_comparison_model(H0=H0, Om0=0.3, OL0=0.7)
        if hasattr(lcdm_model, 'E'):
            H_lcdm = H0 * lcdm_model.E(z_all)
        else:
            # Астрофизическа версия
            H_lcdm = H0 * np.sqrt(0.3 * (1 + z_all)**3 + 0.7)
        
        plt.loglog(z_all, H_linear, '--', color=colors[i], 
                  label=f'Linear H0={H0}', linewidth=2)
        plt.loglog(z_all, H_lcdm, '-', color=colors[i], 
                  label=f'ΛCDM H0={H0}', linewidth=2)
    
    plt.xlabel('Redshift z')
    plt.ylabel('H(z) [km/s/Mpc]')
    plt.title('Хъбъл параметър')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Панел 5: Lookback time
    plt.subplot(2, 3, 5)
    
    for i, H0 in enumerate(H0_values):
        linear_model = LinearUniverse(H0)
        lookback_linear = linear_model.lookback_time(z_all)
        
        lcdm_model = create_lcdm_comparison_model(H0=H0, Om0=0.3, OL0=0.7)
        try:
            age_z0 = lcdm_model.age(0).value
            age_z = lcdm_model.age(z_all).value
        except:
            age_z0 = lcdm_model.age(0)
            age_z = lcdm_model.age(z_all)
        
        lookback_lcdm = (age_z0 - age_z) / 1e9
        
        plt.loglog(z_all, lookback_linear/1e9, '--', color=colors[i], 
                  label=f'Linear H0={H0}', linewidth=2)
        plt.loglog(z_all, lookback_lcdm, '-', color=colors[i], 
                  label=f'ΛCDM H0={H0}', linewidth=2)
    
    plt.xlabel('Redshift z')
    plt.ylabel('Lookback Time [Gyr]')
    plt.title('Време на поглед назад')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Панел 6: Модул на разстояние
    plt.subplot(2, 3, 6)
    
    for i, H0 in enumerate(H0_values):
        linear_model = LinearUniverse(H0)
        mu_linear = linear_model.distance_modulus_at_z(z_all)
        
        lcdm_model = create_lcdm_comparison_model(H0=H0, Om0=0.3, OL0=0.7)
        try:
            mu_lcdm = lcdm_model.distmod(z_all).value
        except:
            mu_lcdm = lcdm_model.distance_modulus(z_all)
        
        plt.semilogx(z_all, mu_linear, '--', color=colors[i], 
                    label=f'Linear H0={H0}', linewidth=2)
        plt.semilogx(z_all, mu_lcdm, '-', color=colors[i], 
                    label=f'ΛCDM H0={H0}', linewidth=2)
    
    plt.xlabel('Redshift z')
    plt.ylabel('Distance Modulus [mag]')
    plt.title('Модул на разстояние')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('high_redshift_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Детайлна таблица за ключови redshift-ове
    print("\n=== ДЕТАЙЛНО СРАВНЕНИЕ ЗА КЛЮЧОВИ REDSHIFT-ОВЕ ===\n")
    print("Използвам H0 = 70.0 km/s/Mpc за сравнение")
    
    linear_model = LinearUniverse(70.0)
    lcdm_model = create_lcdm_comparison_model(H0=70.0, Om0=0.3, OL0=0.7)
    
    print(f"{'z':<6} {'d_L_Lin':<12} {'d_L_LCDM':<12} {'Ratio':<8} {'Age_Lin':<10} {'Age_LCDM':<10} {'μ_Lin':<8} {'μ_LCDM':<8}")
    print("-" * 80)
    
    for z in z_key:
        d_L_linear = linear_model.luminosity_distance_at_z(z)
        age_linear = linear_model.age_at_z(z) / 1e9
        mu_linear = linear_model.distance_modulus_at_z(z)
        
        try:
            d_L_lcdm = lcdm_model.luminosity_distance(z).value
            age_lcdm = lcdm_model.age(z).value / 1e9
            mu_lcdm = lcdm_model.distmod(z).value
        except:
            d_L_lcdm = lcdm_model.luminosity_distance(z)
            age_lcdm = lcdm_model.age(z) / 1e9
            mu_lcdm = lcdm_model.distance_modulus(z)
        
        ratio = d_L_linear / d_L_lcdm
        
        print(f"{z:<6.0f} {d_L_linear:<12.2e} {d_L_lcdm:<12.2e} {ratio:<8.3f} "
              f"{age_linear:<10.3f} {age_lcdm:<10.3f} {mu_linear:<8.2f} {mu_lcdm:<8.2f}")
    
    # Анализ на различията
    print("\n=== АНАЛИЗ НА РАЗЛИЧИЯТА ===\n")
    
    # Намираме къде различията са най-големи
    z_test = np.logspace(0, 3, 1000)
    d_L_linear_test = linear_model.luminosity_distance_at_z(z_test)
    
    try:
        d_L_lcdm_test = lcdm_model.luminosity_distance(z_test).value
    except:
        d_L_lcdm_test = lcdm_model.luminosity_distance(z_test)
    
    ratio_test = d_L_linear_test / d_L_lcdm_test
    
    max_ratio_idx = np.argmax(ratio_test)
    min_ratio_idx = np.argmin(ratio_test)
    
    print(f"Максимално отношение: {ratio_test[max_ratio_idx]:.3f} при z = {z_test[max_ratio_idx]:.1f}")
    print(f"Минимално отношение: {ratio_test[min_ratio_idx]:.3f} при z = {z_test[min_ratio_idx]:.1f}")
    
    # Проверка на рекомбинационната епоха
    z_recomb = 1100
    if z_recomb <= 1000:
        print(f"\nПри z = {z_recomb} (рекомбинация):")
        age_recomb_linear = linear_model.age_at_z(z_recomb) / 1e6  # в My
        print(f"Възраст според линейния модел: {age_recomb_linear:.1f} милиона години")
        print("Това е МНОГО млада Вселена за рекомбинацията!")
    
    print(f"\nПри z = 1000:")
    age_1000_linear = linear_model.age_at_z(1000) / 1e6  # в My
    print(f"Възраст според линейния модел: {age_1000_linear:.1f} милиона години")
    
    try:
        age_1000_lcdm = lcdm_model.age(1000).value / 1e6
    except:
        age_1000_lcdm = lcdm_model.age(1000) / 1e6
    
    print(f"Възраст според ΛCDM модела: {age_1000_lcdm:.1f} милиона години")
    
    print(f"\nТемпературата на CMB при z = 1000:")
    T_cmb_today = 2.725  # K
    T_cmb_1000 = T_cmb_today * (1 + 1000)
    print(f"T_CMB = {T_cmb_1000:.0f} K = {T_cmb_1000 - 273:.0f} °C")
    print("Това е много близо до температурата на рекомбинацията (~3000 K)")


if __name__ == "__main__":
    test_high_redshift_comparison() 