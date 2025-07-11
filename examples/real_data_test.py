#!/usr/bin/env python3
"""
Тестване на линейния космологичен модел с реални данни от Pantheon+ суперновите

Този скрипт зарежда реални данни от Pantheon+ суперновите и сравнява
предсказанията на линейния модел a(t) = k*t със стандартния ΛCDM модел.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from pathlib import Path

# Добавяме пътя към модулите
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.core.linear_universe import LinearUniverse, create_lcdm_comparison_model


def load_pantheon_data(data_path: str) -> tuple:
    """
    Зарежда данните от Pantheon+ суперновите
    
    Args:
        data_path: Път към файла с данни
        
    Returns:
        Tuple с (redshift, distance_modulus, distance_modulus_error)
    """
    try:
        # Четем данните
        data = pd.read_csv(data_path, delim_whitespace=True)
        
        # Извличаме необходимите колони
        z = data['zHD'].values
        mu = data['MU_SH0ES'].values
        mu_err = data['MU_SH0ES_ERR_DIAG'].values
        
        # Филтрираме валидните данни
        valid_mask = (z > 0) & (~np.isnan(mu)) & (~np.isnan(mu_err)) & (z < 2.5)
        
        z_clean = z[valid_mask]
        mu_clean = mu[valid_mask]
        mu_err_clean = mu_err[valid_mask]
        
        print(f"Заредени {len(z_clean)} валидни точки от данни от {len(z)} общо")
        print(f"Диапазон на червеното отместване: {z_clean.min():.4f} - {z_clean.max():.4f}")
        print(f"Диапазон на модула на разстояние: {mu_clean.min():.2f} - {mu_clean.max():.2f}")
        
        return z_clean, mu_clean, mu_err_clean
        
    except Exception as e:
        print(f"Грешка при зареждане на данните: {e}")
        return None, None, None


def fit_models_to_data(z_data, mu_data, mu_err):
    """
    Fitва линейния и ΛCDM модели към данните
    
    Args:
        z_data: Червено отместване
        mu_data: Модул на разстояние
        mu_err: Грешки в модула на разстояние
        
    Returns:
        Tuple с (linear_model, lcdm_model, linear_stats, lcdm_stats)
    """
    print("\n" + "="*60)
    print("FITВАНЕ НА МОДЕЛИ КЪМ ДАННИТЕ")
    print("="*60)
    
    # Fitваме линейния модел
    print("\nFitване на линейния модел...")
    linear_model = LinearUniverse()
    optimal_H0, linear_stats = linear_model.fit_to_data(z_data, mu_data, mu_err)
    linear_model = LinearUniverse(optimal_H0)
    
    print(f"Оптимално H0 за линейния модел: {optimal_H0:.2f} km/s/Mpc")
    print(f"Възраст на Вселената (линеен модел): {linear_model.t0_years/1e9:.2f} млрд години")
    print(f"Χ²_red = {linear_stats['reduced_chi2']:.3f}")
    print(f"RMS остатък = {linear_stats['rms_residual']:.3f} mag")
    
    # Създаваме ΛCDM модел за сравнение
    print("\nСъздаване на ΛCDM модел за сравнение...")
    lcdm_model = create_lcdm_comparison_model(H0=optimal_H0, Om0=0.3, OL0=0.7)
    
    # Изчисляваме статистики за ΛCDM модела
    try:
        # Ако е astropy обект, използваме distmod
        mu_lcdm = lcdm_model.distmod(z_data).value
    except AttributeError:
        # Ако е нашия клас, използваме distance_modulus
        mu_lcdm = lcdm_model.distance_modulus(z_data)
    residuals_lcdm = mu_data - mu_lcdm
    chi2_lcdm = np.sum((residuals_lcdm / mu_err)**2)
    reduced_chi2_lcdm = chi2_lcdm / (len(z_data) - 2)  # 2 параметъра за ΛCDM
    rms_lcdm = np.sqrt(np.mean(residuals_lcdm**2))
    
    lcdm_stats = {
        'chi2_min': chi2_lcdm,
        'reduced_chi2': reduced_chi2_lcdm,
        'rms_residual': rms_lcdm,
        'n_data': len(z_data),
        'degrees_of_freedom': len(z_data) - 2
    }
    
    try:
        # Ако е astropy обект, използваме age
        lcdm_age = lcdm_model.age(0).value / 1e9  # в млрд години
    except:
        # Ако е нашия клас или възникне грешка
        try:
            lcdm_age = lcdm_model.age(0) / 1e9  # в млрд години
        except:
            lcdm_age = 13.8  # стандартна стойност
    
    print(f"Възраст на Вселената (ΛCDM модел): {lcdm_age:.2f} млрд години")
    print(f"Χ²_red = {lcdm_stats['reduced_chi2']:.3f}")
    print(f"RMS остатък = {lcdm_stats['rms_residual']:.3f} mag")
    
    # Сравнение
    print(f"\nСравнение:")
    print(f"ΔΧ²_red = {linear_stats['reduced_chi2'] - lcdm_stats['reduced_chi2']:.3f}")
    if linear_stats['reduced_chi2'] < lcdm_stats['reduced_chi2']:
        print("Линейният модел има по-добро съответствие с данните!")
    else:
        print("ΛCDM модела има по-добро съответствие с данните.")
    
    return linear_model, lcdm_model, linear_stats, lcdm_stats


def plot_comparison(z_data, mu_data, mu_err, linear_model, lcdm_model, 
                   linear_stats, lcdm_stats):
    """
    Създава графики за сравнение на моделите
    
    Args:
        z_data: Червено отместване на данните
        mu_data: Модул на разстояние на данните
        mu_err: Грешки в модула на разстояние
        linear_model: Линеен космологичен модел
        lcdm_model: ΛCDM модел
        linear_stats: Статистики за линейния модел
        lcdm_stats: Статистики за ΛCDM модела
    """
    
    # Създаваме теоретични криви
    z_theory = np.logspace(-3, np.log10(2.0), 1000)
    mu_linear = linear_model.distance_modulus_at_z(z_theory)
    try:
        # Ако е astropy обект, използваме distmod
        mu_lcdm = lcdm_model.distmod(z_theory).value
    except AttributeError:
        # Ако е нашия клас, използваме distance_modulus
        mu_lcdm = lcdm_model.distance_modulus(z_theory)
    
    # Изчисляваме остатъци
    mu_linear_data = linear_model.distance_modulus_at_z(z_data)
    try:
        # Ако е astropy обект, използваме distmod
        mu_lcdm_data = lcdm_model.distmod(z_data).value
    except AttributeError:
        # Ако е нашия клас, използваме distance_modulus
        mu_lcdm_data = lcdm_model.distance_modulus(z_data)
    residuals_linear = mu_data - mu_linear_data
    residuals_lcdm = mu_data - mu_lcdm_data
    
    # Настройваме шрифтовете на български
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Създаваме фигурата
    fig = plt.figure(figsize=(15, 12))
    
    # Горна графика: Диаграма на Хъбъл
    ax1 = plt.subplot(3, 2, (1, 2))
    
    # Данни с грешки
    ax1.errorbar(z_data, mu_data, yerr=mu_err, fmt='o', color='lightgray', 
                alpha=0.6, markersize=2, elinewidth=0.5, capsize=0, 
                label=f'Pantheon+ данни (N={len(z_data)})')
    
    # Теоретични криви
    ax1.plot(z_theory, mu_linear, 'r-', linewidth=2, 
             label=f'Линеен модел (H₀={linear_model.H0_kmsmpc:.1f})')
    try:
        # Ако е astropy обект, използваме distmod
        mu_lcdm = lcdm_model.distmod(z_theory).value
    except AttributeError:
        # Ако е нашия клас, използваме distance_modulus
        mu_lcdm = lcdm_model.distance_modulus(z_theory)
    ax1.plot(z_theory, mu_lcdm, 'b-', linewidth=2, 
             label='ΛCDM модел')
    
    ax1.set_xlabel('Червено отместване z')
    ax1.set_ylabel('Модул на разстояние μ [mag]')
    ax1.set_title('Диаграма на Хъбъл: Сравнение на космологични модели')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 2.0)
    
    # Втора графика: Остатъци за линейния модел
    ax2 = plt.subplot(3, 2, 3)
    ax2.errorbar(z_data, residuals_linear, yerr=mu_err, fmt='ro', alpha=0.6, 
                markersize=3, elinewidth=0.5, capsize=0)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Червено отместване z')
    ax2.set_ylabel('Остатък [mag]')
    ax2.set_title(f'Остатъци: Линеен модел (RMS={linear_stats["rms_residual"]:.3f})')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2.0)
    
    # Трета графика: Остатъци за ΛCDM модела
    ax3 = plt.subplot(3, 2, 4)
    ax3.errorbar(z_data, residuals_lcdm, yerr=mu_err, fmt='bo', alpha=0.6, 
                markersize=3, elinewidth=0.5, capsize=0)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Червено отместване z')
    ax3.set_ylabel('Остатък [mag]')
    ax3.set_title(f'Остатъци: ΛCDM модел (RMS={lcdm_stats["rms_residual"]:.3f})')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 2.0)
    
    # Четвърта графика: Разлика между моделите
    ax4 = plt.subplot(3, 2, 5)
    diff = mu_linear - mu_lcdm
    ax4.plot(z_theory, diff, 'g-', linewidth=2)
    ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Червено отместване z')
    ax4.set_ylabel('Разлика μ_linear - μ_ΛCDM [mag]')
    ax4.set_title('Разлика между моделите')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 2.0)
    
    # Пета графика: Хистограма на остатъците
    ax5 = plt.subplot(3, 2, 6)
    ax5.hist(residuals_linear, bins=50, alpha=0.7, color='red', 
             label=f'Линеен (σ={np.std(residuals_linear):.3f})', density=True)
    ax5.hist(residuals_lcdm, bins=50, alpha=0.7, color='blue', 
             label=f'ΛCDM (σ={np.std(residuals_lcdm):.3f})', density=True)
    ax5.set_xlabel('Остатък [mag]')
    ax5.set_ylabel('Нормализирана честота')
    ax5.set_title('Разпределение на остатъците')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hubble_diagram_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_detailed_analysis(z_data, mu_data, mu_err, linear_model, lcdm_model):
    """
    Създава детайлен анализ на резултатите
    
    Args:
        z_data: Червено отместване на данните
        mu_data: Модул на разстояние на данните
        mu_err: Грешки в модула на разстояние
        linear_model: Линеен космологичен модел
        lcdm_model: ΛCDM модел
    """
    print("\n" + "="*80)
    print("ДЕТАЙЛЕН АНАЛИЗ НА РЕЗУЛТАТИТЕ")
    print("="*80)
    
    # Основни параметри
    print(f"\n🔬 ОСНОВНИ ПАРАМЕТРИ:")
    print(f"{'Параметър':<25} {'Линеен модел':<20} {'ΛCDM модел':<20}")
    print("-" * 65)
    print(f"{'H₀ [km/s/Mpc]':<25} {linear_model.H0_kmsmpc:<20.2f} {70.0:<20.2f}")
    
    try:
        # Ако е astropy обект, използваме age
        lcdm_age = lcdm_model.age(0).value / 1e9
    except:
        # Ако е нашия клас или възникне грешка
        try:
            lcdm_age = lcdm_model.age(0) / 1e9
        except:
            lcdm_age = 13.8
    
    print(f"{'Възраст [млрд г.]':<25} {linear_model.t0_years/1e9:<20.2f} {lcdm_age:<20.2f}")
    
    # Предсказания за ключови червени отмествания
    z_test = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    print(f"\n📊 ПРЕДСКАЗАНИЯ ЗА МОДУЛА НА РАЗСТОЯНИЕ:")
    print(f"{'z':<8} {'Линеен модел':<15} {'ΛCDM модел':<15} {'Разлика':<15}")
    print("-" * 53)
    
    for z in z_test:
        mu_lin = linear_model.distance_modulus_at_z(z)
        try:
            # Ако е astropy обект, използваме distmod
            mu_lcdm = lcdm_model.distmod(z).value
        except AttributeError:
            # Ако е нашия клас, използваме distance_modulus
            mu_lcdm = lcdm_model.distance_modulus(z)
        diff = mu_lin - mu_lcdm
        print(f"{z:<8.1f} {mu_lin:<15.3f} {mu_lcdm:<15.3f} {diff:<15.3f}")
    
    # Времеви анализ
    print(f"\n⏰ ВРЕМЕВИ АНАЛИЗ:")
    print(f"{'z':<8} {'Възраст (лин.) [млрд г.]':<25} {'Lookback (лин.) [млрд г.]':<25}")
    print("-" * 58)
    
    for z in z_test:
        age_lin = linear_model.age_at_z(z) / 1e9
        lookback_lin = linear_model.lookback_time(z) / 1e9
        print(f"{z:<8.1f} {age_lin:<25.3f} {lookback_lin:<25.3f}")
    
    # Статистически анализ на остатъците
    mu_linear_data = linear_model.distance_modulus_at_z(z_data)
    try:
        # Ако е astropy обект, използваме distmod
        mu_lcdm_data = lcdm_model.distmod(z_data).value
    except AttributeError:
        # Ако е нашия клас, използваме distance_modulus
        mu_lcdm_data = lcdm_model.distance_modulus(z_data)
    residuals_linear = mu_data - mu_linear_data
    residuals_lcdm = mu_data - mu_lcdm_data
    
    print(f"\n📈 СТАТИСТИЧЕСКИ АНАЛИЗ НА ОСТАТЪЦИТЕ:")
    print(f"{'Статистика':<20} {'Линеен модел':<15} {'ΛCDM модел':<15}")
    print("-" * 50)
    print(f"{'Средна стойност':<20} {np.mean(residuals_linear):<15.4f} {np.mean(residuals_lcdm):<15.4f}")
    print(f"{'Стандартно отклонение':<20} {np.std(residuals_linear):<15.4f} {np.std(residuals_lcdm):<15.4f}")
    print(f"{'Медиана':<20} {np.median(residuals_linear):<15.4f} {np.median(residuals_lcdm):<15.4f}")
    print(f"{'Максимален остатък':<20} {np.max(np.abs(residuals_linear)):<15.4f} {np.max(np.abs(residuals_lcdm)):<15.4f}")
    
    # Анализ по интервали на z
    print(f"\n🎯 АНАЛИЗ ПО ИНТЕРВАЛИ НА ЧЕРВЕНОТО ОТМЕСТВАНЕ:")
    z_bins = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    
    print(f"{'z интервал':<15} {'N точки':<10} {'RMS (лин.)':<12} {'RMS (ΛCDM)':<12}")
    print("-" * 49)
    
    for z_min, z_max in z_bins:
        mask = (z_data >= z_min) & (z_data < z_max)
        if np.sum(mask) > 0:
            rms_lin = np.sqrt(np.mean(residuals_linear[mask]**2))
            rms_lcdm = np.sqrt(np.mean(residuals_lcdm[mask]**2))
            print(f"{f'[{z_min:.1f}, {z_max:.1f})':<15} {np.sum(mask):<10} {rms_lin:<12.4f} {rms_lcdm:<12.4f}")


def main():
    """Главна функция"""
    print("="*80)
    print("ТЕСТВАНЕ НА ЛИНЕЙНИЯ КОСМОЛОГИЧЕН МОДЕЛ С РЕАЛНИ ДАННИ")
    print("="*80)
    print("Данни: Pantheon+ суперновите")
    print("Модел: a(t) = k*t (линейно разширение)")
    print("Сравнение с: ΛCDM модел")
    print("="*80)
    
    # Определяме пътя към данните
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / "test_2" / "data" / "Pantheon+_Data" / "4_DISTANCES_AND_COVAR" / "Pantheon+SH0ES.dat"
    
    if not data_path.exists():
        print(f"❌ Файлът с данни не е намерен: {data_path}")
        print("Моля, проверете пътя към данните.")
        return
    
    print(f"📁 Зареждане на данни от: {data_path}")
    
    # Зареждаме данните
    z_data, mu_data, mu_err = load_pantheon_data(str(data_path))
    
    if z_data is None:
        print("❌ Неуспешно зареждане на данните.")
        return
    
    # Fitваме моделите
    linear_model, lcdm_model, linear_stats, lcdm_stats = fit_models_to_data(
        z_data, mu_data, mu_err)
    
    # Създаваме графики
    print("\n📊 Създаване на графики...")
    plot_comparison(z_data, mu_data, mu_err, linear_model, lcdm_model, 
                   linear_stats, lcdm_stats)
    
    # Детайлен анализ
    create_detailed_analysis(z_data, mu_data, mu_err, linear_model, lcdm_model)
    
    print(f"\n✅ Анализът е завършен успешно!")
    print(f"📊 Графиката е запазена като 'hubble_diagram_comparison.png'")


if __name__ == "__main__":
    main() 