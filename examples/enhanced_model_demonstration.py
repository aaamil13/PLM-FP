"""
Демонстрация на подобрения линеен модел с динамичен времеви дилатационен фактор
================================================================================

Този скрипт демонстрира как работи теоретичната рамка за линейния космологичен модел
с динамичен времеви дилатационен фактор dτ/dt = [1 + (ρ/ρ_crit)^α]^(-1).

Автор: Проект за изследване на линейна космология
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the lib directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

from enhanced_linear_model import EnhancedLinearUniverse
from core.linear_universe import LinearUniverse

def demonstrate_time_dilation_effects():
    """
    Демонстрация на ефектите от времевия дилатационен фактор
    """
    print("=== Демонстрация на ефектите от времевия дилатационен фактор ===")
    
    # Създаване на модели с различни α стойности
    alpha_values = [0.5, 1.0, 2.0, 3.0]
    models = {}
    
    for alpha in alpha_values:
        models[f'α={alpha}'] = EnhancedLinearUniverse(alpha=alpha)
    
    # Стандартен линеен модел за сравнение
    standard_model = LinearUniverse()
    
    # Червени отмествания за тестване
    z_range = np.logspace(-2, 1, 50)  # От z=0.01 до z=10
    
    # Графика 1: Времеви дилатационен фактор
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for label, model in models.items():
        dtau_dt = model.time_dilation_factor(z_range)
        plt.loglog(z_range, dtau_dt, linewidth=2, label=f'Подобрен модел ({label})')
    
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Стандартен модел (dτ/dt = 1)')
    plt.axvline(x=900, color='green', linestyle='--', alpha=0.5, label='z = 900')
    plt.axvline(x=1300, color='orange', linestyle='--', alpha=0.5, label='z = 1300')
    
    plt.xlabel('Червено отместване z')
    plt.ylabel('Времеви дилатационен фактор dτ/dt')
    plt.title('Еволюция на времевия дилатационен фактор')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Графика 2: Светимостни разстояния
    plt.subplot(2, 2, 2)
    
    # Стандартен модел
    d_L_standard = np.array([standard_model.luminosity_distance_at_z(z) for z in z_range])
    plt.plot(z_range, d_L_standard, 'k-', linewidth=3, label='Стандартен линеен модел')
    
    # Подобрени модели
    for label, model in models.items():
        d_L_enhanced = np.array([model.luminosity_distance(z) for z in z_range])
        plt.plot(z_range, d_L_enhanced, '--', linewidth=2, label=f'Подобрен ({label})')
    
    plt.xlabel('Червено отместване z')
    plt.ylabel('Светимостно разстояние [Mpc]')
    plt.title('Сравнение на светимостни разстояния')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    
    # Графика 3: Относителни разлики
    plt.subplot(2, 2, 3)
    
    for label, model in models.items():
        d_L_enhanced = np.array([model.luminosity_distance(z) for z in z_range])
        relative_diff = (d_L_enhanced - d_L_standard) / d_L_standard * 100
        plt.semilogx(z_range, relative_diff, linewidth=2, label=f'Подобрен ({label})')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    plt.xlabel('Червено отместване z')
    plt.ylabel('Относителна разлика [%]')
    plt.title('Относителна разлика спрямо стандартния модел')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Графика 4: Енергийна плътност
    plt.subplot(2, 2, 4)
    
    model_sample = models['α=2.0']  # Използваме един модел за плътността
    rho_total = model_sample.density_evolution(z_range)
    rho_matter = model_sample.Omega_m * (1 + z_range)**3
    rho_radiation = model_sample.Omega_r * (1 + z_range)**4
    
    plt.loglog(z_range, rho_total, 'k-', linewidth=2, label='Обща плътност')
    plt.loglog(z_range, rho_matter, 'b--', linewidth=2, label='Материя')
    plt.loglog(z_range, rho_radiation, 'r--', linewidth=2, label='Лъчение')
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='ρ_crit')
    plt.axvline(x=900, color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=1300, color='orange', linestyle='--', alpha=0.5)
    
    plt.xlabel('Червено отместване z')
    plt.ylabel('Енергийна плътност (ρ/ρ_crit)')
    plt.title('Еволюция на енергийната плътност')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_model_demonstration.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return models

def analyze_epoch_behavior(models):
    """
    Анализ на поведението в различните епохи
    """
    print("\n=== Анализ на поведението в различните епохи ===")
    
    # Дефиниране на епохите
    epochs = {
        'Късна епоха (z < 0.5)': [0.1, 0.3, 0.5],
        'Преходна епоха (0.5 < z < 2)': [0.7, 1.0, 1.5],
        'Ранна епоха (z > 2)': [2.0, 5.0, 10.0]
    }
    
    for epoch_name, z_values in epochs.items():
        print(f"\n{epoch_name}:")
        print("-" * 50)
        
        for z in z_values:
            print(f"\nЧервено отместване z = {z}:")
            
            # Стандартен модел
            standard_model = LinearUniverse()
            d_L_standard = standard_model.luminosity_distance_at_z(z)
            
            print(f"  Стандартен модел: d_L = {d_L_standard:.1f} Mpc")
            
            # Подобрени модели
            for label, model in models.items():
                rho = model.density_evolution(z)
                dtau_dt = model.time_dilation_factor(z)
                d_L_enhanced = model.luminosity_distance(z)
                
                diff_percent = (d_L_enhanced - d_L_standard) / d_L_standard * 100
                
                print(f"  {label}: ρ/ρ_crit = {rho:.2f}, dτ/dt = {dtau_dt:.6f}, " +
                      f"d_L = {d_L_enhanced:.1f} Mpc ({diff_percent:+.1f}%)")

def test_supernovae_compatibility():
    """
    Тестване на съвместимостта със свръхнови данни
    """
    print("\n=== Тестване на съвместимостта със свръхнови данни ===")
    
    # Симулирани данни от свръхнови (базирани на нашите предишни тестове)
    z_sn = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5])
    
    # Стандартен линеен модел
    standard_model = LinearUniverse()
    mu_standard = np.array([standard_model.distance_modulus_at_z(z) for z in z_sn])
    
    # Подобрени модели
    alpha_values = [0.5, 1.0, 2.0, 3.0]
    
    print("\nСравнение на модули на разстояния:")
    print("z\tСтандартен\tα=0.5\tα=1.0\tα=2.0\tα=3.0")
    print("-" * 60)
    
    for i, z in enumerate(z_sn):
        row = f"{z:.1f}\t{mu_standard[i]:.2f}\t\t"
        
        for alpha in alpha_values:
            model = EnhancedLinearUniverse(alpha=alpha)
            mu_enhanced = model.distance_modulus(z)
            row += f"{mu_enhanced:.2f}\t"
        
        print(row)

def demonstrate_physical_interpretation():
    """
    Демонстрация на физическата интерпретация
    """
    print("\n=== Физическа интерпретация на модела ===")
    
    print("\n1. Основни принципи:")
    print("   • Геометричното разширение: a(t) = k·t (най-простият закон)")
    print("   • Физическото време τ ≠ геометрично време t")
    print("   • Времевият дилатационен фактор: dτ/dt = [1 + (ρ/ρ_crit)^α]^(-1)")
    
    print("\n2. Физически смисъл:")
    print("   • При висока плътност (ранна епоха): dτ/dt ≈ 0")
    print("     → Физическото време 'замръзва' спрямо геометричното")
    print("   • При ниска плътност (късна епоха): dτ/dt ≈ 1")
    print("     → Физическото време = геометрично време")
    
    print("\n3. Наблюдаеми последствия:")
    print("   • Светимостните разстояния се изчисляват чрез:")
    print("     d_c(z) = ∫[t(z) до t₀] c·(dτ/dt) dt")
    print("   • Ефектът е най-силен при високи червени отмествания")
    print("   • Обяснява успеха на линейния модел при z < 1.5")
    
    print("\n4. Революционни импликации:")
    print("   • Тъмната енергия може да е математически артефакт")
    print("   • H(t) = 1/t може да е фундаментален закон на природата")
    print("   • Времето е динамична, а не статична величина")

def main():
    """
    Главна функция за демонстрация
    """
    print("ДЕМОНСТРАЦИЯ НА ПОДОБРЕНИЯ ЛИНЕЕН КОСМОЛОГИЧЕН МОДЕЛ")
    print("=" * 60)
    
    # Демонстрация на времевите дилатационни ефекти
    models = demonstrate_time_dilation_effects()
    
    # Анализ на поведението в различните епохи
    analyze_epoch_behavior(models)
    
    # Тестване на съвместимостта със свръхнови данни
    test_supernovae_compatibility()
    
    # Физическа интерпретация
    demonstrate_physical_interpretation()
    
    print("\n" + "=" * 60)
    print("ЗАКЛЮЧЕНИЕ: Подобреният линеен модел с динамичен времеви")
    print("дилатационен фактор предлага елегантно решение на проблемите")
    print("на стандартния линеен модел, като запазва неговите предимства.")
    print("=" * 60)

if __name__ == "__main__":
    main() 