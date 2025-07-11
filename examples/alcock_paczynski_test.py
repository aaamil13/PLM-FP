#!/usr/bin/env python3
"""
Тест на Alcock-Paczynski за сравнение на геометрията между модели

Този тест измерва как различните космологични модели "виждат" геометрията
на космоса чрез съотношението y(z) = dL_паралелно / dL_перпендикулярно.

Ако един модел е верен, той трябва да предскаже y(z) = 1 за сферични обекти.
Ако предскаже y(z) ≠ 1, значи той "вижда" сферичните обекти като изкривени.

Автор: Изследване на линейния космологичен модел
Дата: 2024
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Използваме non-interactive backend
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os
import sys

# Добавяме пътя към нашите библиотеки
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.core.linear_universe import LinearUniverse

# Физични константи
c_light = 299792.458  # km/s - скорост на светлината

# Стандартни космологични параметри
H0_STD = 70.0  # km/s/Mpc
OMEGA_M_STD = 0.3  # плътност на материята
OMEGA_L_STD = 0.7  # плътност на тъмната енергия

class AlcockPaczynskiTest:
    """
    Клас за изпълнение на тест на Alcock-Paczynski
    """
    
    def __init__(self, H0=H0_STD, omega_m=OMEGA_M_STD, omega_l=OMEGA_L_STD):
        """
        Инициализация на теста
        
        Args:
            H0: Хъбъл константа в km/s/Mpc
            omega_m: Параметър на плътността на материята
            omega_l: Параметър на плътността на тъмната енергия
        """
        self.H0 = H0
        self.omega_m = omega_m
        self.omega_l = omega_l
        
        # Създаваме линейния модел
        self.linear_model = LinearUniverse(H0_kmsmpc=H0)
        
        print(f"Тест на Alcock-Paczynski инициализиран с:")
        print(f"  H₀ = {H0} km/s/Mpc")
        print(f"  Ω_m = {omega_m}")
        print(f"  Ω_Λ = {omega_l}")
        print()
    
    def H_linear(self, z):
        """
        Хъбъл параметър за линейния модел
        H(z) = H₀ * (1+z)
        
        Args:
            z: червено отместване
            
        Returns:
            H(z) в km/s/Mpc
        """
        return self.H0 * (1 + z)
    
    def dA_linear(self, z):
        """
        Ъглово разстояние за линейния модел
        d_A(z) = (c/H₀) * ln(1+z) / (1+z)
        
        Args:
            z: червено отместване
            
        Returns:
            d_A(z) в Mpc
        """
        if np.any(z <= 0):
            return np.zeros_like(z)
        
        return (c_light / self.H0) * np.log(1 + z) / (1 + z)
    
    def H_lcdm(self, z):
        """
        Хъбъл параметър за ΛCDM модел
        H(z) = H₀ * √[Ω_m(1+z)³ + Ω_Λ]
        
        Args:
            z: червено отместване
            
        Returns:
            H(z) в km/s/Mpc
        """
        return self.H0 * np.sqrt(self.omega_m * (1 + z)**3 + self.omega_l)
    
    def _integrand_lcdm(self, z_prime):
        """
        Подынтегрална функция за изчисляване на комовингово разстояние в ΛCDM
        """
        return 1.0 / np.sqrt(self.omega_m * (1 + z_prime)**3 + self.omega_l)
    
    def dc_lcdm(self, z):
        """
        Комовингово разстояние за ΛCDM модел
        d_c(z) = (c/H₀) * ∫[0 до z] dz' / √[Ω_m(1+z')³ + Ω_Λ]
        
        Args:
            z: червено отместване
            
        Returns:
            d_c(z) в Mpc
        """
        if np.isscalar(z):
            if z <= 0:
                return 0.0
            integral, _ = quad(self._integrand_lcdm, 0, z)
            return (c_light / self.H0) * integral
        else:
            # Векторизирано изчисление
            result = np.zeros_like(z)
            for i, z_val in enumerate(z):
                if z_val > 0:
                    integral, _ = quad(self._integrand_lcdm, 0, z_val)
                    result[i] = (c_light / self.H0) * integral
            return result
    
    def dA_lcdm(self, z):
        """
        Ъглово разстояние за ΛCDM модел
        d_A(z) = d_c(z) / (1+z)
        
        Args:
            z: червено отместване
            
        Returns:
            d_A(z) в Mpc
        """
        return self.dc_lcdm(z) / (1 + z)
    
    def y_ap_ratio(self, z):
        """
        Съотношение на Alcock-Paczynski
        y_AP(z) = [H_ΛCDM(z) * d_A_ΛCDM(z)] / [H_Linear(z) * d_A_Linear(z)]
        
        Args:
            z: червено отместване
            
        Returns:
            y_AP(z) - безразмерно съотношение
        """
        # Изчисляваме произведенията H*d_A за двата модела
        lcdm_product = self.H_lcdm(z) * self.dA_lcdm(z)
        linear_product = self.H_linear(z) * self.dA_linear(z)
        
        # Избягваме деление на нула
        mask = linear_product != 0
        if np.isscalar(z):
            if not mask:
                return 0.0
            return lcdm_product / linear_product
        else:
            result = np.zeros_like(z)
            result[mask] = lcdm_product[mask] / linear_product[mask]
            return result
    
    def run_test(self, z_max=3.0, n_points=100):
        """
        Изпълнява теста на Alcock-Paczynski
        
        Args:
            z_max: максимално червено отместване
            n_points: брой точки за изчисление
            
        Returns:
            z_array, y_ap_array: масиви с резултатите
        """
        print("Започвам тест на Alcock-Paczynski...")
        print(f"Диапазон: z = 0.01 до {z_max}")
        print(f"Брой точки: {n_points}")
        print()
        
        # Създаваме масив от z стойности
        z_array = np.linspace(0.01, z_max, n_points)
        
        # Изчисляваме съотношението AP
        y_ap_array = self.y_ap_ratio(z_array)
        
        # Анализираме резултатите
        print("Анализ на резултатите:")
        print(f"  При z = 0.01: y_AP = {y_ap_array[0]:.4f}")
        print(f"  При z = 0.5:  y_AP = {y_ap_array[len(y_ap_array)//4]:.4f}")
        print(f"  При z = 1.0:  y_AP = {y_ap_array[len(y_ap_array)//3]:.4f}")
        print(f"  При z = 2.0:  y_AP = {y_ap_array[2*len(y_ap_array)//3]:.4f}")
        print(f"  При z = {z_max}: y_AP = {y_ap_array[-1]:.4f}")
        print()
        
        # Интерпретация
        if np.max(y_ap_array) > 1.1:
            print("Интерпретация: Линейният модел 'свива' обектите по лъча на зрението")
            print("(вижда ги като сплескани спрямо ΛCDM)")
        elif np.min(y_ap_array) < 0.9:
            print("Интерпретация: Линейният модел 'разтяга' обектите по лъча на зрението")
            print("(вижда ги като издължени спрямо ΛCDM)")
        else:
            print("Интерпретация: Линейният модел е близо до ΛCDM геометрията")
        
        print()
        return z_array, y_ap_array
    
    def plot_results(self, z_array, y_ap_array, save_plot=True):
        """
        Създава графика на резултатите
        
        Args:
            z_array: масив с червени отмествания
            y_ap_array: масив с AP съотношения
            save_plot: дали да запази графиката
        """
        # Настройваме шрифт за кирилица
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.figure(figsize=(12, 8))
        
        # Основна графика
        plt.subplot(2, 1, 1)
        plt.plot(z_array, y_ap_array, 'b-', linewidth=2, label='Линеен модел vs ΛCDM')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='y = 1 (перфектна съвместимост)')
        plt.xlabel('Червено отместване z', fontsize=12)
        plt.ylabel('AP съотношение y(z)', fontsize=12)
        plt.title('Тест на Alcock-Paczynski: Геометрично изкривяване', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Показваме отклонението от 1
        plt.subplot(2, 1, 2)
        deviation = y_ap_array - 1
        plt.plot(z_array, deviation * 100, 'g-', linewidth=2, label='Отклонение от y = 1')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Червено отместване z', fontsize=12)
        plt.ylabel('Отклонение (%)', fontsize=12)
        plt.title('Отклонение от перфектна геометрична съвместимост', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_plot:
            filename = 'alcock_paczynski_test.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Графиката е запазена като {filename}")
        
        plt.close()
    
    def analyze_components(self, z_array):
        """
        Анализира отделните компоненти на съотношението
        
        Args:
            z_array: масив с червени отмествания
        """
        print("Анализ на компонентите:")
        print("=" * 50)
        
        # Изчисляваме всички компоненти
        H_lin = self.H_linear(z_array)
        H_lcdm = self.H_lcdm(z_array)
        dA_lin = self.dA_linear(z_array)
        dA_lcdm = self.dA_lcdm(z_array)
        
        # Показваме няколко ключови стойности
        indices = [len(z_array)//10, len(z_array)//4, len(z_array)//2, 3*len(z_array)//4, -1]
        
        for i in indices:
            z = z_array[i]
            print(f"\nПри z = {z:.2f}:")
            print(f"  H_Linear = {H_lin[i]:.2f} km/s/Mpc")
            print(f"  H_ΛCDM   = {H_lcdm[i]:.2f} km/s/Mpc")
            print(f"  d_A_Linear = {dA_lin[i]:.2f} Mpc")
            print(f"  d_A_ΛCDM   = {dA_lcdm[i]:.2f} Mpc")
            print(f"  Произведение Linear: {H_lin[i] * dA_lin[i]:.2f}")
            print(f"  Произведение ΛCDM:   {H_lcdm[i] * dA_lcdm[i]:.2f}")
            print(f"  y_AP = {self.y_ap_ratio(z):.4f}")


def main():
    """
    Главна функция за изпълнение на теста
    """
    print("=" * 60)
    print("ТЕСТ НА ALCOCK-PACZYNSKI")
    print("Сравнение на геометрията между Линеен модел и ΛCDM")
    print("=" * 60)
    print()
    
    # Създаваме тест обект
    ap_test = AlcockPaczynskiTest()
    
    # Изпълняваме теста
    z_array, y_ap_array = ap_test.run_test(z_max=3.0, n_points=100)
    
    # Анализираме компонентите
    ap_test.analyze_components(z_array)
    
    # Създаваме графика
    ap_test.plot_results(z_array, y_ap_array)
    
    print("\nТестът на Alcock-Paczynski е завършен!")
    print("=" * 60)


if __name__ == "__main__":
    main() 