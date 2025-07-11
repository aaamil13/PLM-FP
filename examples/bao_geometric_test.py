#!/usr/bin/env python3
"""
Геометричен тест на Линейния модел с BAO данни

Този скрипт сравнява предсказанието на линейния модел за AP параметъра
F_AP(z) = H(z) * d_A(z) / c с публикувани BAO данни и с предсказанието на ΛCDM.

Автор: Изследване на линейния космологичен модел
Дата: 2024
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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

class BAOGeometricTest:
    """
    Клас за геометричен тест с BAO данни
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
        
        print(f"BAO геометричен тест инициализиран с:")
        print(f"  H₀ = {H0} km/s/Mpc")
        print(f"  Ω_m = {omega_m}")
        print(f"  Ω_Λ = {omega_l}")
        print()
    
    def load_bao_data(self, filename='../data/bao_data.txt'):
        """
        Зарежда BAO данните от файл
        
        Args:
            filename: път към файла с данни
            
        Returns:
            Dict с BAO данни
        """
        try:
            # Четем данните, пропускайки коментарите
            data = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 6:
                            z_eff = float(parts[0])
                            H_obs = float(parts[1])
                            H_err = float(parts[2])
                            dA_obs = float(parts[3])
                            dA_err = float(parts[4])
                            survey = parts[5]
                            
                            data.append({
                                'z_eff': z_eff,
                                'H_obs': H_obs,
                                'H_err': H_err,
                                'dA_obs': dA_obs,
                                'dA_err': dA_err,
                                'survey': survey
                            })
            
            print(f"Заредени {len(data)} BAO измервания:")
            for d in data:
                print(f"  z = {d['z_eff']:.2f}: H = {d['H_obs']:.0f}±{d['H_err']:.0f}, "
                      f"d_A = {d['dA_obs']:.0f}±{d['dA_err']:.0f} ({d['survey']})")
            print()
            
            return data
            
        except FileNotFoundError:
            print(f"Файлът {filename} не е намерен!")
            return self._create_default_data()
    
    def _create_default_data(self):
        """
        Създава данни по подразбиране ако файлът не е намерен
        """
        print("Използвам вградени BAO данни:")
        data = [
            {'z_eff': 0.15, 'H_obs': 456, 'H_err': 27, 'dA_obs': 664, 'dA_err': 25, 'survey': 'SDSS_MGS'},
            {'z_eff': 0.32, 'H_obs': 787, 'H_err': 20, 'dA_obs': 950, 'dA_err': 23, 'survey': 'BOSS_LOWZ'},
            {'z_eff': 0.57, 'H_obs': 933, 'H_err': 15, 'dA_obs': 1396, 'dA_err': 22, 'survey': 'BOSS_CMASS'},
            {'z_eff': 0.70, 'H_obs': 978, 'H_err': 19, 'dA_obs': 1530, 'dA_err': 35, 'survey': 'eBOSS_LRG'},
            {'z_eff': 1.48, 'H_obs': 1380, 'H_err': 90, 'dA_obs': 1850, 'dA_err': 80, 'survey': 'eBOSS_QSO'},
            {'z_eff': 2.33, 'H_obs': 2220, 'H_err': 70, 'dA_obs': 1662, 'dA_err': 90, 'survey': 'Lyman_alpha'}
        ]
        
        for d in data:
            print(f"  z = {d['z_eff']:.2f}: H = {d['H_obs']:.0f}±{d['H_err']:.0f}, "
                  f"d_A = {d['dA_obs']:.0f}±{d['dA_err']:.0f} ({d['survey']})")
        print()
        
        return data
    
    def H_linear(self, z):
        """Хъбъл параметър за линейния модел"""
        return self.H0 * (1 + z)
    
    def dA_linear(self, z):
        """Ъглово разстояние за линейния модел"""
        if np.any(z <= 0):
            return np.zeros_like(z)
        return (c_light / self.H0) * np.log(1 + z) / (1 + z)
    
    def H_lcdm(self, z):
        """Хъбъл параметър за ΛCDM модел"""
        return self.H0 * np.sqrt(self.omega_m * (1 + z)**3 + self.omega_l)
    
    def _integrand_lcdm(self, z_prime):
        """Подынтегрална функция за ΛCDM комовингово разстояние"""
        return 1.0 / np.sqrt(self.omega_m * (1 + z_prime)**3 + self.omega_l)
    
    def dc_lcdm(self, z):
        """Комовингово разстояние за ΛCDM модел"""
        if np.isscalar(z):
            if z <= 0:
                return 0.0
            integral, _ = quad(self._integrand_lcdm, 0, z)
            return (c_light / self.H0) * integral
        else:
            result = np.zeros_like(z)
            for i, z_val in enumerate(z):
                if z_val > 0:
                    integral, _ = quad(self._integrand_lcdm, 0, z_val)
                    result[i] = (c_light / self.H0) * integral
            return result
    
    def dA_lcdm(self, z):
        """Ъглово разстояние за ΛCDM модел"""
        return self.dc_lcdm(z) / (1 + z)
    
    def calculate_F_AP(self, z_array, model='linear'):
        """
        Изчислява F_AP параметъра за даден модел
        
        Args:
            z_array: масив с червени отмествания
            model: 'linear' или 'lcdm'
            
        Returns:
            F_AP масив
        """
        if model == 'linear':
            H_vals = self.H_linear(z_array)
            dA_vals = self.dA_linear(z_array)
        elif model == 'lcdm':
            H_vals = self.H_lcdm(z_array)
            dA_vals = self.dA_lcdm(z_array)
        else:
            raise ValueError("model трябва да е 'linear' или 'lcdm'")
        
        # F_AP = H * d_A / c
        F_AP = (H_vals * dA_vals) / c_light
        return F_AP
    
    def calculate_observed_F_AP(self, bao_data):
        """
        Изчислява наблюдаваните F_AP стойности от BAO данните
        
        Args:
            bao_data: списък с BAO данни
            
        Returns:
            z_obs, F_AP_obs, F_AP_err масиви
        """
        z_obs = np.array([d['z_eff'] for d in bao_data])
        H_obs = np.array([d['H_obs'] for d in bao_data])
        H_err = np.array([d['H_err'] for d in bao_data])
        dA_obs = np.array([d['dA_obs'] for d in bao_data])
        dA_err = np.array([d['dA_err'] for d in bao_data])
        
        # F_AP_obs = H_obs * dA_obs / c
        F_AP_obs = (H_obs * dA_obs) / c_light
        
        # Разпространение на грешките
        # F_AP_err = F_AP_obs * sqrt((H_err/H_obs)² + (dA_err/dA_obs)²)
        rel_H_err = H_err / H_obs
        rel_dA_err = dA_err / dA_obs
        F_AP_err = F_AP_obs * np.sqrt(rel_H_err**2 + rel_dA_err**2)
        
        return z_obs, F_AP_obs, F_AP_err
    
    def run_test(self, z_max=3.0, n_points=200):
        """
        Изпълнява BAO геометричния тест
        
        Args:
            z_max: максимално червено отместване за теоретичните криви
            n_points: брой точки за теоретичните криви
            
        Returns:
            Dict с резултатите
        """
        print("Започвам BAO геометричен тест...")
        print(f"Теоретични криви: z = 0.01 до {z_max}, {n_points} точки")
        print()
        
        # Зареждаме BAO данните
        bao_data = self.load_bao_data()
        
        # Изчисляваме наблюдаваните F_AP стойности
        z_obs, F_AP_obs, F_AP_err = self.calculate_observed_F_AP(bao_data)
        
        # Създаваме теоретичните криви
        z_theory = np.linspace(0.01, z_max, n_points)
        F_AP_linear = self.calculate_F_AP(z_theory, model='linear')
        F_AP_lcdm = self.calculate_F_AP(z_theory, model='lcdm')
        
        # Анализираме съответствието
        print("Сравнение на теоретичните предсказания с наблюденията:")
        print("=" * 80)
        
        total_chi2_linear = 0
        total_chi2_lcdm = 0
        
        for i, (z, F_obs, F_err) in enumerate(zip(z_obs, F_AP_obs, F_AP_err)):
            # Интерполираме теоретичните стойности
            F_lin_interp = np.interp(z, z_theory, F_AP_linear)
            F_lcdm_interp = np.interp(z, z_theory, F_AP_lcdm)
            
            # Изчисляваме chi-square
            chi2_linear = ((F_obs - F_lin_interp) / F_err)**2
            chi2_lcdm = ((F_obs - F_lcdm_interp) / F_err)**2
            
            total_chi2_linear += chi2_linear
            total_chi2_lcdm += chi2_lcdm
            
            print(f"z = {z:.2f} ({bao_data[i]['survey']}):")
            print(f"  F_AP_obs = {F_obs:.3f} ± {F_err:.3f}")
            print(f"  F_AP_linear = {F_lin_interp:.3f} (χ² = {chi2_linear:.2f})")
            print(f"  F_AP_ΛCDM = {F_lcdm_interp:.3f} (χ² = {chi2_lcdm:.2f})")
            print()
        
        # Обобщени резултати
        print("Обобщени резултати:")
        print(f"  Общ χ² (линеен модел): {total_chi2_linear:.2f}")
        print(f"  Общ χ² (ΛCDM модел): {total_chi2_lcdm:.2f}")
        print(f"  Брой точки: {len(z_obs)}")
        print(f"  Χ²_red (линеен): {total_chi2_linear/len(z_obs):.3f}")
        print(f"  Χ²_red (ΛCDM): {total_chi2_lcdm/len(z_obs):.3f}")
        print()
        
        if total_chi2_linear < total_chi2_lcdm:
            print("🎯 Линейният модел показва по-добро съответствие с BAO данните!")
        elif total_chi2_lcdm < total_chi2_linear:
            print("🎯 ΛCDM моделът показва по-добро съответствие с BAO данните!")
        else:
            print("🎯 И двата модела показват сходно съответствие с BAO данните!")
        
        results = {
            'z_obs': z_obs,
            'F_AP_obs': F_AP_obs,
            'F_AP_err': F_AP_err,
            'z_theory': z_theory,
            'F_AP_linear': F_AP_linear,
            'F_AP_lcdm': F_AP_lcdm,
            'chi2_linear': total_chi2_linear,
            'chi2_lcdm': total_chi2_lcdm,
            'bao_data': bao_data
        }
        
        return results
    
    def plot_results(self, results, save_plot=True):
        """
        Създава графика на резултатите
        
        Args:
            results: резултати от run_test()
            save_plot: дали да запази графиката
        """
        # Настройваме шрифт за кирилица
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.figure(figsize=(14, 10))
        
        # Основна графика
        plt.subplot(2, 1, 1)
        
        # Данни с грешки
        plt.errorbar(results['z_obs'], results['F_AP_obs'], yerr=results['F_AP_err'], 
                    fmt='o', markersize=8, capsize=5, capthick=2, 
                    label='BAO данни', color='red', zorder=5)
        
        # Теоретични криви
        plt.plot(results['z_theory'], results['F_AP_linear'], 
                linewidth=2, label='Линеен модел', color='blue')
        plt.plot(results['z_theory'], results['F_AP_lcdm'], 
                linewidth=2, label='ΛCDM модел', color='green')
        
        plt.xlabel('Червено отместване z', fontsize=12)
        plt.ylabel('F_AP(z) = H(z) × d_A(z) / c', fontsize=12)
        plt.title('BAO геометричен тест: F_AP параметър', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Отклонения от наблюденията
        plt.subplot(2, 1, 2)
        
        # Интерполираме теоретичните стойности за наблюдаваните z
        F_linear_interp = np.interp(results['z_obs'], results['z_theory'], results['F_AP_linear'])
        F_lcdm_interp = np.interp(results['z_obs'], results['z_theory'], results['F_AP_lcdm'])
        
        # Изчисляваме относителните отклонения
        residuals_linear = (results['F_AP_obs'] - F_linear_interp) / results['F_AP_err']
        residuals_lcdm = (results['F_AP_obs'] - F_lcdm_interp) / results['F_AP_err']
        
        # Построяваме остатъците
        plt.errorbar(results['z_obs'], residuals_linear, yerr=np.ones_like(results['z_obs']), 
                    fmt='o', markersize=6, capsize=3, label='Линеен модел', color='blue')
        plt.errorbar(results['z_obs'], residuals_lcdm, yerr=np.ones_like(results['z_obs']), 
                    fmt='s', markersize=6, capsize=3, label='ΛCDM модел', color='green')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='1σ')
        plt.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='2σ')
        plt.axhline(y=-2, color='orange', linestyle='--', alpha=0.5)
        
        plt.xlabel('Червено отместване z', fontsize=12)
        plt.ylabel('Остатъци (σ)', fontsize=12)
        plt.title('Отклонения от BAO данните', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_plot:
            filename = 'bao_geometric_test.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Графиката е запазена като {filename}")
        
        plt.close()
    
    def save_results(self, results, filename='bao_test_results.csv'):
        """
        Запазва резултатите в CSV файл
        
        Args:
            results: резултати от run_test()
            filename: име на файла за запис
        """
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['z_obs', 'F_AP_obs', 'F_AP_err', 'F_AP_linear', 'F_AP_lcdm', 
                           'chi2_linear', 'chi2_lcdm', 'survey'])
            
            # Интерполираме теоретичните стойности
            F_linear_interp = np.interp(results['z_obs'], results['z_theory'], results['F_AP_linear'])
            F_lcdm_interp = np.interp(results['z_obs'], results['z_theory'], results['F_AP_lcdm'])
            
            for i, (z, F_obs, F_err) in enumerate(zip(results['z_obs'], results['F_AP_obs'], results['F_AP_err'])):
                F_lin = F_linear_interp[i]
                F_lcdm = F_lcdm_interp[i]
                chi2_lin = ((F_obs - F_lin) / F_err)**2
                chi2_lcdm = ((F_obs - F_lcdm) / F_err)**2
                survey = results['bao_data'][i]['survey']
                
                writer.writerow([z, F_obs, F_err, F_lin, F_lcdm, chi2_lin, chi2_lcdm, survey])
        
        print(f"Резултатите са запазени в {filename}")


def main():
    """
    Главна функция за изпълнение на BAO теста
    """
    print("=" * 70)
    print("BAO ГЕОМЕТРИЧЕН ТЕСТ")
    print("Сравнение на F_AP параметъра между Линеен модел и ΛCDM")
    print("=" * 70)
    print()
    
    # Създаваме тест обект
    bao_test = BAOGeometricTest()
    
    # Изпълняваме теста
    results = bao_test.run_test(z_max=3.0, n_points=200)
    
    # Създаваме графика
    bao_test.plot_results(results)
    
    # Запазваме резултатите
    bao_test.save_results(results)
    
    print("\nBAO геометричният тест е завършен!")
    print("=" * 70)


if __name__ == "__main__":
    main() 