#!/usr/bin/env python3
"""
Клас PowerLawUniverse за степенни космологични модели

Този модул съдържа имплементация на степенния космологичен модел
a(t) = C*t^n, където n е параметърът на забавяне.
"""

import numpy as np
import warnings
from typing import Union, Tuple, Dict, Optional, List
import scipy.integrate as integrate
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt

# Космологични константи
c = 299792.458  # km/s (скорост на светлината)
H0_to_inv_s = 1.0 / (3.0857e19)  # преобразуване от (km/s)/Mpc в 1/s
Mpc_to_km = 3.0857e19  # Mpc в km


class PowerLawUniverse:
    """
    Клас за степенния космологичен модел: a(t) = C*t^n
    
    Attributes:
        H0_kmsmpc (float): Хъбъл константата в km/s/Mpc
        n (float): Степенен показател (n=1 линеен, n=2/3 материален)
        H0_inv_s (float): Хъбъл константата в 1/s
        t0_s (float): Възраст на Вселената в секунди
        t0_years (float): Възраст на Вселената в години
        C (float): Константа на разширение
    """
    
    def __init__(self, H0_kmsmpc: float = 70.0, n: float = 1.0):
        """
        Инициализира степенния космологичен модел
        
        Args:
            H0_kmsmpc: Хъбъл константата в km/s/Mpc
            n: Степенен показател (n=1 линеен, n=2/3 материален)
        """
        self.H0_kmsmpc = H0_kmsmpc
        self.n = n
        self.H0_inv_s = H0_kmsmpc * H0_to_inv_s
        
        # За степенния модел: H0 = n/t0
        self.t0_s = self.n / self.H0_inv_s
        self.t0_years = self.t0_s / (365.25 * 24 * 3600)  # в години
        
        # Константа на разширение: a(t) = C*t^n, a(t0) = 1
        self.C = 1.0 / (self.t0_s ** self.n)
        
        # Скорост на светлината (за удобство)
        self.c = c
        
        # Проверка за валидност
        if self.n <= 0:
            raise ValueError("Степенният показател n трябва да е положителен")
        if self.n >= 1 and self.n != 1.0:
            warnings.warn("Степенен показател n >= 1 може да даде ускоряващо се разширение")
    
    def __repr__(self) -> str:
        """Представяне на модела"""
        return (f"PowerLawUniverse(H0={self.H0_kmsmpc:.1f} km/s/Mpc, "
                f"n={self.n:.3f}, t0={self.t0_years/1e9:.2f} Gyr)")
    
    def scale_factor(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява мащабния фактор a(t) = C*t^n
        
        Args:
            t: Време в секунди
            
        Returns:
            Мащабен фактор
        """
        return self.C * (t ** self.n)
    
    def hubble_parameter(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява Хъбъл параметъра H(t) = n/t
        
        Args:
            t: Време в секунди
            
        Returns:
            Хъбъл параметър в 1/s
        """
        return self.n / t
    
    def hubble_parameter_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява Хъбъл параметъра H(z) = H0 * (1+z)^(n)
        
        Args:
            z: Червено отместване
            
        Returns:
            Хъбъл параметър в km/s/Mpc
        """
        return self.H0_kmsmpc * ((1 + z) ** self.n)
    
    def time_from_redshift(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява времето на излъчване от червеното отместване
        
        За степенния модел: 1 + z = (t0/t)^n
        Следователно: t = t0 / (1+z)^(1/n)
        
        Args:
            z: Червено отместване
            
        Returns:
            Време на излъчване в секунди
        """
        return self.t0_s / ((1.0 + z) ** (1.0 / self.n))
    
    def redshift_from_time(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява червеното отместване от времето
        
        Args:
            t: Време в секунди
            
        Returns:
            Червено отместване
        """
        return (self.t0_s / t) ** self.n - 1.0
    
    def comoving_distance_at_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява съвместното разстояние от червеното отместване
        
        Аналитична формула за степенния модел:
        r(z) = (c*t0/(1-n)) * [1 - 1/(1+z)^((1-n)/n)]
        
        Args:
            z: Червено отместване
            
        Returns:
            Съвместно разстояние в Mpc
        """
        if abs(self.n - 1.0) < 1e-10:
            # Специален случай: n = 1 (линеен модел)
            return (self.c * self.t0_s / Mpc_to_km) * np.log(1.0 + z)
        else:
            # Общ случай: n != 1
            factor = self.c * self.t0_s / ((1.0 - self.n) * Mpc_to_km)
            term = 1.0 - ((1.0 + z) ** (-(1.0 - self.n) / self.n))
            return factor * term
    
    def luminosity_distance_at_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява светимостното разстояние от червеното отместване
        
        Args:
            z: Червено отместване
            
        Returns:
            Светимостно разстояние в Mpc
        """
        d_C = self.comoving_distance_at_z(z)
        return d_C * (1.0 + z)
    
    def distance_modulus_at_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява модула на разстояние от червеното отместване
        
        μ = m - M = 5 * log10(d_L/Mpc) + 25
        
        Args:
            z: Червено отместване
            
        Returns:
            Модул на разстояние в mag
        """
        d_L_Mpc = self.luminosity_distance_at_z(z)
        return 5.0 * np.log10(d_L_Mpc) + 25.0
    
    def age_at_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява възрастта на Вселената при дадено червено отместване
        
        Args:
            z: Червено отместване
            
        Returns:
            Възраст на Вселената в години
        """
        t_emission = self.time_from_redshift(z)
        return t_emission / (365.25 * 24 * 3600)
    
    def lookback_time(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява времето на поглед назад
        
        Args:
            z: Червено отместване
            
        Returns:
            Време на поглед назад в години
        """
        t_emission = self.time_from_redshift(z)
        t_lookback_s = self.t0_s - t_emission
        return t_lookback_s / (365.25 * 24 * 3600)
    
    def get_model_info(self) -> Dict[str, float]:
        """
        Връща информация за модела
        
        Returns:
            Речник с параметрите на модела
        """
        return {
            'H0_kmsmpc': self.H0_kmsmpc,
            'n': self.n,
            'H0_inv_s': self.H0_inv_s,
            't0_years': self.t0_years,
            't0_Gyr': self.t0_years / 1e9,
            'C': self.C,
            'model': f'Power Law: a(t) = C*t^{self.n:.3f}'
        }
    
    def fit_to_data(self, z_data: np.ndarray, mu_data: np.ndarray, 
                   mu_err: Optional[np.ndarray] = None,
                   fit_n: bool = True, n_bounds: Tuple[float, float] = (0.6, 1.0)) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Fitва модела към данни чрез минимизация на chi^2
        
        Args:
            z_data: Червено отместване на данните
            mu_data: Модул на разстояние на данните
            mu_err: Грешки в модула на разстояние (опционално)
            fit_n: Дали да фитва n или само H0
            n_bounds: Граници за n при фитване
            
        Returns:
            Tuple с оптимални параметри и статистики на fit-a
        """
        
        def chi2_function(params):
            """Функция за минимизиране на chi^2"""
            if fit_n:
                H0, n = params
            else:
                H0 = params[0]
                n = self.n
            
            # Проверка на границите
            if n <= 0 or n > 1.5:
                return 1e10
            
            try:
                temp_model = PowerLawUniverse(H0_kmsmpc=H0, n=n)
                mu_model = temp_model.distance_modulus_at_z(z_data)
                
                if mu_err is not None:
                    chi2 = np.sum(((mu_data - mu_model) / mu_err)**2)
                else:
                    chi2 = np.sum((mu_data - mu_model)**2)
                
                return chi2
            except:
                return 1e10
        
        # Минимизираме chi^2
        if fit_n:
            # Fit и H0 и n
            initial_guess = [self.H0_kmsmpc, self.n]
            bounds = [(50.0, 100.0), n_bounds]
            result = minimize(chi2_function, initial_guess, bounds=bounds, method='L-BFGS-B')
            optimal_H0, optimal_n = result.x
        else:
            # Fit само H0
            result = minimize_scalar(chi2_function, bounds=(50.0, 100.0), method='bounded')
            optimal_H0 = result.x
            optimal_n = self.n
        
        # Оптимален модел
        optimal_model = PowerLawUniverse(H0_kmsmpc=optimal_H0, n=optimal_n)
        
        # Статистики
        mu_model = optimal_model.distance_modulus_at_z(z_data)
        residuals = mu_data - mu_model
        
        if mu_err is not None:
            chi2_min = np.sum((residuals / mu_err)**2)
            reduced_chi2 = chi2_min / (len(z_data) - (2 if fit_n else 1))
        else:
            chi2_min = np.sum(residuals**2)
            reduced_chi2 = chi2_min / (len(z_data) - (2 if fit_n else 1))
        
        rms_residual = np.sqrt(np.mean(residuals**2))
        
        optimal_params = {
            'H0': optimal_H0,
            'n': optimal_n,
            't0_Gyr': optimal_model.t0_years / 1e9
        }
        
        stats = {
            'chi2_min': chi2_min,
            'reduced_chi2': reduced_chi2,
            'rms_residual': rms_residual,
            'n_data': len(z_data),
            'degrees_of_freedom': len(z_data) - (2 if fit_n else 1),
            'success': result.success if hasattr(result, 'success') else True
        }
        
        return optimal_params, stats


def find_optimal_n_for_lcdm_match(z_range: np.ndarray, H0: float = 70.0, 
                                 Om0: float = 0.3, OL0: float = 0.7,
                                 n_range: Tuple[float, float] = (0.6, 1.0)) -> Tuple[float, float, Dict]:
    """
    Намира оптимална стойност на n, която най-добре имитира ΛCDM модела
    
    Args:
        z_range: Диапазон от redshift-ове за сравнение
        H0: Хъбъл константата
        Om0: Плътност на материята
        OL0: Плътност на тъмната енергия
        n_range: Диапазон за търсене на n
        
    Returns:
        Tuple с оптимално n, минимална chi^2 и допълнителна информация
    """
    
    # Създаваме ΛCDM модела за сравнение
    try:
        from astropy.cosmology import FlatLambdaCDM
        lcdm_model = FlatLambdaCDM(H0=H0, Om0=Om0)
        mu_lcdm = lcdm_model.distmod(z_range).value
    except ImportError:
        # Използваме нашата версия
        from .linear_universe import SimpleLCDMUniverse
        lcdm_model = SimpleLCDMUniverse(H0=H0, Om0=Om0, OL0=OL0)
        mu_lcdm = lcdm_model.distance_modulus(z_range)
    
    def objective_function(n):
        """Функция за минимизиране - разлика с ΛCDM"""
        try:
            power_model = PowerLawUniverse(H0_kmsmpc=H0, n=n)
            mu_power = power_model.distance_modulus_at_z(z_range)
            return np.sum((mu_power - mu_lcdm)**2)
        except:
            return 1e10
    
    # Търсим оптималното n
    result = minimize_scalar(objective_function, bounds=n_range, method='bounded')
    
    optimal_n = result.x
    min_chi2 = result.fun
    
    # Създаваме оптимален модел за анализ
    optimal_model = PowerLawUniverse(H0_kmsmpc=H0, n=optimal_n)
    mu_optimal = optimal_model.distance_modulus_at_z(z_range)
    
    # Статистики
    residuals = mu_optimal - mu_lcdm
    rms_residual = np.sqrt(np.mean(residuals**2))
    max_residual = np.max(np.abs(residuals))
    
    analysis_info = {
        'optimal_n': optimal_n,
        'min_chi2': min_chi2,
        'rms_residual': rms_residual,
        'max_residual': max_residual,
        'H0': H0,
        'model_type': f'Power Law: a(t) ∝ t^{optimal_n:.3f}',
        'universe_age_Gyr': optimal_model.t0_years / 1e9,
        'z_range': [z_range.min(), z_range.max()],
        'n_points': len(z_range)
    }
    
    return optimal_n, min_chi2, analysis_info


def compare_models_at_z(z_values: np.ndarray, H0: float = 70.0) -> Dict:
    """
    Сравнява различни модели при дадени redshift-ове
    
    Args:
        z_values: Redshift стойности за сравнение
        H0: Хъбъл константата
        
    Returns:
        Речник с резултатите от сравнението
    """
    
    # Създаваме модели
    linear_model = PowerLawUniverse(H0_kmsmpc=H0, n=1.0)          # n=1 (линеен)
    matter_model = PowerLawUniverse(H0_kmsmpc=H0, n=2.0/3.0)      # n=2/3 (материален)
    
    # Намираме оптималното n за ΛCDM
    optimal_n, _, _ = find_optimal_n_for_lcdm_match(z_values, H0=H0)
    optimal_model = PowerLawUniverse(H0_kmsmpc=H0, n=optimal_n)
    
    # ΛCDM модел
    try:
        from astropy.cosmology import FlatLambdaCDM
        lcdm_model = FlatLambdaCDM(H0=H0, Om0=0.3)
        mu_lcdm = lcdm_model.distmod(z_values).value
    except ImportError:
        from .linear_universe import SimpleLCDMUniverse
        lcdm_model = SimpleLCDMUniverse(H0=H0, Om0=0.3, OL0=0.7)
        mu_lcdm = lcdm_model.distance_modulus(z_values)
    
    # Изчисляваме модулите на разстояние
    mu_linear = linear_model.distance_modulus_at_z(z_values)
    mu_matter = matter_model.distance_modulus_at_z(z_values)
    mu_optimal = optimal_model.distance_modulus_at_z(z_values)
    
    # Сравняваме с ΛCDM
    comparison = {
        'z_values': z_values,
        'models': {
            'Linear (n=1.0)': {
                'n': 1.0,
                'mu': mu_linear,
                'rms_diff_from_lcdm': np.sqrt(np.mean((mu_linear - mu_lcdm)**2)),
                'age_Gyr': linear_model.t0_years / 1e9
            },
            'Matter (n=2/3)': {
                'n': 2.0/3.0,
                'mu': mu_matter,
                'rms_diff_from_lcdm': np.sqrt(np.mean((mu_matter - mu_lcdm)**2)),
                'age_Gyr': matter_model.t0_years / 1e9
            },
            f'Optimal (n={optimal_n:.3f})': {
                'n': optimal_n,
                'mu': mu_optimal,
                'rms_diff_from_lcdm': np.sqrt(np.mean((mu_optimal - mu_lcdm)**2)),
                'age_Gyr': optimal_model.t0_years / 1e9
            },
            'ΛCDM': {
                'n': 'N/A',
                'mu': mu_lcdm,
                'rms_diff_from_lcdm': 0.0,
                'age_Gyr': 13.8  # Приблизително
            }
        }
    }
    
    return comparison 