#!/usr/bin/env python3
"""
Клас LinearUniverse за тестване с реални данни

Този модул съдържа пълната имплементация на линейния космологичен модел
с правилни преобразувания на единиците и функции за сравнение с реални данни.
"""

import numpy as np
import warnings
from typing import Union, Tuple, Dict, Optional
import scipy.integrate as integrate
from scipy.optimize import minimize_scalar

# Космологични константи
c = 299792.458  # km/s (скорост на светлината)
H0_to_inv_s = 1.0 / (3.0857e19)  # преобразуване от (km/s)/Mpc в 1/s
Mpc_to_km = 3.0857e19  # Mpc в km


class LinearUniverse:
    """
    Клас за линейния космологичен модел: a(t) = k*t
    
    Attributes:
        H0_kmsmpc (float): Хъбъл константата в km/s/Mpc
        H0_inv_s (float): Хъбъл константата в 1/s
        t0_s (float): Възраст на Вселената в секунди
        t0_years (float): Възраст на Вселената в години
        k (float): Константа на разширение
    """
    
    def __init__(self, H0_kmsmpc: float = 70.0):
        """
        Инициализира линейния космологичен модел
        
        Args:
            H0_kmsmpc: Хъбъл константата в km/s/Mpc
        """
        self.H0_kmsmpc = H0_kmsmpc
        self.H0_inv_s = H0_kmsmpc * H0_to_inv_s
        
        # За линейния модел: H0 = 1/t0
        self.t0_s = 1.0 / self.H0_inv_s
        self.t0_years = self.t0_s / (365.25 * 24 * 3600)  # в години
        
        # Константа на разширение
        self.k = 1.0 / self.t0_s
        
        # Скорост на светлината (за удобство)
        self.c = c
        
        # Нормализация: a(t0) = 1
        self.a0 = 1.0
        
    def __repr__(self) -> str:
        """Представяне на модела"""
        return (f"LinearUniverse(H0={self.H0_kmsmpc:.1f} km/s/Mpc, "
                f"t0={self.t0_years/1e9:.2f} Gyr)")
    
    def scale_factor(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява мащабния фактор a(t) = k*t
        
        Args:
            t: Време в секунди
            
        Returns:
            Мащабен фактор
        """
        return self.k * t
    
    def hubble_parameter(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява Хъбъл параметъра H(t) = 1/t
        
        Args:
            t: Време в секунди
            
        Returns:
            Хъбъл параметър в 1/s
        """
        return 1.0 / t
    
    def time_from_redshift(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява времето на излъчване от червеното отместване
        
        За линейния модел: 1 + z = a0/a(t) = 1/(k*t)
        Следователно: t = 1/((1+z)*k) = t0/(1+z)
        
        Args:
            z: Червено отместване
            
        Returns:
            Време на излъчване в секунди
        """
        return self.t0_s / (1.0 + z)
    
    def redshift_from_time(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява червеното отместване от времето
        
        Args:
            t: Време в секунди
            
        Returns:
            Червено отместване
        """
        return (self.t0_s / t) - 1.0
    
    def comoving_distance(self, t_emission: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява съвместното разстояние (comoving distance)
        
        d_C = c * ∫[t_e to t0] dt'/a(t') = c * ∫[t_e to t0] dt'/(k*t')
             = (c/k) * ln(t0/t_e)
        
        Args:
            t_emission: Време на излъчване в секунди
            
        Returns:
            Съвместно разстояние в km
        """
        return (c / self.k) * np.log(self.t0_s / t_emission)
    
    def luminosity_distance(self, t_emission: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява светимостното разстояние
        
        d_L = d_C * (1 + z) = d_C * a0/a(t_e) = d_C * (t0/t_e)
        
        Args:
            t_emission: Време на излъчване в секунди
            
        Returns:
            Светимостно разстояние в km
        """
        d_C = self.comoving_distance(t_emission)
        return d_C * (self.t0_s / t_emission)
    
    def comoving_distance_at_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява съвместното разстояние от червеното отместване
        
        Args:
            z: Червено отместване
            
        Returns:
            Съвместно разстояние в Mpc
        """
        t_emission = self.time_from_redshift(z)
        d_C_km = self.comoving_distance(t_emission)
        return d_C_km / Mpc_to_km  # преобразуване в Mpc

    def luminosity_distance_at_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява светимостното разстояние от червеното отместване
        
        Args:
            z: Червено отместване
            
        Returns:
            Светимостно разстояние в Mpc
        """
        t_emission = self.time_from_redshift(z)
        d_L_km = self.luminosity_distance(t_emission)
        return d_L_km / Mpc_to_km  # преобразуване в Mpc
    
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
    
    def angular_diameter_distance(self, t_emission: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява ъгловото разстояние
        
        d_A = d_L / (1 + z)^2 = d_C / (1 + z)
        
        Args:
            t_emission: Време на излъчване в секунди
            
        Returns:
            Ъглово разстояние в km
        """
        d_C = self.comoving_distance(t_emission)
        z = self.redshift_from_time(t_emission)
        return d_C / (1.0 + z)
    
    def lookback_time(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява времето на поглед назад
        
        t_lookback = t0 - t_emission
        
        Args:
            z: Червено отместване
            
        Returns:
            Време на поглед назад в години
        """
        t_emission = self.time_from_redshift(z)
        t_lookback_s = self.t0_s - t_emission
        return t_lookback_s / (365.25 * 24 * 3600)
    
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
    
    def get_model_info(self) -> Dict[str, float]:
        """
        Връща информация за модела
        
        Returns:
            Речник с параметрите на модела
        """
        return {
            'H0_kmsmpc': self.H0_kmsmpc,
            'H0_inv_s': self.H0_inv_s,
            't0_years': self.t0_years,
            't0_Gyr': self.t0_years / 1e9,
            'k': self.k,
            'model': 'Linear: a(t) = k*t'
        }
    
    def fit_to_data(self, z_data: np.ndarray, mu_data: np.ndarray, 
                   mu_err: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, float]]:
        """
        Fitва модела към данни чрез минимизация на chi^2
        
        Args:
            z_data: Червено отместване на данните
            mu_data: Модул на разстояние на данните
            mu_err: Грешки в модула на разстояние (опционално)
            
        Returns:
            Tuple с оптимално H0 и статистики на fit-a
        """
        
        def chi2_function(H0):
            """Функция за минимизиране на chi^2"""
            temp_model = LinearUniverse(H0)
            mu_model = temp_model.distance_modulus_at_z(z_data)
            
            if mu_err is not None:
                chi2 = np.sum(((mu_data - mu_model) / mu_err)**2)
            else:
                chi2 = np.sum((mu_data - mu_model)**2)
            
            return chi2
        
        # Минимизираме chi^2
        result = minimize_scalar(chi2_function, bounds=(50.0, 100.0), method='bounded')
        
        # Оптимален модел
        optimal_H0 = result.x
        optimal_model = LinearUniverse(optimal_H0)
        
        # Статистики
        mu_model = optimal_model.distance_modulus_at_z(z_data)
        residuals = mu_data - mu_model
        
        if mu_err is not None:
            chi2_min = np.sum((residuals / mu_err)**2)
            reduced_chi2 = chi2_min / (len(z_data) - 1)
        else:
            chi2_min = np.sum(residuals**2)
            reduced_chi2 = chi2_min / (len(z_data) - 1)
        
        rms_residual = np.sqrt(np.mean(residuals**2))
        
        stats = {
            'chi2_min': chi2_min,
            'reduced_chi2': reduced_chi2,
            'rms_residual': rms_residual,
            'n_data': len(z_data),
            'degrees_of_freedom': len(z_data) - 1
        }
        
        return optimal_H0, stats


def create_lcdm_comparison_model(H0: float = 70.0, Om0: float = 0.3, 
                               OL0: float = 0.7):
    """
    Създава ΛCDM модел за сравнение
    
    Args:
        H0: Хъбъл константата в km/s/Mpc
        Om0: Плътност на материята днес
        OL0: Плътност на тъмната енергия днес
        
    Returns:
        ΛCDM модел
    """
    try:
        from astropy.cosmology import FlatLambdaCDM
        return FlatLambdaCDM(H0=H0, Om0=Om0)
    except ImportError:
        # Ако astropy не е налично, използваме опростена версия
        return SimpleLCDMUniverse(H0, Om0, OL0)


class SimpleLCDMUniverse:
    """
    Опростена имплементация на ΛCDM модела за сравнение
    """
    
    def __init__(self, H0: float = 70.0, Om0: float = 0.3, OL0: float = 0.7):
        """
        Инициализира ΛCDM модела
        
        Args:
            H0: Хъбъл константата в km/s/Mpc
            Om0: Плътност на материята днес
            OL0: Плътност на тъмната енергия днес
        """
        self.H0 = H0
        self.Om0 = Om0
        self.OL0 = OL0
        
        # Проверка
        if abs(Om0 + OL0 - 1.0) > 0.01:
            warnings.warn("Модел не е плосък: Om0 + OL0 != 1")
    
    def E(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява E(z) = H(z)/H0
        
        Args:
            z: Червено отместване
            
        Returns:
            E(z)
        """
        return np.sqrt(self.Om0 * (1 + z)**3 + self.OL0)
    
    def luminosity_distance(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява светимостното разстояние
        
        Args:
            z: Червено отместване
            
        Returns:
            Светимостно разстояние в Mpc
        """
        if np.isscalar(z):
            z_arr = np.array([z])
        else:
            z_arr = np.array(z)
        
        d_L = np.zeros_like(z_arr)
        
        for i, z_val in enumerate(z_arr):
            # Интеграл за съвместното разстояние
            integrand = lambda z_int: 1.0 / self.E(z_int)
            d_C, _ = integrate.quad(integrand, 0, z_val)
            d_C *= c / self.H0  # в Mpc
            
            # Светимостно разстояние
            d_L[i] = d_C * (1 + z_val)
        
        return d_L[0] if np.isscalar(z) else d_L
    
    def distance_modulus(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява модула на разстояние
        
        Args:
            z: Червено отместване
            
        Returns:
            Модул на разстояние в mag
        """
        d_L = self.luminosity_distance(z)
        return 5.0 * np.log10(d_L) + 25.0
    
    def age(self, z: Union[float, np.ndarray] = 0.0) -> Union[float, np.ndarray]:
        """
        Изчислява възрастта на Вселената
        
        Args:
            z: Червено отместване
            
        Returns:
            Възраст в години
        """
        if np.isscalar(z):
            z_arr = np.array([z])
        else:
            z_arr = np.array(z)
        
        ages = np.zeros_like(z_arr)
        
        for i, z_val in enumerate(z_arr):
            # Интеграл за възрастта
            integrand = lambda z_int: 1.0 / ((1 + z_int) * self.E(z_int))
            age_integral, _ = integrate.quad(integrand, z_val, np.inf)
            ages[i] = age_integral / (self.H0 * H0_to_inv_s) / (365.25 * 24 * 3600)
        
        return ages[0] if np.isscalar(z) else ages
    
    def __repr__(self) -> str:
        """Представяне на модела"""
        return f"SimpleLCDMUniverse(H0={self.H0:.1f}, Om0={self.Om0:.2f}, OL0={self.OL0:.2f})" 