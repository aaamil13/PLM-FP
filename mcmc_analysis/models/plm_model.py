# Файл: mcmc_analysis/models/plm_model.py

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import approx_fprime # Коригиран импорт, съвместим с нови версии на SciPy
import logging

class PLM:
    """
    Клас, представляващ Подобрения линеен модел (ПЛМ).
    Коригирана и числено стабилна версия.
    """
    def __init__(self, H0=70.0, omega_m_h2=0.14, z_crit=900.0, alpha=2.0, epsilon=0.0, beta=1.0, omega_b_h2=0.0224):
        """
        Инициализация на модела с неговите свободни параметри.
        """
        # --- 1. Запазване на параметрите ---
        self.H0 = float(H0)
        self.omega_m_h2 = float(omega_m_h2)
        self.z_crit = float(z_crit)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.beta = float(beta)
        self.omega_b_h2 = float(omega_b_h2)
        
        # --- 2. Изчисляване на производни космологични параметри ---
        self.h = self.H0 / 100.0
        self.Omega_m = self.omega_m_h2 / (self.h**2)
        self.Omega_b = self.omega_b_h2 / (self.h**2)
        
        self.omega_r_h2 = 2.47e-5  # Плътност на радиацията
        self.Omega_r = self.omega_r_h2 / (self.h**2)
        
        self.c = 299792.458  # Скорост на светлината [km/s]
        
        # --- 3. Дефиниране на критичната плътност ---
        # ρ_crit е стойността на ρ_total при z_crit.
        # Нормализираме спрямо днешната критична плътност, така че ρ_total(z=0) = Ω_m + Ω_r.
        self.rho_crit_norm = (self.Omega_m * (1 + self.z_crit)**3) + (self.Omega_r * (1 + self.z_crit)**4)
        
        # --- 4. Връзка време-червено отместване (приближение за началната точка) ---
        # Използваме стандартната връзка за намиране на t0_sec.
        # Това е приближение, но е достатъчно за мащабиране.
        H0_per_sec = self.H0 / 3.086e19  # H0 в 1/s
        self.t0_sec = 1.0 / H0_per_sec   # Възраст на Вселената в секунди (приблизително)
        
        # --- 5. Кеширане за по-бързи изчисления ---
        self._H_of_z_cache = {}
        self._comoving_dist_cache = {}

    def _rho_total_norm(self, z):
        """Изчислява нормализираната плътност ρ_total(z) / ρ_crit_today."""
        return self.Omega_m * (1 + z)**3 + self.Omega_r * (1 + z)**4

    def _time_dilation_factor(self, t):
        """dτ/dt като функция на космологичното време t."""
        # Аргументът 't' трябва да е масив за approx_fprime, затова го третираме като такъв.
        t_val = t[0] if isinstance(t, (list, np.ndarray)) else t
        if t_val <= 0: return 0.0
        
        z_approx = (self.t0_sec / t_val) - 1.0
        
        rho_norm_vs_crit = self._rho_total_norm(z_approx) / self.rho_crit_norm
        return 1.0 / (1.0 + rho_norm_vs_crit**self.alpha)

    def _delta_function(self, t):
        """Функция на деформация δ(t)."""
        if self.epsilon == 0.0:
            return 0.0
        
        # Изчисляваме производната на dτ/dt по времето t.
        # approx_fprime изисква началната точка да е в масив и връща градиент.
        d_dtau_dt_dt = approx_fprime([t], self._time_dilation_factor, epsilon=1e-3 * t)[0]

        z_approx = (self.t0_sec / t) - 1.0
        rho_norm_vs_crit = self._rho_total_norm(z_approx) / self.rho_crit_norm
        
        return self.epsilon * d_dtau_dt_dt * (rho_norm_vs_crit**self.beta)

    def _scale_factor_t(self, t):
        """Мащабен фактор a(t). Нормализираме a(t0) = 1."""
        t_val = t[0] if isinstance(t, (list, np.ndarray)) else t
        if t_val <= 0: return 0.0
        
        # Изчисляваме a(t0) за нормализация
        a_t0 = self.t0_sec * (1.0 + self._delta_function(self.t0_sec))
        
        # Връщаме нормализирания мащабен фактор
        return (t_val / a_t0) * (1.0 + self._delta_function(t_val))

    def _t_of_z(self, z):
        """
        Приближение за намиране на времето t като функция на z.
        """
        return self.t0_sec / (1.0 + z)

    def H_of_z(self, z):
        """Изчислява H(z) в [km/s/Mpc]."""
        z_float = float(z)
        if z_float in self._H_of_z_cache:
            return self._H_of_z_cache[z_float]
            
        if z_float < 0: # Нефизично червено отместване
            return np.inf

        # Намираме времето t, съответстващо на z
        t_z = self._t_of_z(z_float)

        # Изчисляваме производната da/dt в точка t_z
        # approx_fprime изисква началната точка да е в масив и връща градиент.
        da_dt = approx_fprime([t_z], self._scale_factor_t, epsilon=1e-5 * t_z)[0]
        
        # Изчисляваме мащабния фактор a(t) в точка t_z
        a_t = self._scale_factor_t(t_z)

        if a_t <= 1e-9: # Проверка за деление на нула
            return np.inf

        # H(t) = (da/dt) / a(t) в [1/s]
        H_t_per_sec = da_dt / a_t
        
        # Преобразуваме в [km/s/Mpc]
        H_z = H_t_per_sec * 3.086e19

        if not np.isfinite(H_z):
            H_z = np.inf
        
        self._H_of_z_cache[z_float] = H_z
        return H_z

    def comoving_distance(self, z):
        """Комологично разстояние."""
        z_float = float(z)
        if z_float in self._comoving_dist_cache:
            return self._comoving_dist_cache[z_float]
            
        if z_float <= 1e-8:
            return 0.0
        
        def integrand(z_prime):
            H_val = self.H_of_z(z_prime)
            if H_val <= 1e-9 or not np.isfinite(H_val):
                return np.inf # Връща голяма стойност, за да "накаже" интегратора
            return self.c / H_val
        
        try:
            # points=... помага на интегратора при резки промени, напр. около z_crit
            points_to_check = [self.z_crit] if 0 < self.z_crit < z_float else None
            result, _ = integrate.quad(integrand, 0, z_float, limit=200, points=points_to_check)
        except Exception as e:
            logging.warning(f"Грешка в integrate.quad за comoving_distance(z={z_float}): {e}")
            result = np.inf
        
        if not np.isfinite(result):
            result = np.inf

        self._comoving_dist_cache[z_float] = result
        return result

    def angular_diameter_distance(self, z):
        d_c = self.comoving_distance(z)
        if not np.isfinite(d_c): return np.inf
        return d_c / (1.0 + z)

    def luminosity_distance(self, z):
        d_A = self.angular_diameter_distance(z)
        if not np.isfinite(d_A): return np.inf
        return d_A * (1.0 + z)**2

    def distance_modulus(self, z):
        # При всяко извикване на MCMC стъпка се създава нова инстанция на модела,
        # така че кешът автоматично се изчиства.
        
        d_L = self.luminosity_distance(z)
        if d_L <= 0 or not np.isfinite(d_L):
            return -np.inf # Връщаме много малка стойност, за да отхвърлим тези параметри
        
        # Формулата изисква d_L в Mpc. Нашите разстояния вече са в Mpc, тъй като c е в km/s, а H(z) в km/s/Mpc.
        d_L_mpc = d_L
        return 5.0 * np.log10(d_L_mpc) + 25.0

    def calculate_sound_horizon(self, z_star=1090.0):
        """Изчислява звуковия хоризонт r_s до z_star."""
        # Тази функция е за CMB, тя се извиква след другите, така че кешът ще е пълен
        def sound_horizon_integrand(z):
            # По-точна формула за скоростта на звука
            cs_val = self.c / np.sqrt(3.0 * (1.0 + (3.0 * self.Omega_b) / (4.0 * self.Omega_r * (1.0 + z))))
            H_z = self.H_of_z(z)
            if H_z <= 1e-9 or not np.isfinite(H_z):
                return np.inf
            return cs_val / H_z
        
        try:
            r_s, _ = integrate.quad(sound_horizon_integrand, z_star, np.inf, limit=200)
        except Exception as e:
            logging.warning(f"Грешка в integrate.quad за sound_horizon: {e}")
            r_s = np.inf

        if not np.isfinite(r_s):
            return np.inf
        return r_s