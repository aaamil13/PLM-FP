# Файл: mcmc_analysis/models/plm_model-.py

import numpy as np
import scipy.integrate as integrate
import logging

# --- Дефиниране на функции за фазов преход (Strategy Pattern) ---
# Това позволява лесна смяна на модела на преход в бъдеще.

def default_f_bound(z, z_crit, w_crit, f_max):
    """
    Стандартният модел за фазов преход, базиран на tanh.
    """
    # Защитаваме w_crit да не е нула, за да избегнем деление на нула
    w = w_crit if w_crit > 1e-9 else 1e-9
    return f_max * 0.5 * (1.0 - np.tanh((z - z_crit) / w))

# Можете да добавите и други модели тук в бъдеще, напр.:
# def linear_f_bound(z, z_start, z_end, f_max):
#     ...

class PLM:
    """
    Клас, представляващ Подобрения линеен модел, базиран на Фазов Преход (PLM-FP).
    Версия 2.1 - Подобрена стабилност и разширяемост.
    """
    def __init__(self, H0=70.0, omega_m_h2=0.14, z_crit=2.5, w_crit=0.5, f_max=0.85, k=2.0, 
                 omega_b_h2=0.0224, T_CMB=2.7255, f_bound_func=default_f_bound):
        # --- 1. Запазване на параметрите ---
        self.params = {
            'H0': float(H0), 'omega_m_h2': float(omega_m_h2),
            'z_crit': float(z_crit), 'w_crit': float(w_crit),
            'f_max': float(f_max), 'k': float(k),
            'omega_b_h2': float(omega_b_h2), 'T_CMB': float(T_CMB)
        }
        
        # Функция за модела на преход
        self._f_bound_func = f_bound_func

        # --- 2. Изчисляване на производни космологични параметри ---
        self.h = self.params['H0'] / 100.0
        if self.h == 0: # Защита от деление на нула
            self.h = 1e-9
        
        self.Omega_m = self.params['omega_m_h2'] / (self.h**2)
        self.Omega_b = self.params['omega_b_h2'] / (self.h**2)
        
        # ПРЕПОРЪКА 2: Динамично изчисляване на плътността на радиацията
        # Плътност на фотоните + 3 вида безмасови неутрино
        N_eff_neutrino = 3.046 
        self.omega_r_h2 = (2.47e-5) * (self.params['T_CMB'] / 2.7255)**4 * (1 + 0.2271 * N_eff_neutrino)
        self.Omega_r = self.omega_r_h2 / (self.h**2)
        
        self.c = 299792.458  # Скорост на светлината [km/s]
        
        # --- 3. Изчисляване на нормализиращата константа C ---
        H_abs_0 = self._H_abs(0)
        time_dilation_0 = self._time_dilation(0)
        
        self.C = (self.params['H0'] * time_dilation_0 / H_abs_0) if (H_abs_0 > 1e-9 and time_dilation_0 > 1e-9) else self.params['H0']

        # --- 4. Кеширане за по-бързи изчисления ---
        self._H_of_z_cache = {}

    def _f_bound(self, z):
        # ПРЕПОРЪКА 5: Използване на подадената функция за преход
        return self._f_bound_func(z, self.params['z_crit'], self.params['w_crit'], self.params['f_max'])

    def _rho_free_norm(self, z):
        """
        Изчислява плътността на "свободната" енергия, нормализирана спрямо
        днешната критична плътност.
        """
        f_b = self._f_bound(z)
        rho_free_matter = (1.0 - f_b) * self.Omega_m * (1.0 + z)**3
        rho_free_radiation = self.Omega_r * (1.0 + z)**4
        return rho_free_matter + rho_free_radiation

    def _time_dilation(self, z):
        """
        Изчислява фактора на забързване на времето dτ/dt при дадено z.
        """
        rho_free_z = self._rho_free_norm(z)
        
        # Референтна плътност - плътността на свободната материя днес (z=0)
        rho_free_ref = self._rho_free_norm(0)

        # Защита от деление на нула
        if rho_free_z <= 1e-9:
            return 1.0 # Ако няма свободна енергия, няма и забързване на времето

        # ПРЕПОРЪКА 1: Ограничаване на максималната стойност за числена стабилност
        max_dilation = 1e6 # Максимално забързване (предпазна мярка)
        dilation = 1.0 + (rho_free_ref / rho_free_z)**self.params['k']
        return np.clip(dilation, 1.0, max_dilation)

    def _H_abs(self, z):
        """
        Изчислява "абсолютния" Хъбъл параметър, преди корекцията за времето.
        Това е стандартният E(z) от ΛCDM, но без Λ.
        """
        return np.sqrt(self.Omega_m * (1.0 + z)**3 + self.Omega_r * (1.0 + z)**4)

    def H_of_z(self, z):
        """
        Изчислява наблюдавания Хъбъл параметър H(z) в [km/s/Mpc].
        H_obs = C * H_abs / (dτ/dt)
        """
        z_float = float(z)
        if z_float in self._H_of_z_cache:
            return self._H_of_z_cache[z_float]

        if z_float < 0:
            return np.inf

        H_abs_z = self._H_abs(z_float)
        time_dilation_z = self._time_dilation(z_float)
        
        if time_dilation_z <= 1e-9:
            return np.inf
            
        H_z = self.C * H_abs_z / time_dilation_z
        # logging.debug(f"PLM-FP: H_of_z(z={z_float:.2f}) -> H_abs_z={H_abs_z:.2f}, time_dilation_z={time_dilation_z:.2f}, H_z={H_z:.2f}") # Коментирано за производителност

        if not np.isfinite(H_z):
            H_z = np.inf
        
        self._H_of_z_cache[z_float] = H_z
        return H_z

    def comoving_distance(self, z):
        """Комологично разстояние."""
        if z <= 1e-8: return 0.0
        
        def integrand(z_prime):
            H_val = self.H_of_z(z_prime)
            if H_val <= 1e-9 or not np.isfinite(H_val): return np.inf
            return self.c / H_val
        
        try:
            points_to_check = [self.params['z_crit']] if 0 < self.params['z_crit'] < z else None
            result, _ = integrate.quad(integrand, 0, z, limit=200, points=points_to_check)
        except Exception as e:
            logging.warning(f"Грешка в integrate.quad за comoving_distance(z={z}): {e}")
            result = np.inf
        
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
        d_L = self.luminosity_distance(z)
        if d_L <= 0 or not np.isfinite(d_L):
            return -np.inf
        
        # d_L вече е в Mpc
        return 5.0 * np.log10(d_L) + 25.0

    def calculate_sound_horizon(self, z_star=1090.0):
        """Изчислява звуковия хоризонт r_s до z_star."""
        def sound_horizon_integrand(z):
            cs_val = self.c / np.sqrt(3.0 * (1.0 + (3.0 * self.Omega_b) / (4.0 * self.Omega_r * (1.0 + z))))
            H_z = self.H_of_z(z)
            if H_z <= 1e-9 or not np.isfinite(H_z): return np.inf
            return cs_val / H_z
        
        try:
            r_s, _ = integrate.quad(sound_horizon_integrand, z_star, np.inf, limit=200)
        except Exception as e:
            logging.warning(f"Грешка в integrate.quad за sound_horizon: {e}")
            r_s = np.inf

        return r_s
