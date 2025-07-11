"""
Подобрен линеен модел (ПЛМ) за MCMC Анализ
===========================================

Този модул дефинира класа PLM (Подобрен линеен модел), който капсулира
математическата рамка на модела, пригодена за използване в MCMC симулации.

Моделът се базира на следните принципи:
- Геометрия на пространството: a(t) = k * t * (1 + δ(t))
- Темпо на времето: dτ/dt = [1 + (ρ_total(z) / ρ_crit) ^ α]⁻¹
- Функция на деформация δ(t): δ(t) = ε * d/dt(dτ/dt) * (ρ(t)/ρ_crit)^β

Свободните параметри на модела са:
- H0: Хъбъл константа [km/s/Mpc]
- omega_m_h2: Плътност на материята днес (Ω_m,₀ * h²)
- z_crit: Червено отместване на фазовия преход, дефиниращо ρ_crit
- alpha: Показател, определящ рязкостта на прехода (за dτ/dt)
- epsilon: Амплитуда на деформацията δ(t)
- beta: Показател за чувствителност към плътност в δ(t)

Автор: Проект за изследване на линейна космология
"""

import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import numba # Добавяме numba

# === Помощни функции за Numba компилация ===

@numba.jit(nopython=True, nogil=True)
def _numba_density_evolution(Omega_m, Omega_r, z):
    """Numba-компилирана версия на _density_evolution."""
    rho_m = Omega_m * (1 + z)**3
    rho_r = Omega_r * (1 + z)**4
    return rho_m + rho_r

@numba.jit(nopython=True, nogil=True)
def _numba_time_dilation_factor(rho_norm, alpha):
    """Numba-компилирана версия на time_dilation_factor."""
    return 1.0 / (1.0 + rho_norm**alpha)

@numba.jit(nopython=True, nogil=True)
def _numba_calculate_dtau_dt_dt(t, t0_sec, rho_crit, alpha, Omega_m, Omega_r):
    """Numba-компилирана версия на _calculate_dtau_dt_dt."""
    t_vals = np.linspace(t * 0.9, t * 1.1, 100)
    z_vals = (t0_sec / t_vals) - 1.0
    
    dtau_dt_vals = np.empty_like(z_vals)
    for i in numba.prange(len(z_vals)):
        rho = _numba_density_evolution(Omega_m, Omega_r, z_vals[i])
        dtau_dt_vals[i] = _numba_time_dilation_factor(rho / rho_crit, alpha)

    d_dtau_dt_vals = numba_gradient(dtau_dt_vals, t_vals)
    d_dtau_dt_at_t = np.interp(t, t_vals, d_dtau_dt_vals)
    return d_dtau_dt_at_t

@numba.jit(nopython=True, nogil=True)
def _numba_delta_function(z, epsilon, beta, Omega_m, Omega_r, rho_crit, t0_sec, alpha):
    """Numba-компилирана версия на _delta_function."""
    rho_norm = _numba_density_evolution(Omega_m, Omega_r, z) / rho_crit
    d_dtau_dt_val = _numba_calculate_dtau_dt_dt(z, t0_sec, rho_crit, alpha, Omega_m, Omega_r)
    return epsilon * d_dtau_dt_val * (rho_norm**beta)

@numba.jit(nopython=True, nogil=True)
def _numba_scale_factor_t(t, k_s, t0_sec, epsilon, beta, Omega_m, Omega_r, rho_crit, alpha):
    """Numba-компилирана версия на _scale_factor_t."""
    z = (t0_sec / t) - 1.0
    delta_val = _numba_delta_function(z, epsilon, beta, Omega_m, Omega_r, rho_crit, t0_sec, alpha)
    return k_s * t * (1.0 + delta_val)

@numba.jit(nopython=True, nogil=True)
def _numba_H_of_z(z, t0_sec, k_s, epsilon, beta, Omega_m, Omega_r, rho_crit, alpha):
    """Numba-компилирана версия на H_of_z."""
    t = t0_sec / (1 + z)
    t_vals = np.linspace(t * 0.9, t * 1.1, 100)
    
    a_vals = np.empty_like(t_vals)
    for i in numba.prange(len(t_vals)):
        a_vals[i] = _numba_scale_factor_t(t_vals[i], k_s, t0_sec, epsilon, beta, Omega_m, Omega_r, rho_crit, alpha)
    
    da_dt_vals = numba_gradient(a_vals, t_vals)
    da_dt_at_t = np.interp(t, t_vals, da_dt_vals)
    a_at_t = _numba_scale_factor_t(t, k_s, t0_sec, epsilon, beta, Omega_m, Omega_r, rho_crit, alpha)
    
    H_t = (1.0 / a_at_t) * da_dt_at_t
    return H_t

# Custom numba-compatible gradient function
@numba.jit(nopython=True, nogil=True)
def numba_gradient(y, x):
    """
    Numba-съвместима версия на np.gradient за равноотстоящи точки.
    Използва централни разлики за вътрешните точки и крайни за краищата.
    """
    dydx = np.empty_like(y)
    
    # Първа точка (права разлика)
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    
    # Вътрешни точки (централна разлика)
    for i in range(1, len(y) - 1):
        dydx[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
        
    # Последна точка (обратна разлика)
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    return dydx

# === Край на помощните функции ===

class PLM:
    """
    Клас, представляващ Подобрения линеен модел (ПЛМ).
    """
    def __init__(self, H0=70.0, omega_m_h2=0.14, z_crit=900.0, alpha=2.0, epsilon=0.0, beta=1.0, omega_b_h2=0.0224):
        """
        Инициализация на модела с неговите свободни параметри.
        
        Параметри:
        -----------
        H0 : float
            Хъбъл константа днес [km/s/Mpc].
        omega_m_h2 : float
            Плътност на материята днес (Ω_m,₀ * h²).
        z_crit : float
            Червено отместване, при което се случва фазовият преход.
        alpha : float
            Показател на прехода (for dτ/dt).
        epsilon : float
            Амплитуда на деформацията δ(t).
        beta : float
            Показател за чувствителност към плътност в δ(t).
        omega_b_h2 : float
            Плътност на барионната материя днес (Ω_b,₀ * h²).
        """
        self.H0 = float(H0)
        self.omega_m_h2 = float(omega_m_h2)
        self.z_crit = float(z_crit)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.beta = float(beta)
        self.omega_b_h2 = float(omega_b_h2)
        
        self.h = self.H0 / 100.0
        self.Omega_m = self.omega_m_h2 / (self.h**2)
        self.Omega_b = self.omega_b_h2 / (self.h**2)
        
        self.omega_r_h2 = 2.47e-5
        self.Omega_r = self.omega_r_h2 / (self.h**2)
        
        self.c = 299792.458  # km/s
        
        self.rho_crit = self._density_evolution(self.z_crit) # Използваме _density_evolution за инициализация
        
        H0_per_sec = self.H0 / (3.086e19)
        self.t0_sec = 1.0 / H0_per_sec
        self.k_s = self.H0 * self.t0_sec

        self.t_c = self.t0_sec / (1 + 1100)
        self.delta_t_width = 0.01 * self.t0_sec
        
    def _density_evolution(self, z):
        return _numba_density_evolution(self.Omega_m, self.Omega_r, z)

    def time_dilation_factor(self, z):
        rho_norm = self._density_evolution(z) / self.rho_crit
        return _numba_time_dilation_factor(rho_norm, self.alpha)

    def _calculate_dtau_dt_dt(self, z):
        t = self.t0_sec / (1 + z)
        return _numba_calculate_dtau_dt_dt(t, self.t0_sec, self.rho_crit, self.alpha, self.Omega_m, self.Omega_r)

    def _delta_function(self, z):
        rho_norm = self._density_evolution(z) / self.rho_crit
        d_dtau_dt_val = self._calculate_dtau_dt_dt(z)
        return self.epsilon * d_dtau_dt_val * (rho_norm**self.beta)

    def _scale_factor_t(self, t):
        return _numba_scale_factor_t(t, self.k_s, self.t0_sec, self.epsilon, self.beta, self.Omega_m, self.Omega_r, self.rho_crit, self.alpha)

    def H_of_z(self, z):
        return _numba_H_of_z(z, self.t0_sec, self.k_s, self.epsilon, self.beta, self.Omega_m, self.Omega_r, self.rho_crit, self.alpha)

    def comoving_distance(self, z):
        if z <= 1e-8:
            return 0.0
        
        def comoving_integrand(z_prime):
            H_val = self.H_of_z(z_prime)
            if H_val <= 0 or np.isnan(H_val) or np.isinf(H_val):
                return np.inf
            return self.c / H_val
        
        result, abserr = integrate.quad(comoving_integrand, 0, z, limit=100)
        
        if not np.isfinite(result):
            return np.inf
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
        return 5.0 * np.log10(d_L) + 25.0

    def calculate_sound_horizon(self, z_star):
        def sound_horizon_integrand_plm(z):
            cs_val = self.c / np.sqrt(3 * (1 + 3 * self.Omega_b / (4 * self.Omega_r) / (1 + z)))
            H_z = self.H_of_z(z)
            if H_z <= 0 or np.isnan(H_z) or np.isinf(H_z):
                return np.inf
            return cs_val / H_z
        
        r_s, abserr = integrate.quad(sound_horizon_integrand_plm, z_star, np.inf, limit=100)
        
        if not np.isfinite(r_s):
            return np.inf
        return r_s
