# Файл: mcmc_analysis/models/lcdm_model.py

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import quad # Нов импорт

class LCDM:
    """
    Клас-обвивка за стандартния ΛCDM модел, използващ astropy.
    Версия 2.0 - с ръчна имплементация на звуковия хоризонт за съвместимост.
    """
    def __init__(self, H0=70.0, omega_m_h2=0.14, omega_b_h2=0.0224, 
                 n_s=0.96, A_s=2.1e-9, tau_reio=0.054):
        
        self.H0 = float(H0)
        self.omega_m_h2 = float(omega_m_h2)
        self.omega_b_h2 = float(omega_b_h2)
        # ... (другите параметри остават същите)
        
        self.h = self.H0 / 100.0
        self.Om0 = self.omega_m_h2 / (self.h**2)
        self.Ob0 = self.omega_b_h2 / (self.h**2)
        
        # Физически константи
        self.c = 299792.458  # km/s
        self.omega_r_h2 = 2.47e-5 # Плътност на радиацията
        self.Omega_r = self.omega_r_h2 / (self.h**2)

        self.cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Om0, 
                                   Tcmb0=2.7255 * u.K, Ob0=self.Ob0)

    # ... (методите angular_diameter_distance, luminosity_distance, 
    #      distance_modulus, H_of_z остават същите) ...
    def angular_diameter_distance(self, z):
        return self.cosmo.angular_diameter_distance(z).to(u.Mpc).value

    def luminosity_distance(self, z):
        return self.cosmo.luminosity_distance(z).to(u.Mpc).value

    def distance_modulus(self, z):
        mu = self.cosmo.distmod(z).value
        return 0.0 if np.isinf(mu) and z == 0 else mu

    def H_of_z(self, z):
        return self.cosmo.H(z).to(u.km / u.s / u.Mpc).value


    # --- НОВ, КОРИГИРАН МЕТОД ---
    def calculate_sound_horizon(self, z_star):
        """
        Ръчно изчисляване на комологичния звуков хоризонт r_s.
        Това е независимо от версията на astropy.
        """
        def sound_horizon_integrand(z):
            # Скорост на звука в първичната плазма
            cs_val = self.c / np.sqrt(3.0 * (1.0 + (3.0 * self.Ob0) / (4.0 * self.Omega_r * (1.0 + z))))
            
            # Хъбъл параметър от astropy
            H_z = self.H_of_z(z)
            
            if H_z <= 0 or not np.isfinite(H_z):
                return np.inf
            return cs_val / H_z
        
        # Интегрираме от рекомбинацията до безкрайност
        r_s, err = quad(sound_horizon_integrand, z_star, np.inf)
        
        return r_s
