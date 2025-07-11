"""
Стандартен ΛCDM модел за MCMC Анализ
====================================

Този модул дефинира класа LCDM, който служи като обвивка (wrapper)
около `astropy.cosmology` за стандартния ΛCDM модел. Това осигурява
съвместим интерфейс с другите модели в проекта.

Свободните параметри на модела са:
- H0: Хъбъл константа [km/s/Mpc]
- omega_m_h2: Плътност на материята днес (Ω_m,₀ * h²)
- omega_b_h2: Плътност на барионната материя днес (Ω_b,₀ * h²)
- n_s: Спектрален индекс на първичните флуктуации
- A_s: Амплитуда на първичните флуктуации
- tau_reio: Оптична дълбочина на рейонизацията

Забележка: Параметрите n_s, A_s и tau_reio не са нужни за изчисляване
на фоновите разстояния, но са включени за пълнота и за бъдеща
съвместимост с CMB likelihoods.

Автор: Проект за изследване на линейна космология
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

class LCDM:
    """
    Клас-обвивка за стандартния ΛCDM модел, използващ astropy.
    """
    def __init__(self, H0=70.0, omega_m_h2=0.14, omega_b_h2=0.0224, 
                 n_s=0.96, A_s=2.1e-9, tau_reio=0.054):
        """
        Инициализация на модела с неговите свободни параметри.
        
        Параметри:
        -----------
        H0 : float
            Хъбъл константа днес [km/s/Mpc].
        omega_m_h2 : float
            Плътност на цялата материя (барионна + тъмна) днес (Ω_m,₀ * h²).
        omega_b_h2 : float
            Плътност на барионната материя днес (Ω_b,₀ * h²).
        n_s, A_s, tau_reio : float
            Параметри, свързани с CMB (не се използват за фонова космология).
        """
        self.H0 = float(H0)
        self.omega_m_h2 = float(omega_m_h2)
        self.omega_b_h2 = float(omega_b_h2)
        self.n_s = float(n_s)
        self.A_s = float(A_s)
        self.tau_reio = float(tau_reio)
        
        # Производни параметри, нужни за astropy
        self.h = self.H0 / 100.0
        self.Om0 = self.omega_m_h2 / (self.h**2)
        self.Ob0 = self.omega_b_h2 / (self.h**2)
        
        # Създаване на инстанция на astropy cosmology
        # FlatLambdaCDM предполага плоска вселена (Ω_k=0), където Ω_Λ = 1 - Ω_m
        # Tcmb0 е температурата на CMB днес, стандартна стойност.
        # Neff е ефективният брой неутрино видове.
        self.cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Om0, 
                                   Tcmb0=2.7255 * u.K, Ob0=self.Ob0, 
                                   Neff=3.046)

    def angular_diameter_distance(self, z):
        """
        Изчислява ъгловото разстояние d_A [Mpc].
        """
        return self.cosmo.angular_diameter_distance(z).to(u.Mpc).value

    def luminosity_distance(self, z):
        """
        Изчислява светимостното разстояние d_L [Mpc].
        """
        return self.cosmo.luminosity_distance(z).to(u.Mpc).value

    def distance_modulus(self, z):
        """
        Изчислява модула на разстояние μ.
        """
        # astropy връща обект с единица, преобразуваме го към число
        mu = self.cosmo.distmod(z).value
        # Проверка за z=0, където astropy може да върне -inf
        if np.isinf(mu):
            return 0.0 if z == 0 else mu
        return mu

    def H_of_z(self, z):
        """
        Изчислява Хъбъл параметъра H(z) [km/s/Mpc].
        """
        return self.cosmo.H(z).to(u.km / u.s / u.Mpc).value

    def calculate_sound_horizon(self, z_star):
        """
        Изчислява звуковия хоризонт r_s до червено отместване z_star.
        Използва вградената функция на astropy за звуковия хоризонт.
        """
        # astropy.cosmology.sound_horizon(z) изчислява звуковия хоризонт.
        # Трябва да се уверим, че z_star е в правилните единици (бездименсионално).
        return self.cosmo.sound_horizon(z_star).to(u.Mpc).value
