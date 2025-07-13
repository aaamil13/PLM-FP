"""
Likelihood функция за данни от Космическия Микровълнов Фон (CMB)
=================================================================

Този модул дефинира клас за изчисляване на χ² стойността
за даден космологичен модел спрямо компресирани CMB данни.

Използва се сравнение на теоретичните предсказания за звуковия хоризонт (r_s)
и ъгловия размер на звуковия хоризонт (theta_s) с Planck данни.

Автор: Проект за изследване на линейна космология
"""

import numpy as np
import os
import sys

# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.models.plm_model_fp import PLM # Използваме новия модел PLM-FP
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.common_utils.cosmological_parameters import CMBData, PlanckCosmology, PhysicalConstants
from scipy import integrate

class CMBLikelihood:
    """
    Клас за изчисляване на χ² за CMB данни.
    """
    def __init__(self):
        """
        Инициализация и зареждане на CMB референтни данни.
        """
        self.cmb_data = CMBData.get_cmb_summary()
        self.planck_params = PlanckCosmology.get_summary()
        self.constants = PhysicalConstants.get_all_constants()

        # Референтни стойности от Planck (за z_recombination = 1089.8)
        self.z_recombination = 1089.8
        self.rs_fiducial = 144.45 # Mpc, Sound horizon at recombination (approx. from Planck)
        self.theta_s_fiducial = self.cmb_data['constraints']['theta_star'] # Angular size of sound horizon
        self.l_A_fiducial = self.planck_params['l_A'] # Acoustic scale

        # Приблизителни грешки за χ² (за опростен подход)
        # Тези стойности трябва да бъдат калибрирани по-прецизно
        self.rs_err = 0.5 # Mpc (примерна грешка)
        self.theta_s_err = 0.00005 # Примерна грешка за theta_s
        self.l_A_err = 0.5 # Примерна грешка за l_A

    def _sound_horizon_integrand(self, z, Omega_m, Omega_r, H0):
        """
        Интегранд за изчисляване на звуковия хоризонт r_s.
        c_s(z) / H(z), където c_s(z) е скоростта на звука.
        За опростяване, приемаме c_s = c / sqrt(3) за радиационно доминирана епоха.
        По-точното изчисление изисква включване на бариони.
        """
        # Приближение за H(z) за плоска вселена (материя + радиация)
        # H(z) = H0 * sqrt(Omega_m * (1+z)^3 + Omega_r * (1+z)^4)
        H_z = H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_r * (1+z)**4)
        
        # Скорост на звука (c_s = c / sqrt(3) за ултрарелативистичен флуид)
        # За по-прецизно, c_s(z) = c / sqrt(3 * (1 + 3 * Omega_b / (4 * Omega_r) / (1+z)))
        # Но за опростяване, ще използваме константа, тъй като сме в ранна вселена
        c_s = self.constants['c'] / np.sqrt(3) # km/s
        
        return c_s / H_z

    def calculate_sound_horizon(self, model_instance, z_star=1089.8):
        """
        Изчислява звуковия хоризонт r_s до червено отместване z_star.
        """
        # Използваме параметрите на модела в зависимост от типа му
        if isinstance(model_instance, PLM):
            H0 = model_instance.params['H0']
            Omega_m = model_instance.Omega_m
            h = H0 / 100.0
            Omega_r = model_instance.Omega_r # PLM вече изчислява Omega_r
        elif isinstance(model_instance, LCDM):
            H0 = model_instance.H0
            Omega_m = model_instance.Om0
            h = H0 / 100.0
            # LCDM също има Omega_r, но обикновено се извлича от Tcmb0
            Omega_r = (2.47e-5) / (h**2) # Стандартна стойност
        else:
            raise ValueError("Неподдържан модел за изчисляване на r_s")

        # Адаптираме интегранда, за да използва H(z) от модела
        def integrand(z):
            H_z = model_instance.H_of_z(z)
            
            c_s = self.constants['c'] / np.sqrt(3) # km/s
            return c_s / H_z
        
        # ПРАВИЛНИЯТ ИНТЕГРАЛ: от z_recombination до безкрайност
        # Увеличаваме и лимита за по-добра точност
        r_s, _ = integrate.quad(integrand, z_star, np.inf, limit=200)
        
        return r_s

    def log_likelihood(self, model):
        """
        Изчислява логаритъма на likelihood функцията (пропорционален на -0.5 * χ²)
        за CMB данни.
        
        Параметри:
        -----------
        model : object
            Инстанция на космологичен модел (PLM или LCDM),
            която има методи `H_of_z(z)` и `angular_diameter_distance(z)`.
            
        Връща:
        --------
        log_like : float
            Стойността на log(likelihood).
        """
        # Изчисляване на r_s и theta_s от модела
        r_s_model = self.calculate_sound_horizon(model, self.z_recombination)
        
        # Angular diameter distance to recombination
        D_A_recomb_model = model.angular_diameter_distance(self.z_recombination)
        
        # Theta_s = r_s / D_A
        theta_s_model = r_s_model / D_A_recomb_model
        
        # Acoustic scale l_A = pi / theta_s
        l_A_model = np.pi / theta_s_model

        # Създаване на вектор от наблюдения и предсказания
        # Ще използваме r_s и l_A за сравнение, тъй като са ключови за CMB.
        # H0 и Omega_m_h2 също са важни, но са косвено включени.
        # За опростяване, ще използваме само r_s и l_A
        
        # Вектор на разликите
        delta_rs = r_s_model - self.rs_fiducial
        delta_lA = l_A_model - self.l_A_fiducial
        
        # Изчисляване на χ²
        # Приемаме, че r_s и l_A са некорелирани за този опростен модел.
        # Реално, те са силно корелирани.
        chi_squared = (delta_rs / self.rs_err)**2 + (delta_lA / self.l_A_err)**2
        
        # log(likelihood) ∝ -0.5 * χ²
        return -0.5 * chi_squared
