"""
Likelihood функция за данни от Свръхнови (Pantheon+)
=====================================================

Този модул дефинира клас за изчисляване на χ² стойността
за даден космологичен модел спрямо данни от свръхнови тип Ia.
Използва се формулата:

χ²_SN = Δμ^T * C_SN⁻¹ * Δμ

където:
- Δμ е векторът на разликите: μ_observed - μ_model
- C_SN е ковариационната матрица на грешките.

Автор: Проект за изследване на линейна космология
"""

import numpy as np
import os

class SupernovaeLikelihood:
    """
    Клас за изчисляване на χ² за данни от свръхнови.
    """
    def __init__(self, data_path='mcmc_analysis/data/'):
        """
        Инициализация и зареждане на данните.
        
        Параметри:
        -----------
        data_path : str
            Път до директорията, съдържаща данните от Pantheon+.
        """
        self.data_path = data_path
        self.data_file = os.path.join(data_path, 'pantheon_plus_data.txt')
        self.cov_file = os.path.join(data_path, 'pantheon_plus_cov.txt')
        
        self._load_data()

    def _load_data(self):
        """
        Зарежда данните за свръхновите и ковариационната матрица.
        """
        try:
            # Зареждане на основните данни (z, μ, error)
            # Очаква се формат: ред за коментар, следван от колони:
            # name, z_cmb, mb, dmb, z_hel, ...
            # Ще използваме z_cmb за червено отместване и mb за привидна звездна величина,
            # която е свързана с модула на разстояние.
            # Зареждаме колони zCMB (индекс 3) и MU_SH0ES (индекс 10)
            # skiprows=1, за да прескочим заглавния ред.
            data = np.loadtxt(self.data_file, usecols=(3, 10), skiprows=1)
            self.z = data[:, 0]  # Червени отмествания (zCMB)
            self.mu_obs = data[:, 1] # Наблюдавани модули на разстояние (MU_SH0ES)
            
            # Зареждане на ковариационната матрица, прескачайки първия ред
            cov_flat = np.loadtxt(self.cov_file, skiprows=1)
            num_points = len(self.z)
            if len(cov_flat) != num_points**2:
                raise ValueError("Размерът на ковариационната матрица не съответства на броя точки с данни.")
            
            self.cov_matrix = cov_flat.reshape((num_points, num_points))
            
            # Изчисляваме инверсната матрица веднъж за ефективност
            self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
            
            print(f"Успешно заредени {num_points} точки с данни за свръхнови.")

        except FileNotFoundError:
            print(f"ГРЕШКА: Файловете с данни не са намерени в '{self.data_path}'.")
            print("Моля, поставете 'pantheon_plus_data.txt' и 'pantheon_plus_cov.txt' в тази директория.")
            self.z = None
            # Хвърляме грешка, за да спрем изпълнението, ако данните липсват
            raise

    def log_likelihood(self, model, delta_M=0.0, z_local=0.0):
        """
        Изчислява логаритъма на likelihood функцията (пропорционален на -0.5 * χ²).
        
        Параметри:
        -----------
        model : object
            Инстанция на космологичен модел (напр. PLM или LCDM),
            която има метод `distance_modulus(z)`.
        delta_M : float, optional
            Параметър за отместване на абсолютната светимост. 
            По подразбиране е 0.0.
        z_local : float, optional
            Параметър за локално синьо отместване/червено отместване (peculiar velocity).
            По подразбиране е 0.0.
            
        Връща:
        --------
        log_like : float
            Стойността на log(likelihood).
        """
        if self.z is None:
            # Връщаме много лоша стойност, ако данните не са заредени
            return -np.inf

        # Прилагаме корекцията за локално z
        # z_theory = (1 + z_obs) / (1 + z_local) - 1
        # Защита от деление на нула или невалидни стойности за z_local
        if (1 + z_local) <= 0:
            return -np.inf

        z_theory = (1 + self.z) / (1 + z_local) - 1
        
        # Проверяваме за невалидни z_theory стойности (напр. отрицателни)
        if np.any(z_theory < 0):
            return -np.inf

        # Изчисляване на теоретичните стойности за модула на разстояние
        # Векторизираме извикването за по-добра производителност
        mu_model_theory = np.array([model.distance_modulus(z) for z in z_theory])
        
        # Прилагаме отместването delta_M
        mu_model_adjusted = mu_model_theory + delta_M
        
        # Вектор на разликите (residuals)
        delta_mu = self.mu_obs - mu_model_adjusted
        
        # Изчисляване на χ²
        # χ² = Δμ^T * C⁻¹ * Δμ
        chi_squared = np.dot(delta_mu, np.dot(self.inv_cov_matrix, delta_mu))
        
        # log(likelihood) ∝ -0.5 * χ²
        return -0.5 * chi_squared
