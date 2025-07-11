"""
Likelihood функция за данни от Барионни Акустични Осцилации (BAO)
=================================================================

Този модул дефинира клас за изчисляване на χ² стойността
за даден космологичен модел спрямо BAO данни.

Използва се опростена диагонална ковариационна матрица,
базирана на индивидуалните грешки на H(z) и d_A(z).

Автор: Проект за изследване на линейна космология
"""

import numpy as np
import os

class BAOLikelihood:
    """
    Клас за изчисляване на χ² за BAO данни с диагонална ковариация.
    """
    def __init__(self, data_file='data/bao_data.txt'):
        """
        Инициализация и зареждане на BAO данните.
        
        Параметри:
        -----------
        data_file : str
            Път до файла, съдържащ BAO измерванията.
        """
        self.data_file = data_file
        self._load_data()

    def _load_data(self):
        """
        Зарежда BAO данните (z, H(z)_obs, H_err, d_A(z)_obs, d_A_err).
        """
        try:
            # Зареждаме колони: z_eff, H(z), H_err, d_A(z), d_A_err
            # Пропускаме първия ред (коментар)
            data = np.loadtxt(self.data_file, comments='#', usecols=(0, 1, 2, 3, 4))
            
            self.z = data[:, 0]
            self.H_obs = data[:, 1]
            self.H_err = data[:, 2]
            self.dA_obs = data[:, 3]
            self.dA_err = data[:, 4]
            
            # Брой точки с данни за H(z) и d_A(z)
            self.num_points = len(self.z)
            
            # Построяване на диагонална ковариационна матрица
            # За всеки z имаме 2 измервания: H(z) и d_A(z)
            # Приемаме, че H(z) и d_A(z) са некорелирани помежду си
            # и между различните z.
            
            # Създаваме вектор от всички наблюдения: [H(z1), dA(z1), H(z2), dA(z2), ...]
            self.observations = np.empty(2 * self.num_points)
            self.errors_squared = np.empty(2 * self.num_points)
            
            for i in range(self.num_points):
                self.observations[2*i] = self.H_obs[i]
                self.errors_squared[2*i] = self.H_err[i]**2
                
                self.observations[2*i + 1] = self.dA_obs[i]
                self.errors_squared[2*i + 1] = self.dA_err[i]**2
            
            # Обратна на диагоналната ковариационна матрица е просто 1/error^2
            self.inv_cov_diag = 1.0 / self.errors_squared
            
            print(f"Успешно заредени {self.num_points} BAO точки.")

        except FileNotFoundError:
            print(f"ГРЕШКА: Файлът с BAO данни не е намерен в '{self.data_file}'.")
            raise

    def log_likelihood(self, model):
        """
        Изчислява логаритъма на likelihood функцията (пропорционален на -0.5 * χ²)
        за BAO данни.
        
        Параметри:
        -----------
        model : object
            Инстанция на космологичен модел (напр. PLM или LCDM),
            която има методи `H_of_z(z)` и `angular_diameter_distance(z)`.
            
        Връща:
        --------
        log_like : float
            Стойността на log(likelihood).
        """
        # Изчисляване на теоретичните стойности
        model_predictions = np.empty(2 * self.num_points)
        for i in range(self.num_points):
            z_val = self.z[i]
            model_predictions[2*i] = model.H_of_z(z_val)
            model_predictions[2*i + 1] = model.angular_diameter_distance(z_val)
            
        # Вектор на разликите (residuals)
        delta_val = self.observations - model_predictions
        
        # Изчисляване на χ² за диагонална ковариация:
        # χ² = сума [ (observation - model)^2 / error^2 ]
        chi_squared = np.sum(delta_val**2 * self.inv_cov_diag)
        
        # log(likelihood) ∝ -0.5 * χ²
        return -0.5 * chi_squared
