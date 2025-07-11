"""
Анализ на мащабния фактор в линейния модел

Този модул съдържа функции за изчисляване и анализ на мащабния фактор
a(t) = k * t в линейния космологичен модел.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScaleFactorParameters:
    """Параметри на мащабния фактор"""
    k: float = 1.0  # Константа на разширение [1/време]
    a0: float = 1.0  # Начална стойност на мащабния фактор
    t0: float = 1.0  # Начално време


class ScaleFactorAnalyzer:
    """Анализатор на мащабния фактор"""
    
    def __init__(self, params: ScaleFactorParameters):
        self.params = params
    
    def scale_factor(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява мащабния фактор a(t) = k * t
        
        Args:
            t: Време или масив от времена
            
        Returns:
            Мащабен фактор
        """
        return self.params.k * t
    
    def scale_factor_derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява първата производна da/dt = k
        
        Args:
            t: Време или масив от времена
            
        Returns:
            Първа производна на мащабния фактор
        """
        if isinstance(t, np.ndarray):
            return np.full_like(t, self.params.k)
        return self.params.k
    
    def scale_factor_second_derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява втората производна d²a/dt² = 0
        
        Args:
            t: Време или масив от времена
            
        Returns:
            Втора производна на мащабния фактор (винаги 0)
        """
        if isinstance(t, np.ndarray):
            return np.zeros_like(t)
        return 0.0
    
    def hubble_parameter(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява параметъра на Хъбъл H(t) = (da/dt)/a = k/a = 1/t
        
        Args:
            t: Време или масив от времена
            
        Returns:
            Параметър на Хъбъл
        """
        return 1.0 / t
    
    def deceleration_parameter(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява параметъра на забавяне q = -a*d²a/dt²/(da/dt)² = 0
        
        Args:
            t: Време или масив от времена
            
        Returns:
            Параметър на забавяне (винаги 0)
        """
        if isinstance(t, np.ndarray):
            return np.zeros_like(t)
        return 0.0
    
    def expansion_rate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява скоростта на разширение (da/dt)/a₀
        
        Args:
            t: Време или масив от времена
            
        Returns:
            Скорост на разширение
        """
        return self.params.k / self.params.a0
    
    def time_from_scale_factor(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява времето от даден мащабен фактор t = a/k
        
        Args:
            a: Мащабен фактор
            
        Returns:
            Време
        """
        return a / self.params.k
    
    def plot_evolution(self, t_max: float = 10.0, n_points: int = 1000) -> Tuple[plt.Figure, plt.Axes]:
        """
        Създава графики на еволюцията на мащабния фактор
        
        Args:
            t_max: Максимално време за графиката
            n_points: Брой точки в графиката
            
        Returns:
            Figure и Axes обекти
        """
        t = np.linspace(0.1, t_max, n_points)  # Започваме от 0.1 за да избегнем сингулярността
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Еволюция на мащабния фактор', fontsize=16)
        
        # Мащабен фактор
        axes[0, 0].plot(t, self.scale_factor(t), 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Време t')
        axes[0, 0].set_ylabel('Мащабен фактор a(t)')
        axes[0, 0].set_title('a(t) = k * t')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Параметър на Хъбъл
        axes[0, 1].plot(t, self.hubble_parameter(t), 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Време t')
        axes[0, 1].set_ylabel('Параметър на Хъбъл H(t)')
        axes[0, 1].set_title('H(t) = 1/t')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Първа производна
        axes[1, 0].plot(t, self.scale_factor_derivative(t), 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Време t')
        axes[1, 0].set_ylabel('da/dt')
        axes[1, 0].set_title('Първа производна (константа)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Втора производна
        axes[1, 1].plot(t, self.scale_factor_second_derivative(t), 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Време t')
        axes[1, 1].set_ylabel('d²a/dt²')
        axes[1, 1].set_title('Втора производна (нула)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def compare_with_standard_model(self, t_max: float = 10.0, n_points: int = 1000) -> Tuple[plt.Figure, plt.Axes]:
        """
        Сравнява линейния модел със стандартни космологични модели
        
        Args:
            t_max: Максимално време за графиката
            n_points: Брой точки в графиката
            
        Returns:
            Figure и Axes обекти
        """
        t = np.linspace(0.1, t_max, n_points)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Линеен модел
        a_linear = self.scale_factor(t)
        ax.plot(t, a_linear, 'b-', linewidth=2, label='Линеен: a ∝ t')
        
        # Стандартни модели (нормализирани)
        a_radiation = np.sqrt(t)  # a ∝ t^(1/2)
        a_matter = (t)**(2/3)     # a ∝ t^(2/3)
        
        ax.plot(t, a_radiation, 'r--', linewidth=2, label='Лъчение: a ∝ t^(1/2)')
        ax.plot(t, a_matter, 'g--', linewidth=2, label='Материя: a ∝ t^(2/3)')
        
        ax.set_xlabel('Време t')
        ax.set_ylabel('Мащабен фактор a(t)')
        ax.set_title('Сравнение на модели за разширение')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def analyze_singularity(self, t_min: float = 1e-6, t_max: float = 1.0, n_points: int = 1000) -> dict:
        """
        Анализира поведението близо до сингулярността при t=0
        
        Args:
            t_min: Минимално време за анализа
            t_max: Максимално време за анализа
            n_points: Брой точки в анализа
            
        Returns:
            Речник с резултати от анализа
        """
        t = np.linspace(t_min, t_max, n_points)
        
        a = self.scale_factor(t)
        H = self.hubble_parameter(t)
        
        results = {
            'времена': t,
            'мащабен_фактор': a,
            'параметър_хъбъл': H,
            'поведение_при_t_0': {
                'a(t→0)': 0.0,
                'H(t→0)': np.inf,
                'da/dt': self.params.k,
                'd²a/dt²': 0.0
            },
            'линеарност': {
                'наклон': self.params.k,
                'r_squared': 1.0  # Перфектна линеарност
            }
        }
        
        return results


def main():
    """Примерна употреба на модула"""
    # Създаваме параметри
    params = ScaleFactorParameters(k=1.0, a0=1.0, t0=1.0)
    
    # Създаваме анализатор
    analyzer = ScaleFactorAnalyzer(params)
    
    # Тестваме функциите
    print("Тест на мащабния фактор:")
    print(f"a(1.0) = {analyzer.scale_factor(1.0)}")
    print(f"a(5.0) = {analyzer.scale_factor(5.0)}")
    print(f"H(1.0) = {analyzer.hubble_parameter(1.0)}")
    print(f"H(5.0) = {analyzer.hubble_parameter(5.0)}")
    
    # Създаваме графики
    fig, axes = analyzer.plot_evolution(t_max=10.0)
    plt.show()
    
    # Сравняваме с други модели
    fig, ax = analyzer.compare_with_standard_model(t_max=10.0)
    plt.show()
    
    # Анализираме сингулярността
    results = analyzer.analyze_singularity()
    print("\nАнализ на сингулярността:")
    print(f"Поведение при t→0: {results['поведение_при_t_0']}")


if __name__ == "__main__":
    main() 