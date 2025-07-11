"""
Анализ на времевата еволюция в линейния модел

Този модул съдържа функции за анализ на темпото на времето τ(t) ∝ 1/t³
и свързаните с него временни ефекти.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.integrate import quad
from scipy.optimize import fsolve


@dataclass
class TemporalParameters:
    """Параметри на времевата еволюция"""
    tau0: float = 1.0  # Начално темпо на времето
    rho0: float = 1.0  # Начална плътност
    a0: float = 1.0    # Начален мащабен фактор
    k: float = 1.0     # Константа на разширение


class TemporalAnalyzer:
    """Анализатор на времевата еволюция"""
    
    def __init__(self, params: TemporalParameters):
        self.params = params
    
    def time_tempo(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява темпото на времето τ(t) ∝ 1/t³
        
        Args:
            t: Космологично време
            
        Returns:
            Темпо на времето
        """
        return self.params.tau0 * (self.params.a0 / (self.params.k * t))**3
    
    def time_tempo_derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява производната на темпото на времето dτ/dt = -3τ/t
        
        Args:
            t: Космологично време
            
        Returns:
            Производна на темпото
        """
        return -3 * self.time_tempo(t) / t
    
    def proper_time_element(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява елемента на собствено време dτ_proper = τ(t) dt
        
        Args:
            t: Космологично време
            
        Returns:
            Елемент на собствено време
        """
        return self.time_tempo(t)
    
    def integrated_proper_time(self, t1: float, t2: float) -> float:
        """
        Изчислява интегрираното собствено време между t1 и t2
        
        Args:
            t1: Начално време
            t2: Крайно време
            
        Returns:
            Интегрирано собствено време
        """
        def integrand(t):
            return self.time_tempo(t)
        
        result, _ = quad(integrand, t1, t2)
        return result
    
    def time_dilation_factor(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява фактора на забавяне на времето спрямо референтно време
        
        Args:
            t: Космологично време
            
        Returns:
            Фактор на забавяне
        """
        return self.time_tempo(t) / self.params.tau0
    
    def acceleration_of_time(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява "ускорението" на времето d²τ/dt²
        
        Args:
            t: Космологично време
            
        Returns:
            Ускорение на времето
        """
        tau = self.time_tempo(t)
        return 12 * tau / (t**2)
    
    def time_flow_rate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява скоростта на течение на времето dτ/dt_observed
        
        Args:
            t: Космологично време
            
        Returns:
            Скорост на течение
        """
        return self.time_tempo_derivative(t)
    
    def characteristic_time_scale(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява характерното време τ/|dτ/dt| = t/3
        
        Args:
            t: Космологично време
            
        Returns:
            Характерно време
        """
        return t / 3.0
    
    def time_evolution_phases(self, t_max: float = 10.0) -> Dict[str, Dict]:
        """
        Определя фазите на времевата еволюция
        
        Args:
            t_max: Максимално време за анализа
            
        Returns:
            Речник с фазите
        """
        # Ранна фаза (t < 1)
        early_phase = {
            'интервал': (0.01, 1.0),
            'характеристики': 'Много бавно време, високо темпо',
            'доминиращи_процеси': 'Кубично забавяне'
        }
        
        # Средна фаза (1 < t < t_max/2)
        middle_phase = {
            'интервал': (1.0, t_max/2),
            'характеристики': 'Преходно поведение',
            'доминиращи_процеси': 'Постепенно ускорение'
        }
        
        # Късна фаза (t > t_max/2)
        late_phase = {
            'интервал': (t_max/2, t_max),
            'характеристики': 'Бързо време, ниско темпо',
            'доминиращи_процеси': 'Асимптотично поведение'
        }
        
        return {
            'ранна': early_phase,
            'средна': middle_phase,
            'късна': late_phase
        }
    
    def find_half_tempo_time(self) -> float:
        """
        Намира времето при което темпото е половин от началното
        
        Returns:
            Време при τ = τ₀/2
        """
        def equation(t):
            return self.time_tempo(t) - self.params.tau0 / 2
        
        # Започваме търсенето от разумна стойност
        t_half = fsolve(equation, 1.0)[0]
        return t_half
    
    def plot_temporal_evolution(self, t_max: float = 10.0, n_points: int = 1000) -> Tuple[plt.Figure, plt.Axes]:
        """
        Създава графики на времевата еволюция
        
        Args:
            t_max: Максимално време за графиката
            n_points: Брой точки в графиката
            
        Returns:
            Figure и Axes обекти
        """
        t = np.linspace(0.1, t_max, n_points)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Времева еволюция в линейния модел', fontsize=16)
        
        # Темпо на времето
        tempo = self.time_tempo(t)
        axes[0, 0].plot(t, tempo, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Космологично време t')
        axes[0, 0].set_ylabel('Темпо на времето τ(t)')
        axes[0, 0].set_title('τ(t) ∝ 1/t³')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Производна на темпото
        tempo_derivative = self.time_tempo_derivative(t)
        axes[0, 1].plot(t, tempo_derivative, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Космологично време t')
        axes[0, 1].set_ylabel('dτ/dt')
        axes[0, 1].set_title('Скорост на промяна на темпото')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Фактор на забавяне
        dilation = self.time_dilation_factor(t)
        axes[1, 0].plot(t, dilation, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Космологично време t')
        axes[1, 0].set_ylabel('Фактор на забавяне')
        axes[1, 0].set_title('Относително забавяне спрямо t₀')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Характерно време
        char_time = self.characteristic_time_scale(t)
        axes[1, 1].plot(t, char_time, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Космологично време t')
        axes[1, 1].set_ylabel('Характерно време')
        axes[1, 1].set_title('Скала на промените')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_proper_time_integration(self, t_max: float = 10.0, n_points: int = 100) -> Tuple[plt.Figure, plt.Axes]:
        """
        Изчертава интегрираното собствено време
        
        Args:
            t_max: Максимално време за графиката
            n_points: Брой точки в графиката
            
        Returns:
            Figure и Axes обекти
        """
        t_values = np.linspace(0.1, t_max, n_points)
        proper_times = []
        
        for t in t_values:
            proper_time = self.integrated_proper_time(0.1, t)
            proper_times.append(proper_time)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(t_values, proper_times, 'b-', linewidth=2, label='Интегрирано собствено време')
        ax.plot(t_values, t_values, 'r--', linewidth=2, label='Космологично време')
        
        ax.set_xlabel('Космологично време t')
        ax.set_ylabel('Интегрирано собствено време')
        ax.set_title('Сравнение на времевите скали')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def analyze_time_asymmetry(self, t_max: float = 10.0) -> Dict[str, float]:
        """
        Анализира времевата асиметрия в модела
        
        Args:
            t_max: Максимално време за анализа
            
        Returns:
            Речник с резултати
        """
        # Сравняваме времена в ранна и късна фаза
        t_early = 0.1
        t_late = t_max
        
        tempo_early = self.time_tempo(t_early)
        tempo_late = self.time_tempo(t_late)
        
        ratio = tempo_early / tempo_late
        
        # Интегрирано собствено време
        proper_time_total = self.integrated_proper_time(0.1, t_max)
        
        # Половина от времето
        t_half = self.find_half_tempo_time()
        
        results = {
            'съотношение_ранно_късно': ratio,
            'време_при_половин_темпо': t_half,
            'общо_собствено_време': proper_time_total,
            'асиметрия_индекс': np.log10(ratio)
        }
        
        return results
    
    def compare_time_scales(self, t_max: float = 10.0) -> Tuple[plt.Figure, plt.Axes]:
        """
        Сравнява различни времеви скали
        
        Args:
            t_max: Максимално време за графиката
            
        Returns:
            Figure и Axes обекти
        """
        t = np.linspace(0.1, t_max, 1000)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Космологично време
        ax.plot(t, t, 'k-', linewidth=2, label='Космологично време')
        
        # Темпо на времето
        tempo = self.time_tempo(t)
        ax.plot(t, tempo, 'b-', linewidth=2, label='Темпо на времето')
        
        # Характерно време
        char_time = self.characteristic_time_scale(t)
        ax.plot(t, char_time, 'r-', linewidth=2, label='Характерно време')
        
        ax.set_xlabel('Космологично време t')
        ax.set_ylabel('Времеви скали')
        ax.set_title('Сравнение на времевите скали')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax


def main():
    """Примерна употреба на модула"""
    # Създаваме параметри
    params = TemporalParameters(tau0=1.0, rho0=1.0, a0=1.0, k=1.0)
    
    # Създаваме анализатор
    analyzer = TemporalAnalyzer(params)
    
    # Тестваме функциите
    print("Тест на времевата еволюция:")
    print(f"τ(1.0) = {analyzer.time_tempo(1.0):.3f}")
    print(f"τ(5.0) = {analyzer.time_tempo(5.0):.3f}")
    print(f"dτ/dt(1.0) = {analyzer.time_tempo_derivative(1.0):.3f}")
    
    # Анализираме фазите
    phases = analyzer.time_evolution_phases(t_max=10.0)
    print("\nФази на еволюцията:")
    for name, phase in phases.items():
        print(f"{name}: {phase['интервал']}")
    
    # Анализираме асиметрията
    asymmetry = analyzer.analyze_time_asymmetry(t_max=10.0)
    print(f"\nВремева асиметрия:")
    print(f"Съотношение рано/късно: {asymmetry['съотношение_ранно_късно']:.1f}")
    print(f"Време при половин темпо: {asymmetry['време_при_половин_темпо']:.3f}")
    
    # Създаваме графики
    fig, axes = analyzer.plot_temporal_evolution(t_max=10.0)
    plt.show()
    
    # Сравняваме времеви скали
    fig, ax = analyzer.compare_time_scales(t_max=10.0)
    plt.show()


if __name__ == "__main__":
    main() 