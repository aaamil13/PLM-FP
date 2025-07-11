"""
Анализ на еволюцията на плътността в линейния модел

Този модул съдържа функции за анализ на плътността ρ(t) ∝ 1/t³
и свързаните с нея енергийни ефекти.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.integrate import quad
from scipy.optimize import fsolve


@dataclass
class DensityParameters:
    """Параметри на плътността"""
    rho0: float = 1.0  # Начална плътност
    a0: float = 1.0    # Начален мащабен фактор
    k: float = 1.0     # Константа на разширение
    c: float = 1.0     # Скорост на светлината (в естествени единици)


class DensityAnalyzer:
    """Анализатор на плътността"""
    
    def __init__(self, params: DensityParameters):
        self.params = params
    
    def density(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява плътността на материята ρ(t) = ρ₀ * (a₀/(k*t))³
        
        Args:
            t: Космологично време
            
        Returns:
            Плътност на материята
        """
        return self.params.rho0 * (self.params.a0 / (self.params.k * t))**3
    
    def density_derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява производната на плътността dρ/dt = -3ρ/t
        
        Args:
            t: Космологично време
            
        Returns:
            Производна на плътността
        """
        return -3 * self.density(t) / t
    
    def energy_density(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява енергийната плътност E(t) = ρ(t) * c²
        
        Args:
            t: Космологично време
            
        Returns:
            Енергийна плътност
        """
        return self.density(t) * self.params.c**2
    
    def volume_element(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява обемния елемент V(t) = V₀ * (k*t)³
        
        Args:
            t: Космологично време
            
        Returns:
            Обемен елемент
        """
        return (self.params.k * t)**3
    
    def total_energy_comoving(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява общата енергия в съпътстващ обем
        
        Args:
            t: Космологично време
            
        Returns:
            Обща енергия (константа)
        """
        # E_total = ρ(t) * V(t) = ρ₀ * a₀³ / k³ = constant
        return self.params.rho0 * self.params.a0**3 / self.params.k**3 * self.params.c**2
    
    def pressure(self, t: Union[float, np.ndarray], equation_of_state: str = 'dust') -> Union[float, np.ndarray]:
        """
        Изчислява налягането според уравнението на състоянието
        
        Args:
            t: Космологично време
            equation_of_state: Тип уравнение на състоянието ('dust', 'radiation', 'stiff')
            
        Returns:
            Налягане
        """
        rho = self.density(t)
        
        if equation_of_state == 'dust':
            return np.zeros_like(rho) if isinstance(rho, np.ndarray) else 0.0
        elif equation_of_state == 'radiation':
            return rho * self.params.c**2 / 3
        elif equation_of_state == 'stiff':
            return rho * self.params.c**2
        else:
            raise ValueError(f"Неизвестно уравнение на състоянието: {equation_of_state}")
    
    def density_parameter(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява параметъра на плътността Ω(t) = ρ(t)/ρ_critical
        
        Args:
            t: Космологично време
            
        Returns:
            Параметър на плътността
        """
        # За линейния модел H = 1/t, така че ρ_critical = 3H²/(8πG) = 3/(8πG*t²)
        # Приемаме 8πG = 3 за простота, така че ρ_critical = 1/t²
        rho_critical = 1.0 / (t**2)
        return self.density(t) / rho_critical
    
    def jeans_length(self, t: Union[float, np.ndarray], temperature: float = 1.0) -> Union[float, np.ndarray]:
        """
        Изчислява дължината на Джийнс за гравитационна нестабилност
        
        Args:
            t: Космологично време
            temperature: Температура на газа
            
        Returns:
            Дължина на Джийнс
        """
        # Приблизително: λ_J ∝ sqrt(T/ρ)
        rho = self.density(t)
        return np.sqrt(temperature / rho)
    
    def deceleration_from_density(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява параметъра на забавяне от плътността
        
        Args:
            t: Космологично време
            
        Returns:
            Параметър на забавяне
        """
        # За линейния модел q = 0, независимо от плътността
        if isinstance(t, np.ndarray):
            return np.zeros_like(t)
        return 0.0
    
    def cooling_rate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Изчислява скоростта на охлаждане на плътността
        
        Args:
            t: Космологично време
            
        Returns:
            Скорост на охлаждане
        """
        return -self.density_derivative(t) / self.density(t)
    
    def half_density_time(self) -> float:
        """
        Намира времето при което плътността е половин от началната
        
        Returns:
            Време при ρ = ρ₀/2
        """
        def equation(t):
            return self.density(t) - self.params.rho0 / 2
        
        t_half = fsolve(equation, 1.0)[0]
        return t_half
    
    def plot_density_evolution(self, t_max: float = 10.0, n_points: int = 1000) -> Tuple[plt.Figure, plt.Axes]:
        """
        Създава графики на еволюцията на плътността
        
        Args:
            t_max: Максимално време за графиката
            n_points: Брой точки в графиката
            
        Returns:
            Figure и Axes обекти
        """
        t = np.linspace(0.1, t_max, n_points)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Еволюция на плътността в линейния модел', fontsize=16)
        
        # Плътност
        rho = self.density(t)
        axes[0, 0].plot(t, rho, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Космологично време t')
        axes[0, 0].set_ylabel('Плътност ρ(t)')
        axes[0, 0].set_title('ρ(t) ∝ 1/t³')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Производна на плътността
        rho_derivative = self.density_derivative(t)
        axes[0, 1].plot(t, np.abs(rho_derivative), 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Космологично време t')
        axes[0, 1].set_ylabel('|dρ/dt|')
        axes[0, 1].set_title('Скорост на промяна на плътността')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Енергийна плътност
        energy_rho = self.energy_density(t)
        axes[1, 0].plot(t, energy_rho, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Космологично време t')
        axes[1, 0].set_ylabel('Енергийна плътност E(t)')
        axes[1, 0].set_title('E(t) = ρ(t) * c²')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Параметър на плътността
        omega = self.density_parameter(t)
        axes[1, 1].plot(t, omega, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Космологично време t')
        axes[1, 1].set_ylabel('Параметър на плътността Ω(t)')
        axes[1, 1].set_title('Ω(t) = ρ(t)/ρ_critical')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def compare_density_models(self, t_max: float = 10.0, n_points: int = 1000) -> Tuple[plt.Figure, plt.Axes]:
        """
        Сравнява различни модели за еволюция на плътността
        
        Args:
            t_max: Максимално време за графиката
            n_points: Брой точки в графиката
            
        Returns:
            Figure и Axes обекти
        """
        t = np.linspace(0.1, t_max, n_points)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Линеен модел
        rho_linear = self.density(t)
        ax.plot(t, rho_linear, 'b-', linewidth=2, label='Линеен: ρ ∝ 1/t³')
        
        # Стандартни модели (нормализирани)
        rho_radiation = 1.0 / (t**2)  # ρ ∝ 1/t² за лъчение
        rho_matter = 1.0 / (t**2)     # ρ ∝ 1/t² за материя при a ∝ t^(2/3)
        
        ax.plot(t, rho_radiation, 'r--', linewidth=2, label='Лъчение: ρ ∝ 1/t²')
        ax.plot(t, rho_matter, 'g--', linewidth=2, label='Материя: ρ ∝ 1/t²')
        
        ax.set_xlabel('Космологично време t')
        ax.set_ylabel('Плътност ρ(t)')
        ax.set_title('Сравнение на модели за еволюция на плътността')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def analyze_energy_conservation(self, t_max: float = 10.0, n_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Анализира запазването на енергията в модела
        
        Args:
            t_max: Максимално време за анализа
            n_points: Брой точки в анализа
            
        Returns:
            Речник с резултати
        """
        t = np.linspace(0.1, t_max, n_points)
        
        # Плътност и обем
        rho = self.density(t)
        vol = self.volume_element(t)
        
        # Обща енергия
        total_energy = rho * vol
        
        # Производни
        rho_dot = self.density_derivative(t)
        vol_dot = 3 * self.params.k**3 * t**2  # dV/dt = 3k³t²
        
        # Уравнение на непрекъснатостта: dρ/dt + 3H*ρ = 0
        continuity_lhs = rho_dot + 3 * (1.0/t) * rho
        
        results = {
            'време': t,
            'плътност': rho,
            'обем': vol,
            'обща_енергия': total_energy,
            'уравнение_непрекъснатост': continuity_lhs,
            'запазване_енергия': np.abs(total_energy - total_energy[0]) / total_energy[0]
        }
        
        return results
    
    def plot_phase_space(self, t_max: float = 10.0, n_points: int = 1000) -> Tuple[plt.Figure, plt.Axes]:
        """
        Създава фазова диаграма на системата
        
        Args:
            t_max: Максимално време за графиката
            n_points: Брой точки в графиката
            
        Returns:
            Figure и Axes обекти
        """
        t = np.linspace(0.1, t_max, n_points)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Фазова траектория: ρ vs dρ/dt
        rho = self.density(t)
        rho_dot = self.density_derivative(t)
        
        ax.plot(rho, rho_dot, 'b-', linewidth=2)
        ax.set_xlabel('Плътност ρ')
        ax.set_ylabel('Производна на плътността dρ/dt')
        ax.set_title('Фазова диаграма на системата')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Маркираме началото и края
        ax.plot(rho[0], np.abs(rho_dot[0]), 'go', markersize=8, label='Начало')
        ax.plot(rho[-1], np.abs(rho_dot[-1]), 'ro', markersize=8, label='Край')
        ax.legend()
        
        return fig, ax
    
    def stability_analysis(self, t_eval: float = 1.0) -> Dict[str, float]:
        """
        Анализира стабилността на системата
        
        Args:
            t_eval: Време за оценка на стабилността
            
        Returns:
            Речник с резултати
        """
        # Линеаризирано уравнение около равновесие
        # За ρ(t) = ρ₀/t³, малки пертурбации δρ се развиват като δρ ∝ t^λ
        
        # Характерно време на промяна
        char_time = t_eval / 3.0
        
        # Скорост на охлаждане
        cooling = self.cooling_rate(t_eval)
        
        # Критерий за стабилност
        stability_criterion = cooling * char_time
        
        results = {
            'характерно_време': char_time,
            'скорост_охлаждане': cooling,
            'критерий_стабилност': stability_criterion,
            'стабилност': 'стабилна' if stability_criterion < 1 else 'нестабилна'
        }
        
        return results


def main():
    """Примерна употреба на модула"""
    # Създаваме параметри
    params = DensityParameters(rho0=1.0, a0=1.0, k=1.0, c=1.0)
    
    # Създаваме анализатор
    analyzer = DensityAnalyzer(params)
    
    # Тестваме функциите
    print("Тест на плътността:")
    print(f"ρ(1.0) = {analyzer.density(1.0):.3f}")
    print(f"ρ(5.0) = {analyzer.density(5.0):.3f}")
    print(f"dρ/dt(1.0) = {analyzer.density_derivative(1.0):.3f}")
    
    # Анализираме запазването на енергията
    conservation = analyzer.analyze_energy_conservation(t_max=10.0)
    print(f"\nЗапазване на енергията:")
    print(f"Максимално отклонение: {np.max(conservation['запазване_енергия']):.6f}")
    
    # Анализираме стабилността
    stability = analyzer.stability_analysis(t_eval=1.0)
    print(f"\nСтабилност:")
    print(f"Характерно време: {stability['характерно_време']:.3f}")
    print(f"Статус: {stability['стабилност']}")
    
    # Време за половин плътност
    t_half = analyzer.half_density_time()
    print(f"Време за половин плътност: {t_half:.3f}")
    
    # Създаваме графики
    fig, axes = analyzer.plot_density_evolution(t_max=10.0)
    plt.show()
    
    # Сравняваме модели
    fig, ax = analyzer.compare_density_models(t_max=10.0)
    plt.show()
    
    # Фазова диаграма
    fig, ax = analyzer.plot_phase_space(t_max=10.0)
    plt.show()


if __name__ == "__main__":
    main() 