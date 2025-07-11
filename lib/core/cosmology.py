"""
Основни космологични функции за линейния модел

Този модул съдържа базовите математически функции за изчисляване
на космологичните величини в линейния модел a(t) = k*t.
"""

import numpy as np
from typing import Union, Tuple, Optional
import warnings


def validate_time_input(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Валидира входните данни за време
    
    Args:
        t: Време или масив от времена
        
    Returns:
        Валидирано време
        
    Raises:
        ValueError: При невалидни входни данни
    """
    if isinstance(t, (int, float)):
        if t <= 0:
            raise ValueError("Времето трябва да е положително число")
        return float(t)
    
    elif isinstance(t, np.ndarray):
        if np.any(t <= 0):
            raise ValueError("Всички времена трябва да са положителни")
        return t.astype(float)
    
    else:
        raise ValueError("Времето трябва да е число или numpy масив")


def linear_scale_factor(t: Union[float, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява мащабния фактор a(t) = k * t
    
    Args:
        t: Космологично време
        k: Константа на разширение
        
    Returns:
        Мащабен фактор
    """
    t = validate_time_input(t)
    return k * t


def scale_factor_derivative(t: Union[float, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява първата производна на мащабния фактор da/dt = k
    
    Args:
        t: Космологично време
        k: Константа на разширение
        
    Returns:
        Първа производна
    """
    t = validate_time_input(t)
    if isinstance(t, np.ndarray):
        return np.full_like(t, k)
    return k


def scale_factor_second_derivative(t: Union[float, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява втората производна на мащабния фактор d²a/dt² = 0
    
    Args:
        t: Космологично време
        k: Константа на разширение
        
    Returns:
        Втора производна (винаги 0)
    """
    t = validate_time_input(t)
    if isinstance(t, np.ndarray):
        return np.zeros_like(t)
    return 0.0


def hubble_parameter(t: Union[float, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява параметъра на Хъбъл H(t) = (da/dt)/a = k/(kt) = 1/t
    
    За линейния модел a(t) = kt:
    - da/dt = k (константа)
    - H(t) = k/(kt) = 1/t
    
    Args:
        t: Космологично време
        k: Константа на разширение
        
    Returns:
        Параметър на Хъбъл H(t)
        
    Note:
        H₀ (Хъбъл константата) е просто H(t₀), където t₀ е възрастта на Вселената
    """
    t = validate_time_input(t)
    return 1.0 / t


def deceleration_parameter(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Изчислява параметъра на забавяне q = 0 (за линейния модел)
    
    Args:
        t: Космологично време
        
    Returns:
        Параметър на забавяне (винаги 0)
    """
    t = validate_time_input(t)
    if isinstance(t, np.ndarray):
        return np.zeros_like(t)
    return 0.0


def expansion_rate(t: Union[float, np.ndarray], k: float = 1.0, a0: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява скоростта на разширение
    
    Args:
        t: Космологично време
        k: Константа на разширение
        a0: Начален мащабен фактор
        
    Returns:
        Скорост на разширение
    """
    t = validate_time_input(t)
    if isinstance(t, np.ndarray):
        return np.full_like(t, k / a0)
    return k / a0


def cosmic_age(a: Union[float, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява възрастта на Вселената от мащабния фактор
    
    Args:
        a: Мащабен фактор
        k: Константа на разширение
        
    Returns:
        Възраст на Вселената
    """
    if isinstance(a, (int, float)):
        if a <= 0:
            raise ValueError("Мащабният фактор трябва да е положителен")
        return a / k
    elif isinstance(a, np.ndarray):
        if np.any(a <= 0):
            raise ValueError("Всички мащабни фактори трябва да са положителни")
        return a / k
    else:
        raise ValueError("Мащабният фактор трябва да е число или numpy масив")


def lookback_time(z: Union[float, np.ndarray], k: float = 1.0, a0: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява времето назад от червеното отместване
    
    Args:
        z: Червено отместване
        k: Константа на разширение
        a0: Текущ мащабен фактор
        
    Returns:
        Време назад
    """
    # a = a0 / (1 + z)
    # t = a / k
    a = a0 / (1 + z)
    t_observed = a / k
    t_now = a0 / k
    
    return t_now - t_observed


def redshift_from_time(t: Union[float, np.ndarray], t0: float, k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява червеното отместване от времето
    
    Args:
        t: Време на излъчване
        t0: Текущо време
        k: Константа на разширение
        
    Returns:
        Червено отместване
    """
    t = validate_time_input(t)
    if t0 <= 0:
        raise ValueError("Текущото време трябва да е положително")
    
    a = k * t
    a0 = k * t0
    
    z = a0 / a - 1
    return z


def distance_modulus(t: Union[float, np.ndarray], t0: float, k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява модула на разстоянието (опростена версия)
    
    Args:
        t: Време на излъчване
        t0: Текущо време
        k: Константа на разширение
        
    Returns:
        Модул на разстоянието
    """
    # Опростена формула за линейния модел
    # В реалност се изисква интегриране
    
    z = redshift_from_time(t, t0, k)
    
    # Приблизително за малки z: d_L ≈ z/H0
    # За линейния модел H0 = 1/t0
    luminosity_distance = z * t0
    
    # Модул на разстоянието: m - M = 5 * log10(d_L) + 25
    # Тук използваме опростена формула
    return 5 * np.log10(luminosity_distance) if luminosity_distance > 0 else -np.inf


def comoving_distance(t: Union[float, np.ndarray], t0: float, k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява съпътстващото разстояние
    
    Args:
        t: Време на излъчване
        t0: Текущо време
        k: Константа на разширение
        
    Returns:
        Съпътстващо разстояние
    """
    # За линейния модел: d_c = ∫(c dt / a(t)) = ∫(c dt / kt) = c * ln(t)/k
    # Приемаме c = 1 в естествени единици
    
    t = validate_time_input(t)
    if t0 <= 0:
        raise ValueError("Текущото време трябва да е положително")
    
    return np.log(t0) / k - np.log(t) / k


def angular_diameter_distance(t: Union[float, np.ndarray], t0: float, k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява ъгловото разстояние
    
    Args:
        t: Време на излъчване
        t0: Текущо време
        k: Константа на разширение
        
    Returns:
        Ъглово разстояние
    """
    # d_A = d_c / (1 + z)
    d_c = comoving_distance(t, t0, k)
    z = redshift_from_time(t, t0, k)
    
    return d_c / (1 + z)


def luminosity_distance(t: Union[float, np.ndarray], t0: float, k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява разстоянието по светимост
    
    Args:
        t: Време на излъчване
        t0: Текущо време
        k: Константа на разширение
        
    Returns:
        Разстояние по светимост
    """
    # d_L = d_c * (1 + z)
    d_c = comoving_distance(t, t0, k)
    z = redshift_from_time(t, t0, k)
    
    return d_c * (1 + z)


def critical_density(t: Union[float, np.ndarray], G: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява критичната плътност
    
    Args:
        t: Космологично време
        G: Гравитационна константа
        
    Returns:
        Критична плътност
    """
    # ρ_c = 3H²/(8πG)
    # За H = 1/t и 8πG = 3 (нормализация)
    H = hubble_parameter(t)
    return 3 * H**2 / (8 * np.pi * G)


def normalize_parameters(k: float, a0: float, t0: float) -> Tuple[float, float, float]:
    """
    Нормализира параметрите на модела
    
    Args:
        k: Константа на разширение
        a0: Начален мащабен фактор
        t0: Начално време
        
    Returns:
        Нормализирани параметри
    """
    # Нормализация: a0 = 1, t0 = 1
    k_norm = k * t0 / a0
    a0_norm = 1.0
    t0_norm = 1.0
    
    return k_norm, a0_norm, t0_norm


def cosmic_time_to_conformal(t: Union[float, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Преобразува космологично време в конформно време
    
    Args:
        t: Космологично време
        k: Константа на разширение
        
    Returns:
        Конформно време
    """
    # η = ∫(dt/a) = ∫(dt/kt) = ln(t)/k
    t = validate_time_input(t)
    
    # Добавяме малка константа за да избегнем log(0)
    if isinstance(t, np.ndarray):
        t_safe = np.where(t > 1e-10, t, 1e-10)
    else:
        t_safe = max(t, 1e-10)
    
    return np.log(t_safe) / k


def conformal_time_to_cosmic(eta: Union[float, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Преобразува конформно време в космологично време
    
    Args:
        eta: Конформно време
        k: Константа на разширение
        
    Returns:
        Космологично време
    """
    # t = exp(k * η)
    return np.exp(k * eta)


def sound_horizon(t: Union[float, np.ndarray], c_s: float = 1.0/np.sqrt(3)) -> Union[float, np.ndarray]:
    """
    Изчислява звуковия хоризонт
    
    Args:
        t: Космологично време
        c_s: Скорост на звука
        
    Returns:
        Звуков хоризонт
    """
    # r_s = ∫(c_s * dt / a) = c_s * η
    eta = cosmic_time_to_conformal(t)
    return c_s * eta


def particle_horizon(t: Union[float, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """
    Изчислява частичния хоризонт
    
    Args:
        t: Космологично време
        k: Константа на разширение
        
    Returns:
        Частичен хоризонт
    """
    # d_H = a(t) * ∫(dt/a) = a(t) * η = k*t * ln(t)/k = t * ln(t)
    t = validate_time_input(t)
    
    # Избягваме log(0)
    if isinstance(t, np.ndarray):
        t_safe = np.where(t > 1e-10, t, 1e-10)
        return t_safe * np.log(t_safe)
    else:
        t_safe = max(t, 1e-10)
        return t_safe * np.log(t_safe)


def validate_cosmological_parameters(k: float, a0: float, t0: float) -> None:
    """
    Валидира космологичните параметри
    
    Args:
        k: Константа на разширение
        a0: Начален мащабен фактор
        t0: Начално време
        
    Raises:
        ValueError: При невалидни параметри
    """
    if k <= 0:
        raise ValueError("Константата на разширение трябва да е положителна")
    
    if a0 <= 0:
        raise ValueError("Началният мащабен фактор трябва да е положителен")
    
    if t0 <= 0:
        raise ValueError("Началното време трябва да е положително")
    
    # Предупреждение за нереалистични стойности
    if k > 1e6:
        warnings.warn("Константата на разширение е много голяма")
    
    if a0 > 1e6:
        warnings.warn("Началният мащабен фактор е много голям")
    
    if t0 > 1e6:
        warnings.warn("Началното време е много голямо")


def hubble_parameter_matter_model(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Изчислява параметъра на Хъбъл за материалния модел H(t) = (2/3)/t
    
    За материалния модел a(t) ∝ t^(2/3):
    - da/dt ∝ (2/3)t^(-1/3)
    - H(t) = (da/dt)/a = (2/3)/t
    
    Args:
        t: Космологично време
        
    Returns:
        Параметър на Хъбъл за материалния модел
    """
    t = validate_time_input(t)
    return (2.0/3.0) / t


def compare_hubble_evolution(t: Union[float, np.ndarray]) -> dict:
    """
    Сравнява еволюцията на Хъбъл параметъра в различни модели
    
    Args:
        t: Космологично време
        
    Returns:
        Речник с H(t) за различни модели
    """
    t = validate_time_input(t)
    
    return {
        'linear': hubble_parameter(t),
        'matter': hubble_parameter_matter_model(t),
        'ratio_linear_to_matter': hubble_parameter(t) / hubble_parameter_matter_model(t)
    }


def cosmic_age_from_hubble(H0: float, model: str = 'linear') -> float:
    """
    Изчислява възрастта на Вселената от Хъбъл константата
    
    Args:
        H0: Хъбъл константата (в подходящи единици)
        model: Тип модел ('linear' или 'matter')
        
    Returns:
        Възраст на Вселената
    """
    if model == 'linear':
        return 1.0 / H0
    elif model == 'matter':
        return (2.0/3.0) / H0
    else:
        raise ValueError(f"Неизвестен модел: {model}")


# Алиаси за удобство
a = linear_scale_factor
H = hubble_parameter
q = deceleration_parameter
H_matter = hubble_parameter_matter_model 