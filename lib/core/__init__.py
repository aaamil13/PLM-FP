"""
Основни функции за линейния космологичен модел

Този пакет съдържа основните функции за изчисляване на:
- Мащабен фактор и производни
- Плътност и енергия
- Временна еволюция
- Космологични параметри
"""

from .cosmology import *
from .linear_universe import *
# from .temporal import *
# from .energy import *

__version__ = "1.0.0"
__author__ = "Linear Cosmology Model Team"

__all__ = [
    # Космологични функции
    'linear_scale_factor',
    'hubble_parameter',
    'hubble_parameter_matter_model',
    'deceleration_parameter',
    'expansion_rate',
    'compare_hubble_evolution',
    'cosmic_age_from_hubble',
    
    # Класове за моделиране
    'LinearUniverse',
    'SimpleLCDMUniverse',
    'create_lcdm_comparison_model',
    
    # Помощни функции
    'validate_time_input',
    'normalize_parameters',
    'cosmic_time_to_conformal',
    
    # Алиаси
    'a', 'H', 'H_matter', 'q'
] 