"""
Compositional Generalization Analysis Package
This package provides tools for analyzing compositional generalization
in diffusion models.
"""

__version__ = '0.1.0'

from .analysis import CompositionalGeneralizationAnalyzer
from .data import ComponentConfig
from .data import CompositePromptGenerator
from .metrics import calculate_mae
from .metrics import calculate_mse
from .metrics import get_metric
from .models import ActivationHook
from .models import DiffusionModelWrapper
from .models import ModelType
from .models import get_model_type

__all__ = [
  'ActivationHook',
  'ComponentConfig',
  'CompositePromptGenerator',
  'CompositionalGeneralizationAnalyzer',
  'DiffusionModelWrapper',
  'ModelType',
  'calculate_mae',
  'calculate_mse',
  'get_metric',
  'get_model_type',
]
