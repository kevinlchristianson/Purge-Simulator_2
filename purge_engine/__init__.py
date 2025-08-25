"""
Purge Engine Package - UI-agnostic purge simulation engine

This package provides clean interfaces for purge pipeline simulation
without any coupling to UI frameworks.
"""

from .types import Inputs, Results, StrategyConfig, IPSConfig
from .engine import run_simulation

__all__ = ['Inputs', 'Results', 'StrategyConfig', 'IPSConfig', 'run_simulation']