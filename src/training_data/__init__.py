"""
External training data loading for competitive programming.
Ensures proper train/test separation with OJBench as evaluation-only.
"""

from .external_problem_loader import ExternalProblemLoader, load_external_training_data

__all__ = ['ExternalProblemLoader', 'load_external_training_data']