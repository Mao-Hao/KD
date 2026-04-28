"""Neural network models for PDE field approximation.

Provides FieldModel — a configurable MLP surrogate that handles
normalization internally so autograd derivatives are automatically
correct in the original coordinate space.

FieldModelTrainer trains a FieldModel on observed data with support
for early stopping, RNG isolation, and automatic normalization.
"""

from kd2.models.field_model import FieldModel
from kd2.models.trainer import FieldModelTrainer, TrainingResult

__all__ = ["FieldModel", "FieldModelTrainer", "TrainingResult"]
