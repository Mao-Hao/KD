"""kd2 - Symbolic regression platform for PDE discovery."""

from kd2.api import Model
from kd2.data import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
    generate_advection_data,
    generate_burgers_data,
    generate_diffusion_data,
    load_burgers,
    load_chafee_infante,
    load_kdv,
    load_pde_compound,
    load_pde_divide,
)
from kd2.inspect import preview
from kd2.search.result import ExperimentResult
from kd2.search.sga import SGAConfig
from kd2.viz.engine import VizEngine

__version__ = "0.1.0"

__all__ = [
    "AxisInfo",
    "DataTopology",
    "ExperimentResult",
    "FieldData",
    "Model",
    "PDEDataset",
    "SGAConfig",
    "TaskType",
    "VizEngine",
    "__version__",
    "generate_advection_data",
    "generate_burgers_data",
    "generate_diffusion_data",
    "load_burgers",
    "load_chafee_infante",
    "load_kdv",
    "load_pde_compound",
    "load_pde_divide",
    "preview",
]
