"""Search algorithm interfaces for kd2."""

from __future__ import annotations

from kd2.search.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    RunnerCallback,
    VizDataCollector,
)
from kd2.search.protocol import (
    IterativeSearchAlgorithm,
    PlatformComponents,
    SearchAlgorithm,
)
from kd2.search.recorder import VizRecorder
from kd2.search.result import (
    ExperimentResult,
    ResultBuilder,
    ResultTargetProvider,
    RunResult,
)
from kd2.search.runner import ExperimentRunner

__all__ = [
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "ExperimentResult",
    "ExperimentRunner",
    "IterativeSearchAlgorithm",
    "LoggingCallback",
    "PlatformComponents",
    "ResultBuilder",
    "ResultTargetProvider",
    "RunResult",
    "RunnerCallback",
    "SearchAlgorithm",
    "VizDataCollector",
    "VizRecorder",
]
