"""VizExtension protocol for plugin-provided plots.

Algorithms can optionally implement this protocol to provide
custom plots beyond the universal set. The platform handles
figure creation, styling, saving, and resource cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from matplotlib.axes import Axes


@dataclass
class PlotInfo:
    """Descriptor for a plugin-provided plot.

    Attributes:
        name: Machine-readable identifier (e.g. "aic_landscape").
        title: Human-readable title (e.g. "AIC vs Complexity").
        description: Optional longer description.
    """

    name: str
    title: str
    description: str = ""


@runtime_checkable
class VizExtension(Protocol):
    """Protocol for algorithms that provide custom visualizations.

    The platform creates a Figure+Axes, calls ``render_plot(name, ax)``,
    then saves and closes the figure. The algorithm only draws on the
    provided Axes.
    """

    def list_plots(self) -> list[PlotInfo]:
        """Return descriptors for all available plugin plots."""
        ...

    def render_plot(self, name: str, ax: Axes) -> None:
        """Render the named plot onto the provided Axes."""
        ...

    def get_plot_data(self, name: str) -> Any:
        """Return JSON-exportable data for the named plot."""
        ...
