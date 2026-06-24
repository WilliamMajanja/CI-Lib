"""CI-Lib Visualization Suite — Unified Python + R visualization interface."""

from __future__ import annotations

from typing import Dict, List
from ci_lib.viz.python_viz import PythonVisualizationEngine
from ci_lib.viz.r_bridge import (
    RVisualizationEngine,
    RScriptRunner,
    get_r_engine,
    check_r_setup,
    R_AVAILABLE,
)

__all__ = [
    "PythonVisualizationEngine",
    "RVisualizationEngine",
    "RScriptRunner",
    "get_r_engine",
    "check_r_setup",
    "R_AVAILABLE",
    "VisualizationSuite",
]


class VisualizationSuite:
    """Unified entry point for all visualizations (Python + R)."""

    def __init__(self):
        self.python = PythonVisualizationEngine()
        self.r = get_r_engine()
        self.r_available = R_AVAILABLE or getattr(self.r, 'available', False)

    def capabilities(self) -> Dict[str, List[str]]:
        return {
            "python": self.python.list_libraries(),
            "r": self.r_available,
            "engines": {
                "python": True,
                "r_rpy2": R_AVAILABLE,
                "r_subprocess": getattr(self.r, 'available', False),
            },
        }