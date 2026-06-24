"""R-Python bridge for visualization using rpy2."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import tempfile
import subprocess

import numpy as np
import pandas as pd

R_AVAILABLE = False
rpy2_robjects = None
rpy2_pandas = None
r_interface = None

_home = os.path.expanduser("~")
_r_libs = os.environ.get("R_LIBS_USER") or os.path.join(_home, "R", "library")
os.environ.setdefault("R_LIBS_USER", _r_libs)
os.environ.setdefault("R_HOME", "/usr/lib/R")

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr, isinstalled
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
    import logging

    rpy2_logger.setLevel(logging.ERROR)

    # Verify R is actually reachable
    _test = robjects.r('pi')
    if float(str(_test[0])) > 3.0:
        rpy2_robjects = robjects
        rpy2_pandas = pandas2ri
        r_interface = robjects.r
        base = importr('base')
        utils = importr('utils')
        R_AVAILABLE = True
except Exception:
    pass


class RVisualizationEngine:
    """Bridge to R visualization libraries via rpy2."""
    
    def __init__(self):
        self.packages_loaded: Dict[str, Any] = {}
        self._init_r_environment()
    
    def _init_r_environment(self) -> None:
        """Initialize R environment and load core packages."""
        if not R_AVAILABLE:
            return
        
        try:
            core_packages = [
                'ggplot2', 'plotly', 'dplyr', 'tidyr', 'readr',
                'RColorBrewer', 'viridis', 'scales', 'patchwork'
            ]
            for pkg in core_packages:
                if isinstalled(pkg):
                    self.packages_loaded[pkg] = importr(pkg)
        except Exception:
            pass
    
    def load_package(self, package: str) -> bool:
        """Load an R package."""
        if not R_AVAILABLE:
            return False
        try:
            if not isinstalled(package):
                return False
            self.packages_loaded[package] = importr(package)
            return True
        except Exception:
            return False
    
    def install_package(self, package: str, repos: str = "https://cloud.r-project.org/") -> bool:
        """Install an R package."""
        if not R_AVAILABLE:
            return False
        try:
            utils = importr('utils')
            utils.install_packages(package, repos=repos)
            return self.load_package(package)
        except Exception:
            return False
    
    def r_code(self, code: str) -> Any:
        """Execute arbitrary R code."""
        if not R_AVAILABLE:
            raise RuntimeError("R not available via rpy2")
        return r_interface(code)
    
    def df_to_r(self, df: pd.DataFrame) -> Any:
        """Convert pandas DataFrame to R data.frame."""
        if not R_AVAILABLE:
            raise RuntimeError("R not available")
        with localconverter(rpy2_robjects.default_converter + pandas2ri.converter):
            return rpy2_robjects.conversion.py2rpy(df)
    
    def r_to_df(self, r_obj: Any) -> pd.DataFrame:
        """Convert R object to pandas DataFrame."""
        if not R_AVAILABLE:
            raise RuntimeError("R not available")
        with localconverter(rpy2_robjects.default_converter + pandas2ri.converter):
            return rpy2_robjects.conversion.rpy2py(r_obj)
    
    def ggplot(self, data: pd.DataFrame, mapping: Dict[str, str], **kwargs) -> Any:
        """Create a ggplot object."""
        if not R_AVAILABLE:
            raise RuntimeError("R not available")
        
        if 'ggplot2' not in self.packages_loaded:
            self.load_package('ggplot2')
        
        ggplot2 = self.packages_loaded['ggplot2']
        r_data = self.df_to_r(data)
        
        aes_mapping = robjects.r['aes'](**mapping)
        p = ggplot2.ggplot(r_data, aes_mapping)
        
        for geom, params in kwargs.items():
            if hasattr(ggplot2, f'geom_{geom}'):
                geom_func = getattr(ggplot2, f'geom_{geom}')
                p = p + geom_func(**params)
        
        return p
    
    def save_plot(self, plot_obj: Any, filename: str, width: int = 800, height: int = 600, dpi: int = 100) -> str:
        """Save R plot to file."""
        if not R_AVAILABLE:
            raise RuntimeError("R not available")
        
        grdevices = importr('grDevices')
        grdevices.png(filename, width=width, height=height, res=dpi)
        r_interface['print'](plot_obj)
        grdevices.dev_off()
        return filename
    
    def plotly_ggplot(self, ggplot_obj: Any) -> Dict:
        """Convert ggplot to plotly JSON."""
        if not R_AVAILABLE:
            raise RuntimeError("R not available")
        
        if 'plotly' not in self.packages_loaded:
            self.load_package('plotly')
        
        plotly = self.packages_loaded['plotly']
        py_plot = plotly.ggplotly(ggplot_obj)
        
        py_plot_json = r_interface('jsonlite::toJSON')(py_plot.rx2('x'), auto_unbox=True)
        return json.loads(str(py_plot_json))
    
    def run_r_script(self, script: str, data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
        """Run R script with optional data inputs."""
        if not R_AVAILABLE:
            raise RuntimeError("R not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            if data:
                for name, df in data.items():
                    r_df = self.df_to_r(df)
                    r_interface.assign(name, r_df)
            
            r_interface(script)
            
            result = {}
            for name in ['result', 'plot_data', 'summary']:
                try:
                    result[name] = self.r_to_df(r_interface[name])
                except Exception:
                    pass
            
            return result


class RScriptRunner:
    """Run R scripts via subprocess (fallback when rpy2 unavailable)."""
    
    def __init__(self, rscript_path: str = "Rscript"):
        self.rscript_path = rscript_path
        self._check_r()
    
    def _check_r(self) -> bool:
        try:
            result = subprocess.run([self.rscript_path, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            self.available = result.returncode == 0
            return self.available
        except Exception:
            self.available = False
            return False
    
    def run_script(self, script: str, args: List[str] = None, 
                   input_data: Dict[str, pd.DataFrame] = None) -> Dict:
        """Run R script file with optional data."""
        if not self.available:
            raise RuntimeError("Rscript not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "script.R"
            script_path.write_text(script)
            
            if input_data:
                for name, df in input_data.items():
                    csv_path = Path(tmpdir) / f"{name}.csv"
                    df.to_csv(csv_path, index=False)
            
            cmd = [self.rscript_path, str(script_path)]
            if args:
                cmd.extend(args)
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=tmpdir, timeout=300)
            
            output = {}
            if result.returncode == 0:
                for f in Path(tmpdir).glob("*.csv"):
                    output[f.stem] = pd.read_csv(f)
                for f in Path(tmpdir).glob("*.json"):
                    output[f.stem] = json.loads(f.read_text())
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output": output
            }
    
    def install_packages(self, packages: List[str]) -> bool:
        """Install R packages via script."""
        if not self.available:
            return False
        
        script = f"""
        install.packages(c({", ".join(f'"{p}"' for p in packages)}), 
                        repos = "https://cloud.r-project.org/")
        """
        result = self.run_script(script)
        return result["success"]


def get_r_engine() -> Union[RVisualizationEngine, RScriptRunner]:
    """Get the best available R engine."""
    if R_AVAILABLE:
        return RVisualizationEngine()
    else:
        return RScriptRunner()


def check_r_setup() -> Dict[str, Any]:
    """Check R environment setup."""
    return {
        "rpy2_available": R_AVAILABLE,
        "rscript_available": False,
        "packages": {}
    }