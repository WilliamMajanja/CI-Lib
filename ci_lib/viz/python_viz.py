"""Python visualization engine — unified interface for all Python viz libraries."""

from __future__ import annotations

import io
import base64
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

# Core plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

# Interactive
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.io import export_png
    from bokeh.models import ColumnDataSource, HoverTool, Legend
    import bokeh.palettes as bp
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    import altair as alt
    ALTAR_AVAILABLE = True
except ImportError:
    ALTAR_AVAILABLE = False

# Network
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# Statistical
try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Geospatial
try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# 3D
try:
    import pyvista as pv
    PVISTA_AVAILABLE = True
except ImportError:
    PVISTA_AVAILABLE = False

sns.set_theme(style="whitegrid")


class PythonVisualizationEngine:
    """Unified access to all Python visualization capabilities."""

    def __init__(self):
        self.libraries: Dict[str, bool] = {
            "matplotlib": True,
            "seaborn": True,
            "plotly": PLOTLY_AVAILABLE,
            "bokeh": BOKEH_AVAILABLE,
            "altair": ALTAR_AVAILABLE,
            "networkx": NETWORKX_AVAILABLE,
            "pyvis": PYVIS_AVAILABLE,
            "statsmodels": STATSMODELS_AVAILABLE,
            "folium": FOLIUM_AVAILABLE,
            "pyvista": PVISTA_AVAILABLE,
        }

    def list_libraries(self) -> List[str]:
        return [name for name, avail in self.libraries.items() if avail]

    # ── Matplotlib / Seaborn ──────────────────────────────────────────

    def mpl_line(self, data: Dict[str, List[float]], title: str = "",
                 xlabel: str = "", ylabel: str = "", figsize: Tuple[int,int] = (8,4)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        for label, values in data.items():
            ax.plot(values, label=label, linewidth=1.5)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        return fig

    def mpl_scatter(self, x: np.ndarray, y: np.ndarray, c: Any = None,
                    title: str = "", xlabel: str = "", ylabel: str = "",
                    figsize: Tuple[int,int] = (7,5)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(x, y, c=c, cmap="viridis", s=40, alpha=0.8, edgecolors="k", linewidth=0.3)
        if c is not None and len(np.unique(c)) > 1:
            plt.colorbar(sc, ax=ax)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); fig.tight_layout()
        return fig

    def mpl_hist(self, data: np.ndarray, bins: int = 30, title: str = "",
                 xlabel: str = "", figsize: Tuple[int,int] = (8,4)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(data, bins=bins, edgecolor="white", alpha=0.7)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        return fig

    def mpl_heatmap(self, matrix: np.ndarray, xticklabels: List[str] = None,
                    yticklabels: List[str] = None, title: str = "",
                    figsize: Tuple[int,int] = (8,6), annot: bool = True) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(matrix, annot=annot, fmt=".2f", cmap="viridis",
                    xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
        ax.set_title(title); fig.tight_layout()
        return fig

    def mpl_boxplot(self, data: Dict[str, List[float]], title: str = "",
                    figsize: Tuple[int,int] = (10,5)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(data.values(), labels=data.keys(), patch_artist=True)
        ax.set_title(title); ax.grid(True, alpha=0.3); fig.tight_layout()
        return fig

    def mpl_pairplot(self, df: pd.DataFrame, hue: str = None) -> Figure:
        g = sns.pairplot(df, hue=hue, diag_kind="kde")
        g.fig.suptitle("Pairplot", y=1.02)
        return g.fig

    def mpl_correlation_matrix(self, df: pd.DataFrame, figsize: Tuple[int,int] = (10,8)) -> Figure:
        corr = df.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix"); fig.tight_layout()
        return fig

    def mpl_3d_surface(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                       title: str = "", figsize: Tuple[int,int] = (9,6)) -> Figure:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_title(title); fig.tight_layout()
        return fig

    def mpl_violin(self, data: Dict[str, List[float]], title: str = "",
                   figsize: Tuple[int,int] = (10,5)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        parts = ax.violinplot(data.values(), showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(data.keys())
        ax.set_title(title); ax.grid(True, alpha=0.3); fig.tight_layout()
        return fig

    # ── Plotly ────────────────────────────────────────────────────────

    def plotly_line(self, data: Dict[str, List[float]], title: str = "",
                    xlabel: str = "", ylabel: str = "") -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = go.Figure()
        for label, values in data.items():
            fig.add_trace(go.Scatter(y=values, mode='lines', name=label))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel,
                         template="plotly_white", hovermode="x unified")
        return fig.to_json()

    def plotly_scatter(self, x: np.ndarray, y: np.ndarray, labels: np.ndarray = None,
                       title: str = "") -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = px.scatter(x=x, y=y, color=labels, title=title,
                        template="plotly_white", opacity=0.7)
        return fig.to_json()

    def plotly_3d_scatter(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          labels: np.ndarray = None, title: str = "") -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = px.scatter_3d(x=x, y=y, z=z, color=labels, title=title,
                           template="plotly_white", opacity=0.7)
        return fig.to_json()

    def plotly_heatmap(self, matrix: np.ndarray, title: str = "") -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = px.imshow(matrix, title=title, template="plotly_white",
                       color_continuous_scale="Viridis")
        return fig.to_json()

    def plotly_boxplot(self, data: Dict[str, List[float]], title: str = "") -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = go.Figure()
        for name, vals in data.items():
            fig.add_trace(go.Box(y=vals, name=name))
        fig.update_layout(title=title, template="plotly_white")
        return fig.to_json()

    def plotly_density(self, x: np.ndarray, title: str = "") -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = px.histogram(x=x, marginal="box", title=title, template="plotly_white")
        return fig.to_json()

    def plotly_sunburst(self, labels: List[str], parents: List[str],
                        values: List[float], title: str = "") -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = px.sunburst(names=labels, parents=parents, values=values, title=title)
        return fig.to_json()

    def plotly_parallel_coordinates(self, df: pd.DataFrame, color_col: str,
                                    title: str = "") -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = px.parallel_coordinates(df, color=color_col, title=title,
                                     template="plotly_white")
        return fig.to_json()

    def plotly_treemap(self, labels: List[str], parents: List[str],
                       values: List[float], title: str = "") -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = px.treemap(names=labels, parents=parents, values=values, title=title)
        return fig.to_json()

    # ── Bokeh ─────────────────────────────────────────────────────────

    def bokeh_line(self, data: Dict[str, List[float]], title: str = "",
                   width: int = 700, height: int = 400) -> Optional[str]:
        if not BOKEH_AVAILABLE: return None
        p = figure(title=title, width=width, height=height,
                  tools="pan,wheel_zoom,box_zoom,reset,hover,save")
        p.grid.grid_line_alpha = 0.3
        colors = bp.Category10[10]
        for i, (label, values) in enumerate(data.items()):
            p.line(range(len(values)), values, legend_label=label,
                  line_width=2, color=colors[i % 10])
        p.legend.click_policy = "hide"
        return self._bokeh_to_html(p)

    def bokeh_scatter(self, x: np.ndarray, y: np.ndarray,
                      title: str = "", width: int = 600, height: int = 500) -> Optional[str]:
        if not BOKEH_AVAILABLE: return None
        source = ColumnDataSource(data=dict(x=x, y=y))
        p = figure(title=title, width=width, height=height,
                  tools="pan,wheel_zoom,box_zoom,reset,hover,save")
        p.circle('x', 'y', source=source, size=6, alpha=0.6)
        p.grid.grid_line_alpha = 0.3
        return self._bokeh_to_html(p)

    def bokeh_histogram(self, data: np.ndarray, bins: int = 30,
                        title: str = "", width: int = 600, height: int = 400) -> Optional[str]:
        if not BOKEH_AVAILABLE: return None
        hist, edges = np.histogram(data, bins=bins)
        p = figure(title=title, width=width, height=height,
                  tools="pan,wheel_zoom,save")
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
              fill_color="steelblue", line_color="white", alpha=0.7)
        p.grid.grid_line_alpha = 0.3
        return self._bokeh_to_html(p)

    def _bokeh_to_html(self, p: Any) -> str:
        from bokeh.resources import CDN
        from bokeh.embed import file_html
        return file_html(p, CDN, "plot")

    # ── Altair ────────────────────────────────────────────────────────

    def altair_scatter(self, df: pd.DataFrame, x: str, y: str,
                       color: str = None, title: str = "") -> Optional[str]:
        if not ALTAR_AVAILABLE: return None
        chart = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
            x=x, y=y, color=color if color else alt.value("steelblue")
        ).properties(title=title, width=500, height=400).interactive()
        return chart.to_json()

    def altair_line(self, df: pd.DataFrame, x: str, y: str,
                    color: str = None, title: str = "") -> Optional[str]:
        if not ALTAR_AVAILABLE: return None
        chart = alt.Chart(df).mark_line().encode(
            x=x, y=y, color=color if color else alt.value("steelblue")
        ).properties(title=title, width=600, height=400).interactive()
        return chart.to_json()

    def altair_heatmap(self, df: pd.DataFrame, x: str, y: str, color: str,
                       title: str = "") -> Optional[str]:
        if not ALTAR_AVAILABLE: return None
        chart = alt.Chart(df).mark_rect().encode(
            x=alt.X(x, bin=True), y=alt.Y(y, bin=True),
            color=alt.Color(color, aggregate="count")
        ).properties(title=title, width=500, height=400).interactive()
        return chart.to_json()

    def altair_boxplot(self, df: pd.DataFrame, x: str, y: str,
                       title: str = "") -> Optional[str]:
        if not ALTAR_AVAILABLE: return None
        chart = alt.Chart(df).mark_boxplot().encode(
            x=x, y=alt.Y(y, scale=alt.Scale(zero=False))
        ).properties(title=title, width=400, height=400).interactive()
        return chart.to_json()

    # ── NetworkX ──────────────────────────────────────────────────────

    def network_graph(self, edges: List[Tuple[int,int]],
                      labels: Dict[int,str] = None,
                      title: str = "Network Graph",
                      figsize: Tuple[int,int] = (8,6)) -> Figure:
        if not NETWORKX_AVAILABLE: return self.mpl_scatter([0], [0])
        G = nx.Graph()
        G.add_edges_from(edges)
        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=labels is not None, labels=labels,
                node_color="steelblue", node_size=500, edge_color="gray",
                font_size=10, ax=ax)
        ax.set_title(title); fig.tight_layout()
        return fig

    def pyvis_graph(self, edges: List[Tuple[int,int]],
                    labels: Dict[int,str] = None,
                    title: str = "Interactive Network") -> Optional[str]:
        if not PYVIS_AVAILABLE: return None
        net = Network(height="500px", width="100%", notebook=False, heading=title)
        for a, b in edges:
            net.add_node(a, label=str(labels.get(a, a)) if labels else str(a))
            net.add_node(b, label=str(labels.get(b, b)) if labels else str(b))
            net.add_edge(a, b)
        net.toggle_physics(True)
        return net.generate_html()

    # ── Statsmodels ────────────────────────────────────────────────────

    def ts_decomposition(self, series: np.ndarray, period: int = 12,
                         model: str = "additive",
                         figsize: Tuple[int,int] = (10,8)) -> Optional[Figure]:
        if not STATSMODELS_AVAILABLE: return None
        ts = pd.Series(series)
        result = seasonal_decompose(ts, model=model, period=period)
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        result.observed.plot(ax=axes[0]); axes[0].set_title("Observed")
        result.trend.plot(ax=axes[1]); axes[1].set_title("Trend")
        result.seasonal.plot(ax=axes[2]); axes[2].set_title("Seasonal")
        result.resid.plot(ax=axes[3]); axes[3].set_title("Residual")
        fig.suptitle("Time Series Decomposition", fontsize=14)
        fig.tight_layout()
        return fig

    def ts_acf_pacf(self, series: np.ndarray, lags: int = 40,
                    figsize: Tuple[int,int] = (12,4)) -> Optional[Figure]:
        if not STATSMODELS_AVAILABLE: return None
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        plot_acf(pd.Series(series), lags=lags, ax=axes[0])
        plot_pacf(pd.Series(series), lags=lags, ax=axes[1])
        fig.tight_layout()
        return fig

    def qq_plot(self, data: np.ndarray, figsize: Tuple[int,int] = (6,6)) -> Optional[Figure]:
        if not STATSMODELS_AVAILABLE: return None
        fig, ax = plt.subplots(figsize=figsize)
        sm.qqplot(data, line='s', ax=ax)
        ax.set_title("Q-Q Plot"); fig.tight_layout()
        return fig

    # ── Folium ────────────────────────────────────────────────────────

    def folium_map(self, locations: List[Tuple[float, float]],
                   popups: List[str] = None, center: Tuple[float,float] = (20,0),
                   zoom: int = 2) -> Optional[str]:
        if not FOLIUM_AVAILABLE: return None
        m = folium.Map(location=center, zoom_start=zoom)
        for i, (lat, lon) in enumerate(locations):
            pop = popups[i] if popups else f"Point {i}"
            folium.Marker([lat, lon], popup=pop,
                         icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
        return m._repr_html_()

    def folium_heatmap(self, locations: List[Tuple[float, float]],
                       weights: List[float] = None,
                       center: Tuple[float,float] = (20,0),
                       zoom: int = 2) -> Optional[str]:
        if not FOLIUM_AVAILABLE: return None
        from folium.plugins import HeatMap
        m = folium.Map(location=center, zoom_start=zoom)
        data = [[loc[0], loc[1], w] for loc, w in
                zip(locations, weights if weights else [1]*len(locations))]
        HeatMap(data, radius=15, blur=10).add_to(m)
        return m._repr_html_()

    # ── Generic ───────────────────────────────────────────────────────

    def fig_to_base64(self, fig: Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode()

    def fig_to_html(self, fig: Figure) -> str:
        b64 = self.fig_to_base64(fig)
        return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:900px"/>'

    def to_streamlit(self, fig: Figure):
        import streamlit as st
        st.pyplot(fig)

    def to_streamlit_plotly(self, fig_json: str):
        import streamlit as st
        import plotly.io as pio
        fig = pio.from_json(fig_json)
        st.plotly_chart(fig, use_container_width=True)