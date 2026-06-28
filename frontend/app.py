"""CI-Lib Visualization Suite — Streamlit Frontend."""

from __future__ import annotations

import json
import subprocess
import urllib.request
import urllib.error
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyBboxPatch

# ── Viz engine ──────────────────────────────────────────────────────
from ci_lib.viz.python_viz import PythonVisualizationEngine, PLOTLY_AVAILABLE
viz = PythonVisualizationEngine()

# ── Tank & Dozer modules ────────────────────────────────────────────
from tankdozer import __app_name__, __version__

# ══════════════════════════════════════════════════════════════════════
# FUTURISTIC CYBERPUNK THEME
# ══════════════════════════════════════════════════════════════════════
FUTURISTIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Fira+Code:wght@300;400;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a1a2e 100%);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
}

h1 {
    background: linear-gradient(90deg, #00f0ff, #ff00ff, #00f0ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
}

@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

h2 {
    color: #00f0ff !important;
    border-bottom: 1px solid rgba(0, 240, 255, 0.3);
    padding-bottom: 6px;
}

h3 {
    color: #ff00ff !important;
}

.stMarkdown, p, li, span, div {
    color: #c0c0e0 !important;
}

/* ── Cards & Containers ─────────────────────────────────── */
.css-1r6slb0, .css-12oz5g7, .stTabs [data-baseweb="tab-panel"], .css-1offfwp, .css-1v3fvcr {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(0, 240, 255, 0.12);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
}

/* ── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(0, 240, 255, 0.15);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 1px;
    color: #8888aa !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(255, 0, 255, 0.2)) !important;
    color: #00f0ff !important;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(90deg, #00f0ff, #0088ff) !important;
    border: none !important;
    color: #000 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-radius: 8px;
    transition: all 0.3s ease;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 35px rgba(0, 240, 255, 0.5) !important;
}
.stButton > button:active {
    transform: translateY(0px);
}

/* Danger button override */
div[data-testid="stButton"][kind="secondary"] > button {
    background: linear-gradient(90deg, #ff0044, #ff0088) !important;
    box-shadow: 0 0 20px rgba(255, 0, 68, 0.3);
}

/* ── Sidebar ─────────────────────────────────────────────── */
.css-1d391kg, .css-1wrcr25, section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1f 0%, #150626 100%) !important;
    border-right: 1px solid rgba(0, 240, 255, 0.15);
}

section[data-testid="stSidebar"] .css-1v3fvcr {
    background: rgba(255, 255, 255, 0.02) !important;
    border: none !important;
    backdrop-filter: none !important;
    box-shadow: none !important;
}

.sidebar-header {
    font-family: 'Orbitron', sans-serif;
    background: linear-gradient(90deg, #00f0ff, #ff00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    font-size: 18px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.sidebar-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00f0ff, #ff00ff, transparent);
    margin: 12px 0;
}

/* ── Metric Cards ────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: rgba(0, 240, 255, 0.05);
    border: 1px solid rgba(0, 240, 255, 0.15);
    border-radius: 10px;
    padding: 12px;
    backdrop-filter: blur(4px);
}

div[data-testid="stMetric"] > div:first-child {
    font-family: 'Orbitron', sans-serif !important;
    color: #ff00ff !important;
    font-size: 11px !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}

div[data-testid="stMetric"] > div:first-child + div {
    font-family: 'Fira Code', monospace !important;
    color: #00f0ff !important;
    font-size: 24px !important;
    font-weight: 700;
    text-shadow: 0 0 15px rgba(0, 240, 255, 0.5);
}

/* ── Text Input / Select / Slider ────────────────────────── */
.stTextInput > div > div, .stSelectbox > div > div, .stSlider > div {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(0, 240, 255, 0.2) !important;
    border-radius: 8px !important;
    color: #00f0ff !important;
}

.stTextInput > div > div:focus-within, .stSelectbox > div > div:focus-within {
    border-color: #ff00ff !important;
    box-shadow: 0 0 15px rgba(255, 0, 255, 0.2) !important;
}

.stSlider label {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 11px !important;
    color: #8888aa !important;
}

/* ── Terminal output box ─────────────────────────────────── */
.terminal-box {
    background: #0a0a0a;
    border: 1px solid #00f0ff;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Fira Code', 'Courier New', monospace;
    font-size: 13px;
    color: #00f0ff;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    box-shadow: inset 0 0 30px rgba(0, 240, 255, 0.05), 0 0 20px rgba(0, 240, 255, 0.1);
}

.terminal-box::-webkit-scrollbar {
    width: 6px;
}
.terminal-box::-webkit-scrollbar-track {
    background: #0a0a0a;
}
.terminal-box::-webkit-scrollbar-thumb {
    background: #00f0ff;
    border-radius: 3px;
}

.terminal-prompt::before {
    content: "❯ ";
    color: #ff00ff;
    font-weight: bold;
}

/* ── Info / Success / Error boxes ────────────────────────── */
.stAlert {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(0, 240, 255, 0.15) !important;
    border-radius: 8px !important;
    backdrop-filter: blur(4px);
}

/* ── Dividers ────────────────────────────────────────────── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00f0ff, #ff00ff, transparent);
    margin: 20px 0;
}

/* ── Progress bar ────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #00f0ff, #ff00ff) !important;
}

/* ── Text area ───────────────────────────────────────────── */
.stTextArea textarea {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(0, 240, 255, 0.2) !important;
    color: #00f0ff !important;
    border-radius: 8px !important;
    font-family: 'Fira Code', monospace !important;
}

/* ── Remove plain white background on metric values ─────── */
div[data-testid="stMetricLabel"] p {
    color: #ff00ff !important;
}

/* ── Info callout ────────────────────────────────────────── */
div.stInfo {
    background: rgba(0, 240, 255, 0.08) !important;
    border-left: 3px solid #00f0ff !important;
}

div.stError {
    background: rgba(255, 0, 68, 0.1) !important;
    border-left: 3px solid #ff0044 !important;
}

div.stSuccess {
    background: rgba(0, 255, 65, 0.08) !important;
    border-left: 3px solid #00ff41 !important;
}

/* ── Expander ────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: 'Orbitron', sans-serif !important;
    color: #00f0ff !important;
    background: rgba(0, 240, 255, 0.03) !important;
    border-radius: 8px !important;
}

/* ── Checkbox ────────────────────────────────────────────── */
.stCheckbox label {
    color: #c0c0e0 !important;
}

/* ── DataFrames ──────────────────────────────────────────── */
.stDataFrame {
    background: transparent !important;
}
.stDataFrame table {
    color: #c0c0e0 !important;
}
.stDataFrame thead tr th {
    background: rgba(0, 240, 255, 0.1) !important;
    color: #00f0ff !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 11px !important;
}
.stDataFrame tbody tr:nth-child(even) {
    background: rgba(255, 255, 255, 0.02) !important;
}

/* ── Spinner ─────────────────────────────────────────────── */
.stSpinner {
    color: #00f0ff !important;
}

/* ── Number input ────────────────────────────────────────── */
.stNumberInput input {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(0, 240, 255, 0.2) !important;
    color: #00f0ff !important;
    border-radius: 8px !important;
}

/* ── Pulsing action button for Tank & Dozer ──────────────── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 0, 255, 0.3); }
    50% { box-shadow: 0 0 40px rgba(255, 0, 255, 0.7); }
}

.td-run-button > button {
    animation: pulse-glow 2s ease-in-out infinite;
    background: linear-gradient(90deg, #ff00ff, #ff0088) !important;
    font-size: 14px !important;
    padding: 10px 30px !important;
}

/* ── Quick action sidebar buttons ────────────────────────── */
.td-quick-btn > button {
    font-size: 10px !important;
    padding: 4px 8px !important;
    letter-spacing: 1px !important;
}
"""


# ── Configuration ────────────────────────────────────────────────────

st.set_page_config(page_title="CI-Lib Viz Suite", layout="wide")

# Inject futuristic CSS
st.markdown(f"<style>{FUTURISTIC_CSS}</style>", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────

st.sidebar.markdown(
    '<div class="sidebar-header">⬡ CI-Lib</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div style="font-family: Fira Code, monospace; font-size: 11px; color: #8888aa;">'
    'Computational Intelligence Suite</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

API_BASE = st.sidebar.text_input(
    "API Endpoint", value="http://localhost:8000/api",
    help="FastAPI backend URL",
)

LIB_STATUS = viz.list_libraries()
st.sidebar.markdown(
    f'<div style="font-family: Fira Code, monospace; font-size: 11px; color: #8888aa;">'
    f'Libraries: {", ".join(LIB_STATUS) if LIB_STATUS else "matplotlib"}</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="font-family: Orbitron, sans-serif; font-size: 12px; color: #00f0ff; '
    'letter-spacing: 2px;">⚡ QUICK ACTIONS</div>',
    unsafe_allow_html=True,
)

def _queue_td(module: str, cmd: str, args: str = ""):
    st.session_state.td_module = module
    st.session_state.td_cmd = cmd
    st.session_state.td_args = args
    st.session_state.td_auto_run = True
    st.toast(f"Switch to 'Tank & Dozer' tab to run: {module} {cmd}", icon="⚡")

td_col1, td_col2 = st.sidebar.columns(2)
with td_col1:
    if st.button("🚀 IR Init", key="side_ir_init", help="tankdozer ir init"):
        _queue_td("ir", "init", "--severity medium")
with td_col2:
    if st.button("📊 Status", key="side_ir_status", help="tankdozer ir status"):
        _queue_td("ir", "status")

td_col3, td_col4 = st.sidebar.columns(2)
with td_col3:
    if st.button("🔍 Quick Scan", key="side_scan", help="tankdozer scan quick"):
        _queue_td("scan", "quick")
with td_col4:
    if st.button("📄 Report", key="side_report", help="tankdozer report generate"):
        _queue_td("report", "generate")

st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="font-family: Orbitron, sans-serif; font-size: 11px; color: #8888aa; '
    'letter-spacing: 1px;">▸ LINKS</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<a href="http://localhost:8000/docs" target="_blank" '
    'style="color: #00f0ff; text-decoration: none; font-family: Fira Code, monospace; '
    'font-size: 12px;">📡 API Docs</a>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<a href="https://github.com/WilliamMajanja/CI-Lib" target="_blank" '
    'style="color: #ff00ff; text-decoration: none; font-family: Fira Code, monospace; '
    'font-size: 12px;">⌨ GitHub</a>',
    unsafe_allow_html=True,
)

# Initialize session state for Tank & Dozer
if "td_module" not in st.session_state:
    st.session_state.td_module = "ir"
    st.session_state.td_cmd = "init"
    st.session_state.td_args = ""
    st.session_state.td_output = ""
    st.session_state.td_auto_run = False


def api_call(method: str, path: str, body: dict | None = None) -> Any:
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        st.error(f"API Error {e.code}: {detail}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def run_tankdozer(module: str, command: str, extra_args: str = "") -> str:
    """Run a tankdozer CLI command and return its output."""
    cmd = ["tankdozer", module, command]
    if extra_args:
        cmd.extend(extra_args.split())
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            output += f"\n[!] Exit code: {result.returncode}"
        return output
    except FileNotFoundError:
        return "[!] tankdozer not found. Run: pip install -e ."
    except subprocess.TimeoutExpired:
        return "[!] Command timed out after 30s"
    except Exception as e:
        return f"[!] Error: {e}"


# ── Page layout ──────────────────────────────────────────────────────

st.markdown(
    '<h1 style="text-align: center; font-size: 32px; margin-bottom: 4px;">'
    '⬡ CI-Lib Visualization Suite</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align: center; font-family: Fira Code, monospace; '
    'color: #8888aa; font-size: 13px; letter-spacing: 3px;">'
    'COMPUTATIONAL INTELLIGENCE // ADVANCED ANALYTICS // SECURITY OPS</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

tab_names = [
    "Clustering", "Evolutionary", "Swarm", "Fuzzy Inference",
    "Neural Network", "Optimization", "Utilities", "Benchmarks",
    "Viz Gallery", "Interactive", "Statistical", "Network",
    "Geospatial", "3D Plots", "Tank & Dozer",
]

tabs = st.tabs(tab_names)

# =====================================================================
# TAB 1 — Clustering
# =====================================================================

with tabs[0]:
    st.header("Clustering")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("K-Means")
        n_clusters = st.slider("Clusters", 2, 10, 3, key="km_k")
        init_method = st.selectbox("Init", ["kmeans++", "random"], key="km_init")
        n_samples_km = st.slider("Samples", 20, 500, 150, key="km_n")
        seed_km = st.number_input("Seed", 0, 9999, 42, key="km_seed")
        if st.button("Run K-Means", key="btn_km"):
            rng = np.random.default_rng(seed_km)
            X = np.vstack([
                rng.normal(loc=(2, 2), scale=0.8, size=(n_samples_km // 3, 2)),
                rng.normal(loc=(6, 6), scale=0.8, size=(n_samples_km // 3, 2)),
                rng.normal(loc=(2, 6), scale=0.8, size=(n_samples_km // 3, 2)),
            ])
            r = api_call("POST", "/clustering/kmeans", {
                "n_clusters": n_clusters, "init": init_method,
                "seed": int(seed_km), "data": X.tolist(),
            })
            if r:
                st.metric("Inertia", f"{r['inertia']:.3f}")
                if r["silhouette"] is not None:
                    st.metric("Silhouette", f"{r['silhouette']:.3f}")
                fig = plot_2d_decision_space(
                    X, np.array(r["labels"]),
                    centers=np.array(r["centers"]),
                    title=f"K-Means (k={n_clusters})",
                )
                st.pyplot(fig)

    with col2:
        st.subheader("DBSCAN")
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.05, key="db_eps")
        min_pts = st.slider("Min Samples", 2, 20, 5, key="db_min")
        n_samples_db = st.slider("Samples", 20, 500, 200, key="db_n")
        seed_db = st.number_input("Seed", 0, 9999, 0, key="db_seed")
        if st.button("Run DBSCAN", key="btn_db"):
            rng = np.random.default_rng(seed_db)
            X = np.vstack([
                rng.normal(loc=(3, 3), scale=0.5, size=(n_samples_db // 2, 2)),
                rng.normal(loc=(7, 3), scale=0.5, size=(n_samples_db // 2, 2)),
                rng.uniform(0, 10, size=(max(5, n_samples_db // 10), 2)),
            ])
            r = api_call("POST", "/clustering/dbscan", {
                "eps": eps, "min_samples": int(min_pts), "data": X.tolist(),
            })
            if r:
                st.metric("Clusters Found", r["n_clusters"])
                noise = int(np.sum(np.array(r["labels"]) == -1))
                st.metric("Noise Points", noise)
                fig = plot_2d_decision_space(
                    X, np.array(r["labels"]),
                    title=f"DBSCAN (eps={eps}, min={min_pts})",
                )
                st.pyplot(fig)

# =====================================================================
# TAB 2 — Evolutionary
# =====================================================================
with tabs[1]:
    st.header("Evolutionary Algorithms")
    col1, col2 = st.columns(2)
    for col, algo_name, algo_key, extra_params in [
        (col1, "Genetic Algorithm", "ga", {}),
        (col2, "Differential Evolution", "de", {}),
    ]:
        with col:
            st.subheader(algo_name)
            bfn = st.selectbox("Benchmark", ["sphere","rosenbrock","rastrigin","ackley"], key=f"{algo_key}_bfn")
            nd = st.slider("Dimensions", 2, 10, 2, key=f"{algo_key}_nd")
            pop = st.slider("Population", 10, 200, 50, key=f"{algo_key}_pop")
            gen = st.slider("Generations", 10, 500, 100, key=f"{algo_key}_gen")
            sd = st.number_input("Seed", 0, 9999, 42, key=f"{algo_key}_sd")
            bnd = st.text_input("Bounds", "-5,5", key=f"{algo_key}_bnd")
            if algo_key == "ga":
                mr = st.slider("Mutation Rate", 0.0, 1.0, 0.1, key="ga_mr")
                cr = st.slider("Crossover Rate", 0.0, 1.0, 0.8, key="ga_cr")
                el = st.slider("Elitism", 0, 10, 2, key="ga_el")
                btn = st.button("Run GA", key="btn_ga")
                params = {"mutation_rate": mr, "crossover_rate": cr, "elitism": int(el)}
            else:
                F = st.slider("F (mutation)", 0.1, 2.0, 0.8, 0.05, key="de_F")
                CR = st.slider("CR (crossover)", 0.0, 1.0, 0.9, key="de_CR")
                strat = st.selectbox("Strategy", ["best/1/bin", "rand/1/bin"], key="de_strat")
                btn = st.button("Run DE", key="btn_de")
                params = {"F": F, "CR": CR, "strategy": strat}

            if btn:
                try:
                    bounds = [float(x) for x in bnd.split(",")]
                except ValueError:
                    st.error("Bounds must be comma-separated numbers, e.g. -5,5"); continue
                endpoint = "/evolutionary/genetic" if algo_key == "ga" else "/evolutionary/differential"
                r = api_call("POST", endpoint, {
                    "n_dims": int(nd), "pop_size": int(pop), "generations": int(gen),
                    "benchmark_fn": bfn, "bounds": [bounds] * nd, "seed": int(sd), **params,
                })
                if r:
                    st.metric("Best Fitness", f"{r['best_fitness']:.6e}")
                    st.text(f"Solution: {[f'{v:.4f}' for v in r['best_solution']]}")
                    fig = plot_convergence(r["history"], f"{algo_name} Convergence")
                    st.pyplot(fig)

# =====================================================================
# TAB 3 — Swarm
# =====================================================================
with tabs[2]:
    st.header("Swarm Intelligence")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Particle Swarm Optimization")
        bfn = st.selectbox("Benchmark", ["sphere","rosenbrock","rastrigin","ackley"], key="pso_bfn")
        nd = st.slider("Dimensions", 2, 10, 2, key="pso_nd")
        np_ = st.slider("Particles", 5, 200, 30, key="pso_np")
        it = st.slider("Iterations", 10, 500, 100, key="pso_it")
        w = st.slider("Inertia (w)", 0.1, 1.5, 0.7, 0.05, key="pso_w")
        c1 = st.slider("Cognitive (c1)", 0.0, 3.0, 1.5, 0.1, key="pso_c1")
        c2 = st.slider("Social (c2)", 0.0, 3.0, 1.5, 0.1, key="pso_c2")
        sd = st.number_input("Seed", 0, 9999, 42, key="pso_sd")
        if st.button("Run PSO", key="btn_pso"):
            r = api_call("POST", "/swarm/pso", {
                "n_dims": int(nd), "n_particles": int(np_), "generations": int(it),
                "benchmark_fn": bfn, "bounds": [[-5,5]]*nd, "w": w, "c1": c1, "c2": c2, "seed": int(sd),
            })
            if r:
                st.metric("Best Fitness", f"{r['best_fitness']:.6e}")
                fig = plot_convergence(r["history"], "PSO Convergence"); st.pyplot(fig)
    with col2:
        st.subheader("Ant Colony Optimization (TSP)")
        nc = st.slider("Cities", 5, 30, 8, key="aco_nc")
        na = st.slider("Ants", 2, 100, 20, key="aco_na")
        aco_it = st.slider("Iterations", 10, 200, 50, key="aco_it")
        alpha = st.slider("Alpha", 0.0, 5.0, 1.0, 0.1, key="aco_alpha")
        beta = st.slider("Beta", 0.0, 5.0, 2.0, 0.1, key="aco_beta")
        evap = st.slider("Evaporation", 0.05, 0.95, 0.5, 0.05, key="aco_evap")
        sd = st.number_input("Seed", 0, 9999, 42, key="aco_sd")
        if st.button("Run ACO", key="btn_aco"):
            r = api_call("POST", "/swarm/aco", {
                "n_ants": int(na), "n_cities": int(nc), "iterations": int(aco_it),
                "alpha": alpha, "beta": beta, "evaporation": evap, "seed": int(sd),
            })
            if r:
                st.metric("Best Distance", f"{r['best_distance']:.3f}")
                coords = np.array(r.get("city_coordinates", r["distance_matrix"]))
                fig, ax = plt.subplots(figsize=(5,5))
                route = r["best_route"]
                ax.scatter(coords[:,0], coords[:,1], c="steelblue", s=80, zorder=5)
                for i in range(len(route)):
                    a, b = route[i], route[(i+1)%len(route)]
                    ax.plot([coords[a,0], coords[b,0]], [coords[a,1], coords[b,1]],
                           color="crimson", linewidth=1.2, alpha=0.7)
                for i, (x,y) in enumerate(coords):
                    ax.annotate(str(i), (x,y), fontsize=9, fontweight="bold")
                ax.set_title(f"ACO Tour (dist={r['best_distance']:.2f})")
                ax.set_aspect("equal"); st.pyplot(fig)

# =====================================================================
# TAB 4 — Fuzzy Inference
# =====================================================================
with tabs[3]:
    st.header("Fuzzy Inference System")
    st.markdown("**Tipping Problem** — rate service (0–10) and food (0–10).")
    col1, col2 = st.columns(2)
    with col1:
        svc = st.slider("Service Rating", 0.0, 10.0, 6.0, 0.5)
        fd = st.slider("Food Quality", 0.0, 10.0, 7.0, 0.5)
        method = st.selectbox("Defuzzify", ["centroid","bisector","max_membership"])
    with col2:
        if st.button("Compute Tip", key="btn_fis"):
            r = api_call("POST", "/fuzzy/default-fis", {"inputs": {"service": svc, "food": fd}, "method": method})
            if r:
                tip_val = r["output"]["tip"]
                st.metric("Suggested Tip", f"{tip_val:.1f}%")
                st.progress(min(tip_val/30, 1.0))

# =====================================================================
# TAB 5 — Neural Network
# =====================================================================
with tabs[4]:
    st.header("Neural Network — XOR Demo")
    col1, col2 = st.columns(2)
    with col1:
        lr = st.slider("Learning Rate", 0.001, 1.0, 0.1, 0.005, key="nn_lr")
        epochs = st.slider("Epochs", 100, 5000, 1000, key="nn_ep")
        activation = st.selectbox("Activation", ["sigmoid","tanh","relu","leaky_relu"], key="nn_act")
        batch = st.selectbox("Batch Size", [None,1,2,4],
                            format_func=lambda x:"Full" if x is None else str(x), key="nn_bs")
        sd = st.number_input("Seed", 0, 9999, 42, key="nn_sd")
    with col2:
        if st.button("Train Network", key="btn_nn"):
            X = [[0,0],[0,1],[1,0],[1,1]]; y = [[0],[1],[1],[0]]
            r = api_call("POST", "/neural/train", {
                "layer_sizes": [2,4,1], "activation": activation,
                "learning_rate": lr, "epochs": int(epochs),
                "batch_size": batch, "seed": int(sd), "X": X, "y": y,
            })
            if r:
                st.metric("Initial MSE", f"{r['initial_mse']:.6f}")
                st.metric("Final MSE", f"{r['final_mse']:.6f}")
                pred_r = api_call("POST", "/neural/predict", {"X": X})
                if pred_r:
                    st.subheader("Predictions")
                    for inp, out in zip(X, pred_r["predictions"]):
                        st.text(f"  {inp} -> {out[0]:.4f}")

# =====================================================================
# TAB 6 — Optimization
# =====================================================================
with tabs[5]:
    st.header("Classic Optimization")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Simulated Annealing")
        bfn_sa = st.selectbox("Benchmark", ["sphere","rosenbrock","rastrigin","ackley"], key="sa_bfn")
        nd_sa = st.slider("Dimensions", 2, 10, 2, key="sa_nd")
        init_t = st.slider("Initial Temp", 1.0, 500.0, 100.0, key="sa_t")
        cr_sa = st.slider("Cooling Rate", 0.8, 0.999, 0.99, 0.001, key="sa_cr")
        sched = st.selectbox("Schedule", ["exponential","linear","logarithmic"], key="sa_sched")
        sd_sa = st.number_input("Seed", 0, 9999, 42, key="sa_sd")
        if st.button("Run SA", key="btn_sa"):
            r = api_call("POST", "/optimization/simulated-annealing", {
                "n_dims": int(nd_sa), "benchmark_fn": bfn_sa, "bounds": [[-5,5]]*nd_sa,
                "initial_temp": init_t, "cooling_rate": cr_sa, "cooling_schedule": sched, "seed": int(sd_sa),
            })
            if r:
                st.metric("Best Fitness", f"{r['best_fitness']:.6e}")
                fig = plot_convergence(r["history"], "SA Convergence"); st.pyplot(fig)
    with col2:
        st.subheader("Gradient Descent")
        bfn_gd = st.selectbox("Benchmark", ["sphere","rosenbrock","rastrigin"], key="gd_bfn")
        nd_gd = st.slider("Dimensions", 2, 10, 2, key="gd_nd")
        lr_gd = st.slider("Learning Rate", 0.0001, 1.0, 0.01, 0.0005, key="gd_lr")
        meth = st.selectbox("Method", ["sgd","momentum","adam"], key="gd_meth")
        mom = st.slider("Momentum", 0.0, 0.99, 0.9, 0.01, key="gd_mom")
        sd_gd = st.number_input("Seed", 0, 9999, 42, key="gd_sd")
        if st.button("Run GD", key="btn_gd"):
            r = api_call("POST", "/optimization/gradient-descent", {
                "n_dims": int(nd_gd), "benchmark_fn": bfn_gd,
                "learning_rate": lr_gd, "momentum": mom, "method": meth, "seed": int(sd_gd),
            })
            if r:
                st.metric("Best Value", f"{r['best_fitness']:.6e}")
                fig = plot_convergence(r["history"], f"GD ({meth}) Convergence"); st.pyplot(fig)

# =====================================================================
# TAB 7 — Utilities
# =====================================================================
with tabs[6]:
    st.header("Utilities")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Normalize Data")
        norm_data = st.text_area("Enter data as [[x1,y1],[x2,y2],...]",
                                "[[1,2],[3,4],[5,6],[7,8]]", key="norm_in")
        norm_meth = st.selectbox("Method", ["minmax","zscore"], key="norm_meth")
        if st.button("Normalize", key="btn_norm"):
            try:
                r = api_call("POST", "/utils/normalize", {"data": json.loads(norm_data), "method": norm_meth})
                if r: st.json(r["data"])
            except json.JSONDecodeError: st.error("Invalid JSON")
    with col2:
        st.subheader("Metrics")
        met_yt = st.text_input("y_true", "[0,1,1,0,1]", key="met_yt")
        met_yp = st.text_input("y_pred", "[0,1,0,0,1]", key="met_yp")
        if st.button("Compute", key="btn_met"):
            try:
                r = api_call("POST", "/utils/metrics", {"y_true": json.loads(met_yt), "y_pred": json.loads(met_yp)})
                if r:
                    for k, v in r.items():
                        if v is not None and not isinstance(v, list):
                            st.metric(k.replace("_"," ").title(), f"{v:.4f}" if isinstance(v,float) else v)
            except json.JSONDecodeError: st.error("Invalid JSON")

# =====================================================================
# TAB 8 — Benchmarks
# =====================================================================
with tabs[7]:
    st.header("Benchmark Functions")
    bfns = ["sphere","rosenbrock","rastrigin","ackley","griewank","schwefel"]
    col1, col2 = st.columns(2)
    with col1:
        bfn_name = st.selectbox("Function", bfns, key="bm_fn")
        dims = st.slider("Dimensions", 1, 10, 2, key="bm_dims")
        x_vals = st.text_input("x (comma-separated)", "1.0, 2.0", key="bm_x")
        if st.button("Evaluate", key="btn_bm"):
            try:
                x = [float(v) for v in x_vals.split(",")]
                if len(x) != dims: st.error(f"Expected {dims} values, got {len(x)}")
                else:
                    r = api_call("POST", "/utils/benchmarks/evaluate", {"fn_name": bfn_name, "x": x})
                    if r: st.metric(f"{bfn_name} value", f"{r['value']:.6f}")
            except ValueError: st.error("Invalid x values")
    with col2:
        st.subheader("Quick Visualization")
        if st.button("Plot Landscape (2D)", key="btn_bm_plot"):
            r0 = api_call("GET", "/utils/benchmarks")
            if r0:
                res = 50; x = np.linspace(-5,5,res); y = np.linspace(-5,5,res)
                Xg, Yg = np.meshgrid(x, y); Z = np.zeros_like(Xg)
                for i in range(res):
                    for j in range(res):
                        resp = api_call("POST", "/utils/benchmarks/evaluate",
                                       {"fn_name": bfn_name, "x": [float(Xg[i,j]), float(Yg[i,j])]})
                        if resp: Z[i,j] = resp["value"]
                fig, ax = plt.subplots(figsize=(5,4))
                c = ax.contourf(Xg, Yg, Z, levels=30, cmap="viridis")
                plt.colorbar(c, ax=ax)
                ax.set_title(f"{bfn_name} Landscape"); st.pyplot(fig)

# =====================================================================
# TAB 9 — Viz Gallery
# =====================================================================
with tabs[8]:
    st.header("Visualization Gallery")
    st.markdown("Demo plots using available libraries.")
    gal_col1, gal_col2, gal_col3 = st.columns(3)

    x = np.linspace(0, 4*np.pi, 200)
    y1 = np.sin(x); y2 = np.cos(x); y3 = np.sin(x) * np.cos(x)

    with gal_col1:
        st.subheader("Matplotlib Line")
        fig = viz.mpl_line({"sin(x)": y1, "cos(x)": y2, "sin(x)*cos(x)": y3},
                          title="Trigonometric Functions", xlabel="x", ylabel="y")
        st.pyplot(fig)

    with gal_col2:
        st.subheader("Matplotlib Histogram")
        data = np.random.default_rng(42).normal(0, 1, 1000)
        fig = viz.mpl_hist(data, bins=40, title="Normal Distribution", xlabel="Value")
        st.pyplot(fig)

    with gal_col3:
        st.subheader("Matplotlib Boxplot")
        groups = {f"Group {i}": list(np.random.default_rng(i).normal(0, 1, 100))
                 for i in range(4)}
        fig = viz.mpl_boxplot(groups, title="Group Distributions")
        st.pyplot(fig)

    gal_col4, gal_col5, gal_col6 = st.columns(3)
    with gal_col4:
        st.subheader("Scatter Plot")
        rng = np.random.default_rng(42)
        xs = rng.normal(0, 1, 300); ys = xs * 0.7 + rng.normal(0, 0.5, 300)
        fig = viz.mpl_scatter(xs, ys, title="Correlated Data", xlabel="X", ylabel="Y")
        st.pyplot(fig)

    with gal_col5:
        st.subheader("Correlation Heatmap")
        df = pd.DataFrame(np.random.default_rng(42).normal(0, 1, (50, 6)),
                         columns=list("ABCDEF"))
        fig = viz.mpl_correlation_matrix(df)
        st.pyplot(fig)

    with gal_col6:
        st.subheader("3D Surface")
        Xg, Yg = np.meshgrid(np.linspace(-3,3,40), np.linspace(-3,3,40))
        Z = np.sin(np.sqrt(Xg**2 + Yg**2))
        fig = viz.mpl_3d_surface(Xg, Yg, Z, title="3D Surface Plot")
        st.pyplot(fig)

    if PLOTLY_AVAILABLE:
        st.markdown("---")
        st.subheader("Plotly Interactive")
        fig_json = viz.plotly_line({"sin(x)": y1, "cos(x)": y2}, title="Plotly Interactive")
        if fig_json:
            import plotly.io as pio
            st.plotly_chart(pio.from_json(fig_json), use_container_width=True)

# =====================================================================
# TAB 10 — Interactive (Plotly/Bokeh/Altair)
# =====================================================================
with tabs[9]:
    st.header("Interactive Visualizations")
    int_col1, int_col2 = st.columns(2)

    rng = np.random.default_rng(123)
    n = 200
    df_demo = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n) * 0.5 + rng.normal(0, 0.5, n),
        "category": np.random.choice(list("ABC"), n),
        "value": rng.exponential(1, n),
    })

    with int_col1:
        if PLOTLY_AVAILABLE and st.button("Generate Interactive Scatter (Plotly)", key="btn_ply_scatter"):
            import plotly.express as px
            fig = px.scatter(df_demo, x="x", y="y", color="category",
                           title="Interactive Scatter", template="plotly_white",
                           opacity=0.7, size="value")
            st.plotly_chart(fig, use_container_width=True)

        if st.button("Generate Interactive 3D Scatter (Plotly)", key="btn_ply_3d"):
            import plotly.express as px
            df_3d = pd.DataFrame({
                "x": rng.normal(0, 1, 150), "y": rng.normal(0, 1, 150),
                "z": rng.normal(0, 1, 150),
                "group": np.random.choice(list("ABCD"), 150),
            })
            fig = px.scatter_3d(df_3d, x="x", y="y", z="z", color="group",
                              title="3D Interactive Scatter", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with int_col2:
        if st.button("Generate Histogram (Plotly)", key="btn_ply_hist"):
            import plotly.express as px
            fig = px.histogram(df_demo, x="value", color="category", marginal="box",
                             title="Distribution by Category", template="plotly_white",
                             barmode="overlay", opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("Generate Boxplot (Plotly)", key="btn_ply_box"):
            import plotly.express as px
            fig = px.box(df_demo, x="category", y="value", color="category",
                        title="Boxplot by Category", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    int_col3, int_col4 = st.columns(2)
    with int_col3:
        if st.button("Sunburst Chart (Plotly)", key="btn_sunburst"):
            import plotly.express as px
            data = dict(
                character=["Root", "A", "B", "C", "A1", "A2", "B1", "B2", "C1", "C2"],
                parent=["", "Root", "Root", "Root", "A", "A", "B", "B", "C", "C"],
                value=[0, 10, 20, 30, 5, 5, 10, 10, 15, 15],
            )
            fig = px.sunburst(data, names='character', parents='parent', values='value',
                            title="Hierarchical Sunburst")
            st.plotly_chart(fig, use_container_width=True)

    with int_col4:
        if st.button("Parallel Coordinates (Plotly)", key="btn_parallel"):
            import plotly.express as px
            df_para = pd.DataFrame({
                "sepal_length": rng.uniform(4, 8, 100),
                "sepal_width": rng.uniform(2, 4.5, 100),
                "petal_length": rng.uniform(1, 7, 100),
                "petal_width": rng.uniform(0.1, 2.5, 100),
                "species": np.random.choice(["setosa","versicolor","virginica"], 100),
            })
            fig = px.parallel_coordinates(df_para, color="species",
                                        title="Parallel Coordinates",
                                        template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 11 — Statistical
# =====================================================================
with tabs[10]:
    st.header("Statistical Visualizations")
    st.markdown("Time series decomposition, ACF/PACF, Q-Q plots, and more.")

    stat_col1, stat_col2 = st.columns(2)

    rng = np.random.default_rng(42)
    t = np.arange(200)
    trend = 0.05 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 0.5, 200)
    series = trend + seasonal + noise

    with stat_col1:
        if st.button("Time Series Decomposition", key="btn_ts_decomp"):
            fig = viz.ts_decomposition(series, period=12)
            if fig: st.pyplot(fig)
            else: st.info("Install statsmodels for time series decomposition")

        if st.button("Density + Histogram (Seaborn)", key="btn_density"):
            fig, ax = plt.subplots(figsize=(8,4))
            import seaborn as sns
            sns.histplot(series, kde=True, bins=30, ax=ax)
            ax.set_title("Distribution with KDE"); st.pyplot(fig)

        if st.button("ECDF Plot", key="btn_ecdf"):
            fig, ax = plt.subplots(figsize=(7,4))
            import seaborn as sns
            sns.ecdfplot(series, ax=ax)
            ax.set_title("Empirical Cumulative Distribution"); st.pyplot(fig)

    with stat_col2:
        if st.button("ACF / PACF Plots", key="btn_acf"):
            fig = viz.ts_acf_pacf(series, lags=30)
            if fig: st.pyplot(fig)
            else: st.info("Install statsmodels for ACF/PACF plots")

        if st.button("Q-Q Plot", key="btn_qq"):
            fig = viz.qq_plot(series)
            if fig: st.pyplot(fig)
            else: st.info("Install statsmodels for Q-Q plots")

    stat_col3, stat_col4 = st.columns(2)
    with stat_col3:
        if st.button("Violin Plot", key="btn_violin"):
            groups = {f"Group {i}": list(np.random.default_rng(i).normal(i, 0.5, 100))
                     for i in range(4)}
            fig = viz.mpl_violin(groups, title="Violin Plot Comparison")
            st.pyplot(fig)

    with stat_col4:
        if st.button("Joint Plot (Seaborn)", key="btn_joint"):
            fig, ax = plt.subplots(figsize=(7,6))
            import seaborn as sns
            df_joint = pd.DataFrame({"x": rng.normal(0, 1, 300),
                                    "y": rng.normal(0.5, 1, 300)})
            sns.histplot(df_joint, x="x", y="y", ax=ax, bins=30, cmap="viridis", cbar=True)
            ax.set_title("Bivariate Histogram"); st.pyplot(fig)

# =====================================================================
# TAB 12 — Network
# =====================================================================
with tabs[11]:
    st.header("Network / Graph Visualizations")
    net_col1, net_col2 = st.columns(2)

    edges = [(0,1),(1,2),(2,3),(3,0),(0,4),(1,5),(2,6),(3,7),
             (4,5),(5,6),(6,7),(7,4),(0,5),(2,7)]
    labels = {i: f"Node {i}" for i in range(8)}

    with net_col1:
        if st.button("Static Network Graph (NetworkX)", key="btn_nx"):
            fig = viz.network_graph(edges, labels, title="Network Graph")
            st.pyplot(fig)

        st.subheader("Custom Network")
        net_size = st.slider("Nodes", 5, 50, 12, key="net_nodes")
        edge_prob = st.slider("Edge Probability", 0.1, 1.0, 0.3, 0.05, key="net_prob")
        if st.button("Generate Random Network", key="btn_rand_net"):
            import networkx as nx
            G = nx.erdos_renyi_graph(net_size, edge_prob, seed=42)
            fig, ax = plt.subplots(figsize=(8,6))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color="steelblue",
                    node_size=400, edge_color="gray", font_size=9, ax=ax)
            ax.set_title(f"Random Graph (n={net_size}, p={edge_prob})")
            st.pyplot(fig)

    with net_col2:
        if st.button("Interactive Network (PyVis)", key="btn_pyvis"):
            html = viz.pyvis_graph(edges, labels, title="Interactive Network")
            if html:
                st.components.v1.html(html, height=500)
            else:
                st.info("Install pyvis for interactive network graphs: pip install pyvis")

    st.markdown("---")
    net_col3, net_col4 = st.columns(2)
    with net_col3:
        st.subheader("Degree Distribution")
        if st.button("Plot Degree Distribution", key="btn_degree"):
            import networkx as nx
            G = nx.erdos_renyi_graph(100, 0.15, seed=42)
            degrees = [d for _, d in G.degree()]
            fig, ax = plt.subplots(figsize=(7,4))
            ax.hist(degrees, bins=20, edgecolor="white", alpha=0.7)
            ax.set_title("Degree Distribution"); ax.set_xlabel("Degree"); ax.set_ylabel("Count")
            st.pyplot(fig)

    with net_col4:
        st.subheader("Community Detection")
        if st.button("Detect Communities", key="btn_communities"):
            import networkx as nx
            G = nx.karate_club_graph()
            communities = nx.community.louvain_communities(G, seed=42)
            node_colors = np.zeros(len(G))
            for i, comm in enumerate(communities):
                for node in comm:
                    node_colors[node] = i
            fig, ax = plt.subplots(figsize=(8,6))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, node_color=node_colors, cmap="Set2",
                   with_labels=True, node_size=300, ax=ax)
            ax.set_title(f"Zachary Karate Club ({len(communities)} communities)")
            st.pyplot(fig)

# =====================================================================
# TAB 13 — Geospatial
# =====================================================================
with tabs[12]:
    st.header("Geospatial Visualizations")
    st.markdown("World map visualizations using available libraries.")

    geo_col1, geo_col2 = st.columns(2)

    with geo_col1:
        st.subheader("World Cities Demo")
        if st.button("Show World Cities (Folium)", key="btn_folium"):
            cities = [
                (40.7128, -74.0060, "New York"),
                (51.5074, -0.1278, "London"),
                (35.6762, 139.6503, "Tokyo"),
                (-33.8688, 151.2093, "Sydney"),
                (55.7558, 37.6173, "Moscow"),
                (19.0760, 72.8777, "Mumbai"),
                (-23.5505, -46.6333, "Sao Paulo"),
                (31.2304, 121.4737, "Shanghai"),
                (28.6139, 77.2090, "Delhi"),
                (1.3521, 103.8198, "Singapore"),
            ]
            html = viz.folium_map([(lat, lon) for lat, lon, _ in cities],
                                 popups=[name for _, _, name in cities],
                                 center=(20, 0), zoom=2)
            if html:
                st.components.v1.html(html, height=500)
            else:
                st.info("Install folium for maps: pip install folium")

    with geo_col2:
        st.subheader("Quick Geo Viz")
        st.markdown("""
        **Available libraries:**  
        - `folium` — Leaflet maps  
        - `geopandas` — Shapefile/GeoJSON support  
        - `contextily` — Basemaps
        
        Install with:  
        ```bash
        pip install folium geopandas contextily
        ```
        """)

    geo_col3, geo_col4 = st.columns(2)
    with geo_col3:
        if st.button("Coordinate Points Map", key="btn_coords"):
            import numpy as np
            rng = np.random.default_rng(42)
            coords = []
            names = []
            continents = {
                "NA": (25, -100), "SA": (-15, -60), "EU": (50, 10),
                "AF": (5, 20), "AS": (35, 100), "OC": (-25, 135),
            }
            for cont, (clat, clon) in continents.items():
                for _ in range(5):
                    lat = clat + rng.normal(0, 15)
                    lon = clon + rng.normal(0, 15)
                    coords.append((lat, lon))
                    names.append(f"{cont}-{_}")
            html = viz.folium_heatmap(coords, center=(20, 0), zoom=2)
            if html:
                st.components.v1.html(html, height=500)
            else:
                st.info("Install folium for heatmaps")

# =====================================================================
# TAB 14 — 3D Plots
# =====================================================================
with tabs[13]:
    st.header("3D Visualizations")
    st.markdown("3D surface plots, scatter plots, and more.")

    d3_col1, d3_col2 = st.columns(2)

    with d3_col1:
        st.subheader("3D Surface (Matplotlib)")
        res = st.slider("Resolution", 20, 100, 40, key="d3_res")
        func_choice = st.selectbox("Function", ["sinc", "peaks", "ripple", "saddle"], key="d3_func")
        if st.button("Generate 3D Surface", key="btn_3d_surface"):
            x = np.linspace(-3, 3, res); y = np.linspace(-3, 3, res)
            Xg, Yg = np.meshgrid(x, y)
            if func_choice == "sinc":
                R = np.sqrt(Xg**2 + Yg**2) + 1e-10
                Z = np.sin(R) / R
            elif func_choice == "peaks":
                Z = 3*(1-Xg)**2*np.exp(-(Xg**2) - (Yg+1)**2) - 10*(Xg/5 - Xg**3 - Yg**5)*np.exp(-(Xg**2) - Yg**2) - 1/3*np.exp(-((Xg+1)**2) - Yg**2)
            elif func_choice == "ripple":
                Z = np.sin(5 * np.sqrt(Xg**2 + Yg**2)) * np.cos(3 * Xg)
            else:  # saddle
                Z = Xg**2 - Yg**2
            fig = viz.mpl_3d_surface(Xg, Yg, Z, title=f"{func_choice.title()} Surface")
            st.pyplot(fig)

    with d3_col2:
        st.subheader("3D Scatter (Matplotlib)")
        n_3d = st.slider("Points", 50, 500, 200, key="d3_n")
        if st.button("Generate 3D Scatter", key="btn_3d_scatter"):
            rng = np.random.default_rng(42)
            x = rng.normal(0, 1, n_3d)
            y = rng.normal(0, 1, n_3d)
            z = x**2 + y**2 + rng.normal(0, 0.2, n_3d)
            labels = np.digitize(z, [0.5, 1.5, 3.0])
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': '3d'})
            sc = ax.scatter(x, y, z, c=labels, cmap="viridis", s=40, alpha=0.7)
            fig.colorbar(sc, ax=ax, shrink=0.5)
            ax.set_title("3D Scatter Plot"); st.pyplot(fig)

    if PLOTLY_AVAILABLE:
        st.markdown("---")
        st.subheader("3D Interactive (Plotly)")
        ply_col1, ply_col2 = st.columns(2)
        with ply_col1:
            if st.button("3D Scatter (Plotly)", key="btn_ply_3d_scatter"):
                rng = np.random.default_rng(99)
                import plotly.express as px
                df = pd.DataFrame({
                    "x": rng.normal(0, 1, 300), "y": rng.normal(0, 1, 300),
                    "z": rng.normal(0, 1, 300),
                    "cluster": np.random.choice(list("KLMN"), 300),
                })
                fig = px.scatter_3d(df, x="x", y="y", z="z", color="cluster",
                                   title="3D Interactive Scatter", template="plotly_white",
                                   opacity=0.6)
                st.plotly_chart(fig, use_container_width=True)
        with ply_col2:
            if st.button("3D Surface (Plotly)", key="btn_ply_3d_surface"):
                import plotly.graph_objects as go
                x = np.linspace(-3, 3, 50); y = np.linspace(-3, 3, 50)
                Xg, Yg = np.meshgrid(x, y)
                R = np.sqrt(Xg**2 + Yg**2) + 1e-10
                Z = np.sin(R) / R
                fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorscale="Viridis")])
                fig.update_layout(title="3D Surface (Plotly)", template="plotly_white",
                                 scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
                st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 15 — Tank & Dozer CLI
# =====================================================================

with tabs[14]:
    st.markdown(
        '<h2 style="text-align: center; margin-bottom: 0;">⬡ TANK & DOZER</h2>'
        '<p style="text-align: center; font-family: Fira Code, monospace; color: #8888aa; '
        'font-size: 12px; letter-spacing: 2px; margin-top: 0;">'
        'CYBERSECURITY INCIDENT RESPONSE FRAMEWORK</p>',
        unsafe_allow_html=True,
    )

    # ── Banner art ──────────────────────────────────────────
    banner = """
    ▄▄▄█████▓ ▄▄▄       ███▄    █  ██ ▄█▀ ▓█████▄  ▒█████   ▒█████  ▒███████▒
    ▓  ██▒ ▓▒▒████▄     ██ ▀█   █  ██▄█▒ ▒██▀ ██▌▒██▒  ██▒▒██▒  ██▒▒ ▒ ▒ ▄▀░
    ▒ ▓██░ ▒░▒██  ▀█▄  ▓██  ▀█ ██▒▓███▄░ ░██   █▌▒██░  ██▒▒██░  ██▒░ ▒ ▄▀▒░
    ░ ▓██▓ ░ ░██▄▄▄▄██ ▓██▒  ▐▌██▒▓██ █▄ ░▓█▄   ▌▒██   ██░▒██   ██░  ▄▀▒   ░
      ▒██▒ ░  ▓█   ▓██▒▒██░   ▓██░▒██▒ █▄░▒████▓ ░ ████▓▒░░ ████▓▒░▒███████▒
      ▒ ░░    ▒▒   ▓▒█░░ ▒░   ▒ ▒ ▒ ▒▒ ▓▒ ▒▒▓  ▒ ░ ▒░▒░▒░ ░ ▒░▒░▒░ ░▒▒ ▓░▒░▒
        ░      ▒   ▒▒ ░░ ░░   ░ ▒░░ ░▒ ▒░ ░ ▒  ▒   ░ ▒ ▒░   ░ ▒ ▒░ ░░▒ ▒ ░ ▒
      ░        ░   ▒      ░   ░ ░ ░ ░░ ░  ░ ░  ░ ░ ░ ░ ▒  ░ ░ ░ ▒  ░ ░ ░ ░ ░
                   ░  ░         ░ ░  ░      ░        ░ ░      ░ ░      ░ ░
    """
    st.markdown(
        f'<pre style="text-align: center; font-family: Fira Code, monospace; '
        f'font-size: 10px; color: #00f0ff; line-height: 1.1; '
        f'background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; '
        f'border: 1px solid rgba(0,240,255,0.15);">'
        f'{banner}</pre>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<p style="text-align: center; font-family: Orbitron, sans-serif; '
        f'color: #ff00ff; font-size: 14px; letter-spacing: 3px;">'
        f'v{__version__} // {__app_name__}</p>',
        unsafe_allow_html=True,
    )

    # ── Command Builder ─────────────────────────────────────
    st.markdown("---")
    td_col_a, td_col_b = st.columns([1, 2])

    with td_col_a:
        st.markdown(
            '<p style="font-family: Orbitron, sans-serif; color: #00f0ff; '
            'font-size: 14px; letter-spacing: 2px;">▸ COMMAND BUILDER</p>',
            unsafe_allow_html=True,
        )

        td_module_configs = {
            "ir": {
                "label": "Incident Response",
                "commands": {
                    "init": {"args": [{"key": "case_name", "label": "Case Name", "default": "untitled"}, {"key": "severity", "label": "Severity", "type": "select", "options": ["low", "medium", "high", "critical"], "default": "medium"}]},
                    "status": {"args": []},
                    "close": {"args": []},
                },
            },
            "analyze": {
                "label": "Analyze",
                "commands": {
                    "logs": {"args": [{"key": "file", "label": "CSV File", "default": "sample.csv"}, {"key": "clusters", "label": "Clusters", "type": "number", "default": "3"}]},
                    "network": {"args": []},
                    "traffic": {"args": []},
                },
            },
            "scan": {
                "label": "Scan",
                "commands": {
                    "quick": {"args": []},
                    "full": {"args": [{"key": "target", "label": "Target", "default": "127.0.0.1"}, {"key": "rate", "label": "Rate", "type": "number", "default": "1000"}]},
                    "web": {"args": []},
                },
            },
            "ioc": {
                "label": "IOC Enrichment",
                "commands": {
                    "enrich": {"args": [{"key": "indicator", "label": "Indicator", "default": "185.130.5.190"}, {"key": "type", "label": "Type", "type": "select", "options": ["ip", "domain", "hash", "url"], "default": "ip"}]},
                    "bulk": {"args": []},
                    "cluster": {"args": [{"key": "file", "label": "CSV File", "default": "iocs.csv"}]},
                },
            },
            "report": {
                "label": "Report",
                "commands": {
                    "generate": {"args": []},
                    "executive": {"args": []},
                    "timeline": {"args": []},
                },
            },
        }

        td_module = st.selectbox(
            "Module",
            options=list(td_module_configs.keys()),
            format_func=lambda x: td_module_configs[x]["label"],
            key="td_module_sel",
        )

        td_cmd_keys = list(td_module_configs[td_module]["commands"].keys())
        td_cmd = st.selectbox(
            "Command",
            options=td_cmd_keys,
            key="td_cmd_sel",
        )

        # Build extra args input fields
        td_args_list = []
        cmd_config = td_module_configs[td_module]["commands"][td_cmd]
        for arg_cfg in cmd_config["args"]:
            ak = f"td_arg_{arg_cfg['key']}"
            if arg_cfg.get("type") == "select":
                val = st.selectbox(
                    f"--{arg_cfg['key']}",
                    options=arg_cfg["options"],
                    index=arg_cfg["options"].index(arg_cfg["default"]),
                    key=ak,
                )
                td_args_list.append(f"--{arg_cfg['key']} {val}")
            elif arg_cfg.get("type") == "number":
                val = st.number_input(
                    f"--{arg_cfg['key']}",
                    value=int(arg_cfg["default"]),
                    key=ak,
                )
                td_args_list.append(f"--{arg_cfg['key']} {val}")
            else:
                val = st.text_input(
                    arg_cfg["label"],
                    value=arg_cfg["default"],
                    key=ak,
                )
                td_args_list.append(val)

        td_extra_args = " ".join(td_args_list)

        # Preview command line
        cmd_preview = f"tankdozer {td_module} {td_cmd} {td_extra_args}"
        st.markdown(
            f'<div style="font-family: Fira Code, monospace; font-size: 12px; '
            f'color: #ff00ff; background: rgba(0,0,0,0.3); padding: 8px 12px; '
            f'border-radius: 6px; border: 1px solid rgba(255,0,255,0.2); '
            f'margin-top: 12px;">$ {cmd_preview}</div>',
            unsafe_allow_html=True,
        )

        run_clicked = st.button(
            "▶ EXECUTE COMMAND",
            key="td_run",
            help="Run the configured tankdozer command",
        )

    with td_col_b:
        st.markdown(
            '<p style="font-family: Orbitron, sans-serif; color: #00f0ff; '
            'font-size: 14px; letter-spacing: 2px;">▸ TERMINAL OUTPUT</p>',
            unsafe_allow_html=True,
        )

        if run_clicked or st.session_state.get("td_auto_run"):
            if st.session_state.get("td_auto_run"):
                td_module = st.session_state.td_module
                td_cmd = st.session_state.td_cmd
                td_extra_args = st.session_state.td_args
                st.session_state.td_auto_run = False
            else:
                st.session_state.td_module = td_module
                st.session_state.td_cmd = td_cmd
                st.session_state.td_args = td_extra_args

            with st.spinner("Executing command..."):
                output = run_tankdozer(td_module, td_cmd, td_extra_args)
                st.session_state.td_output = output
                st.rerun()

        # Terminal output display
        output_text = st.session_state.get("td_output", "")
        st.markdown(
            f'<div class="terminal-box">'
            f'<span style="color: #ff00ff;">┌─</span> '
            f'<span style="color: #8888aa;">tankdozer terminal</span>'
            f'<span style="color: #ff00ff;">─' + "─" * 40 + '</span><br>'
            f'{output_text}'
            f'<br><span style="color: #ff00ff;">└─</span> '
            f'<span style="color: #8888aa;">exit</span></div>',
            unsafe_allow_html=True,
        )

        if output_text:
            if st.button("🗑 Clear Output", key="td_clear"):
                st.session_state.td_output = ""
                st.rerun()

    # ── Quick Reference ─────────────────────────────────────
    with st.expander("📖 Command Reference"):
        st.markdown("""
        | Module | Command | Description |
        |--------|---------|-------------|
        | `ir` | `init [name] --severity` | Start a new incident response case |
        | `ir` | `status` | Show current case status |
        | `ir` | `close` | Close the current case |
        | `analyze` | `logs <file> --clusters` | Detect anomalies in log data |
        | `analyze` | `network` | Analyze network flow patterns |
        | `analyze` | `traffic` | Traffic pattern analysis |
        | `scan` | `quick` | Quick port scan (top 100 ports) |
        | `scan` | `full <target> --rate` | Full port scan (1-65535) |
        | `scan` | `web` | Web application recon |
        | `ioc` | `enrich <indicator> --type` | Enrich an IOC |
        | `ioc` | `bulk` | Bulk IOC enrichment |
        | `ioc` | `cluster <file>` | Cluster IOCs by similarity |
        | `report` | `generate` | Full incident report |
        | `report` | `executive` | Executive summary |
        | `report` | `timeline` | Incident timeline |
        """)


# =====================================================================
# Helpers
# =====================================================================

def plot_convergence(history: list[float], title: str = "Convergence"):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history, color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best Fitness")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    return fig


def plot_2d_decision_space(
    positions: np.ndarray, labels: np.ndarray | None = None,
    centers: np.ndarray | None = None, title: str = "Data",
):
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is not None:
        scatter = ax.scatter(positions[:,0], positions[:,1], c=labels,
                            cmap="viridis", s=40, alpha=0.8, edgecolors="k", linewidth=0.3)
        if len(np.unique(labels)) > 1: plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(positions[:,0], positions[:,1], s=40, alpha=0.7)
    if centers is not None:
        ax.scatter(centers[:,0], centers[:,1], c="red", marker="X", s=200,
                  label="Centroids", edgecolors="black", linewidth=1.5)
        ax.legend()
    ax.set_title(title); ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig