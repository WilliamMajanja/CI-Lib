# CI-Lib: A Computational Intelligence Library

**A Pure-NumPy Framework for Optimisation, Learning, and Decision-Making**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CI-Lib is a pure-NumPy computational intelligence library spanning seven
core domains: neural networks, evolutionary algorithms, swarm intelligence,
fuzzy logic, clustering, classical optimisation, and utility functions. It
is designed for **research, education, and practical application** with a
focus on algorithmic transparency, minimal dependencies, and a consistent API.

The library is accompanied by a **FastAPI REST backend**, an interactive
**Streamlit dashboard**, a **best-in-class visualization suite** (10+ Python
libraries + 60+ R packages), and **Tank & Dozer**, a cybersecurity incident
response CLI that demonstrates practical applications of CI algorithms.

## Table of Contents

- [Modules](#modules)
- [Quick Start](#quick-start)
- [Library Usage](#library-usage)
- [Visualization Suite](#visualization-suite)
- [REST API](#rest-api)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Tank & Dozer CLI](#tank--dozer-cli)
- [Docker](#docker)
- [Local Development](#local-development-without-docker)
- [Running Experiments](#running-experiments)
- [Tests](#tests)
- [R Setup](#r-setup)
- [Thesis](#thesis)
- [License](#license)
- [References](#references)

## Modules

| Module | Algorithms | Key Classes |
|--------|-----------|-------------|
| **neural** | Feedforward networks, backpropagation | `FeedForwardNetwork` |
| **evolutionary** | Genetic algorithm, differential evolution | `GeneticAlgorithm`, `DifferentialEvolution` |
| **swarm** | Particle swarm optimisation, ant colony | `ParticleSwarmOptimizer`, `AntColonyOptimizer` |
| **fuzzy** | Mamdani fuzzy inference | `FuzzySet`, `FuzzyVariable`, `FuzzyRule`, `FuzzyInferenceSystem` |
| **clustering** | K-Means, DBSCAN | `KMeans`, `DBSCAN` |
| **optimization** | Simulated annealing, gradient descent | `SimulatedAnnealing`, `GradientDescent` |
| **utils** | Preprocessing, metrics, benchmarks | `normalize`, `mse`, `sphere`, `rosenbrock`, … |
| **viz** | Python + R visualization suite | `PythonVisualizationEngine`, `VisualizationSuite` |

## Quick Start

### Installation

```bash
# Core library (NumPy only)
pip install -e .

# With web interface (Streamlit dashboard)
pip install -e ".[web]"

# With visualization libraries (Plotly, Bokeh, Seaborn, etc.)
pip install -e ".[viz]"

# With R integration
pip install -e ".[r]"

# With all extras (dev + web + viz + r)
pip install -e ".[all]"

# Or via pip from requirements
pip install -r requirements.txt
pip install -r requirements-web.txt
```

### Library Usage

```python
import numpy as np
from ci_lib import GeneticAlgorithm, KMeans, FeedForwardNetwork
from ci_lib.utils import sphere

# Evolutionary optimisation
ga = GeneticAlgorithm(sphere, n_dims=5, bounds=np.array([[-5, 5]] * 5), seed=42)
best, fitness, history = ga.evolve(generations=200)
print(f"Best fitness: {fitness:.6f}")

# Clustering
X = np.random.default_rng(42).standard_normal((200, 2))
km = KMeans(n_clusters=3, seed=42)
labels = km.fit_predict(X)
print(f"Silhouette score: {km.silhouette_score(X):.3f}")

# Neural network (XOR problem)
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([[0],[1],[1],[0]])
nn = FeedForwardNetwork([2, 4, 1], activation="sigmoid", learning_rate=0.5, seed=42)
nn.fit(X_xor, y_xor, epochs=2000)
print(f"Final MSE: {nn.score(X_xor, y_xor):.6f}")
```

## Visualization Suite

A comprehensive visualization engine supporting **10+ Python libraries** and **60+ R packages**.

### Python Visualization Libraries

| Library | Type | Usage |
|---------|------|-------|
| **Matplotlib** | Static plots | All basic chart types + 3D |
| **Seaborn** | Statistical plots | Heatmap, pairplot, violin, KDE |
| **Plotly** | Interactive plots | Scatter, 3D, sunburst, parallel coords |
| **Bokeh** | Interactive dashboards | Linked pan/zoom, server-side apps |
| **Altair** | Declarative viz | Grammar of graphics (Vega-Lite) |
| **NetworkX** | Graph analysis | Degree dist, communities, layout |
| **PyVis** | Interactive networks | Force-directed, physics engine |
| **Statsmodels** | Time series | Decomposition, ACF/PACF, Q-Q |
| **Folium** | Maps | Leaflet maps, heatmaps, markers |
| **PyVista** | 3D rendering | Surface, volume, mesh visualization |

### Python Usage

```python
from ci_lib.viz import PythonVisualizationEngine

viz = PythonVisualizationEngine()

# Matplotlib line plot
fig = viz.mpl_line({"sin(x)": y1, "cos(x)": y2}, title="Trig", xlabel="x", ylabel="y")

# Correlation heatmap (seaborn)
fig = viz.mpl_correlation_matrix(df)

# 3D surface
fig = viz.mpl_3d_surface(X, Y, Z, title="Surface Plot")

# Time series decomposition
fig = viz.ts_decomposition(series, period=12)

# Network graph
fig = viz.network_graph(edges, labels)

# Interactive plotly exports
fig_json = viz.plotly_scatter(x, y, labels)  # returns JSON for Plotly
```

### R Integration

The `ci_lib.viz.r_bridge` module provides two engines for R visualizations:

1. **RVisualizationEngine** (via `rpy2`) — Call R directly from Python
2. **RScriptRunner** (via subprocess) — Run R scripts as standalone processes

```python
from ci_lib.viz.r_bridge import get_r_engine

engine = get_r_engine()
if engine.available:
    engine.run_r_script('''
        library(ggplot2)
        p <- ggplot(mtcars, aes(wt, mpg)) + geom_point()
        ggsave("plot.png", p)
    ''')
```

### R Visualization Templates

Pre-built R templates are available in `ci_lib/viz/r_scripts/`:
- `ggplot_templates.R` — Scatter/density, correlation heatmaps, PCA biplots,
  time series, boxplots, violin plots, ridge plots, dendrograms, ROC curves

### Streamlit Frontend Tabs

The dashboard includes 14 tabs covering all visualization types:

| Tab | Libraries | Features |
|-----|-----------|----------|
| Clustering–Benchmarks | API + matplotlib | CI algorithm demos |
| **Viz Gallery** | matplotlib, plotly | Demo of all chart types |
| **Interactive** | plotly, bokeh, altair | 3D scatter, sunburst, parallel coords |
| **Statistical** | statsmodels, seaborn | TS decomposition, ACF/PACF, Q-Q |
| **Network** | networkx, pyvis | Graph viz, communities, interactive |
| **Geospatial** | folium | World maps, heatmaps, markers |
| **3D Plots** | matplotlib, plotly | Surface, scatter, interactive 3D |

### Shiny Integration

R Shiny apps can be launched from Python:

```python
from frontend.shiny_integration import ShinyAppBridge

shiny = ShinyAppBridge()
if shiny.available:
    url = shiny.launch(app_name="CI-Lib Dashboard", port=3838)
    print(f"Shiny app at {url}")
    shiny.stop()
```

## REST API

Start the FastAPI backend:

```bash
python -m backend.app
```

The API listens on `http://localhost:8000` with auto-generated docs at
`/docs` (Swagger) and `/redoc` (ReDoc).

**Available endpoints (16 total):**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/clustering/kmeans` | K-Means clustering |
| `POST` | `/api/clustering/dbscan` | DBSCAN clustering |
| `POST` | `/api/evolutionary/genetic` | Genetic algorithm |
| `POST` | `/api/evolutionary/differential` | Differential evolution |
| `POST` | `/api/swarm/pso` | Particle swarm optimisation |
| `POST` | `/api/swarm/aco` | Ant colony optimisation (TSP) |
| `POST` | `/api/fuzzy/default-fis` | Mamdani FIS (tipping problem) |
| `POST` | `/api/fuzzy/custom-fis` | Custom 1-input Mamdani FIS |
| `POST` | `/api/neural/train` | Train feedforward network |
| `POST` | `/api/optimization/simulated-annealing` | Simulated annealing |
| `POST` | `/api/optimization/gradient-descent` | Gradient descent |
| `POST` | `/api/utils/normalize` | Data normalisation |
| `POST` | `/api/utils/metrics` | Classification/regression metrics |
| `GET` | `/api/utils/benchmarks` | List benchmark functions |
| `POST` | `/api/utils/benchmarks/evaluate` | Evaluate benchmark function |

## Streamlit Dashboard

```bash
streamlit run frontend/app.py --server.port=8501
```

The dashboard provides eight interactive tabs:

1. **Clustering** — K-Means and DBSCAN with 2-D scatter plots
2. **Evolutionary** — GA and DE with convergence plots
3. **Swarm** — PSO convergence and ACO TSP tour visualisation
4. **Fuzzy Inference** — Mamdani tipping problem with live sliders
5. **Neural Network** — XOR training with prediction table
6. **Optimization** — SA and GD with convergence tracking
7. **Utilities** — Data normalisation and metrics computation
8. **Benchmarks** — Benchmark function evaluation and landscape plots

## Tank & Dozer CLI

A cybersecurity incident response framework that applies CI algorithms:

```bash
tankdozer ir init --severity high        # Start IR case
tankdozer analyze logs sample.csv        # Anomaly detection via K-Means + DBSCAN
tankdozer scan quick                      # Port scan simulation
tankdozer ioc enrich 185.130.5.190       # IOC enrichment
tankdozer report generate                 # Generate incident report
```

## Docker

```bash
docker compose up --build -d
# Backend:  http://localhost:8000
# Frontend: http://localhost:8501
```

## Local Development (without Docker)

For environments without Docker (e.g., Raspberry Pi), use a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies (core + web + dev + viz)
pip install -e ".[all]"

# Or step by step:
pip install -e ".[dev]"          # Core + dev tools
pip install -e ".[web]"          # FastAPI + Streamlit
pip install -e ".[viz]"          # All visualization libraries
pip install rpy2                 # R integration (optional)

# Run backend (Terminal 1)
python -m backend.app
# API: http://localhost:8000 | Docs: http://localhost:8000/docs

# Run frontend (Terminal 2)
streamlit run frontend/app.py --server.port=8501
# Dashboard: http://localhost:8501

# Run R setup (Terminal 3, optional)
Rscript install_r_packages.R
```

## Running Experiments

Generate convergence comparison figures for the thesis:

```bash
python experiments/convergence_comparison.py
```

This produces:
- CSV results in `experiments/results/`
- PNG figures in `thesis/figures/`

## R Setup

Install R (if not already installed):

```bash
# Debian/Ubuntu/Raspberry Pi
sudo apt-get install r-base r-base-dev

# macOS
brew install r

# Windows: Download from https://cran.r-project.org/
```

Then install R visualization packages:

```bash
Rscript install_r_packages.R
```

Or install individual packages as needed:

```bash
Rscript -e 'install.packages(c("ggplot2", "plotly", "dplyr"), repos="https://cloud.r-project.org/")'
```

The Python-R bridge uses `rpy2`:

```bash
pip install rpy2
```

## Tests

```bash
python -m pytest tests/ -v
```

The test suite covers all algorithms with 20+ test classes, including:
- Correctness on canonical problems (XOR, tipping, TSP)
- Convergence validation on benchmark functions
- Edge cases (invalid inputs, boundary conditions)
- Reproducibility with fixed random seeds

## Thesis

A LaTeX thesis skeleton is provided in the `thesis/` directory:

```bash
cd thesis
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Key thesis materials:
- `thesis/main.tex` — Full LaTeX document
- `thesis/references.bib` — 25+ academic references
- `thesis/figures/` — Generated experiment figures

## License

MIT — see [LICENSE](LICENSE) for details.

## References

The library implements algorithms founded on the following seminal works:

1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*.
2. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*.
3. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *IEEE ICNN*.
4. Dorigo, M., et al. (1996). Ant system: optimization by a colony of cooperating agents. *IEEE Trans. SMC*.
5. Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*.
6. Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. *Int. J. Man-Machine Studies*.
7. Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Trans. Information Theory*.
8. Ester, M., et al. (1996). A density-based algorithm for discovering clusters. *KDD*.
9. Kirkpatrick, S., et al. (1983). Optimization by simulated annealing. *Science*.
10. Storn, R., & Price, K. (1997). Differential evolution. *J. Global Optimization*.

See `thesis/references.bib` for the complete bibliography.
