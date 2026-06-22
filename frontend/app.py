"""Streamlit frontend for the Computational Intelligence Library."""

import io
import json
import urllib.request
import urllib.error
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = st.sidebar.text_input("API Base URL", value="http://localhost:8000/api")


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


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CI-Lib Demo", layout="wide")
st.title("Computational Intelligence Library — Interactive Demo")
st.markdown("---")

tabs = st.tabs([
    "Clustering",
    "Evolutionary",
    "Swarm",
    "Fuzzy Inference",
    "Neural Network",
    "Optimization",
    "Utilities",
    "Benchmarks",
])

# ---------------------------------------------------------------------------
# Helper: plot convergence
# ---------------------------------------------------------------------------

def plot_convergence(history: list[float], title: str = "Convergence"):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history, color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def plot_2d_decision_space(
    positions: np.ndarray, labels: np.ndarray | None = None,
    centers: np.ndarray | None = None, title: str = "Data",
):
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is not None:
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], c=labels,
            cmap="viridis", s=40, alpha=0.8, edgecolors="k", linewidth=0.3,
        )
        if len(np.unique(labels)) > 1:
            plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], s=40, alpha=0.7)
    if centers is not None:
        ax.scatter(
            centers[:, 0], centers[:, 1], c="red", marker="X",
            s=200, label="Centroids", edgecolors="black", linewidth=1.5,
        )
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ===========================================================================
# TAB 1 — Clustering
# ===========================================================================

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

# ===========================================================================
# TAB 2 — Evolutionary
# ===========================================================================

with tabs[1]:
    st.header("Evolutionary Algorithms")
    col1, col2 = st.columns(2)

    for col, algo_name, algo_key, extra_params in [
        (col1, "Genetic Algorithm", "ga", {}),
        (col2, "Differential Evolution", "de", {}),
    ]:
        with col:
            st.subheader(algo_name)
            bfn = st.selectbox(
                "Benchmark", ["sphere", "rosenbrock", "rastrigin", "ackley"],
                key=f"{algo_key}_bfn",
            )
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
            else:
                F = st.slider("F (mutation)", 0.1, 2.0, 0.8, 0.05, key="de_F")
                CR = st.slider("CR (crossover)", 0.0, 1.0, 0.9, key="de_CR")
                strat = st.selectbox("Strategy", ["best/1/bin", "rand/1/bin"], key="de_strat")
                btn = st.button("Run DE", key="btn_de")

            if btn:
                try:
                    bounds = [float(x) for x in bnd.split(",")]
                except ValueError:
                    st.error("Bounds must be comma-separated numbers, e.g. -5,5")
                    continue

                if algo_key == "ga":
                    r = api_call("POST", "/evolutionary/genetic", {
                        "n_dims": int(nd), "pop_size": int(pop),
                        "generations": int(gen), "benchmark_fn": bfn,
                        "bounds": [bounds] * nd,
                        "seed": int(sd), "mutation_rate": mr,
                        "crossover_rate": cr, "elitism": int(el),
                    })
                else:
                    r = api_call("POST", "/evolutionary/differential", {
                        "n_dims": int(nd), "pop_size": int(pop),
                        "generations": int(gen), "benchmark_fn": bfn,
                        "bounds": [bounds] * nd,
                        "seed": int(sd), "F": F, "CR": CR, "strategy": strat,
                    })
                if r:
                    st.metric("Best Fitness", f"{r['best_fitness']:.6e}")
                    st.text(f"Solution: {[f'{v:.4f}' for v in r['best_solution']]}")
                    fig = plot_convergence(r["history"], f"{algo_name} Convergence")
                    st.pyplot(fig)

# ===========================================================================
# TAB 3 — Swarm
# ===========================================================================

with tabs[2]:
    st.header("Swarm Intelligence")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Particle Swarm Optimization")
        bfn = st.selectbox("Benchmark", ["sphere", "rosenbrock", "rastrigin", "ackley"], key="pso_bfn")
        nd = st.slider("Dimensions", 2, 10, 2, key="pso_nd")
        np_ = st.slider("Particles", 5, 200, 30, key="pso_np")
        it = st.slider("Iterations", 10, 500, 100, key="pso_it")
        w = st.slider("Inertia (w)", 0.1, 1.5, 0.7, 0.05, key="pso_w")
        c1 = st.slider("Cognitive (c1)", 0.0, 3.0, 1.5, 0.1, key="pso_c1")
        c2 = st.slider("Social (c2)", 0.0, 3.0, 1.5, 0.1, key="pso_c2")
        sd = st.number_input("Seed", 0, 9999, 42, key="pso_sd")
        if st.button("Run PSO", key="btn_pso"):
            r = api_call("POST", "/swarm/pso", {
                "n_dims": int(nd), "n_particles": int(np_),
                "generations": int(it), "benchmark_fn": bfn,
                "bounds": [[-5, 5]] * nd,
                "w": w, "c1": c1, "c2": c2, "seed": int(sd),
            })
            if r:
                st.metric("Best Fitness", f"{r['best_fitness']:.6e}")
                fig = plot_convergence(r["history"], "PSO Convergence")
                st.pyplot(fig)

    with col2:
        st.subheader("Ant Colony Optimization (TSP)")
        nc = st.slider("Cities", 5, 30, 8, key="aco_nc")
        na = st.slider("Ants", 2, 100, 20, key="aco_na")
        aco_it = st.slider("Iterations", 10, 200, 50, key="aco_it")
        alpha = st.slider("Alpha (pheromone)", 0.0, 5.0, 1.0, 0.1, key="aco_alpha")
        beta = st.slider("Beta (heuristic)", 0.0, 5.0, 2.0, 0.1, key="aco_beta")
        evap = st.slider("Evaporation", 0.05, 0.95, 0.5, 0.05, key="aco_evap")
        sd = st.number_input("Seed", 0, 9999, 42, key="aco_sd")
        if st.button("Run ACO", key="btn_aco"):
            r = api_call("POST", "/swarm/aco", {
                "n_ants": int(na), "n_cities": int(nc),
                "iterations": int(aco_it), "alpha": alpha,
                "beta": beta, "evaporation": evap, "seed": int(sd),
            })
            if r:
                st.metric("Best Distance", f"{r['best_distance']:.3f}")
                st.text(f"Best Route: {r['best_route']}")
                coords = np.array(r.get("city_coordinates", r["distance_matrix"]))
                fig, ax = plt.subplots(figsize=(5, 5))
                route = r["best_route"]
                ax.scatter(coords[:, 0], coords[:, 1], c="steelblue", s=80, zorder=5)
                for i in range(len(route)):
                    a, b = route[i], route[(i + 1) % len(route)]
                    ax.plot(
                        [coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                        color="crimson", linewidth=1.2, alpha=0.7,
                    )
                for i, (x, y) in enumerate(coords):
                    ax.annotate(str(i), (x, y), fontsize=9, fontweight="bold")
                ax.set_title(f"ACO Tour (dist={r['best_distance']:.2f})")
                ax.set_aspect("equal")
                st.pyplot(fig)

# ===========================================================================
# TAB 4 — Fuzzy Inference
# ===========================================================================

with tabs[3]:
    st.header("Fuzzy Inference System")
    st.markdown("**Tipping Problem** — rate service (0–10) and food (0–10) to get a tip percentage.")
    col1, col2 = st.columns(2)
    with col1:
        svc = st.slider("Service Rating", 0.0, 10.0, 6.0, 0.5)
        fd = st.slider("Food Quality", 0.0, 10.0, 7.0, 0.5)
        method = st.selectbox("Defuzzify", ["centroid", "bisector", "max_membership"])
    with col2:
        if st.button("Compute Tip", key="btn_fis"):
            r = api_call("POST", "/fuzzy/default-fis", {
                "inputs": {"service": svc, "food": fd},
                "method": method,
            })
            if r:
                tip_val = r["output"]["tip"]
                st.metric("Suggested Tip", f"{tip_val:.1f}%")
                st.progress(min(tip_val / 30, 1.0))
                if tip_val < 10:
                    st.info("Low tip — consider improving service!")
                elif tip_val < 20:
                    st.success("Medium tip — decent experience.")
                else:
                    st.success("High tip — excellent!")

# ===========================================================================
# TAB 5 — Neural Network
# ===========================================================================

with tabs[4]:
    st.header("Neural Network — XOR Demo")
    st.markdown("Train a simple feedforward network on the XOR problem.")
    col1, col2 = st.columns(2)
    with col1:
        lr = st.slider("Learning Rate", 0.001, 1.0, 0.1, 0.005, key="nn_lr")
        epochs = st.slider("Epochs", 100, 5000, 1000, key="nn_ep")
        activation = st.selectbox("Activation", ["sigmoid", "tanh", "relu", "leaky_relu"], key="nn_act")
        batch = st.selectbox("Batch Size", [None, 1, 2, 4], format_func=lambda x: "Full" if x is None else str(x), key="nn_bs")
        sd = st.number_input("Seed", 0, 9999, 42, key="nn_sd")
    with col2:
        if st.button("Train Network", key="btn_nn"):
            X = [[0, 0], [0, 1], [1, 0], [1, 1]]
            y = [[0], [1], [1], [0]]
            r = api_call("POST", "/neural/train", {
                "layer_sizes": [2, 4, 1],
                "activation": activation,
                "learning_rate": lr,
                "epochs": int(epochs),
                "batch_size": batch,
                "seed": int(sd),
                "X": X, "y": y,
            })
            if r:
                st.metric("Initial MSE", f"{r['initial_mse']:.6f}")
                st.metric("Final MSE", f"{r['final_mse']:.6f}")
                st.metric("Layers", r["n_layers"])
                st.text(f"Weights: {r['weights_shapes']}")
                pred_r = api_call("POST", "/neural/predict", {"X": X})
                if pred_r:
                    st.subheader("Predictions")
                    for inp, out in zip(X, pred_r["predictions"]):
                        st.text(f"  {inp} -> {out[0]:.4f}")

# ===========================================================================
# TAB 6 — Optimization
# ===========================================================================

with tabs[5]:
    st.header("Classic Optimization")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Simulated Annealing")
        bfn_sa = st.selectbox("Benchmark", ["sphere", "rosenbrock", "rastrigin", "ackley"], key="sa_bfn")
        nd_sa = st.slider("Dimensions", 2, 10, 2, key="sa_nd")
        init_t = st.slider("Initial Temp", 1.0, 500.0, 100.0, key="sa_t")
        cr_sa = st.slider("Cooling Rate", 0.8, 0.999, 0.99, 0.001, key="sa_cr")
        sched = st.selectbox("Schedule", ["exponential", "linear", "logarithmic"], key="sa_sched")
        sd_sa = st.number_input("Seed", 0, 9999, 42, key="sa_sd")
        if st.button("Run SA", key="btn_sa"):
            r = api_call("POST", "/optimization/simulated-annealing", {
                "n_dims": int(nd_sa), "benchmark_fn": bfn_sa,
                "bounds": [[-5, 5]] * nd_sa,
                "initial_temp": init_t, "cooling_rate": cr_sa,
                "cooling_schedule": sched, "seed": int(sd_sa),
            })
            if r:
                st.metric("Best Fitness", f"{r['best_fitness']:.6e}")
                fig = plot_convergence(r["history"], "SA Convergence")
                st.pyplot(fig)

    with col2:
        st.subheader("Gradient Descent")
        bfn_gd = st.selectbox("Benchmark", ["sphere", "rosenbrock", "rastrigin"], key="gd_bfn")
        nd_gd = st.slider("Dimensions", 2, 10, 2, key="gd_nd")
        lr_gd = st.slider("Learning Rate", 0.0001, 1.0, 0.01, 0.0005, key="gd_lr")
        meth = st.selectbox("Method", ["sgd", "momentum", "adam"], key="gd_meth")
        mom = st.slider("Momentum", 0.0, 0.99, 0.9, 0.01, key="gd_mom")
        sd_gd = st.number_input("Seed", 0, 9999, 42, key="gd_sd")
        if st.button("Run GD", key="btn_gd"):
            r = api_call("POST", "/optimization/gradient-descent", {
                "n_dims": int(nd_gd), "benchmark_fn": bfn_gd,
                "learning_rate": lr_gd, "momentum": mom,
                "method": meth, "seed": int(sd_gd),
            })
            if r:
                st.metric("Best Value", f"{r['best_fitness']:.6e}")
                fig = plot_convergence(r["history"], f"GD ({meth}) Convergence")
                st.pyplot(fig)

# ===========================================================================
# TAB 7 — Utilities
# ===========================================================================

with tabs[6]:
    st.header("Utilities")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Normalize Data")
        norm_data = st.text_area(
            "Enter data as [[x1,y1],[x2,y2],...]",
            "[[1,2],[3,4],[5,6],[7,8]]",
            key="norm_in",
        )
        norm_meth = st.selectbox("Method", ["minmax", "zscore"], key="norm_meth")
        if st.button("Normalize", key="btn_norm"):
            try:
                d = json.loads(norm_data)
                r = api_call("POST", "/utils/normalize", {"data": d, "method": norm_meth})
                if r:
                    st.text(f"Result ({r['method']}):")
                    st.json(r["data"])
            except json.JSONDecodeError:
                st.error("Invalid JSON")
    with col2:
        st.subheader("Metrics")
        met_yt = st.text_input("y_true", "[0,1,1,0,1]", key="met_yt")
        met_yp = st.text_input("y_pred", "[0,1,0,0,1]", key="met_yp")
        if st.button("Compute", key="btn_met"):
            try:
                r = api_call("POST", "/utils/metrics", {
                    "y_true": json.loads(met_yt),
                    "y_pred": json.loads(met_yp),
                })
                if r:
                    for k, v in r.items():
                        if v is not None:
                            st.metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else v)
            except json.JSONDecodeError:
                st.error("Invalid JSON")

# ===========================================================================
# TAB 8 — Benchmarks
# ===========================================================================

with tabs[7]:
    st.header("Benchmark Functions")
    bfns = ["sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel"]
    col1, col2 = st.columns(2)
    with col1:
        bfn_name = st.selectbox("Function", bfns, key="bm_fn")
        dims = st.slider("Dimensions", 1, 10, 2, key="bm_dims")
        x_vals = st.text_input("x (comma-separated)", "1.0, 2.0", key="bm_x")
        if st.button("Evaluate", key="btn_bm"):
            try:
                x = [float(v) for v in x_vals.split(",")]
                if len(x) != dims:
                    st.error(f"Expected {dims} values, got {len(x)}")
                else:
                    r = api_call("POST", "/utils/benchmarks/evaluate", {
                        "fn_name": bfn_name, "x": x,
                    })
                    if r:
                        st.metric(f"{bfn_name} value", f"{r['value']:.6f}")
                        st.text(f"Bounds: {r['bounds']}")
                        st.text(f"Optimum: {r['optimum_value']}")
            except ValueError:
                st.error("Invalid x values")
    with col2:
        st.subheader("Quick Visualization")
        if st.button("Plot Landscape (2D)", key="btn_bm_plot"):
            fn2 = bfn_name
            r0 = api_call("GET", "/utils/benchmarks")
            if r0:
                res = 50
                x = np.linspace(-5, 5, res)
                y = np.linspace(-5, 5, res)
                Xg, Yg = np.meshgrid(x, y)
                Z = np.zeros_like(Xg)
                for i in range(res):
                    for j in range(res):
                        resp = api_call("POST", "/utils/benchmarks/evaluate", {
                            "fn_name": fn2, "x": [float(Xg[i, j]), float(Yg[i, j])],
                        })
                        if resp:
                            Z[i, j] = resp["value"]
                fig, ax = plt.subplots(figsize=(5, 4))
                c = ax.contourf(Xg, Yg, Z, levels=30, cmap="viridis")
                plt.colorbar(c, ax=ax)
                ax.set_title(f"{fn2} Landscape")
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                st.pyplot(fig)
