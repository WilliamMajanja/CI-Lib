"""
Convergence comparison of CI algorithms on benchmark functions.

Generates figures and tables for thesis Chapter 5 (Experimental Evaluation).

Usage:
    python experiments/convergence_comparison.py

Output:
    experiments/results/  -- CSV tables
    thesis/figures/       -- PNG figures
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ci_lib import (
    GeneticAlgorithm,
    DifferentialEvolution,
    ParticleSwarmOptimizer,
    SimulatedAnnealing,
)
from ci_lib.utils.benchmarks import sphere, rastrigin, rosenbrock, ackley

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
FIGURE_DIR = Path(__file__).resolve().parent.parent / "thesis" / "figures"
SEED = 42
N_DIMS = 10
POP_SIZE = 50
GENERATIONS = 500
N_TRIALS = 10

ALGORITHMS = {
    "GA": lambda fn, nd, b: GeneticAlgorithm(fn, nd, b, pop_size=POP_SIZE, seed=SEED),
    "DE": lambda fn, nd, b: DifferentialEvolution(fn, nd, b, pop_size=POP_SIZE, seed=SEED),
    "PSO": lambda fn, nd, b: ParticleSwarmOptimizer(fn, nd, b, n_particles=POP_SIZE, seed=SEED),
    "SA": lambda fn, nd, b: SimulatedAnnealing(fn, nd, b, seed=SEED),
}

BENCHMARKS = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock,
    "Ackley": ackley,
}

COLORS = {"GA": "#1f77b4", "DE": "#ff7f0e", "PSO": "#2ca02c", "SA": "#d62728"}
LINESTYLES = {"GA": "-", "DE": "--", "PSO": "-.", "SA": ":"}


def make_bounds(nd: int, b: tuple[float, float]) -> np.ndarray:
    return np.array([b] * nd)


def run_single(algo_name: str, fn, bounds: np.ndarray, nd: int) -> list[float]:
    algo = ALGORITHMS[algo_name](fn, nd, bounds)
    if algo_name == "SA":
        _, _, history = algo.optimize()
    elif algo_name in ("GA", "DE"):
        _, _, history = algo.evolve(generations=GENERATIONS)
    else:
        _, _, history = algo.optimize(iterations=GENERATIONS)
    return history


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    for bm_name, bm_fn in BENCHMARKS.items():
        bounds = make_bounds(N_DIMS, bm_fn.bounds)
        print(f"\nBenchmark: {bm_name} ({N_DIMS}-D)")

        results: dict[str, list[list[float]]] = {name: [] for name in ALGORITHMS}

        for trial in range(N_TRIALS):
            print(f"  Trial {trial + 1}/{N_TRIALS}")
            for name in ALGORITHMS:
                hist = run_single(name, bm_fn, bounds, N_DIMS)
                results[name].append(hist)

        # Write CSV
        csv_path = OUTPUT_DIR / f"{bm_name.lower()}_{N_DIMS}d.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["generation"] + [f"{name}_trial{t}" for name in ALGORITHMS for t in range(N_TRIALS)]
            writer.writerow(header)
            max_len = max(len(h) for trial_hists in results.values() for h in trial_hists)
            for gen in range(max_len):
                row = [gen]
                for name in ALGORITHMS:
                    for t in range(N_TRIALS):
                        row.append(results[name][t][gen] if gen < len(results[name][t]) else "")
                writer.writerow(row)
        print(f"  Saved: {csv_path}")

        # Compute median convergence
        generations_actual = min(len(h) for trial_hists in results.values() for h in trial_hists)
        fig, ax = plt.subplots(figsize=(8, 4))
        for name in ALGORITHMS:
            arr = np.array([h[:generations_actual] for h in results[name]])
            median = np.median(arr, axis=0)
            q25 = np.percentile(arr, 25, axis=0)
            q75 = np.percentile(arr, 75, axis=0)
            gens = np.arange(generations_actual)
            ax.plot(gens, median, color=COLORS[name], ls=LINESTYLES[name], label=name, linewidth=1.5)
            ax.fill_between(gens, q25, q75, color=COLORS[name], alpha=0.1)

        ax.set_yscale("log")
        ax.set_xlabel("Generation / Iteration", fontsize=11)
        ax.set_ylabel("Best Fitness", fontsize=11)
        ax.set_title(f"{bm_name} ({N_DIMS}-D) — Convergence Comparison", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig_path = FIGURE_DIR / f"{bm_name.lower()}_convergence_{N_DIMS}d.png"
        fig.savefig(fig_path, dpi=150)
        print(f"  Saved figure: {fig_path}")
        plt.close(fig)

        # Print final results table
        print(f"\n  Final Results ({bm_name}, {N_DIMS}-D):")
        print(f"  {'Algorithm':<12} {'Median Best':<16} {'Best Trial':<16} {'Worst Trial':<16}")
        print(f"  {'-'*60}")
        for name in ALGORITHMS:
            finals = [h[-1] for h in results[name]]
            median_val = np.median(finals)
            best_val = np.min(finals)
            worst_val = np.max(finals)
            print(f"  {name:<12} {median_val:<16.6e} {best_val:<16.6e} {worst_val:<16.6e}")

    print("\nDone. Results saved to experiments/results/")
    print("Figures saved to thesis/figures/")


if __name__ == "__main__":
    main()
