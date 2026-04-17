"""
ci_lib — Computational Intelligence Library
============================================

A pure-NumPy library implementing core computational intelligence algorithms
for optimization, learning, and decision-making.

Modules
-------
neural        : Feedforward neural networks with backpropagation.
evolutionary  : Genetic algorithms and differential evolution.
swarm         : Particle swarm optimization and ant colony optimization.
fuzzy         : Fuzzy sets, variables, rules, and Mamdani inference.
clustering    : K-Means and DBSCAN clustering.
optimization  : Simulated annealing and gradient descent.
utils         : Preprocessing, metrics, and benchmark functions.
"""

__version__ = "1.0.0"

from ci_lib.neural import FeedForwardNetwork
from ci_lib.evolutionary import GeneticAlgorithm, DifferentialEvolution
from ci_lib.swarm import ParticleSwarmOptimizer, AntColonyOptimizer
from ci_lib.fuzzy import FuzzyVariable, FuzzySet, FuzzyRule, FuzzyInferenceSystem
from ci_lib.clustering import KMeans, DBSCAN
from ci_lib.optimization import SimulatedAnnealing, GradientDescent

__all__ = [
    "FeedForwardNetwork",
    "GeneticAlgorithm",
    "DifferentialEvolution",
    "ParticleSwarmOptimizer",
    "AntColonyOptimizer",
    "FuzzyVariable",
    "FuzzySet",
    "FuzzyRule",
    "FuzzyInferenceSystem",
    "KMeans",
    "DBSCAN",
    "SimulatedAnnealing",
    "GradientDescent",
]
