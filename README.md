# Partial-Realpart-Analysis- & Computational Intelligence Library

A repository combining universal set theory, partial real-part analysis, and a full **Computational Intelligence** toolkit built on pure NumPy.

## ci_lib — Computational Intelligence Library

A pure-NumPy Python package with 7 modules covering the core pillars of computational intelligence:

| Module | Algorithms | Key Classes |
|--------|-----------|-------------|
| **neural** | Feedforward networks, backpropagation | `FeedForwardNetwork` |
| **evolutionary** | Genetic algorithm, differential evolution | `GeneticAlgorithm`, `DifferentialEvolution` |
| **swarm** | Particle swarm optimization, ant colony | `ParticleSwarmOptimizer`, `AntColonyOptimizer` |
| **fuzzy** | Mamdani fuzzy inference | `FuzzySet`, `FuzzyVariable`, `FuzzyRule`, `FuzzyInferenceSystem` |
| **clustering** | K-Means, DBSCAN | `KMeans`, `DBSCAN` |
| **optimization** | Simulated annealing, gradient descent | `SimulatedAnnealing`, `GradientDescent` |
| **utils** | Preprocessing, metrics, benchmarks | `normalize`, `mse`, `sphere`, `rosenbrock`, … |

### Quick Start

```python
import numpy as np
from ci_lib import GeneticAlgorithm, KMeans, FeedForwardNetwork
from ci_lib.utils import sphere, mse

# --- Optimize with a Genetic Algorithm ---
ga = GeneticAlgorithm(sphere, n_dims=5, bounds=np.array([[-5, 5]] * 5), seed=42)
best, fitness, history = ga.evolve(generations=200)
print(f"Best fitness: {fitness:.6f}")

# --- Cluster data with K-Means ---
X = np.random.default_rng(42).standard_normal((200, 2))
km = KMeans(n_clusters=3, seed=42)
labels = km.fit_predict(X)

# --- Train a Neural Network ---
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([[0],[1],[1],[0]])
nn = FeedForwardNetwork([2, 8, 1], activation='sigmoid', learning_rate=0.5, seed=42)
nn.fit(X_xor, y_xor, epochs=2000)
print(nn.predict(X_xor))
```

### Installation

```bash
pip install numpy
# Run from the repository root:
python -c "import ci_lib; print(ci_lib.__version__)"
```

### Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Original Content

### Usage
Libre Office Math to open the .math file.
Terminal for R-Graphics Device, List Recursion visualisation & Component Specific Debugging.

## Use Cases
- Banking
- Data Segmentation
- Math
- Data Compression
- Data Visualisation
- BioInformatics

## License
MIT
# Use Cases
 ##Banking 
 ##Data Segmentation
 ##Math
 ##Data Compression 
 ##Data Visualisation
 ##BioInformatics 
 

