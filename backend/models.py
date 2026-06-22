"""Pydantic request/response models for all API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


class KMeansRequest(BaseModel):
    n_clusters: int = Field(default=3, ge=2)
    max_iter: int = Field(default=300, ge=1)
    tol: float = Field(default=1e-4, ge=0)
    init: str = Field(default="kmeans++")
    seed: int | None = None
    data: list[list[float]]


class KMeansResponse(BaseModel):
    labels: list[int]
    centers: list[list[float]]
    inertia: float
    silhouette: float | None


class DBSCANRequest(BaseModel):
    eps: float = Field(default=0.5, gt=0)
    min_samples: int = Field(default=5, ge=1)
    data: list[list[float]]


class DBSCANResponse(BaseModel):
    labels: list[int]
    n_clusters: int
    core_sample_indices: list[int]


class OptimizerRequest(BaseModel):
    n_dims: int = Field(default=2, ge=1, le=20)
    pop_size: int = Field(default=50, ge=4, le=500)
    generations: int = Field(default=100, ge=1, le=10000)
    bounds: list[list[float]]
    seed: int | None = None
    benchmark_fn: str = Field(default="sphere")


class OptimizerResponse(BaseModel):
    best_solution: list[float]
    best_fitness: float
    history: list[float]
    n_dims: int


class GeneticRequest(OptimizerRequest):
    mutation_rate: float = Field(default=0.1, ge=0, le=1)
    crossover_rate: float = Field(default=0.8, ge=0, le=1)
    elitism: int = Field(default=2, ge=0)


class DifferentialRequest(OptimizerRequest):
    F: float = Field(default=0.8, gt=0, le=2)
    CR: float = Field(default=0.9, ge=0, le=1)
    strategy: str = Field(default="best/1/bin")


class PSORequest(OptimizerRequest):
    n_particles: int = Field(default=30, ge=2, le=500)
    w: float = Field(default=0.7, gt=0)
    c1: float = Field(default=1.5, ge=0)
    c2: float = Field(default=1.5, ge=0)
    w_decay: bool = True


class ACORequest(BaseModel):
    n_ants: int = Field(default=20, ge=2, le=200)
    alpha: float = Field(default=1.0, ge=0)
    beta: float = Field(default=2.0, ge=0)
    evaporation: float = Field(default=0.5, gt=0, le=1)
    q: float = Field(default=100.0, gt=0)
    iterations: int = Field(default=100, ge=1)
    seed: int | None = None
    n_cities: int = Field(default=8, ge=3, le=50)


class ACOResponse(BaseModel):
    best_route: list[int]
    best_distance: float
    history: list[float]
    distance_matrix: list[list[float]]
    city_coordinates: list[list[float]]


class FISRequest(BaseModel):
    inputs: dict[str, float]
    method: str = Field(default="centroid")


class FISResponse(BaseModel):
    output: dict[str, float]


class NeuralTrainRequest(BaseModel):
    layer_sizes: list[int] = Field(min_length=2)
    activation: str = Field(default="sigmoid")
    learning_rate: float = Field(default=0.01, gt=0)
    epochs: int = Field(default=1000, ge=1)
    batch_size: int | None = None
    seed: int | None = None
    X: list[list[float]]
    y: list[list[float]]


class NeuralTrainResponse(BaseModel):
    final_mse: float
    initial_mse: float
    n_layers: int
    weights_shapes: list[list[int]]


class NeuralPredictRequest(BaseModel):
    X: list[list[float]]


class NeuralPredictResponse(BaseModel):
    predictions: list[list[float]]


class SARequest(BaseModel):
    n_dims: int = Field(default=2, ge=1, le=20)
    bounds: list[list[float]]
    initial_temp: float = Field(default=100.0, gt=0)
    cooling_rate: float = Field(default=0.99, gt=0, le=1)
    min_temp: float = Field(default=1e-8, ge=0)
    max_iter_per_temp: int = Field(default=10, ge=1)
    cooling_schedule: str = Field(default="exponential")
    seed: int | None = None
    benchmark_fn: str = Field(default="sphere")


class GDRequest(BaseModel):
    n_dims: int = Field(default=2, ge=1, le=20)
    learning_rate: float = Field(default=0.01, gt=0)
    momentum: float = Field(default=0.9, ge=0, le=1)
    method: str = Field(default="sgd")
    max_iter: int = Field(default=1000, ge=1)
    tol: float = Field(default=1e-8, ge=0)
    seed: int | None = None
    benchmark_fn: str = Field(default="sphere")


class NormalizeRequest(BaseModel):
    data: list[list[float]]
    method: str = Field(default="minmax")


class NormalizeResponse(BaseModel):
    data: list[list[float]]
    method: str


class MetricsRequest(BaseModel):
    y_true: list[Any]
    y_pred: list[Any]


class MetricsResponse(BaseModel):
    accuracy: float | None
    mse: float | None
    rmse: float | None
    mae: float | None
    r2: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    confusion_matrix: list[list[int]] | None


class BenchmarkEvalRequest(BaseModel):
    fn_name: str
    x: list[float]


class BenchmarkEvalResponse(BaseModel):
    fn_name: str
    value: float
    bounds: list[float]
    optimum_value: float


class HealthResponse(BaseModel):
    status: str
    version: str
    app: str
