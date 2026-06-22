"""FastAPI router definitions for all CI module endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
import numpy as np

from ci_lib.clustering import KMeans, DBSCAN
from ci_lib.evolutionary import GeneticAlgorithm, DifferentialEvolution
from ci_lib.swarm import ParticleSwarmOptimizer, AntColonyOptimizer
from ci_lib.fuzzy import FuzzyInferenceSystem, FuzzyVariable, FuzzySet, FuzzyRule
from ci_lib.neural import FeedForwardNetwork
from ci_lib.optimization import SimulatedAnnealing, GradientDescent
from ci_lib.utils import normalize, metrics, benchmarks as bm
from backend.models import (
    KMeansRequest, KMeansResponse,
    DBSCANRequest, DBSCANResponse,
    GeneticRequest, DifferentialRequest,
    PSORequest, ACORequest, ACOResponse,
    FISRequest, FISResponse,
    NeuralTrainRequest, NeuralTrainResponse,
    NeuralPredictRequest, NeuralPredictResponse,
    SARequest, GDRequest,
    NormalizeRequest, NormalizeResponse,
    MetricsRequest, MetricsResponse,
    BenchmarkEvalRequest, BenchmarkEvalResponse,
    OptimizerResponse,
    HealthResponse,
)
from backend.config import settings

router = APIRouter()

_BENCHMARK_FNS = {
    "sphere": bm.sphere,
    "rosenbrock": bm.rosenbrock,
    "rastrigin": bm.rastrigin,
    "ackley": bm.ackley,
    "griewank": bm.griewank,
    "schwefel": bm.schwefel,
}


def _get_benchmark(name: str) -> bm._BenchmarkFunction:
    fn = _BENCHMARK_FNS.get(name)
    if fn is None:
        raise HTTPException(400, f"Unknown benchmark '{name}'. Choose from {list(_BENCHMARK_FNS)}")
    return fn


def _make_bounds(n_dims: int, bounds: list[list[float]]) -> np.ndarray:
    b = np.asarray(bounds, dtype=float)
    if b.ndim == 1:
        b = np.tile(b, (n_dims, 1))
    return b


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.version, app=settings.app_name)


@router.post("/clustering/kmeans", response_model=KMeansResponse)
async def kmeans_endpoint(req: KMeansRequest) -> KMeansResponse:
    X = np.asarray(req.data, dtype=np.float64)
    model = KMeans(
        n_clusters=req.n_clusters, max_iter=req.max_iter,
        tol=req.tol, init=req.init, seed=req.seed,
    )
    model.fit(X)
    sil = None
    if req.n_clusters >= 2:
        try:
            sil = model.silhouette_score(X)
        except Exception:
            sil = None
    return KMeansResponse(
        labels=model.labels_.tolist(),
        centers=model.cluster_centers_.tolist(),
        inertia=model.inertia_,
        silhouette=sil,
    )


@router.post("/clustering/dbscan", response_model=DBSCANResponse)
async def dbscan_endpoint(req: DBSCANRequest) -> DBSCANResponse:
    X = np.asarray(req.data, dtype=np.float64)
    model = DBSCAN(eps=req.eps, min_samples=req.min_samples)
    model.fit(X)
    return DBSCANResponse(
        labels=model.labels_.tolist(),
        n_clusters=model.n_clusters_,
        core_sample_indices=model.core_sample_indices_.tolist(),
    )


@router.post("/evolutionary/genetic", response_model=OptimizerResponse)
async def genetic_endpoint(req: GeneticRequest) -> OptimizerResponse:
    fn = _get_benchmark(req.benchmark_fn)
    bounds = _make_bounds(req.n_dims, req.bounds)
    ga = GeneticAlgorithm(
        fitness_fn=fn, n_dims=req.n_dims, bounds=bounds,
        pop_size=req.pop_size, mutation_rate=req.mutation_rate,
        crossover_rate=req.crossover_rate, elitism=req.elitism, seed=req.seed,
    )
    best_sol, best_fit, history = ga.evolve(generations=req.generations)
    return OptimizerResponse(
        best_solution=best_sol.tolist(), best_fitness=best_fit,
        history=history, n_dims=req.n_dims,
    )


@router.post("/evolutionary/differential", response_model=OptimizerResponse)
async def differential_endpoint(req: DifferentialRequest) -> OptimizerResponse:
    fn = _get_benchmark(req.benchmark_fn)
    bounds = _make_bounds(req.n_dims, req.bounds)
    de = DifferentialEvolution(
        fitness_fn=fn, n_dims=req.n_dims, bounds=bounds,
        pop_size=req.pop_size, F=req.F, CR=req.CR,
        strategy=req.strategy, seed=req.seed,
    )
    best_sol, best_fit, history = de.evolve(generations=req.generations)
    return OptimizerResponse(
        best_solution=best_sol.tolist(), best_fitness=best_fit,
        history=history, n_dims=req.n_dims,
    )


@router.post("/swarm/pso", response_model=OptimizerResponse)
async def pso_endpoint(req: PSORequest) -> OptimizerResponse:
    fn = _get_benchmark(req.benchmark_fn)
    bounds = _make_bounds(req.n_dims, req.bounds)
    pso = ParticleSwarmOptimizer(
        fitness_fn=fn, n_dims=req.n_dims, bounds=bounds,
        n_particles=req.n_particles, w=req.w, c1=req.c1, c2=req.c2,
        w_decay=req.w_decay, seed=req.seed,
    )
    best_pos, best_fit, history = pso.optimize(iterations=req.generations)
    return OptimizerResponse(
        best_solution=best_pos.tolist(), best_fitness=best_fit,
        history=history, n_dims=req.n_dims,
    )


@router.post("/swarm/aco", response_model=ACOResponse)
async def aco_endpoint(req: ACORequest) -> ACOResponse:
    rng = np.random.default_rng(req.seed)
    coords = rng.random((req.n_cities, 2)) * 100
    dm = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    aco = AntColonyOptimizer(
        distance_matrix=dm, n_ants=req.n_ants, alpha=req.alpha,
        beta=req.beta, evaporation=req.evaporation, q=req.q, seed=req.seed,
    )
    best_route, best_dist, history = aco.optimize(iterations=req.iterations)
    return ACOResponse(
        best_route=best_route, best_distance=best_dist,
        history=history, distance_matrix=dm.tolist(),
        city_coordinates=coords.tolist(),
    )


@router.post("/fuzzy/default-fis", response_model=FISResponse)
async def fuzzy_default_fis(req: FISRequest) -> FISResponse:
    service = FuzzyVariable("service", (0, 10))
    service.add_set("poor", FuzzySet.triangular("poor", 0, 0, 5))
    service.add_set("good", FuzzySet.triangular("good", 0, 5, 10))
    service.add_set("excellent", FuzzySet.triangular("excellent", 5, 10, 10))

    food = FuzzyVariable("food", (0, 10))
    food.add_set("rancid", FuzzySet.trapezoidal("rancid", 0, 0, 1, 3))
    food.add_set("delicious", FuzzySet.trapezoidal("delicious", 7, 9, 10, 10))

    tip = FuzzyVariable("tip", (0, 30))
    tip.add_set("low", FuzzySet.triangular("low", 0, 5, 10))
    tip.add_set("medium", FuzzySet.triangular("medium", 10, 15, 20))
    tip.add_set("high", FuzzySet.triangular("high", 20, 25, 30))

    fis = FuzzyInferenceSystem()
    fis.add_variable(service, "input")
    fis.add_variable(food, "input")
    fis.add_variable(tip, "output")

    fis.add_rule(FuzzyRule([(service, "poor"), (food, "rancid")], (tip, "low"), "or"))
    fis.add_rule(FuzzyRule([(service, "good")], (tip, "medium"), "and"))
    fis.add_rule(FuzzyRule([(service, "excellent"), (food, "delicious")], (tip, "high"), "and"))

    result = fis.compute(req.inputs, defuzzify_method=req.method)
    return FISResponse(output=result)


@router.post("/fuzzy/custom-fis", response_model=FISResponse)
async def fuzzy_custom_fis(req: FISRequest) -> FISResponse:
    inp = FuzzyVariable("input", (0, 10))
    inp.add_set("low", FuzzySet.triangular("low", 0, 0, 5))
    inp.add_set("high", FuzzySet.triangular("high", 5, 10, 10))

    out = FuzzyVariable("output", (0, 10))
    out.add_set("low", FuzzySet.triangular("low", 0, 0, 5))
    out.add_set("high", FuzzySet.triangular("high", 5, 10, 10))

    fis = FuzzyInferenceSystem()
    fis.add_variable(inp, "input")
    fis.add_variable(out, "output")
    fis.add_rule(FuzzyRule([(inp, "low")], (out, "high"), "and"))
    fis.add_rule(FuzzyRule([(inp, "high")], (out, "low"), "and"))

    result = fis.compute(req.inputs, defuzzify_method=req.method)
    return FISResponse(output=result)


@router.post("/neural/train", response_model=NeuralTrainResponse)
async def neural_train(req: NeuralTrainRequest) -> NeuralTrainResponse:
    X = np.asarray(req.X, dtype=np.float64)
    y = np.asarray(req.y, dtype=np.float64)

    nn = FeedForwardNetwork(
        layer_sizes=req.layer_sizes, activation=req.activation,
        learning_rate=req.learning_rate, seed=req.seed,
    )
    initial_mse = nn.score(X, y)
    nn.fit(X, y, epochs=req.epochs, batch_size=req.batch_size)
    final_mse = nn.score(X, y)

    return NeuralTrainResponse(
        final_mse=final_mse, initial_mse=initial_mse,
        n_layers=len(req.layer_sizes),
        weights_shapes=[list(w.shape) for w in nn.weights],
    )


@router.post("/optimization/simulated-annealing", response_model=OptimizerResponse)
async def sa_endpoint(req: SARequest) -> OptimizerResponse:
    fn = _get_benchmark(req.benchmark_fn)
    bounds = _make_bounds(req.n_dims, req.bounds)
    sa = SimulatedAnnealing(
        objective_fn=fn, n_dims=req.n_dims, bounds=bounds,
        initial_temp=req.initial_temp, cooling_rate=req.cooling_rate,
        min_temp=req.min_temp, max_iter_per_temp=req.max_iter_per_temp,
        cooling_schedule=req.cooling_schedule, seed=req.seed,
    )
    best_sol, best_fit, history = sa.optimize()
    return OptimizerResponse(
        best_solution=best_sol.tolist(), best_fitness=best_fit,
        history=history, n_dims=req.n_dims,
    )


@router.post("/optimization/gradient-descent", response_model=OptimizerResponse)
async def gd_endpoint(req: GDRequest) -> OptimizerResponse:
    fn = _get_benchmark(req.benchmark_fn)
    gd = GradientDescent(
        objective_fn=fn, n_dims=req.n_dims,
        learning_rate=req.learning_rate, momentum=req.momentum,
        method=req.method, seed=req.seed,
    )
    best_sol, best_val, history = gd.optimize(max_iter=req.max_iter, tol=req.tol)
    return OptimizerResponse(
        best_solution=best_sol.tolist(), best_fitness=best_val,
        history=history, n_dims=req.n_dims,
    )


@router.post("/utils/normalize", response_model=NormalizeResponse)
async def normalize_endpoint(req: NormalizeRequest) -> NormalizeResponse:
    X = np.asarray(req.data, dtype=np.float64)
    result = normalize(X, method=req.method)
    return NormalizeResponse(data=result.tolist(), method=req.method)


@router.post("/utils/metrics", response_model=MetricsResponse)
async def metrics_endpoint(req: MetricsRequest) -> MetricsResponse:
    yt, yp = np.asarray(req.y_true), np.asarray(req.y_pred)
    resp = MetricsResponse(
        accuracy=None, mse=None, rmse=None, mae=None,
        r2=None, precision=None, recall=None, f1=None, confusion_matrix=None,
    )
    try:
        resp.accuracy = metrics.accuracy(yt, yp)
    except Exception:
        pass
    try:
        resp.mse = metrics.mse(yt, yp)
        resp.rmse = metrics.rmse(yt, yp)
        resp.mae = metrics.mae(yt, yp)
        resp.r2 = metrics.r_squared(yt, yp)
    except Exception:
        pass
    try:
        p, r, f = metrics.precision_recall_f1(yt, yp)
        resp.precision = p
        resp.recall = r
        resp.f1 = f
        cm = metrics.confusion_matrix(yt, yp)
        resp.confusion_matrix = cm.tolist()
    except Exception:
        pass
    return resp


@router.get("/utils/benchmarks")
async def list_benchmarks() -> dict:
    return {
        "benchmarks": {
            name: {"bounds": fn.bounds, "optimum_value": fn._optimum_value}
            for name, fn in _BENCHMARK_FNS.items()
        }
    }


@router.post("/utils/benchmarks/evaluate", response_model=BenchmarkEvalResponse)
async def evaluate_benchmark(req: BenchmarkEvalRequest) -> BenchmarkEvalResponse:
    fn = _get_benchmark(req.fn_name)
    x = np.asarray(req.x, dtype=np.float64)
    value = fn(x)
    return BenchmarkEvalResponse(
        fn_name=req.fn_name, value=value,
        bounds=list(fn.bounds), optimum_value=fn._optimum_value,
    )
