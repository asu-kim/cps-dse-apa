from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import (SessionLocal, Body, Motor, Battery, Tires, Communication, ComputingUnit, DroneBody, DroneMotor,
                      DroneBattery, Application, Component, EdgeServer)
from fastapi.middleware.cors import CORSMiddleware
from pulp import LpProblem, LpVariable, PULP_CBC_CMD, LpMinimize, LpInteger, value
from typing import List, Annotated
from pydantic import BaseModel,Field
import random
import math
import heapq
from skopt import Optimizer
from skopt.space import Categorical, Integer
import warnings
import time


app = FastAPI()


# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

warnings.filterwarnings("ignore", category=UserWarning)
# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Request Model
class OptimizationRequest(BaseModel):
    budget: float
    farm_size: float
    crop_type: str
    applications: Annotated[List[str], Field(default_factory=list)]

def solve_ilp(possible_configurations, budget, farm_size, edge_cost, extra_cost,α: float = 0.0, β: float = 0.0, γ: float = 0.0):
    fleets = []
    for i, config in enumerate(possible_configurations):
        model = LpProblem(f"SingleConfig_{i}", LpMinimize)
        x = LpVariable("units", lowBound=0, cat=LpInteger)

        cov  = config["coverage"]
        cost = config["cost"] + extra_cost

        # cover the farm
        model += x * cov >= farm_size
        # stay within budget
        model += x * cost + edge_cost <= budget
        # Enforce minimum runtime per‐unit
        MIN_RUNTIME = 0.2
        if config["type"] == "Drone":
            model += config["runtime_hours"] * x >= MIN_RUNTIME * x


        model += α * config["norm_cost"] * x + β * config["norm_coverage"] * x - γ * config["norm_payload"] * x

        status = model.solve(PULP_CBC_CMD(msg=False))
        if status == 1:  # feasible
            qty = int(x.value())
            objective_val = value(model.objective)
            #print(f"[ILP] Config {i}: qty={qty}, cost={config['cost']}, obj={objective_val}")
            fleets.append((i, qty, objective_val))


    # sort by total fleet
    fleets.sort(key=lambda t: t[2])  # use the stored objective
    fleets = [(i, qty) for i, qty, _ in fleets[:20]]
    return fleets[:20]

def solve_area_payload(possible_configurations, farm_size,edge_cost, extra_cost, α: float = 0.0, β: float = 0.0, γ: float = 0.0):
    fleets = []
    for i, config in enumerate(possible_configurations):
        model = LpProblem(f"SingleConfig_{i}", LpMinimize)
        x = LpVariable("units", lowBound=0, cat=LpInteger)

        cov = config["coverage"]
        # cover the farm
        model += x * cov >= farm_size
        # Enforce minimum runtime per‐unit
        MIN_RUNTIME = 0.25
        if config["type"] == "Drone":
            model += config["runtime_hours"] * x >= MIN_RUNTIME * x


        model += α * config["norm_cost"] * x + β * config["norm_coverage"] * x - γ * config["norm_payload"] * x

        status = model.solve(PULP_CBC_CMD(msg=False))
        if status == 1:  # feasible
            qty = int(x.value())
            fleets.append((i, qty))
    return fleets[:20]


def solve_area_payload(possible_configurations, farm_size,edge_cost, extra_cost, α: float = 0.0, β: float = 0.0, γ: float = 0.0):
    fleets = []
    for i, config in enumerate(possible_configurations):
        model = LpProblem(f"SingleConfig_{i}", LpMinimize)
        x = LpVariable("units", lowBound=0, cat=LpInteger)

        cov = config["coverage"]
        # cover the farm
        model += x * cov >= farm_size
        # Enforce minimum runtime per‐unit
        MIN_RUNTIME = 0.25
        if config["type"] == "Drone":
            model += config["runtime_hours"] * x >= MIN_RUNTIME * x

        model += α * config["norm_cost"] * x + β * config["norm_coverage"] * x - γ * config["norm_payload"] * x

        status = model.solve(PULP_CBC_CMD(msg=False))
        if status == 1:  # feasible
            qty = int(x.value())
            fleets.append((i, qty))
    return fleets[:20]


def solve_cost_area(possible_configurations, budget, farm_size, edge_cost, extra_cost,α: float = 0.0, β: float = 0.0, γ: float = 0.0):
    fleets = []
    for i, config in enumerate(possible_configurations):
        model = LpProblem(f"SingleConfig_{i}", LpMinimize)
        x = LpVariable("units", lowBound=0, cat=LpInteger)

        cov = config["coverage"]
        cost = config["cost"] + extra_cost

        # cover the farm
        model += x * cov >= farm_size
        # stay within budget
        model += x * cost + edge_cost <= budget
        # Enforce minimum runtime per‐unit
        MIN_RUNTIME = 0.25
        if config["type"] == "Drone":
            model += config["runtime_hours"] * x >= MIN_RUNTIME * x

        # objective
        model += (α * config["norm_cost"]  * x
                  + β * config["norm_coverage"] * x
                  - γ * config["norm_payload"] * x)

        status = model.solve(PULP_CBC_CMD(msg=False))
        if status == 1:  # feasible
            qty = int(x.value())
            fleets.append((i, qty))

    fleets.sort(key=lambda t: (
            t[1] * (possible_configurations[t[0]]["cost"] + extra_cost) + edge_cost
            - possible_configurations[t[0]]["norm_coverage"],)
        # - possible_configurations[t[0]]["norm_payload"],
    )
    return fleets[:20]


def solve_payload_cost(possible_configurations, budget, edge_cost, extra_cost,α: float = 0.0, β: float = 0.0, γ: float = 0.0):
    fleets = []
    for i, config in enumerate(possible_configurations):
        model = LpProblem(f"SingleConfig_{i}", LpMinimize)
        x = LpVariable("units", lowBound=1, cat=LpInteger)

        cost = config["cost"] + extra_cost

        # stay within budget
        model += x * cost + edge_cost <= budget
        # Enforce minimum runtime per‐unit
        MIN_RUNTIME = 0.25
        if config["type"] == "Drone":
            model += config["runtime_hours"] * x >= MIN_RUNTIME * x

        # objective
        model += (α * (config["norm_cost"] + extra_cost) * x
                  + β * config["norm_coverage"] * x
                  - γ * config["norm_payload"] * x)

        status = model.solve(PULP_CBC_CMD(msg=False))
        if status == 1:  # feasible
            qty = int(x.value())
            fleets.append((i, qty))

    # sort by total fleet cost
    fleets.sort(key=lambda t: (
        t[1] * (possible_configurations[t[0]]["norm_cost"] + extra_cost) + edge_cost)
    )
    return fleets[:20]

def solve_simulated_annealing(configs, budget, farm_size, edge_cost, extra_cost,
                              α=0.5, β=0.3, γ=0.2, max_iter=1000,
                              T_start=100.0, T_end=0.1, cooling_rate=0.95, max_qty=10):
    def objective(config, x):
        if config["type"] == "Drone" and config["runtime_hours"] * x < 0.25 * x:
            return float("inf")
        if x * (config["cost"] + extra_cost) + edge_cost > budget:
            return float("inf")
        if x * config["coverage"] < farm_size:
            return float("inf")
        return α * config["norm_cost"] * x + β * config["norm_coverage"] * x - γ * config["norm_payload"] * x

    T = T_start
    current_config = random.choice(configs)
    current_qty = random.randint(1, max_qty)
    current_score = objective(current_config, current_qty)

    best = (current_config, current_qty, current_score)
    seen = set()
    results = []

    while T > T_end and len(results) < 1000:
        new_config = random.choice(configs)
        new_qty = random.randint(1, max_qty)
        new_score = objective(new_config, new_qty)

        if new_score < current_score:
            accept = True
        else:
            delta = new_score - current_score
            accept_prob = math.exp(-delta / T)
            accept = random.random() < accept_prob

        if accept:
            current_config, current_qty, current_score = new_config, new_qty, new_score
            key = (configs.index(current_config), current_qty)
            if key not in seen and current_score < float("inf"):
                results.append((key[0], key[1], current_score))
                seen.add(key)

        T *= cooling_rate

    results.sort(key=lambda x: x[2])
    return [(i, qty) for i, qty, _ in results[:20]]

def solve_bayesian_optimization(configs, budget, farm_size, edge_cost, extra_cost,
                                α=0.5, β=0.3, γ=0.2, n_calls=50, max_qty=10):
    # 1) Define the search space: config index ∈ {0…d-1}, quantity ∈ {1…max_qty}
    d = len(configs)
    space = [
        Categorical(list(range(d)), name="config_idx"),
        Integer(1, max_qty, name="qty")
    ]

    # 2) Define the (to-minimize) objective with hard penalties for infeasible points
    INF = 1e6
    def objective(params):
        idx, x = params
        cfg = configs[int(idx)]
        cost_total = x * (cfg["cost"] + extra_cost) + edge_cost
        # feasibility checks
        if cost_total > budget:
            return INF
        if x * cfg["coverage"] < farm_size:
            return INF
        if cfg.get("type") == "Drone" and cfg["runtime_hours"] * x < 0.25 * x:
            return INF
        # weighted objective (same as your others)
        return α * cfg["norm_cost"] * x + β * cfg["norm_coverage"] * x - γ * cfg["norm_payload"] * x

    # 3) Create the BO optimizer and run ask/tell loop
    opt = Optimizer(
        dimensions=space,
        base_estimator="GP",   # gaussian process surrogate
        acq_func="EI",         # expected improvement
        random_state=0
    )

    results = []
    for _ in range(n_calls):
        next_pt = opt.ask()
        fval = objective(next_pt)
        opt.tell(next_pt, fval)
        # collect only feasible ones
        if fval < INF:
            idx, x = next_pt
            results.append((int(idx), int(x), fval))

    # 4) Sort by objective and return top 20 (config_idx, quantity)
    results.sort(key=lambda t: t[2])
    top20 = [(i, qty) for i, qty, _ in results[:20]]
    return top20

def solve_pg_dse(configs, budget, farm_size, edge_cost, extra_cost, pop_size=50, generations=20,max_qty=10):
    # === Step 1: PSP - Pareto-optimal Subspace Pruning ===
    def pareto_prune(configs):
        pareto_set = []
        for c in configs:
            dominated = False
            for other in configs:
                if (other["coverage"] >= c["coverage"] and
                    other["payload"] >= c["payload"] and
                    other["cost"] <= c["cost"] and
                    (other["coverage"] > c["coverage"] or
                     other["payload"] > c["payload"] or
                     other["cost"] < c["cost"])):
                    dominated = True
                    break
            if not dominated:
                pareto_set.append(c)
        return pareto_set

    configs = pareto_prune(configs)

    def is_feasible(config, x):
        cost_total = x * (config["cost"] + extra_cost) + edge_cost
        if config["type"] == "Drone" and config["runtime_hours"] * x < 0.25 * x:
            return False
        if cost_total > budget or x * config["coverage"] < farm_size:
            return False
        return True

    def mutate(ind):
        config, qty = ind
        qty = max(1, qty + random.choice([-1, 0, 1]))
        new_config = random.choice(configs) if random.random() < 0.3 else config
        return (new_config, qty)

    def crossover(p1, p2):
        return (random.choice([p1[0], p2[0]]), random.choice([p1[1], p2[1]]))

    def update_pareto_front(candidates):
        pareto = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if (other[2] >= c[2] and  # coverage
                    other[3] >= c[3] and  # payload
                    other[4] <= c[4] and  # cost
                    (other[2] > c[2] or other[3] > c[3] or other[4] < c[4])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(c)
        return pareto

    # === Step 2: PEGA - Pareto-optimal Elite Genetic Algorithm ===
    population = []
    while len(population) < pop_size:
        config = random.choice(configs)
        qty = random.randint(1, max_qty)
        if is_feasible(config, qty):
            population.append((config, qty))

    archive = []

    for _ in range(generations):
        scored = []
        for cfg, qty in population:
            if is_feasible(cfg, qty):
                scored.append((cfg, qty, cfg["coverage"] * qty, cfg["payload"] * qty, cfg["cost"] * qty))
        archive = update_pareto_front(archive + scored)
        survivors = scored[:pop_size // 2]

        new_gen = []
        while len(new_gen) < pop_size:
            if len(survivors) < 2:
                break
            p1, p2 = random.sample(survivors, 2)
            child = crossover(p1, p2)
            mutated = mutate(child)
            if is_feasible(*mutated):
                new_gen.append(mutated)

        population = new_gen

    seen = set()
    final = []
    for config, qty, _, _, _ in archive:
        key = (configs.index(config), qty)
        if key not in seen:
            final.append(key)
            seen.add(key)
        if len(final) >= 20:
            break
    return final

def solve_random_search(configs, budget, farm_size, edge_cost, extra_cost, α=0.5, β=0.3, γ=0.2, trials=500, max_qty=10):
    candidates = []
    for _ in range(trials):
        config = random.choice(configs)
        x = random.randint(1, max_qty)
        cost_total = x * (config["cost"] + extra_cost) + edge_cost
        if config["type"] == "Drone" and config["runtime_hours"] * x < 0.25 * x:
            continue
        if cost_total > budget or x * config["coverage"] < farm_size:
            continue
        obj = α * config["norm_cost"] * x + β * config["norm_coverage"] * x - γ * config["norm_payload"] * x
        candidates.append((configs.index(config), x, obj))
    candidates.sort(key=lambda c: c[2])
    return [(i, qty) for i, qty, _ in candidates[:20]]

def solve_genetic_algorithm(configs, budget, farm_size, edge_cost, extra_cost, α=0.5, β=0.3, γ=0.2,
                            pop_size=50, generations=20, max_qty=10):
    d = len(configs)
    alpha = 1.54468
    p = alpha / d

    def fitness(config, x):
        if config["type"] == "Drone" and config["runtime_hours"] * x < 0.25 * x:
            return float("inf")
        if x * (config["cost"] + extra_cost) + edge_cost > budget:
            return float("inf")
        if x * config["coverage"] < farm_size:
            return float("inf")
        return α * config["norm_cost"] * x + β * config["norm_coverage"] * x - γ * config["norm_payload"] * x

    population = [(random.choice(configs), random.randint(1, max_qty)) for _ in range(pop_size)]
    archive = []

    for _ in range(generations):
        scored = [(c, x, fitness(c, x)) for c, x in population if fitness(c, x) < float("inf")]
        if not scored:
            break
        scored.sort(key=lambda s: s[2])
        archive.extend(scored)
        survivors = scored[:pop_size // 2]
        if len(survivors) < 2:
            break
        population = [(c, x) for c, x, _ in survivors]
        while len(population) < pop_size:
            if len(survivors) >= 2:
                p1, p2 = random.sample(survivors, 2)
            else:
                p1 = p2 = random.choice(survivors)
            c = random.choice([p1[0], p2[0]])
            x = random.randint(1, max_qty)
            # FastGA mutation
            if random.random() < p:
                c = random.choice(configs)
            population.append((c, x))

    archive.sort(key=lambda s: s[2])
    unique_solutions = {}
    for c, x, score in archive:
        key = (configs.index(c), x)
        if key not in unique_solutions or score < unique_solutions[key]:
            unique_solutions[key] = score
    return [(i, qty) for (i, qty), _ in sorted(unique_solutions.items(), key=lambda t: t[1])[:20]]


def solve_discrete(configs, budget, farm_size, edge_cost, extra_cost, α=0.5, β=0.3, γ=0.2, max_qty=10):
    d = len(configs)
    candidates = []
    for config in configs:
        for qty in range(1, max_qty + 1):
            if random.random() > 1 / d:
                continue
            total_cost = qty * (config["cost"] + extra_cost) + edge_cost
            if total_cost > budget or qty * config["coverage"] < farm_size:
                continue
            if config["type"] == "Drone" and config["runtime_hours"] * qty < 0.25 * qty:
                continue
            obj = α * config["norm_cost"] * qty + β * config["norm_coverage"] * qty - γ * config["norm_payload"] * qty
            candidates.append((config, qty, obj))
    candidates.sort(key=lambda x: x[2])
    seen = set()
    final = []
    for config, qty, _ in candidates:
        key = (configs.index(config), qty)
        if key not in seen:
            final.append(key)
            seen.add(key)
        if len(final) >= 20:
            break
    return final

def solve_lengler(configs, budget, farm_size, edge_cost, extra_cost,
                  α=0.5, β=0.3, γ=0.2, max_qty=1000, top_k=20):

    def objective(cfg, x):
        return α*cfg["norm_cost"]*x + β*cfg["norm_coverage"]*x - γ*cfg["norm_payload"]*x

    def is_feasible(cfg, x):
        total = x * (cfg["cost"] + extra_cost) + edge_cost
        if total > budget or x * cfg["coverage"] < farm_size:
            return False
        if cfg["type"] == "Drone" and cfg["runtime_hours"] * x < 0.25 * x:
            return False
        return True

    # min-heap of size <= top_k, storing (-score, idx, qty)
    best = []

    def consider(idx, qty):
        sc = objective(configs[idx], qty)
        entry = (-sc, idx, qty)
        if len(best) < top_k:
            heapq.heappush(best, entry)
        else:
            # best[0] is the *worst* of our top-K (it's the largest negative)
            if entry[0] > best[0][0]:
                heapq.heapreplace(best, entry)

    # 1) scan the entire (config × qty) space, but never store more than top_k
    for idx, cfg in enumerate(configs):
        for qty in range(1, max_qty + 1):
            if not is_feasible(cfg, qty):
                continue
            consider(idx, qty)

    # 2) unpack the heap into a ranked list of (idx, qty)
    top = sorted(best, reverse=True)  # largest scores first
    return [(idx, qty) for _, idx, qty in top]



def solve_portfolio(configs, budget, farm_size, edge_cost, extra_cost, max_qty=10):
    results = []
    optimizers = [
        lambda *args, **kw: solve_genetic_algorithm(*args, **kw, max_qty=max_qty),
        lambda *args, **kw: solve_lengler(*args, **kw, max_qty=max_qty),
        lambda *args, **kw: solve_discrete(*args, **kw, max_qty=max_qty),
        lambda *args, **kw: solve_random_search(*args, **kw, max_qty=max_qty)
    ]

    for solver in optimizers:
        try:
            out = solver(configs, budget, farm_size, edge_cost, extra_cost)
            if out:
                config, qty = out[0]
                total_cost = qty * (configs[config]["cost"] + extra_cost) + edge_cost
                if total_cost <= budget:
                    results.append((config, qty))
        except Exception as e:
            print(f"Optimizer {solver.__name__} failed: {e}")

    def obj(idx, qty):
        cfg = configs[idx]
        return (cfg["norm_cost"] * qty + 0.3 * cfg["norm_coverage"] * qty - 0.2 * cfg["norm_payload"] * qty)

    results.sort(key=lambda pair: obj(*pair))
    return results[:20] if results else []

def timed(label, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    elapsed = round((end - start) * 1000, 3)  # ms
    print(f"{label} runtime: {elapsed} ms")
    return result, elapsed

# **API Route**
@app.post("/get_options")
def get_options(request: OptimizationRequest, db: Session = Depends(get_db)):

    # decide overall compute‐mode: if *any* application needs Onboard, we do Onboard; else Off-board
    edge = db.query(EdgeServer).first()
    edge_cost = edge.cost if edge else 0.0
    apps = db.query(Application).filter(Application.name.in_(request.applications)).all()
    if len(apps) != len(request.applications):
        return {
            "message": "One or more requested applications not found in database."
        }
    allowed_per_app = []
    for app in apps:
        if app.platform == "Both":
            allowed_per_app.append({"Drone", "Rover"})
        else:
            allowed_per_app.append({app.platform})

    common_platforms = set.intersection(*allowed_per_app)
    if not common_platforms:
        # No single platform can satisfy every chosen application
        return {
            "message": "No platform—Drone or Rover—can handle all selected applications simultaneously."
        }
    processing_modes = {app.processing_mode for app in apps}
    computing_mode = "Onboard" if "Onboard" in processing_modes else "Off-board"

    # map each crop to its allowed platform(s):
    crop_platform_map = {
        "Indoor": {"Rover"},
        "Fiber": {"Drone"},
        "Legume": {"Drone"},
        "Paddy": {"Drone"},
        "Cereal": {"Rover", "Drone"},
        "Tree": {"Rover", "Drone"},
        "Orchard": {"Rover", "Drone"},
        "Oilseed": {"Rover", "Drone"},
        # any others default to Both:
    }

    crop_platform = crop_platform_map.get(request.crop_type, {"Rover", "Drone"})

    # intersect with per-application platforms:
    allowed_platforms = common_platforms & crop_platform

    if not allowed_platforms:
        return {
            "message": f"No feasible configurations for crop '{request.crop_type}' on platform(s) "
                       f"{sorted(crop_platform)} with those applications."}

    needed_comps = []
    for app in apps:
        needed_comps.extend(app.components) # these are full Component objects
    # Deduplicate by primary key:
    needed_comps = {c.id: c for c in needed_comps}.values()

    response = get_possible_options(db, request.budget, request.farm_size, request.crop_type,
                                    computing_mode,list(needed_comps),edge_cost)
    configs = response["possible_configurations"]

    if not configs:
        return {"message": "No valid configurations found."}  # Prevent frontend errors

    extra_cost = sum(c.cost for c in needed_comps)

    max_cost = max(config["cost"] for config in configs) or 1
    max_cov = max(config["coverage"] for config in configs) or 1
    max_payload = max(config["payload"] for config in configs) or 1

    # Inject normalized values into config for reuse
    for config in configs:
        config["norm_cost"] = config["cost"] / max_cost
        config["norm_coverage"] = config["coverage"] / max_cov
        config["norm_payload"] = config["payload"] / max_payload

    min_unit_cost = min(cfg["cost"] + extra_cost for cfg in configs)
    min_coverage = min(cfg["coverage"] for cfg in configs)


    max_by_budget = int((request.budget - edge_cost) // min_unit_cost)
    max_by_coverage = math.ceil(request.farm_size / min_coverage)


    max_qty = min(max_by_budget, max_by_coverage)


    # 6) Run all four solvers on that single “all_configs” list:
    (fleets, ilp_time) = timed("ILP", solve_ilp, configs, request.budget, request.farm_size,
                               edge_cost, extra_cost, α=0.611, β=0.278, γ=0.111)
    (area_payload_counts, area_payload_time) = timed(
        "Area-Payload", solve_area_payload, configs, request.farm_size,
        edge_cost, extra_cost, β=0.5, γ=0.5)
    (cost_area_counts, cost_area_time) = timed(
        "Cost-Area", solve_cost_area, configs, request.budget, request.farm_size,
        edge_cost, extra_cost, α=0.5, β=0.5)
    (payload_cost_counts, payload_cost_time) = timed(
        "Payload-Cost", solve_payload_cost, configs, request.budget,
        edge_cost, extra_cost, α=0.5, γ=0.5)
    (simulated_annealing, sa_time) = timed(
        "Simulated Annealing", solve_simulated_annealing, configs,
        request.budget, request.farm_size, edge_cost, extra_cost, α=0.611, β=0.278, γ=0.111,
        max_iter=1000, T_start=100, T_end=0.1, cooling_rate=0.95, max_qty=max_qty)
    (bayesian, bo_time) = timed(
        "Bayesian Optimization", solve_bayesian_optimization, configs, request.budget,
        request.farm_size, edge_cost, extra_cost, α=0.611, β=0.278, γ=0.111, n_calls=50, max_qty=max_qty)
    (pg_dse, pg_time) = timed(
        "PG-DSE", solve_pg_dse, configs, request.budget, request.farm_size,
        edge_cost, extra_cost, pop_size=50, generations=20, max_qty=max_qty)
    (random_search, rs_time) = timed(
        "Random Search", solve_random_search, configs, request.budget, request.farm_size, edge_cost,
        edge_cost, α=0.611, β=0.278, γ=0.111, trials=500, max_qty=max_qty)
    (genetic_algorithm, ga_time) = timed(
        "Genetic Algorithm", solve_genetic_algorithm, configs, request.budget, request.farm_size,
        edge_cost, extra_cost, α=0.611, β=0.278, γ=0.111, pop_size=50, generations=20, max_qty=max_qty)
    (discrete, disc_time) = timed(
        "Discrete", solve_discrete, configs, request.budget, request.farm_size,
        edge_cost, extra_cost, α=0.5, γ=0.5, max_qty=max_qty)
    (lengler, lengler_time) = timed(
        "Lengler", solve_lengler, configs, request.budget, request.farm_size,
        edge_cost, extra_cost, α=0.611, β=0.278, γ=0.111, max_qty=max_qty, top_k=20)
    (portfolio, portfolio_time) = timed(
        "Portfolio", solve_portfolio, configs, request.budget,
        request.farm_size, edge_cost, extra_cost, max_qty=max_qty)


    full_objective = []
    for idx, qty in fleets:
        base_config   = configs[idx]
        unit_cost  = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        full_objective.append({
            "type":            base_config["type"],
            "body":            base_config["body"],
            "motor":           base_config["motor"],
            "battery":         base_config["battery"],
            "computing_device":base_config["computing_device"],
            "computing_type":  base_config["computing_type"],
            "computing_cost":  base_config["computing_cost"],
            "computing_perf":  base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server":     {"name": edge.name, "cost": edge_cost},
            "base_cost":       base_config["cost"],
            "additional_components": [
            {"name":c.name, "category":c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost":            base_config["cost"] + extra_cost + edge_cost,
            "coverage":        base_config["coverage"],
            "payload":         base_config["payload"],
            "runtime_hours":   base_config["runtime_hours"],
            "quantity":        qty,
            "total_cost":      round(qty*(base_config["cost"]+extra_cost) + edge_cost,2),
            "total_coverage":  round(qty*base_config["coverage"],2),
        })

    full_objective.sort(key=lambda c: c["total_cost"])

    area_payload_list = []
    for idx, qty in area_payload_counts:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        area_payload_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    area_payload_list.sort(key=lambda c: c["coverage"])

    cost_area_list = []
    for idx, qty in cost_area_counts:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        cost_area_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    cost_area_list.sort(key=lambda c: c["total_cost"])

    payload_cost_list = []
    for idx, qty in payload_cost_counts:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        payload_cost_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    payload_cost_list.sort(key=lambda c: c["total_cost"])

    simulated_annealing_list = []
    for idx, qty in simulated_annealing:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        simulated_annealing_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    simulated_annealing_list.sort(key=lambda c: c["total_cost"])
    #
    bayesian_list = []
    for idx, qty in bayesian:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        bayesian_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    bayesian_list.sort(key=lambda c: c["total_cost"])

    pg_dse_list = []
    for idx, qty in pg_dse:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        pg_dse_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    pg_dse_list.sort(key=lambda c: c["total_cost"])

    random_search_list = []
    for idx, qty in random_search:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        random_search_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    random_search_list.sort(key=lambda c: c["total_cost"])


    genetic_algorithm_list = []
    for idx, qty in genetic_algorithm:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        genetic_algorithm_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    genetic_algorithm_list.sort(key=lambda c: c["total_cost"])

    discrete_list = []
    for idx, qty in discrete:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        discrete_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    discrete_list.sort(key=lambda c: c["total_cost"])

    lengler_list = []
    for idx, qty in lengler:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        lengler_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    lengler_list.sort(key=lambda c: c["total_cost"])

    portfolio_list = []
    for idx, qty in portfolio:
        base_config = configs[idx]
        unit_cost = base_config["cost"] + extra_cost
        total_cost = unit_cost * qty + edge_cost
        portfolio_list.append({
            "type": base_config["type"],
            "body": base_config["body"],
            "motor": base_config["motor"],
            "battery": base_config["battery"],
            "computing_device": base_config["computing_device"],
            "computing_type": base_config["computing_type"],
            "computing_cost": base_config["computing_cost"],
            "computing_perf": base_config["computing_perf"],
            "computing_power_watts": base_config["computing_power_watts"],
            "edge_server": {"name": edge.name, "cost": edge_cost},
            "base_cost": base_config["cost"],
            "additional_components": [
                {"name": c.name, "category": c.category} for c in needed_comps
            ],
            "additional_cost": extra_cost,
            "cost": base_config["cost"] + extra_cost + edge_cost,
            "coverage": base_config["coverage"],
            "payload": base_config["payload"],
            "runtime_hours": base_config["runtime_hours"],
            "quantity": qty,
            "total_cost": round(qty * (base_config["cost"] + extra_cost) + edge_cost, 2),
            "total_coverage": round(qty * base_config["coverage"], 2),
        })

    portfolio_list.sort(key=lambda c: c["total_cost"])


    #filter by requested platform
    def filtered_by_platform(sols):
        out = []
        if isinstance(sols, dict):
            sols = [
                configs[idx] | {"quantity": qty}
                for idx, qty in sols.items()
            ]
        for sol in sols:
            if sol["type"] in allowed_platforms:
                out.append(sol)
        return out

    full_objective      = filtered_by_platform(full_objective)
    area_payload        = filtered_by_platform(area_payload_list)
    cost_area           = filtered_by_platform(cost_area_list)
    payload_cost        = filtered_by_platform(payload_cost_list)
    simulated_annealing = filtered_by_platform(simulated_annealing_list)
    bayesian = filtered_by_platform(bayesian_list)
    pg_dse = filtered_by_platform(pg_dse_list)
    random_search = filtered_by_platform(random_search_list)
    genetic_algorithm = filtered_by_platform(genetic_algorithm_list)
    discrete = filtered_by_platform(discrete_list)
    lengler = filtered_by_platform(lengler_list)
    portfolio = filtered_by_platform(portfolio_list)

    def nonempty(sols):
        return [sol for sol in sols if sol]

    full_objective      = nonempty(full_objective)
    area_payload        = nonempty(area_payload)
    cost_area           = nonempty(cost_area)
    payload_cost        = nonempty(payload_cost)
    simulated_annealing = nonempty(simulated_annealing)
    bayesian = nonempty(bayesian)
    pg_dse = nonempty(pg_dse)
    random_search = nonempty(random_search)
    genetic_algorithm = nonempty(genetic_algorithm)
    discrete = nonempty(discrete)
    lengler = nonempty(lengler)
    portfolio = nonempty(portfolio)

    if not (full_objective or area_payload or cost_area or payload_cost):
        return {"message": "No feasible configurations after applying crop + application platform constraints."}

    return {
        "proposed_approach": full_objective,
        # "area_payload": area_payload,
        # "cost_area": cost_area,
        # "payload_cost": payload_cost,
        "simulated_annealing": simulated_annealing,
        "bayesian": bayesian,
        "pg_dse": pg_dse,
        "random_search": random_search,
        "genetic_algorithm": genetic_algorithm,
        "discrete": discrete,
        "lengler": lengler,
        "portfolio": portfolio,
    "runtimes_ms": {
        "ilp": ilp_time,
        "area_payload": area_payload_time,
        "cost_area": cost_area_time,
        "payload_cost": payload_cost_time,
        "simulated_annealing": sa_time,
        "bayesian": bo_time,
        "pg_dse": pg_time,
        "random_search": rs_time,
        "genetic_algorithm": ga_time,
        "discrete": disc_time,
        "lengler": lengler_time,
        "portfolio": portfolio_time
    }
    }

# **Function to Generate Possible Configurations**
def get_possible_options(db: Session, budget: float, farm_size: float, crop_type: str,
                         computing_mode: str, needed_comps: List[Component], edge_cost: float):
    bodies = db.query(Body).all()
    motors = db.query(Motor).all()
    batteries = db.query(Battery).all()
    tires = db.query(Tires).all()
    drone_bodies = db.query(DroneBody).all()
    drone_motors = db.query(DroneMotor).all()
    drone_batteries = db.query(DroneBattery).all()
    communications = db.query(Communication).all()
    computing_units = db.query(ComputingUnit).all()
    edge = db.query(EdgeServer).first()
    edge_cost = edge.cost if edge else 0.0


    possible_configurations = []

    size_rank = {"Small": 1, "Medium": 2, "Large": 3}

    # Compute weight factor
    def compute_weight_factor(W_max):
        return 1 / (1 + (W_max / 100))

    # Newtons of weight, divided by your 100 N normalization constant
    def compute_factor(mass_kg: float) -> float:
        return (mass_kg * 9.81) / 100.0

    # Compute max payload for rovers
    def compute_max_payload(body, motor, battery, tires, num_motors=4):
        torque = motor.torque
        wheel_radius = tires.wheel_radius
        W_motors = torque  * num_motors / ( wheel_radius * 9.8 )
        bf = compute_factor(body.mass_kg)
        mf = compute_factor(motor.mass_kg)
        batf = compute_factor(battery.mass_kg)
        tf = compute_factor(tires.mass_kg)
        W_max = W_motors - (bf + mf + batf + tf)
        return round(W_max, 2)

    # Compute area coverage for rovers
    def compute_area_coverage(battery, motor, tires, weight_factor):
        bf = compute_factor(battery.mass_kg)
        mf = compute_factor(motor.mass_kg)
        tf = compute_factor(tires.mass_kg)
        return bf * mf * tf * weight_factor * tires.wheel_radius * 10000

    # **Compute Drone Payload (Uses Factor & Torque)**
    def compute_drone_payload(drone_body, drone_motor, drone_battery, num_motors=4):
        torque = drone_motor.torque
        W_motors = torque * num_motors / 9.8
        bf = compute_factor(drone_body.mass_kg)
        mf = compute_factor(drone_motor.mass_kg)
        batf = compute_factor(drone_battery.mass_kg)

        # Subtract Battery Weight from W_max Calculation
        W_max = W_motors - (bf + mf + batf)

        # print(f"Drone Payload Computed: W_max = {W_max} (W_motors: {W_motors}, Battery Weight: {drone_battery.mass_kg})")

        return round(W_max, 2)

    # **Compute Drone Coverage (Battery & Motor Factor)**
    def compute_drone_coverage(drone_battery, drone_motor, weight_factor):
        bf = compute_factor(drone_battery.mass_kg)
        mf = compute_factor(drone_motor.mass_kg)

        return bf * mf * weight_factor * 1000

    def pick_compute():
        if computing_mode == "Off-board":
            cands = [c for c in computing_units if c.unit_type in ("CPU","GPU","TPU")]
            return min(cands, key=lambda c: c.cost)
        else:
            cands = [c for c in computing_units if c.unit_type in ("GPU","TPU")]
            return max(cands, key=lambda c: c.performance)

    print("\n FINDING CONFIGURATIONS...")

    # Precompute the total extra‐components draw in watts:
    comps_draw = sum(c.power_watts for c in needed_comps)

    # **Rovers**
    for body in bodies:
        for motor in motors:
            #Require motor.size exactly match body.size
            if motor.size != body.size:
                continue
            for battery in batteries:
                for tire in tires:
                    computing    = pick_compute()
                    best = None
                    best_total_comm_cost = float("inf")

                    for comm in communications:
                        # how many comm nodes do we need to cover farm_size?
                        hops = math.ceil(math.sqrt(farm_size) / comm.range_km)
                        if hops < 1:
                            hops = 1

                        total_comm_cost = comm.cost * hops
                        total_comm_power = comm.power_watts * hops

                        # keep the cheapest multi-hop option
                        if total_comm_cost < best_total_comm_cost:
                            best_total_comm_cost = total_comm_cost
                            best = (comm, hops, total_comm_power)

                    if best is None:
                        continue

                    comm, hops, comm_power = best


                    total_cost = body.cost + motor.cost + battery.cost + tire.cost + computing.cost + total_comm_cost
                    if total_cost > budget:
                        continue

                    W_max = compute_max_payload(body, motor, battery, tire)
                    # skip if motor can’t lift this body+battery+tire combo
                    if W_max <= 0:
                        continue

                    weight_factor = compute_weight_factor(W_max)
                    area_coverage = compute_area_coverage(battery, motor, tire, weight_factor)

                    # estimate runtime (hours)
                    motor_draw   = motor.power_watts
                    compute_draw = computing.power_watts
                    cap_wh       = battery.capacity_wh
                    draw_total   = motor_draw  + compute_draw + comps_draw
                    battery_factor = compute_factor(battery.mass_kg)
                    runtime_hrs  = round((cap_wh * battery_factor)/draw_total, 2) if draw_total>0 else 0.0


                    possible_configurations.append({
                        "type": "Rover",
                        "body": f"{body.material} {body.size}",
                        "motor": motor.size,
                        "battery": battery.size,
                        "tires": tire.size,
                        "computing_device": computing.model,
                        "computing_type": computing.unit_type,
                        "computing_cost": computing.cost,
                        "computing_perf": computing.performance,
                        "computing_power_watts": computing.power_watts,
                        "cost": total_cost,
                        "coverage": area_coverage,
                        "payload": W_max,
                        "runtime_hours": runtime_hrs,

                    })

    # **Drones**
    for drone_body in drone_bodies:
        for drone_motor in drone_motors:
            # force drone_motor.size == drone_body.size
            if drone_motor.size != drone_body.size:
                continue
            for drone_battery in drone_batteries:
                computing    = pick_compute()
                best = None
                best_total_comm_cost = float("inf")

                for comm in communications:
                    # how many comm nodes do we need to cover farm_size?
                    hops = math.ceil(math.sqrt(farm_size) / comm.range_km)
                    if hops < 1:
                        hops = 1

                    total_comm_cost = comm.cost * hops
                    total_comm_power = comm.power_watts * hops

                    # keep the cheapest multi-hop option
                    if total_comm_cost < best_total_comm_cost:
                        best_total_comm_cost = total_comm_cost
                        best = (comm, hops, total_comm_power)

                if best is None:
                    continue

                comm, hops, comm_power = best

                total_cost = drone_body.cost + drone_motor.cost + drone_battery.cost + computing.cost + best_total_comm_cost
                if total_cost > budget:
                    continue

                W_max = compute_drone_payload(drone_body, drone_motor, drone_battery)
                if W_max <= 0:
                    continue

                weight_factor = compute_weight_factor(W_max)
                area_coverage = compute_drone_coverage(drone_battery, drone_motor, weight_factor)

                motor_draw = drone_motor.power_watts
                compute_draw = computing.power_watts
                cap_wh = drone_battery.capacity_wh
                draw_total = motor_draw  + compute_draw + comps_draw
                drone_battery_factor = compute_factor(drone_battery.mass_kg)
                runtime_hrs = round((cap_wh * drone_battery_factor) / draw_total, 2) if draw_total > 0 else 0.0

                possible_configurations.append({
                    "type": "Drone",
                    "body": f"{drone_body.material} {drone_body.size}",
                    "motor": drone_motor.size,
                    "battery": drone_battery.size,
                    "computing_device": computing.model,
                    "computing_type": computing.unit_type,
                    "computing_cost": computing.cost,
                    "computing_perf": computing.performance,
                    "computing_power_watts": computing.power_watts,
                    "cost": total_cost,
                    "coverage": area_coverage,
                    "payload": W_max,
                    "runtime_hours": runtime_hrs,

                })

    return {"possible_configurations": possible_configurations}