"""Process parameter optimization: find cutting speed, feed and depth of cut with the lowest cost per part
that still meet the roughness requirement and the spindle power limit.

Every trial cut costs material and machine time, so the question is: how good a setting can you find with only
20 experiments? Three strategies get the same budget:

- DOE + RSM  (classic): Box-Behnken design (15 runs), quadratic response surface, 5 confirmation runs
- Random     (naive):   20 Latin hypercube runs, take the best one
- Bayesian   (ML):      6 Latin hypercube runs, then 14 runs chosen one by one by a Gaussian process surrogate
                        with constrained expected improvement

Each strategy recommends one setting that it has actually tested. Trial cuts come from the model calibrated on real
C45 experiments, with the real measurement scatter; the noiseless model gives the true cost of every recommendation
and the reference optimum, so the comparison is fair.
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from optimize import calibration
from optimize.plant import BOUNDS, PLANT, VARS, Part, evaluate, experiment

warnings.filterwarnings("ignore", module="sklearn.gaussian_process")

BUDGET = 20
# safety margin from the real measurement scatter: recommend only settings whose roughness is at least one standard
# deviation of the replicate scatter below the drawing limit (about 11 % with the C45 data)
SAFETY = math.exp(-calibration.fit()["ra"]["replicate_sd_log"])
LO = np.array([BOUNDS[v][0] for v in VARS])
HI = np.array([BOUNDS[v][1] for v in VARS])


def pass_options(prt: Part) -> list[float]:
    """Depths of cut that remove the stock in whole passes (deepest first), within the insert's range."""
    lo, hi = BOUNDS["ap"]
    return [prt.stock / k for k in range(1, 40) if lo <= prt.stock / k <= hi]


def to_x(u: np.ndarray, prt: Part) -> dict:
    """Unit cube -> real settings. The third coordinate chooses the number of passes (a CNC programmer sets
    passes, and every pass option gets the same share of the search space)."""
    u = np.asarray(u, dtype=float)
    v = BOUNDS["v"][0] + u[0] * (BOUNDS["v"][1] - BOUNDS["v"][0])
    f = BOUNDS["f"][0] + u[1] * (BOUNDS["f"][1] - BOUNDS["f"][0])
    opts = pass_options(prt)
    ap = opts[min(int(u[2] * len(opts)), len(opts) - 1)]
    return {"v": float(v), "f": float(f), "ap": float(ap)}


# ---- the truth (vectorised grid search) ----------------------------------------------------------------------------
def true_optimum(prt: Part, margin: float = 1.0, n: int = 61) -> dict:
    """Best setting on a fine grid using the noiseless model (only possible because the plant is simulated)."""
    best = None
    opts = pass_options(prt)
    for u in itertools.product(np.linspace(0, 1, n), np.linspace(0, 1, 2 * n),
                               (np.arange(len(opts)) + 0.5) / len(opts)):
        e = evaluate(to_x(np.array(u), prt), prt)
        if e["ra_um"] <= margin * prt.ra_max and e["power_kw"] <= PLANT.p_max:
            if best is None or e["cost_eur"] < best["cost_eur"]:
                best = e
    return best


# ---- shared helpers -------------------------------------------------------------------------------------------------
def run(u: np.ndarray, prt: Part, rng) -> dict:
    x = to_x(u, prt)
    r = experiment(x, prt, rng)
    r["u"] = np.asarray(u, dtype=float)
    return r


def power_ok(u: np.ndarray, prt: Part) -> bool:
    """Spindle power is predicted from the Kienzle equation, no trial cut needed."""
    return evaluate(to_x(u, prt), prt)["power_kw"] <= PLANT.p_max


def recommend(runs: list[dict], prt: Part, ra_pred=None, cost_pred=None) -> dict:
    """Pick a tested setting whose roughness is safely inside the limit. With a cost model, rank by the model's
    de-noised prediction instead of the single noisy measurement (the lowest measured cost is often just luck).
    If a model of roughness is given, its prediction must also be inside the limit."""
    ok = []
    for r in runs:
        if r["power_kw"] > PLANT.p_max or r["ra_um"] > SAFETY * prt.ra_max:
            continue
        if ra_pred is not None and ra_pred(r["u"]) > SAFETY * prt.ra_max:
            continue
        ok.append(r)
    if not ok:                                   # nothing safe: fall back to the smoothest tested setting
        return min(runs, key=lambda r: r["ra_um"])
    if cost_pred is not None:
        return min(ok, key=lambda r: cost_pred(r["u"]))
    return min(ok, key=lambda r: r["cost_eur"])


# ---- strategy 1: classic DOE + response surface ---------------------------------------------------------------------
def box_behnken() -> np.ndarray:
    pts = []
    for i, j in itertools.combinations(range(3), 2):
        for a, b in itertools.product((0.0, 1.0), repeat=2):
            p = [0.5, 0.5, 0.5]
            p[i], p[j] = a, b
            pts.append(p)
    pts += [[0.5, 0.5, 0.5]] * 3
    return np.array(pts)


def _quad_model(U, y):
    pf = PolynomialFeatures(2)
    m = Ridge(alpha=1e-3).fit(pf.fit_transform(U), y)
    return lambda u: m.predict(pf.transform(np.atleast_2d(u)))


def doe_rsm(prt: Part, rng, budget: int = BUDGET) -> dict:
    runs = [run(u, prt, rng) for u in box_behnken()]
    U = np.array([r["u"] for r in runs])
    cost_m = _quad_model(U, np.log([r["cost_eur"] for r in runs]))
    ra_m = _quad_model(U, np.log([r["ra_um"] for r in runs]))
    cand = qmc.LatinHypercube(d=3, seed=rng).random(4000)
    feas = [u for u in cand if math.exp(ra_m(u)[0]) <= SAFETY * prt.ra_max and power_ok(u, prt)]
    feas = sorted(feas, key=lambda u: cost_m(u)[0])
    for u in feas[: budget - len(runs)]:          # confirmation runs at the best predicted settings
        runs.append(run(u, prt, rng))
    U = np.array([r["u"] for r in runs])        # refit on all 20 runs
    cost_m = _quad_model(U, np.log([r["cost_eur"] for r in runs]))
    ra_m = _quad_model(U, np.log([r["ra_um"] for r in runs]))
    return {"name": "DOE + response surface", "runs": runs,
            "best": recommend(runs, prt, lambda u: math.exp(ra_m(u)[0]), lambda u: cost_m(u)[0])}


# ---- strategy 2: random search ---------------------------------------------------------------------------------------
def random_search(prt: Part, rng, budget: int = BUDGET) -> dict:
    runs = [run(u, prt, rng) for u in qmc.LatinHypercube(d=3, seed=rng).random(budget)]
    return {"name": "Random (Latin hypercube)", "runs": runs, "best": recommend(runs, prt)}


# ---- strategy 3: Bayesian optimization ------------------------------------------------------------------------------
def _gp(U, y):
    k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=[0.3] * 3, length_scale_bounds=(0.03, 3), nu=2.5) \
        + WhiteKernel(1e-2, (1e-5, 1e-1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2,
                                    random_state=0).fit(U, y)


def bayes_opt(prt: Part, rng, budget: int = BUDGET, n_init: int = 6) -> dict:
    runs = [run(u, prt, rng) for u in qmc.LatinHypercube(d=3, seed=rng).random(n_init)]
    limit = math.log(SAFETY * prt.ra_max)
    gp_c = gp_r = None
    while len(runs) < budget:
        U = np.array([r["u"] for r in runs])
        gp_c = _gp(U, np.log([r["cost_eur"] for r in runs]))
        gp_r = _gp(U, np.log([r["ra_um"] for r in runs]))
        cand = qmc.LatinHypercube(d=3, seed=rng).random(3000)
        cand = cand[[power_ok(u, prt) for u in cand]]
        mu, sd = gp_c.predict(cand, return_std=True)
        mr, sr = gp_r.predict(cand, return_std=True)
        p_feas = norm.cdf((limit - mr) / np.maximum(sr, 1e-9))
        feas_runs = [math.log(r["cost_eur"]) for r in runs if r["ra_um"] <= SAFETY * prt.ra_max]
        if feas_runs:   # constrained expected improvement over the best safe run so far
            best = min(feas_runs)
            z = (best - mu) / np.maximum(sd, 1e-9)
            ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
            acq = ei * p_feas
        else:           # no safe run yet: first find the feasible region
            acq = p_feas
        runs.append(run(cand[int(np.argmax(acq))], prt, rng))
    U = np.array([r["u"] for r in runs])
    gp_c = _gp(U, np.log([r["cost_eur"] for r in runs]))
    gp_r = _gp(U, np.log([r["ra_um"] for r in runs]))
    return {"name": "Bayesian optimization (GP)", "runs": runs, "model": (gp_c, gp_r),
            "best": recommend(runs, prt, lambda u: math.exp(gp_r.predict(np.atleast_2d(u))[0]),
                              lambda u: gp_c.predict(np.atleast_2d(u))[0])}


STRATEGIES = {"doe": doe_rsm, "random": random_search, "bayes": bayes_opt}


# ---- benchmark ----------------------------------------------------------------------------------------------------
def score(result: dict, prt: Part, opt: dict) -> dict:
    """How good is the recommended setting really (noiseless model)?"""
    b = result["best"]
    true = evaluate({k: b[k] for k in VARS}, prt)
    return {"strategy": result["name"], "true_cost": true["cost_eur"],
            "gap_pct": 100 * (true["cost_eur"] / opt["cost_eur"] - 1),
            "feasible": true["ra_um"] <= prt.ra_max and true["power_kw"] <= PLANT.p_max, "setting": true}


def benchmark(prt: Part, seeds: int = 20) -> dict:
    """Run every strategy with many random seeds (= many repeats of the whole experiment campaign)."""
    opt = true_optimum(prt)
    opt_safe = true_optimum(prt, margin=SAFETY)        # best possible with the same roughness safety margin
    rows = {k: [] for k in STRATEGIES}
    for s in range(seeds):
        for k, fn in STRATEGIES.items():
            rows[k].append(score(fn(prt, np.random.default_rng(1000 * s + hash(k) % 997)), prt, opt))
    summary = {}
    for k, rs in rows.items():
        gaps = np.array([r["gap_pct"] for r in rs])
        summary[k] = {"strategy": rs[0]["strategy"], "median_gap_pct": float(np.median(gaps)),
                      "mean_gap_pct": float(gaps.mean()), "p90_gap_pct": float(np.percentile(gaps, 90)),
                      "feasible_pct": 100 * float(np.mean([r["feasible"] for r in rs]))}
    return {"part": prt.name, "optimum": opt, "optimum_with_margin": opt_safe,
            "margin_cost_pct": 100 * (opt_safe["cost_eur"] / opt["cost_eur"] - 1), "summary": summary, "rows": rows}
