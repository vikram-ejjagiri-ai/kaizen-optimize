"""Tests: real-data calibration, optimizers on cases with known answers, and the benchmark model check."""
import math

import numpy as np
import pytest

from optimize import calibration, inventory, maintenance, process, scheduling
from optimize.advisor import check
from optimize.plant import PLANT, evaluate, experiment, handbook, part, passes, roughness, tool_life


# ---- 1 cutting data: calibration on the real C45 experiments ---------------------------------------------------------
def test_roughness_model_fits_real_data():
    cal = calibration.fit()
    assert cal["ra"]["n"] == 40 and cal["ra"]["r2"] > 0.9


def test_roughness_physics_direction():
    assert roughness(200, 0.30, 1.0) > roughness(200, 0.12, 1.0)          # more feed -> rougher
    assert roughness(240, 0.15, 1.0) < roughness(80, 0.15, 1.0)           # faster -> smoother in this data


def test_taylor_fit_from_real_wear_curves():
    t = calibration.fit()["tool_life"]
    assert 0.1 < t["k"] < 0.6                                              # tool life falls with speed
    assert tool_life(240, 0.1, 0.2) == pytest.approx(t["C_T"] * 240 ** -t["k"], rel=1e-9)
    assert tool_life(200, 0.3, 2.0) < tool_life(200, 0.1, 0.2)             # heavier cut -> shorter life


def test_trial_cut_noise_matches_replicate_scatter():
    A, rng = part("A"), np.random.default_rng(0)
    x = handbook(A)
    meas = np.log([experiment(x, A, rng)["ra_um"] for _ in range(3000)])
    assert np.std(meas) == pytest.approx(calibration.fit()["ra"]["replicate_sd_log"], rel=0.08)
    assert np.exp(np.mean(meas)) == pytest.approx(evaluate(x, A)["ra_um"], rel=0.02)


def test_passes_and_power_limit():
    A = part("A")
    assert passes(2.0, A) == 2 and passes(1.9, A) == 3
    assert evaluate({"v": 240, "f": 0.4, "ap": 3.0}, part("B"))["power_kw"] > PLANT.p_max   # heavy cut not allowed


def test_optimum_beats_handbook_and_meets_spec():
    for prt in PLANT.parts:
        opt = process.true_optimum(prt, n=25)
        assert opt["cost_eur"] < evaluate(handbook(prt), prt)["cost_eur"]
        assert opt["ra_um"] <= prt.ra_max and opt["power_kw"] <= PLANT.p_max


def test_bayesian_optimization_is_close_and_safe():
    A = part("A")
    opt = process.true_optimum(A, n=25)
    gaps = []
    for s in range(3):
        r = process.score(process.bayes_opt(A, np.random.default_rng(s)), A, opt)
        assert r["feasible"]
        gaps.append(r["gap_pct"])
    assert np.median(gaps) < 15


# ---- 2 scheduling on the published benchmark ---------------------------------------------------------------------------
def tiny():
    p = np.array([[3, 2, 4], [2, 3, 2]])
    s = np.zeros((2, 3, 3), dtype=int)
    s[0, 0, 1] = s[0, 1, 0] = 1
    return scheduling.Instance("tiny", p, s, np.array([3, 2, 6]), np.array([2, 1, 1]))


def test_evaluate_by_hand():
    r = scheduling.evaluate(tiny(), {0: [0, 1], 1: [2]})
    # machine 0: job 0 ends 3, setup 1, job 1 ends 6 (due 2 -> 4 late x w1); machine 1: job 2 ends 2
    assert r["weighted_tardiness"] == 4 and r["late_jobs"] == 1 and r["setup_total"] == 1


def test_windows_are_respected():
    inst = tiny().with_windows({0: [(2, 5)]})
    r = scheduling.evaluate(inst, {0: [0, 1], 1: [2]})
    for row in r["rows"]:
        if row["machine"] == 0:
            assert row["end"] <= 2 or row["setup_start"] >= 7


def test_every_job_scheduled_once_and_search_never_worse():
    inst = scheduling.load("medium", 2)
    atcs = scheduling.dispatch(inst, "ATCS")
    assert sorted(r["job"] for r in atcs["rows"]) == list(range(inst.n))
    ig = scheduling.iterated_greedy(inst, atcs, time_limit=3)
    assert ig["weighted_tardiness"] <= atcs["weighted_tardiness"]
    assert sorted(r["job"] for r in ig["rows"]) == list(range(inst.n))


def test_model_matches_published_benchmark():
    """Where CP-SAT proves optimality, our optimum equals the published best value."""
    pub = scheduling.published("small")
    for i in (1, 2, 17):
        r = scheduling.solve(scheduling.load("small", i), time_limit=10)
        assert r["status"] == "OPTIMAL" and r["weighted_tardiness"] == pub.loc[i, "Best wT"]


def test_shorter_processing_times():
    inst = scheduling.load("medium", 1)
    assert inst.scaled(0.7).p.sum() < inst.p.sum()


# ---- 3 maintenance on the real fleet data -------------------------------------------------------------------------------
def test_weibull_recovers_known_parameters_with_censoring():
    rng = np.random.default_rng(1)
    life = 500 * rng.weibull(2.5, 6000)
    cens = rng.uniform(100, 700, 6000)
    t, d = np.minimum(life, cens), (life <= cens).astype(float)
    b, e = maintenance.fit_weibull(t, d)
    assert b == pytest.approx(2.5, rel=0.06) and e == pytest.approx(500, rel=0.04)


def test_kaplan_meier_by_hand():
    t, s = maintenance.kaplan_meier(np.array([1, 2, 2, 3, 4.0]), np.array([1, 1, 0, 1, 0.0]))
    assert list(t) == [1, 2, 3]
    assert s == pytest.approx([0.8, 0.6, 0.3])


@pytest.mark.skipif(not maintenance.CSV.exists(), reason="SCANIA file not downloaded (see README)")
def test_real_component_wears_out_and_optimum_is_sensible():
    df = maintenance.load()
    assert len(df) == 23550 and int(df.failed.sum()) == 2272
    b, e = maintenance.fit_weibull(df.t.to_numpy(float), df.failed.to_numpy(float))
    assert b > 1.5                                                         # wear-out: preventive replacement pays
    W = lambda x: maintenance.reliability(x, b, e)
    T = maintenance.best_T(W, 20, 3 * e)
    assert 0.3 * e < T < e
    assert maintenance.cost_rate(T, W) < maintenance.COSTS["breakdown_eur"] / (e * math.gamma(1 + 1 / b))


# ---- 4 spare parts on real demand ------------------------------------------------------------------------------------------
def test_inventory_simulation_balances():
    demand = np.array([[2, 0, 3, 1, 0, 4, 2, 1]], dtype=float)
    r = inventory.simulate(2, 4, demand, warmup=2)
    assert 0 <= r["fill_rate_pct"] <= 100 and r["orders_per_month"] >= 0
    full = inventory.simulate(100, 1, demand, start_stock=100)
    assert full["fill_rate_pct"] == 100.0


def test_portfolio_and_optimizer_beats_rule_in_sample():
    df = inventory.load_demand()
    parts = inventory.portfolio(df)
    assert len(parts) > 300
    y = df[parts[0]].to_numpy(float)[: inventory.TRAIN_MONTHS]
    paths = inventory._paths(y, 0)
    s, Q = inventory.optimize_part(y, seed=0)
    rs, rq = inventory.rule_of_thumb(y)
    opt = inventory.simulate(s, Q, paths, warmup=inventory.WARMUP)["eur_per_month"]
    rule = inventory.simulate(rs, rq, paths, warmup=inventory.WARMUP)["eur_per_month"]
    assert opt <= rule + 1e-9


# ---- advisor ---------------------------------------------------------------------------------------------------------------
def test_advisor_number_check():
    facts = ["[F1] Best replacement age 425 (95 % interval 410 to 443)."]
    assert check("Replace at 425 [F1].", facts) == []
    assert check("Replace at 380 [F1].", facts)
