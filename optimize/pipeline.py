"""Run the whole chain once on the real datasets and save the results for the dashboard.

  1 cutting data   C45 turning experiments -> calibrated model -> Bayesian optimization vs DOE vs random
                   -> cycle time and tool use per part with the recommended settings
  3 maintenance    SCANIA fleet lifetimes -> censored Weibull -> best replacement age -> due components
  2 schedule       benchmark instances (unrelated machines, sequence-dependent setups, weighted tardiness):
                   (a) our solver vs published results, (b) this week's order book with the new cycle times and
                   maintenance windows
  4 inventory      real spare-part demand -> (s, Q) policies tested on 12 months the optimizer never saw

    python -m optimize.pipeline            # full run (~12 min)
    python -m optimize.pipeline --quick    # smaller benchmarks (~3 min)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from optimize import calibration, inventory, maintenance, process, scheduling
from optimize.plant import PLANT, VARS, evaluate, handbook

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results.json"
WEEK_INSTANCE = ("medium", 1)            # benchmark instance used as "this week's order book"
WEEK_MIX = {"A": 0.30, "B": 0.25, "C": 0.30, "D": 0.15}   # assumed share of machining time per part family


def _clean(o):
    """numpy -> plain Python for JSON."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if k not in ("u", "model", "seqs")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run(quick: bool = False, log=print) -> dict:
    t0 = time.time()
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "quick": quick, "parts": {}, "benchmark": {}}

    # 1) cutting data -------------------------------------------------------------------------------------------------
    cal = calibration.fit()
    out["calibration"] = cal
    log(f"[cutting] Ra model R² {cal['ra']['r2']:.2f}; Taylor T = {cal['tool_life']['C_T']:.0f} · vc^-{cal['tool_life']['k']:.2f}")
    seeds = 5 if quick else 15
    for prt in PLANT.parts:
        fam = prt.name[0]
        bo = process.bayes_opt(prt, np.random.default_rng(42))
        rec = evaluate({k: bo["best"][k] for k in VARS}, prt)
        out["parts"][fam] = {"name": prt.name, "ra_max": prt.ra_max, "handbook": evaluate(handbook(prt), prt),
                             "recommended": rec, "optimum": process.true_optimum(prt),
                             "runs": [{k: r[k] for k in ("v", "f", "ap", "ra_um", "cost_eur", "power_kw")}
                                      for r in bo["runs"]]}
        log(f"[cutting] {prt.name}: {rec['cost_eur']:.2f} €/part (handbook {out['parts'][fam]['handbook']['cost_eur']:.2f}); "
            f"benchmark with {seeds} campaigns per strategy")
        b = process.benchmark(prt, seeds=seeds)
        out["benchmark"][fam] = {"summary": b["summary"], "margin_cost_pct": b["margin_cost_pct"]}
    hb = sum(WEEK_MIX[f] * out["parts"][f]["handbook"]["cycle_min"] for f in WEEK_MIX)
    op = sum(WEEK_MIX[f] * out["parts"][f]["recommended"]["cycle_min"] for f in WEEK_MIX)
    factor = op / hb
    out["cycle_factor"] = factor

    # 3) maintenance (before the schedule: it decides the maintenance windows) ----------------------------------------
    mp = maintenance.run(boot=60 if quick else 200, log=log)
    out["maintenance"] = mp

    # 2) schedule -----------------------------------------------------------------------------------------------------
    n_bench = 4 if quick else 10
    out["schedule_benchmark"] = scheduling.benchmark("medium", range(1, n_bench + 1), 10 if quick else 25, log=log)
    out["schedule_model_check"] = scheduling.verify_model("small", max_jobs=7, time_limit=10)
    out["week"] = week_plan(factor, mp["T_star"], quick, log)

    # 4) inventory ----------------------------------------------------------------------------------------------------
    inv = inventory.run(max_parts=60 if quick else None, log=log)
    inv.pop("per_part")
    out["inventory"] = inv
    edges = {f: (out["parts"][f]["handbook"]["edges_per_part"], out["parts"][f]["recommended"]["edges_per_part"])
             for f in WEEK_MIX}
    out["insert_use_change_pct"] = 100 * (sum(WEEK_MIX[f] * e[1] for f, e in edges.items())
                                          / sum(WEEK_MIX[f] * e[0] for f, e in edges.items()) - 1)

    out["seconds"] = round(time.time() - t0, 1)
    out = _clean(out)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / "reports" / "results.md").write_text(report(out), encoding="utf-8")
    log(f"done in {out['seconds']} s -> {RESULTS}")
    return out


def week_plan(factor: float, T_star: float, quick: bool = False, log=print) -> dict:
    """This week's order book (a benchmark instance) with the new cycle times and maintenance windows."""
    base = scheduling.load(*WEEK_INSTANCE)
    # each machine carries a wearing component like SCANIA Component X; ages are drawn from real components still
    # in service that are approaching the replacement age, so some (not all) machines need maintenance this week
    rng = np.random.default_rng(3)
    in_service = maintenance.load().query("failed == 0").t.to_numpy()
    pool = in_service[(in_service > T_star - 3 * maintenance.SOON) & (in_service < T_star)]
    ages = {k: float(rng.choice(pool)) for k in range(base.m)}
    horizon = int(base.p.mean() * base.n / base.m)
    windows = maintenance.machine_windows(ages, T_star, horizon, length=int(2 * base.p.mean()))
    week = {}
    for label, inst in (("handbook", base), ("optimized", base.scaled(factor))):
        inst = inst.with_windows(windows)
        edd, atcs = scheduling.dispatch(inst, "EDD"), scheduling.dispatch(inst, "ATCS")
        ig = scheduling.iterated_greedy(inst, min((edd, atcs), key=lambda r: r["weighted_tardiness"]),
                                        8 if quick else 25)
        for name, r in (("EDD", edd), ("ATCS", atcs), ("Iterated greedy", ig)):
            week[f"{label}|{name}"] = {"kpis": {k: r[k] for k in ("weighted_tardiness", "late_jobs", "setup_total",
                                                                  "makespan")}, "rows": r["rows"]}
        log(f"[schedule] this week, {label} cycle times: EDD {edd['weighted_tardiness']}, "
            f"iterated greedy {ig['weighted_tardiness']}")
    return {"instance": f"{WEEK_INSTANCE[0]}-{WEEK_INSTANCE[1]}", "jobs": base.n, "machines": base.m,
            "due": base.due.tolist(), "weight": base.w.tolist(), "component_age": ages, "windows": windows,
            "results": week}


def load() -> dict | None:
    if not RESULTS.exists():
        return None
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    return r if "calibration" in r else None          # ignore results of the old simulated version


def headline(r: dict) -> dict:
    """Key numbers for the report, the dashboard and the AI advisor."""
    p = r["parts"]
    w = r["week"]["results"]
    mc = r["maintenance"]["compare"]
    rtf = mc["Run to failure"]["eur_per_1000_time_units"]
    opt_key = next(k for k in mc if k.startswith("Optimized"))
    inv = r["inventory"]["summary"]
    best_inv = min(inv, key=lambda k: inv[k]["eur_per_month_all_parts"])
    return {
        "cost_per_part": {f: (x["handbook"]["cost_eur"], x["recommended"]["cost_eur"]) for f, x in p.items()},
        "mix_saving_pct": 100 * (1 - sum(WEEK_MIX[f] * p[f]["recommended"]["cost_eur"] for f in WEEK_MIX)
                                 / sum(WEEK_MIX[f] * p[f]["handbook"]["cost_eur"] for f in WEEK_MIX)),
        "cycle_factor": r["cycle_factor"],
        "bo_median_gap": {f: b["summary"]["bayes"]["median_gap_pct"] for f, b in r["benchmark"].items()},
        "doe_median_gap": {f: b["summary"]["doe"]["median_gap_pct"] for f, b in r["benchmark"].items()},
        "sched_ig_median_gap_pct": r["schedule_benchmark"]["ig_median_gap_pct"],
        "week_wt": {k: v["kpis"]["weighted_tardiness"] for k, v in w.items()},
        "week_late": {k: v["kpis"]["late_jobs"] for k, v in w.items()},
        "T_star": r["maintenance"]["T_star"], "beta": r["maintenance"]["fit"]["beta"],
        "maint_saving_vs_rtf_pct": 100 * (1 - mc[opt_key]["eur_per_1000_time_units"] / rtf),
        "inventory_best_policy": best_inv,
        "inventory": inv,
        "insert_use_change_pct": r["insert_use_change_pct"],
    }


def report(r: dict) -> str:
    h = headline(r)
    cal = r["calibration"]
    L = [f"# Kaizen Optimize: results on real data ({r['generated']}{', quick run' if r['quick'] else ''})", "",
         "## 1. Cutting data (model calibrated on real C45 turning experiments)", "",
         f"Roughness model: R² {cal['ra']['r2']:.2f} on {cal['ra']['n']} measurements, leave-one-run-out error "
         f"{cal['ra']['cv_mape_pct']:.0f} % (2-level design, corners are extrapolated in this check). Tool life: "
         f"T = {cal['tool_life']['C_T']:.0f} · vc^-{cal['tool_life']['k']:.2f} min (VB = 0.3 mm, f 0.1, ap 0.2), "
         f"valid {cal['tool_life']['valid_speed'][0]}–{cal['tool_life']['valid_speed'][1]} m/min.", "",
         "| Part | Strategy | Median gap to model optimum | 90th percentile gap | Meets spec |", "|---|---|---|---|---|"]
    for f, b in r["benchmark"].items():
        for s in b["summary"].values():
            L.append(f"| {r['parts'][f]['name']} | {s['strategy']} | {s['median_gap_pct']:.1f} % | {s['p90_gap_pct']:.1f} % | "
                     f"{s['feasible_pct']:.0f} % |")
    L += ["", "| Part | Handbook €/part | Recommended €/part | Model optimum | Recommended setting |", "|---|---|---|---|---|"]
    for f, p in r["parts"].items():
        x = p["recommended"]
        L.append(f"| {p['name']} | {p['handbook']['cost_eur']:.2f} | {x['cost_eur']:.2f} | {p['optimum']['cost_eur']:.2f} | "
                 f"{x['v']:.0f} m/min, {x['f']:.3f} mm, {x['passes']} pass(es), Ra {x['ra_um']:.2f} µm |")
    sb = r["schedule_benchmark"]
    L += ["", f"## 2. Scheduling (benchmark: {sb['instances']} instances, 54 jobs, 5 machines)", "",
          f"Model check: {r['schedule_model_check']['equal_to_published']} of {r['schedule_model_check']['proven_optimal']} "
          "small instances solved to proven optimality give exactly the published best value.", "",
          "| Method | Mean gap to best known | Median gap |", "|---|---|---|"]
    for x in sb["ranking"]:
        L.append(f"| {x['method']} | {x['mean_gap_pct']:.1f} % | {x['median_gap_pct']:.1f} % |")
    L += ["", f"This week's order book ({r['week']['instance']}), cycle times x {r['cycle_factor']:.2f} with the new cutting data:", "",
          "| Cycle times | Method | Weighted tardiness | Late jobs | Setup | Makespan |", "|---|---|---|---|---|---|"]
    for k, v in r["week"]["results"].items():
        a, b = k.split("|")
        q = v["kpis"]
        L.append(f"| {a} | {b} | {q['weighted_tardiness']} | {q['late_jobs']} | {q['setup_total']} | {q['makespan']} |")
    m = r["maintenance"]
    L += ["", "## 3. Maintenance (SCANIA Component X, real fleet data)", "",
          f"{m['vehicles']} vehicles, {m['repairs']} repairs, {m['censored']} censored. Weibull β {m['fit']['beta']:.2f} "
          f"(95 % CI {m['ci95']['beta'][0]:.2f}–{m['ci95']['beta'][1]:.2f}), η {m['fit']['eta']:.0f}. Best replacement age "
          f"{m['T_star']:.0f} (95 % CI {m['ci95']['T_star'][0]:.0f}–{m['ci95']['T_star'][1]:.0f}); hold-out Kaplan-Meier "
          f"optimum {m['T_holdout_km']:.0f}.", "", "| Policy | Cost per 1000 time units | On hold-out vehicles | Ends in breakdown |",
          "|---|---|---|---|"]
    for k, v in m["compare"].items():
        ho = "–" if v["holdout_eur_per_1000_time_units"] is None else f"{v['holdout_eur_per_1000_time_units']:.0f} €"
        L.append(f"| {k} | {v['eur_per_1000_time_units']:.0f} € | {ho} | {v['share_ending_in_breakdown_pct']:.0f} % |")
    inv = r["inventory"]
    L += ["", f"## 4. Spare-part inventory ({inv['parts']} real parts, tested on {inv['test_months']} unseen months)", "",
          "| Policy | Cost per month | Fill rate | Average stock |", "|---|---|---|---|"]
    for k, v in inv["summary"].items():
        L.append(f"| {k} | {v['eur_per_month_all_parts']:.0f} € | {v['fill_rate_pct']:.1f} % | {v['avg_stock_units']:.0f} units |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(quick=ap.parse_args().quick)
