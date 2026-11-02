"""Stage 2 on real benchmark data: unrelated parallel machines with sequence-dependent setup times, minimise the total
weighted tardiness  sum_j w_j * max(0, C_j - d_j).

Data: PMSP_SDST_TWT benchmark by Sungbum Jun (MIT licence, https://github.com/sjun-dgu/PMSP_SDST_TWT).
Every instance gives, for each machine, the processing time of each job and a full setup matrix (setup from job i to
job j depends on the machine), plus a due date and a customer weight per job. The repository also publishes the best
known objective value per instance and the results of CP (IBM CP Optimizer), MILP, a hybrid genetic algorithm,
simulated annealing, dispatch rules and several LLMs, so our solver can be compared with published numbers.

Model conventions (same as the published CP model): machines are free at time 0, the first job on a machine has no
setup, setups only occur between consecutive jobs on the same machine, preemption is not allowed.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from optimize.config import DATA

BENCH_DIR = DATA / "scheduling"
SETS = {"small": "Test_Small", "medium": "Test_Medium", "large": "Test_Large"}
PUBLISHED_COLUMNS = ["Best wT", "CP", "MILP", "HGA", "SA", "EDD", "ATCS", "MDD", "GPT4.1", "Gemini"]


@dataclass
class Instance:
    name: str
    p: np.ndarray            # p[m, j] processing time of job j on machine m
    s: np.ndarray            # s[m, i, j] setup from job i to job j on machine m
    due: np.ndarray
    w: np.ndarray
    blocked: dict = field(default_factory=dict)   # maintenance windows per machine: {k: [(start, length), ...]}

    @property
    def n(self) -> int:
        return self.p.shape[1]

    @property
    def m(self) -> int:
        return self.p.shape[0]

    def scaled(self, factor: float, name: str | None = None) -> "Instance":
        """Same orders with shorter or longer processing times (e.g. after new cutting data). Setups stay."""
        return Instance(name or self.name, np.maximum(1, np.rint(self.p * factor)).astype(int), self.s, self.due, self.w,
                        self.blocked)

    def with_windows(self, blocked: dict) -> "Instance":
        return Instance(self.name, self.p, self.s, self.due, self.w, blocked)


_CACHE: dict[str, np.lib.npyio.NpzFile] = {}


def load(set_name: str, idx: int) -> Instance:
    """The original JSON instances (216 MB) are stored as one compressed .npz per set; values are unchanged."""
    if set_name not in _CACHE:
        _CACHE[set_name] = np.load(BENCH_DIR / f"{SETS[set_name]}.npz")
    z = _CACHE[set_name]
    return Instance(f"{set_name}-{idx}", z[f"p_{idx}"].astype(int), z[f"s_{idx}"].astype(int),
                    z[f"due_{idx}"].astype(int), z[f"w_{idx}"].astype(int))


def instance_ids(set_name: str) -> list[int]:
    if set_name not in _CACHE:
        _CACHE[set_name] = np.load(BENCH_DIR / f"{SETS[set_name]}.npz")
    return sorted(int(k[2:]) for k in _CACHE[set_name].files if k.startswith("p_"))


def published(set_name: str) -> pd.DataFrame:
    df = pd.read_excel(BENCH_DIR / "Best_Solutions_EO.xlsx", sheet_name=SETS[set_name])
    df = df.rename(columns={"Unnamed: 0": "instance"}).dropna(subset=["instance"])
    df["instance"] = df["instance"].astype(int)
    return df.set_index("instance")


# ---- evaluation ---------------------------------------------------------------------------------------------------
def evaluate(inst: Instance, seqs: dict[int, list[int]], blocked: dict[int, list[tuple[int, int]]] | None = None) -> dict:
    """Simulate machine sequences; returns schedule rows and KPIs. Maintenance windows (inst.blocked) are respected:
    a job whose setup + processing would overlap a window starts after it."""
    blocked = inst.blocked if blocked is None else blocked
    rows = []
    for k, seq in seqs.items():
        t, prev = 0, None
        for j in seq:
            su = 0 if prev is None else int(inst.s[k, prev, j])
            start = _after_blocks(t, su + int(inst.p[k, j]), (blocked or {}).get(k, []))
            rows.append({"job": j, "machine": k, "setup_start": start, "start": start + su,
                         "end": start + su + int(inst.p[k, j]), "setup": su})
            t, prev = start + su + int(inst.p[k, j]), j
    return {"rows": rows, **kpis(inst, rows)}


def _after_blocks(t: int, dur: int, blocks: list[tuple[int, int]]) -> int:
    for a, d in sorted(blocks):
        if t < a + d and t + dur > a:
            t = a + d
    return t


def kpis(inst: Instance, rows: list[dict]) -> dict:
    tard = {r["job"]: max(0, r["end"] - int(inst.due[r["job"]])) for r in rows}
    return {"weighted_tardiness": int(sum(int(inst.w[j]) * t for j, t in tard.items())),
            "late_jobs": int(sum(t > 0 for t in tard.values())),
            "setup_total": int(sum(r["setup"] for r in rows)),
            "makespan": int(max((r["end"] for r in rows), default=0))}


# ---- dispatch rules (baselines) -----------------------------------------------------------------------------------
def dispatch(inst: Instance, rule: str = "EDD", k_atcs: float = 2.0) -> dict:
    """List scheduling: repeatedly pick the next job by the rule and put it on the machine where it finishes first.
    EDD = earliest due date. ATCS = apparent tardiness cost with setups (Lee, Bhaskaran & Pinedo 1997)."""
    free = {k: 0 for k in range(inst.m)}
    last: dict[int, int | None] = {k: None for k in range(inst.m)}
    seqs: dict[int, list[int]] = {k: [] for k in range(inst.m)}
    todo = set(range(inst.n))
    p_bar, s_bar = inst.p.mean(), max(inst.s.mean(), 1e-9)
    while todo:
        def finish(j, k):
            su = 0 if last[k] is None else inst.s[k, last[k], j]
            return _after_blocks(free[k], su + inst.p[k, j], inst.blocked.get(k, [])) + su + inst.p[k, j]
        if rule == "EDD":
            j = min(todo, key=lambda j: (inst.due[j], j))
            k = min(range(inst.m), key=lambda k: finish(j, k))
        else:  # ATCS: score every (job, machine) pair, take the highest priority
            best = None
            for j in todo:
                for k in range(inst.m):
                    su = 0 if last[k] is None else inst.s[k, last[k], j]
                    slack = max(0.0, inst.due[j] - inst.p[k, j] - free[k])
                    score = inst.w[j] / inst.p[k, j] * math.exp(-slack / (k_atcs * p_bar)) * math.exp(-su / (0.5 * s_bar))
                    if best is None or score > best[0]:
                        best = (score, j, k)
            _, j, k = best
        seqs[k].append(j)
        free[k] = finish(j, k)
        last[k] = j
        todo.remove(j)
    out = evaluate(inst, seqs)
    out["seqs"] = seqs
    return out


# ---- iterated greedy local search (Ruiz & Stützle style) --------------------------------------------------------------
def _machine_cost(inst: Instance, k: int, seq: list[int]) -> int:
    t, prev, cost = 0, None, 0
    pk, sk, due, w = inst.p[k], inst.s[k], inst.due, inst.w
    blocks = inst.blocked.get(k)
    for j in seq:
        dur = (0 if prev is None else sk[prev, j]) + pk[j]
        if blocks:
            t = _after_blocks(t, dur, blocks)
        t += dur
        if t > due[j]:
            cost += w[j] * (t - due[j])
        prev = j
    return int(cost)


def _best_insert(inst, seqs, costs, j):
    """Cheapest machine and position for job j (all positions on all machines are tried)."""
    best = None
    for k, seq in seqs.items():
        base = costs[k]
        for pos in range(len(seq) + 1):
            c = _machine_cost(inst, k, seq[:pos] + [j] + seq[pos:]) - base
            if best is None or c < best[0]:
                best = (c, k, pos)
    return best


def _local_search(inst, seqs, costs):
    """Take each job out and put it back at its best place (any machine, any position) until nothing improves.
    The old position is one of the candidates, so a move never makes the schedule worse."""
    improved = True
    while improved:
        improved = False
        for j in random.sample(range(inst.n), inst.n):
            before = sum(costs.values())
            k0 = next(k for k, sq in seqs.items() if j in sq)
            seqs[k0] = [x for x in seqs[k0] if x != j]
            costs[k0] = _machine_cost(inst, k0, seqs[k0])
            dc, k, pos = _best_insert(inst, seqs, costs, j)
            seqs[k] = seqs[k][:pos] + [j] + seqs[k][pos:]
            costs[k] += dc
            if sum(costs.values()) < before:
                improved = True
    return seqs, costs


def iterated_greedy(inst: Instance, start: dict, time_limit: float = 10.0, d: int = 4, temp: float = 0.4,
                    seed: int = 1) -> dict:
    """Destroy d random jobs, rebuild them greedily at their best positions, improve with local search; accept
    worse solutions with a small probability (constant-temperature rule from Ruiz & Stützle 2007)."""
    random.seed(seed)
    seqs = {k: list(v) for k, v in start["seqs"].items()}
    costs = {k: _machine_cost(inst, k, v) for k, v in seqs.items()}
    seqs, costs = _local_search(inst, seqs, costs)
    best_seqs, best = {k: list(v) for k, v in seqs.items()}, sum(costs.values())
    T = temp * inst.w.mean() * inst.p.mean() / 10
    t0, it = time.time(), 0
    while time.time() - t0 < time_limit:
        it += 1
        cur = {k: list(v) for k, v in seqs.items()}
        removed = random.sample(range(inst.n), d)
        for k in cur:
            cur[k] = [x for x in cur[k] if x not in removed]
        cc = {k: _machine_cost(inst, k, v) for k, v in cur.items()}
        for j in removed:
            dc, k, pos = _best_insert(inst, cur, cc, j)
            cur[k] = cur[k][:pos] + [j] + cur[k][pos:]
            cc[k] += dc
        cur, cc = _local_search(inst, cur, cc)
        new, old = sum(cc.values()), sum(costs.values())
        if new < old or random.random() < math.exp(-(new - old) / max(T, 1e-9)):
            seqs, costs = cur, cc
            if new < best:
                best_seqs, best = {k: list(v) for k, v in cur.items()}, new
    out = evaluate(inst, best_seqs)
    out.update(seqs=best_seqs, iterations=it, seconds=round(time.time() - t0, 1))
    return out


# ---- CP-SAT ---------------------------------------------------------------------------------------------------------
def solve(inst: Instance, time_limit: float = 30.0, hint: dict | None = None, workers: int = 8) -> dict:
    """Exact model, stopped after `time_limit` seconds. Returns the schedule, KPIs, solver status and lower bound."""
    blocked = inst.blocked
    m = cp_model.CpModel()
    n, M = inst.n, inst.m
    H = int(inst.p.max(axis=0).sum() + inst.s.max() * n + sum(a + d for b in blocked.values() for a, d in b) + 1)
    pres, start, end, su = {}, {}, {}, {}
    for j in range(n):
        lits = []
        for k in range(M):
            pres[j, k] = m.NewBoolVar(f"x{j}_{k}")
            start[j, k] = m.NewIntVar(0, H, f"s{j}_{k}")
            end[j, k] = m.NewIntVar(0, H, f"e{j}_{k}")
            su[j, k] = m.NewIntVar(0, int(inst.s[k].max()), f"su{j}_{k}")
            m.Add(end[j, k] == start[j, k] + int(inst.p[k, j])).OnlyEnforceIf(pres[j, k])
            lits.append(pres[j, k])
        m.AddExactlyOne(lits)

    for k in range(M):
        arcs, busy = [], []
        for j in range(n):
            first = m.NewBoolVar(f"f{j}_{k}")
            arcs += [(0, j + 1, first), (j + 1, 0, m.NewBoolVar(f"l{j}_{k}")), (j + 1, j + 1, pres[j, k].Not())]
            m.Add(su[j, k] == 0).OnlyEnforceIf(first)
            m.Add(su[j, k] == 0).OnlyEnforceIf(pres[j, k].Not())
            if blocked.get(k):          # setup + processing must not overlap a maintenance window
                ss = m.NewIntVar(0, H, f"ss{j}_{k}")
                m.Add(ss == start[j, k] - su[j, k])
                size = m.NewIntVar(0, H, f"sz{j}_{k}")
                m.Add(size == end[j, k] - ss)
                busy.append(m.NewOptionalIntervalVar(ss, size, end[j, k], pres[j, k], f"b{j}_{k}"))
        for i in range(n):
            for j in range(n):
                if i != j:
                    lit = m.NewBoolVar(f"a{i}_{j}_{k}")
                    arcs.append((i + 1, j + 1, lit))
                    m.Add(su[j, k] == int(inst.s[k, i, j])).OnlyEnforceIf(lit)
                    m.Add(start[j, k] >= end[i, k] + int(inst.s[k, i, j])).OnlyEnforceIf(lit)
        m.AddCircuit(arcs)
        if blocked.get(k):
            fixed = [m.NewIntervalVar(a, d, a + d, f"mw{k}_{a}") for a, d in blocked[k]]
            m.AddNoOverlap(busy + fixed)

    tard = []
    for j in range(n):
        c = m.NewIntVar(0, H, f"c{j}")
        for k in range(M):
            m.Add(c == end[j, k]).OnlyEnforceIf(pres[j, k])
        t = m.NewIntVar(0, H, f"t{j}")
        m.AddMaxEquality(t, [0, c - int(inst.due[j])])
        tard.append(int(inst.w[j]) * t)
    m.Minimize(sum(tard))

    if hint:
        rows = {r["job"]: r for r in hint["rows"]}
        for j, r in rows.items():
            for k in range(M):
                m.AddHint(pres[j, k], r["machine"] == k)
            m.AddHint(start[j, r["machine"]], r["start"])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = 1
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"no schedule found ({solver.StatusName(status)})")
    seqs = {k: [] for k in range(M)}
    for j in range(n):
        k = next(k for k in range(M) if solver.Value(pres[j, k]))
        seqs[k].append((solver.Value(start[j, k]), j))
    seqs = {k: [j for _, j in sorted(v)] for k, v in seqs.items()}
    out = evaluate(inst, seqs)                # re-simulate: the reported KPIs never rely on the solver alone
    out.update(seqs=seqs, status=solver.StatusName(status), bound=float(solver.BestObjectiveBound()),
               solver_objective=float(solver.ObjectiveValue()), seconds=round(solver.WallTime(), 1))
    return out


# ---- benchmark against published results ----------------------------------------------------------------------------
def benchmark(set_name: str = "medium", instances=range(1, 11), time_limit: float = 20.0, log=print) -> dict:
    """Our methods on the published instances: EDD, ATCS, and iterated greedy started from the better rule."""
    pub = published(set_name)
    rows = []
    for i in instances:
        inst = load(set_name, i)
        edd, atcs = dispatch(inst, "EDD"), dispatch(inst, "ATCS")
        ig = iterated_greedy(inst, min((edd, atcs), key=lambda r: r["weighted_tardiness"]), time_limit)
        best = float(pub.loc[i, "Best wT"])
        r = {"instance": i, "jobs": inst.n, "machines": inst.m, "ours_EDD": edd["weighted_tardiness"],
             "ours_ATCS": atcs["weighted_tardiness"], "ours_IG": ig["weighted_tardiness"],
             "ig_iterations": ig["iterations"], "gap_to_best_pct": 100 * (ig["weighted_tardiness"] - best) / best}
        for c in PUBLISHED_COLUMNS:
            if c in pub.columns:
                r[f"pub_{c}"] = float(pub.loc[i, c])
        rows.append(r)
        log(f"[schedule] {set_name}-{i}: iterated greedy {r['ours_IG']} vs best known {best:.0f} "
            f"({r['gap_to_best_pct']:+.1f} %)")
    df = pd.DataFrame(rows)
    method = {"ours_EDD": "EDD rule (ours)", "ours_ATCS": "ATCS rule (ours)", "ours_IG": "Iterated greedy (ours)"}
    method.update({f"pub_{c}": f"{c} (published)" for c in PUBLISHED_COLUMNS if c != "Best wT"})
    ranking = []
    for col, name in method.items():
        if col in df:
            g = 100 * (df[col] - df["pub_Best wT"]) / df["pub_Best wT"]
            ranking.append({"method": name, "mean_gap_pct": float(g.mean()), "median_gap_pct": float(g.median())})
    ranking.sort(key=lambda x: x["mean_gap_pct"])
    return {"set": set_name, "instances": len(df), "time_limit_s": time_limit, "rows": rows, "ranking": ranking,
            "ig_median_gap_pct": float(df["gap_to_best_pct"].median()),
            "ig_matched_best": int((df["gap_to_best_pct"] <= 1e-9).sum())}


def verify_model(set_name: str = "small", max_jobs: int = 7, time_limit: float = 10.0) -> dict:
    """Check that our objective and rules equal the benchmark's: where CP-SAT proves optimality, our optimum must
    equal the published best value."""
    pub = published(set_name)
    out = []
    for i in instance_ids(set_name):
        inst = load(set_name, i)
        if inst.n > max_jobs:
            continue
        r = solve(inst, time_limit, workers=8)
        if r["status"] == "OPTIMAL":
            out.append({"instance": i, "jobs": inst.n, "ours": r["weighted_tardiness"], "published_best": float(pub.loc[i, "Best wT"])})
    return {"proven_optimal": len(out), "equal_to_published": sum(o["ours"] == o["published_best"] for o in out), "rows": out}
