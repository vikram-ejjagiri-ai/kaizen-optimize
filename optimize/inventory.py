"""Stage 4 on real demand data: reorder policy (s, Q) for a spare-parts store.

Data: monthly sales of 2,674 car spare parts, January 1998 - March 2002 (Hyndman et al. 2008, "Forecasting with
Exponential Smoothing", R package expsmooth, dataset `carparts`). Typical spare-part demand: small, lumpy, many zero
months - exactly the case where textbook formulas that assume normally distributed demand go wrong.

Policy (s, Q), reviewed at the end of every month: when stock on hand + on order <= s, order Q units. Lead time
1 month. Demand that cannot be served from stock is lost.

Method:
  * train on the first 39 months (Jan 1998 - Mar 2001), test on the last 12 months (Apr 2001 - Mar 2002) that the
    optimizer never saw;
  * three policies per part:
      - rule of thumb:  s = 1 month of average demand, Q = 3 months of average demand
      - textbook:       s = demand in lead time + 1.65 sigma (95 % service, normal demand), Q = EOQ
      - optimized:      grid search over (s, Q); every candidate is simulated on the same 300 bootstrap demand
                        paths (common random numbers), lowest expected cost wins - once with all training months,
                        once with only the last 12 months (demand of these parts falls over their life cycle:
                        2.0 units/month in 1998, 0.9 in the last training year, 0.6 in the test year).
Costs are assumptions (the dataset has no prices), the same for every part and stated in COSTS.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from optimize.config import DATA

CSV = DATA / "inventory" / "carparts_monthly.csv"
TRAIN_MONTHS = 39
MONTHS = pd.period_range("1998-01", periods=51, freq="M")
COSTS = {"unit_value_eur": 60.0,           # assumed average part value
         "holding_pct_per_month": 2.0,      # capital, space, obsolescence
         "order_eur": 25.0,                 # cost of placing and receiving one order
         "lost_sale_eur": 45.0}             # margin + goodwill lost per unit not available
LEAD = 1
PATHS, HORIZON, WARMUP = 300, 36, 12
RECENT = 12                       # months used by the drift-aware policy


def load_demand() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df.index = MONTHS[: len(df)]
    return df


def portfolio(df: pd.DataFrame, min_share_nonzero: float = 0.5, min_mean: float = 1.0) -> list[str]:
    """Parts with a complete history that sell in at least half of the training months."""
    tr = df.iloc[:TRAIN_MONTHS]
    full = df.columns[df.notna().all()]
    sel = [c for c in full if (tr[c] > 0).mean() >= min_share_nonzero and tr[c].mean() >= min_mean]
    return sel


def _h() -> float:
    return COSTS["unit_value_eur"] * COSTS["holding_pct_per_month"] / 100


def simulate(s: int, Q: int, demand: np.ndarray, start_stock: int | None = None, warmup: int = 0) -> dict:
    """demand: array (paths, months). Vectorised over paths. The first `warmup` months run the policy but are not
    counted, so the result does not depend on the starting stock. Returns cost per month and fill rate."""
    demand = np.atleast_2d(demand)
    P, T = demand.shape
    on_hand = np.full(P, s + Q if start_stock is None else start_stock, dtype=float)
    pipeline = np.zeros((P, LEAD + 1))
    hold = order = lost = served = 0.0
    for t in range(T):
        on_hand += pipeline[:, 0]
        pipeline = np.roll(pipeline, -1, axis=1)
        pipeline[:, -1] = 0
        d = demand[:, t]
        sold = np.minimum(on_hand, d)
        on_hand -= sold
        need = (on_hand + pipeline.sum(axis=1)) <= s
        pipeline[need, LEAD - 1] += Q
        if t >= warmup:
            lost += (d - sold).sum()
            served += sold.sum()
            hold += on_hand.sum()
            order += need.sum()
    n = P * (T - warmup)
    total = served + lost
    return {"s": int(s), "Q": int(Q),
            "eur_per_month": (hold * _h() + order * COSTS["order_eur"] + lost * COSTS["lost_sale_eur"]) / n,
            "fill_rate_pct": 100 * served / total if total else 100.0,
            "avg_stock": hold / n, "orders_per_month": order / n}


def _paths(train: np.ndarray, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(train, size=(PATHS, HORIZON), replace=True)


def rule_of_thumb(train: np.ndarray) -> tuple[int, int]:
    mu = train.mean()
    return int(math.ceil(mu)), max(1, int(math.ceil(3 * mu)))


def textbook(train: np.ndarray) -> tuple[int, int]:
    mu, sd = train.mean(), train.std(ddof=1)
    s = int(math.ceil(mu * LEAD + 1.65 * sd * math.sqrt(LEAD)))
    eoq = math.sqrt(2 * COSTS["order_eur"] * mu / max(_h(), 1e-9))
    return s, max(1, int(round(eoq)))


def optimize_part(train: np.ndarray, seed: int = 0) -> tuple[int, int]:
    paths = _paths(train, seed)
    mx = int(max(train.max(), 1))
    best = None
    for s in range(0, 2 * mx + 3):
        for Q in range(1, 4 * mx + 5):
            c = simulate(s, Q, paths, warmup=WARMUP)["eur_per_month"]
            if best is None or c < best[0]:
                best = (c, s, Q)
    return best[1], best[2]


def run(max_parts: int | None = None, log=print) -> dict:
    df = load_demand()
    parts = portfolio(df)[:max_parts] if max_parts else portfolio(df)
    log(f"[inventory] {len(parts)} spare parts with a full history; train {MONTHS[0]}..{MONTHS[TRAIN_MONTHS - 1]}, "
        f"test {MONTHS[TRAIN_MONTHS]}..{MONTHS[-1]}")
    rows = []
    for i, p in enumerate(parts):
        y = df[p].to_numpy(dtype=float)
        train, test = y[:TRAIN_MONTHS], y[TRAIN_MONTHS:]
        recent = train[-RECENT:]
        pol = {"Rule of thumb": rule_of_thumb(train), "Textbook (normal + EOQ)": textbook(train),
               "Optimized on full history": optimize_part(train, seed=i),
               "Optimized on last 12 months": optimize_part(recent, seed=i)}
        for name, (s, Q) in pol.items():
            # run the policy through the real training months, count only the 12 test months it never saw
            r = simulate(s, Q, y[None, :], start_stock=s + Q, warmup=TRAIN_MONTHS)
            rows.append({"part": p, "policy": name, **r, "train_mean": float(train.mean()),
                         "test_demand": float(test.sum())})
    res = pd.DataFrame(rows)
    summ = {}
    for name, g in res.groupby("policy", sort=False):
        demand = g["test_demand"].sum()
        summ[name] = {"eur_per_month_all_parts": float(g["eur_per_month"].sum()),
                      "fill_rate_pct": float((g["fill_rate_pct"] * g["test_demand"]).sum() / max(demand, 1)),
                      "avg_stock_units": float(g["avg_stock"].sum()),
                      "orders_per_month": float(g["orders_per_month"].sum())}
    example = parts[int(np.argmax([df[p].iloc[:TRAIN_MONTHS].std() for p in parts]))]
    return {"parts": len(parts), "train_months": TRAIN_MONTHS, "test_months": len(MONTHS) - TRAIN_MONTHS,
            "costs": COSTS, "lead_time_months": LEAD, "summary": summ,
            "example_part": {"id": example, "demand": df[example].fillna(0).astype(int).tolist(),
                             "months": [str(m) for m in MONTHS],
                             "policies": res[res.part == example][["policy", "s", "Q", "eur_per_month",
                                                                   "fill_rate_pct"]].to_dict("records")},
            "per_part": res.to_dict("records")}
