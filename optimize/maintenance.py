"""Stage 3 on real fleet data: when should a wearing component be replaced preventively?

Data: SCANIA Component X dataset (Kharazian et al. 2024, Swedish National Data Service, DOI 10.5878/jvb5-d390),
file train_tte.csv: 23,550 trucks, each with the operating time observed (anonymised time steps) and whether
Component X was repaired in that time (2,272) or not yet (21,278 - right-censored: we only know it lasted at least
that long). Time is in the dataset's anonymised units; the method is the same with hours or kilometres.

Method
  1. Split the vehicles 70 / 30 at random. Fit a Weibull distribution on the 70 % (maximum likelihood with
     right-censoring) and get uncertainty by bootstrapping vehicles.
  2. Check the fit against the Kaplan-Meier curve of the 30 % hold-out vehicles (no distribution assumed).
  3. Age replacement: replace preventively at age T, or after a failure if it comes first.
        cost per time unit  C(T) = (Cp * R(T) + Cf * (1 - R(T))) / integral_0^T R(t) dt
     The best T is found from the Weibull fit and checked on the hold-out Kaplan-Meier curve.
  4. Sensitivity: the best T depends on how much more a breakdown costs than a planned repair.
  5. Planning: how many components still in service are already past T, or will reach it soon.
Costs are assumptions (the dataset has no costs) and are stated in COSTS.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import integrate, optimize

from optimize.config import DATA

CSV = DATA / "maintenance" / "scania_component_x_train_tte.csv"
COSTS = {"planned_eur": 1500.0,      # part + planned workshop slot
         "breakdown_eur": 5100.0}    # part + secondary damage + towing / unplanned downtime
SOON = 15.0                          # "due soon" window in the dataset's time units
SEED = 7


def load() -> pd.DataFrame:
    if not CSV.exists():
        raise FileNotFoundError(
            f"{CSV.name} is missing. Download train_tte.csv from the SCANIA Component X dataset "
            "(https://doi.org/10.5878/jvb5-d390, open access) and save it as " + str(CSV))
    df = pd.read_csv(CSV)
    return df.rename(columns={"length_of_study_time_step": "t", "in_study_repair": "failed"})


def fit_weibull(t: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """Maximum likelihood for right-censored data: failures contribute the density, censored units the survival."""
    def nll(p):
        beta, eta = np.exp(p)
        z = t / eta
        return -(np.sum(d * (np.log(beta / eta) + (beta - 1) * np.log(z))) - np.sum(z ** beta))
    res = optimize.minimize(nll, x0=[math.log(1.5), math.log(t.mean())], method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    beta, eta = np.exp(res.x)
    return float(beta), float(eta)


def kaplan_meier(t: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(t)
    t, d = t[order], d[order]
    times = np.unique(t[d == 1])
    at_risk = np.array([np.sum(t >= x) for x in times])
    events = np.array([np.sum((t == x) & (d == 1)) for x in times])
    return times, np.cumprod(1 - events / at_risk)


def reliability(t, beta, eta):
    return np.exp(-(np.asarray(t, dtype=float) / eta) ** beta)


def cost_rate(T: float, R) -> float:
    """C(T) for any survival function R (Weibull or Kaplan-Meier)."""
    x = np.linspace(0.0, T, 1500)                    # trapezoid rule: works for smooth and step-shaped R
    y = np.array([float(R(v)) for v in x])
    up = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))
    r = float(R(T))
    return (COSTS["planned_eur"] * r + COSTS["breakdown_eur"] * (1 - r)) / up


def best_T(R, lo: float, hi: float) -> float:
    return float(optimize.minimize_scalar(lambda T: cost_rate(T, R), bounds=(lo, hi), method="bounded").x)


def run(boot: int = 200, log=print) -> dict:
    df = load()
    rng = np.random.default_rng(SEED)
    is_train = rng.random(len(df)) < 0.7
    tr, te = df[is_train], df[~is_train]
    t, d = tr.t.to_numpy(float), tr.failed.to_numpy(float)
    beta, eta = fit_weibull(t, d)
    W = lambda x: reliability(x, beta, eta)
    t_max = float(df.t.max())
    T_star = best_T(W, 20, min(3 * eta, 5 * t_max))
    log(f"[maintenance] {len(df)} vehicles ({int(df.failed.sum())} repairs, {int((1 - df.failed).sum())} censored); "
        f"Weibull beta {beta:.2f}, eta {eta:.0f}; best replacement age {T_star:.0f}")

    # bootstrap (resample vehicles) for the uncertainty of beta, eta and T*
    bs = []
    for _ in range(boot):
        i = rng.integers(0, len(t), len(t))
        b, e = fit_weibull(t[i], d[i])
        bs.append((b, e, best_T(lambda x: reliability(x, b, e), 20, min(3 * e, 5 * t_max))))
    bs = np.array(bs)
    ci = {k: [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
          for j, k in enumerate(("beta", "eta", "T_star"))}

    # hold-out check with Kaplan-Meier (no distribution assumed); KM is only defined up to the longest observation
    km_t, km_s = kaplan_meier(te.t.to_numpy(float), te.failed.to_numpy(float))
    KM = lambda x: 1.0 if x < km_t[0] else float(km_s[np.searchsorted(km_t, x, side="right") - 1])
    grid = np.linspace(40, min(km_t[-1], t_max), 120)
    km_costs = np.array([cost_rate(T, KM) for T in grid])
    T_km = float(grid[int(np.argmin(km_costs))])
    rtf_weibull = COSTS["breakdown_eur"] / (eta * math.gamma(1 + 1 / beta))

    policies = {"Run to failure": None, f"Replace at characteristic life ({eta:.0f})": eta,
                f"Optimized age ({T_star:.0f})": T_star}
    compare = {}
    for name, T in policies.items():
        c = rtf_weibull if T is None else cost_rate(T, W)
        holdout = None if T is None or T > km_t[-1] else cost_rate(T, KM)
        fail_share = 1.0 if T is None else 1 - float(W(T))
        compare[name] = {"age": T, "eur_per_1000_time_units": 1000 * c,
                         "holdout_eur_per_1000_time_units": None if holdout is None else 1000 * holdout,
                         "share_ending_in_breakdown_pct": 100 * fail_share}

    sens = []
    for ratio in (1.5, 2, 3.4, 5, 10):
        cf = COSTS["planned_eur"] * ratio
        f = lambda T: (COSTS["planned_eur"] * W(T) + cf * (1 - W(T))) / integrate.quad(W, 0, T)[0]
        Tr = float(optimize.minimize_scalar(f, bounds=(20, 5 * t_max), method="bounded").x)
        sens.append({"breakdown_vs_planned": ratio, "best_age": Tr})

    in_service = df[df.failed == 0].t.to_numpy(float)
    planning = {"in_service": int(len(in_service)), "past_T": int(np.sum(in_service >= T_star)),
                "due_within_soon": int(np.sum((in_service < T_star) & (in_service >= T_star - SOON))),
                "soon_window": SOON}
    curve = [{"age": float(T), "weibull": cost_rate(T, W), "km_holdout": float(cost_rate(T, KM))
              if T <= km_t[-1] else None} for T in np.linspace(60, min(2.2 * eta, 5 * t_max), 50)]
    sv = np.linspace(0, t_max, 80)
    survival = {"t": sv.tolist(), "weibull": W(sv).tolist(),
                "km_holdout": [KM(x) if x <= km_t[-1] else None for x in sv]}
    return {"vehicles": len(df), "repairs": int(df.failed.sum()), "censored": int((1 - df.failed).sum()),
            "train": int(is_train.sum()), "holdout": int((~is_train).sum()),
            "fit": {"beta": beta, "eta": eta}, "ci95": ci, "T_star": T_star, "T_holdout_km": T_km,
            "costs": COSTS, "compare": compare, "sensitivity": sens, "planning": planning,
            "curve": curve, "survival": survival, "time_unit": "anonymised time steps (SCANIA)"}


def machine_windows(ages: dict[int, float], T_star: float, horizon: int, length: int) -> dict[int, list[tuple[int, int]]]:
    """Illustrative link to the schedule: a machine whose component reaches T* within SOON time units gets one
    maintenance window in this planning horizon (earlier if it is closer to T*)."""
    out = {}
    for k, age in ages.items():
        left = T_star - age
        if left <= SOON:
            start = int(max(0.0, left) / SOON * max(horizon - length, 0))
            out[k] = [(start, length)]
    return out
