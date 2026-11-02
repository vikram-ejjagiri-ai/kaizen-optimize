"""Calibrate the cutting model on real turning experiments.

Data: M. Gombár (University of West Bohemia, 2026), "Dataset and MATLAB analysis scripts for surface roughness and
tool wear study in C45 steel turning", Mendeley Data, DOI 10.17632/yz37pvtztx.1, licence CC BY 4.0.
Dry longitudinal turning of C45 (12 050.1) with a KNUX 190408-EL P20 insert, nose radius 0.8 mm.

What is fitted from the data
  * Surface roughness Ra: 40 measurements of a 2^3 factorial design (vc 8.8 / 351.7 m/min, f 0.1 / 0.5 mm,
    ap 0.1 / 3.0 mm, 5 replicates). Log-linear model with the speed x feed interaction that the data shows:
        ln Ra = b0 + b1 ln vc + b2 ln f + b3 ln ap + b4 ln vc * ln f
    checked with leave-one-run-out cross-validation (all 5 replicates of a run are left out together).
  * Measurement scatter of Ra: pooled standard deviation of ln Ra over the replicates.
  * Tool life: flank wear VB measured over time at 70, 110, 140 and 240 m/min (f 0.1 mm, ap 0.2 mm). For each
    speed VB = C * t^m is fitted and solved for VB = 0.3 mm (ISO 3685 criterion), then Taylor T = C_T * vc^-k.
    6 and 25 m/min are excluded: at such low speeds wear is dominated by built-up edge, not by Taylor-type wear.
  * Scatter of tool life: residual standard deviation of the Taylor fit.

What is NOT in the data and therefore assumed (literature values, stated in the README)
  * how tool life changes with feed and depth of cut (data only at f = 0.1, ap = 0.2):
        T = C_T * vc^-k * (f / 0.1)^-FEED_EXP * (ap / 0.2)^-DEPTH_EXP
  * specific cutting force for the spindle power (Kienzle, catalogue values for C45).
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pandas as pd

from optimize.config import DATA

XLSX = DATA / "cutting" / "C45_turning_Gombar_2026.xlsx"
VB_CRIT = 0.30                     # mm, ISO 3685
TAYLOR_SPEEDS = (70, 110, 140, 240)
FEED_EXP, DEPTH_EXP = 1.0, 0.3     # assumed: tool life ~ f^-1.0 * ap^-0.3 (typical extended-Taylor exponents)
REF_F, REF_AP = 0.10, 0.20         # conditions of the wear tests


def _design(df: pd.DataFrame) -> np.ndarray:
    lv, lf, la = np.log(df.vc), np.log(df.f), np.log(df.ap)
    return np.c_[np.ones(len(df)), lv, lf, la, lv * lf]


@functools.lru_cache(maxsize=1)
def fit() -> dict:
    doe = pd.read_excel(XLSX, sheet_name="5_DOE_RzRa")
    X, y = _design(doe), np.log(doe.Ra.to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    # leave-one-run-out cross-validation (a run = one corner of the design with its 5 replicates)
    pred = np.empty_like(y)
    for run in doe.run.unique():
        tr, te = (doe.run != run).to_numpy(), (doe.run == run).to_numpy()
        b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = X[te] @ b
    ra, ra_cv = doe.Ra.to_numpy(), np.exp(pred)
    fitted = np.exp(X @ beta)
    sd_rep = float(np.sqrt(doe.assign(l=np.log(doe.Ra)).groupby("run").l.var(ddof=1).mean()))

    # tool life from flank wear curves
    vb = pd.read_excel(XLSX, sheet_name="3_VB_TimeSeries")
    life = {}
    for v, g in vb[(vb.tau > 0) & vb.vc.isin(TAYLOR_SPEEDS)].groupby("vc"):
        m, c = np.polyfit(np.log(g.tau), np.log(g.VB.clip(lower=1e-4)), 1)
        life[int(v)] = float((VB_CRIT / math.exp(c)) ** (1 / m))
    lv, lt = np.log(list(life)), np.log(list(life.values()))
    slope, icpt = np.polyfit(lv, lt, 1)
    resid = lt - (icpt + slope * lv)
    return {
        "ra": {"coef": beta.tolist(), "terms": ["1", "ln vc", "ln f", "ln ap", "ln vc * ln f"],
               "r2": float(1 - ((ra - fitted) ** 2).sum() / ((ra - ra.mean()) ** 2).sum()),
               "cv_rmse_um": float(np.sqrt(((ra - ra_cv) ** 2).mean())),
               "cv_mape_pct": float(100 * np.mean(np.abs(ra - ra_cv) / ra)),
               "replicate_sd_log": sd_rep, "n": len(doe),
               "measured": doe[["vc", "f", "ap", "Ra"]].to_dict("records"), "cv_pred": ra_cv.tolist()},
        "tool_life": {"C_T": float(math.exp(icpt)), "k": float(-slope), "life_min_at_speed": life,
                      "resid_sd_log": float(resid.std(ddof=1)) if len(resid) > 2 else 0.1,
                      "feed_exp_assumed": FEED_EXP, "depth_exp_assumed": DEPTH_EXP,
                      "ref_f": REF_F, "ref_ap": REF_AP, "valid_speed": [min(TAYLOR_SPEEDS), max(TAYLOR_SPEEDS)]},
    }


def roughness(v, f, ap):
    b = fit()["ra"]["coef"]
    lv, lf = np.log(v), np.log(f)
    return np.exp(b[0] + b[1] * lv + b[2] * lf + b[3] * np.log(ap) + b[4] * lv * lf)


def tool_life(v, f, ap):
    t = fit()["tool_life"]
    return t["C_T"] * v ** (-t["k"]) * (f / REF_F) ** (-FEED_EXP) * (ap / REF_AP) ** (-DEPTH_EXP)
