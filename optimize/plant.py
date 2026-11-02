"""The plant: CNC turning of C45 steel parts on lathes, with a process model calibrated on real experiments.

- surface roughness Ra and tool life: fitted to real turning tests on C45 with a 0.8 mm nose-radius P20 insert
  (Gombár 2026, Mendeley Data, CC BY 4.0) - see calibration.py
- measurement scatter of trial cuts: taken from the replicates (roughness) and the Taylor fit (tool life)
- spindle power: Kienzle equation with catalogue values for C45 (assumption)
- cost per part = machine time + tool changes + tool inserts + energy (cost rates are assumptions)

The calibrated model is the "digital twin" that stands in for the machine: trial cuts are drawn from it with the real
scatter, and its noiseless optimum is the reference every optimizer is compared with.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from optimize import calibration

# ---- decision variables and their limits (catalogue values for a P-grade insert on C45) -----------------------
BOUNDS = {"v": (70.0, 240.0),     # cutting speed, m/min - range where tool wear was measured (no extrapolation)
          "f": (0.10, 0.40),      # feed, mm/rev - inside the tested feed range 0.1 - 0.5
          "ap": (0.5, 3.0)}       # depth of cut, mm
VARS = list(BOUNDS)


@dataclass(frozen=True)
class Part:
    name: str
    diameter: float      # mm
    length: float        # mm, machined length
    stock: float         # mm, radial material to remove
    ra_max: float        # µm, drawing requirement


@dataclass(frozen=True)
class Plant:
    nose_radius: float = 0.8           # mm
    taylor_C: float = 230.0            # extended Taylor constant (gives ~25 min at 200 m/min, 0.2 mm, 2 mm)
    taylor_n: float = 0.25
    taylor_a: float = 0.5
    taylor_b: float = 0.2
    kc11: float = 1700.0               # N/mm², specific cutting force of C45
    mc: float = 0.25
    kappa: float = 95.0                # tool cutting edge angle, degrees
    eta: float = 0.8                   # drive efficiency
    p_max: float = 7.5                 # kW spindle power limit (compact CNC lathe)
    machine_rate: float = 1.5          # €/min (machine + operator)
    handling_min: float = 1.0          # min load/unload per part
    tool_change_min: float = 2.0       # min per insert edge change
    insert_cost: float = 12.0          # € per cutting edge
    energy_price: float = 0.25         # €/kWh
    parts: tuple = field(default_factory=lambda: (
        Part("A shaft", 60, 200, 4.0, 3.2),
        Part("B flange", 120, 60, 5.0, 3.2),
        Part("C pin", 30, 120, 3.0, 2.5),   # tightest part; Ra <= 2.0 is too close to this insert's capability (~1.7 µm)
        Part("D axle", 80, 320, 4.0, 3.2),
    ))


PLANT = Plant()


def part(name: str) -> Part:
    return next(p for p in PLANT.parts if p.name.startswith(name))


# ---- noiseless process model ("the truth") ----------------------------------------------------------------------
def roughness(v, f, ap, pl: Plant = PLANT):
    """Ra in µm, model fitted to the real C45 experiments (calibration.py)."""
    return calibration.roughness(v, f, ap)


def tool_life(v, f, ap, pl: Plant = PLANT):
    """Minutes of cutting per insert edge until VB = 0.3 mm: Taylor fit on real wear data (calibration.py)."""
    return calibration.tool_life(v, f, ap)


def power(v, f, ap, pl: Plant = PLANT):
    """Spindle power in kW (Kienzle)."""
    h = f * math.sin(math.radians(pl.kappa))            # chip thickness
    b = ap / math.sin(math.radians(pl.kappa))           # chip width
    fc = pl.kc11 * h ** (1 - pl.mc) * b                 # N
    return fc * v / 60000 / pl.eta


def passes(ap, prt: Part) -> int:
    return math.ceil(prt.stock / ap - 1e-9)


def cutting_time(v, f, ap, prt: Part):
    """Minutes of cutting for one part: passes x (path length / feed rate). The last pass is set to the remaining
    stock, so the depth actually cut is stock / passes."""
    passes = math.ceil(prt.stock / ap - 1e-9)            # whole passes: 4 mm stock at 3 mm depth still needs 2
    rpm = 1000 * v / (math.pi * prt.diameter)
    return passes * prt.length / (f * rpm)


def evaluate(x: dict, prt: Part, pl: Plant = PLANT) -> dict:
    """Everything a process engineer wants to know about one setting. "ap" is the maximum depth of cut; the CNC
    program removes the stock in whole passes, so the depth actually cut is stock / passes."""
    v, f, ap_max = x["v"], x["f"], x["ap"]
    n = passes(ap_max, prt)
    ap = prt.stock / n
    tc = cutting_time(v, f, ap, prt)
    T = tool_life(v, f, ap, pl)
    edges = tc / T                                       # insert edges used per part
    p = power(v, f, ap, pl)
    ra = float(roughness(v, f, ap, pl))
    machine_min = tc + pl.handling_min + pl.tool_change_min * edges
    cost = (pl.machine_rate * machine_min + pl.insert_cost * edges + pl.energy_price * p * tc / 60)
    return {"v": v, "f": f, "ap": ap_max, "passes": n, "ap_cut": ap, "cutting_min": tc, "cycle_min": machine_min, "tool_life_min": T,
            "edges_per_part": edges, "power_kw": p, "ra_um": ra, "cost_eur": cost,
            "feasible": ra <= prt.ra_max and p <= pl.p_max}


# ---- what an experiment on the real machine returns: noisy measurements ------------------------------------------
def experiment(x: dict, prt: Part, rng: np.random.Generator, pl: Plant = PLANT) -> dict:
    """One trial cut with the real scatter: roughness as between the replicates of the C45 tests, tool life as the
    residual scatter of the Taylor fit."""
    true = evaluate(x, prt, pl)
    cal = calibration.fit()
    ra = true["ra_um"] * rng.lognormal(0, cal["ra"]["replicate_sd_log"])
    T = true["tool_life_min"] * rng.lognormal(0, cal["tool_life"]["resid_sd_log"])
    tc = true["cutting_min"]
    edges = tc / T
    cost = (pl.machine_rate * (tc + pl.handling_min + pl.tool_change_min * edges) + pl.insert_cost * edges
            + pl.energy_price * true["power_kw"] * tc / 60)
    return {**x, "ra_um": ra, "cost_eur": cost, "power_kw": true["power_kw"]}


# ---- a typical "handbook" starting point used as the baseline ----------------------------------------------------
HANDBOOK = {"v": 180.0, "f": 0.15, "ap": 1.5}


def handbook(prt: Part) -> dict:
    """Catalogue starting values; parts with a tighter roughness limit get the usual finishing feed of 0.10 mm."""
    return {**HANDBOOK, "f": 0.15 if prt.ra_max >= 3.2 else 0.10}
