"""Kaizen Optimize dashboard:  streamlit run app.py"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optimize import pipeline
from optimize.advisor import advise, fact_sheet
from optimize.llm import LLM

st.set_page_config(page_title="Kaizen Optimize", page_icon="⚙️", layout="wide")
GREEN, ORANGE, BLUE, GREY, RED = "#2f6b1f", "#c8702c", "#3b6fb6", "#7b6e63", "#b3261e"
PAL = [GREEN, ORANGE, BLUE, "#8a4fa3", "#5a8f8a", GREY, "#a3a04f"]


@st.cache_data(show_spinner=False)
def results():
    return pipeline.load()


with st.sidebar:
    st.header("⚙️ Kaizen Optimize")
    st.caption("Four optimization problems of a machining plant, each solved on a real public dataset.")
    quick = st.checkbox("Quick run (smaller benchmarks, ~3 min)", value=True)
    if st.button("Recompute everything"):
        box = st.empty()
        with st.spinner("Optimizing ..."):
            pipeline.run(quick=quick, log=lambda m: box.caption(m))
        results.clear()
    st.divider()
    st.markdown("**Data sources**")
    st.caption("1 · C45 turning experiments, Gombár 2026 (Mendeley Data, CC BY 4.0)  \n"
               "2 · PMSP-SDST-TWT scheduling benchmark, Jun 2025 (MIT)  \n"
               "3 · SCANIA Component X fleet data (Swedish National Data Service)  \n"
               "4 · carparts spare-part sales, Hyndman et al. 2008")
    st.caption("Costs are assumptions. The four datasets are linked through one plant story; those links are "
               "illustrative. Details in README.md.")

R = results()
if R is None:
    st.warning("No results yet. Click **Recompute everything** in the sidebar, or run `python -m optimize.pipeline`.")
    st.stop()
H = pipeline.headline(R)

st.title("Kaizen Optimize")
st.caption(f"Cutting data → schedule → maintenance → spare parts · real data · results from {R['generated']}"
           + (" (quick run)" if R.get("quick") else ""))
tab_o, tab_p, tab_s, tab_m, tab_i, tab_a = st.tabs(
    ["Overview", "1 · Cutting data", "2 · Schedule", "3 · Maintenance", "4 · Spare parts", "🤖 AI advisor"])

# ---------------------------------------------------------------------------------------------------------------------
with tab_o:
    inv = R["inventory"]["summary"]
    best_inv = H["inventory_best_policy"]
    c = st.columns(4)
    c[0].metric("Machining cost per part", f"-{H['mix_saving_pct']:.0f} %", "vs handbook cutting data",
                delta_color="off")
    c[1].metric("Weighted tardiness this week", f"{H['week_wt']['optimized|Iterated greedy']:,}",
                f"{H['week_wt']['optimized|Iterated greedy'] - H['week_wt']['handbook|EDD']:,} vs today",
                delta_color="inverse")
    c[2].metric("Maintenance cost", f"-{H['maint_saving_vs_rtf_pct']:.0f} %", "vs run to failure", delta_color="off")
    c[3].metric("Spare-part fill rate", f"{inv[best_inv]['fill_rate_pct']:.1f} %",
                f"{inv[best_inv]['eur_per_month_all_parts'] - inv['Textbook (normal + EOQ)']['eur_per_month_all_parts']:,.0f} €/month vs textbook",
                delta_color="inverse")
    st.markdown("#### Where the data comes from and what is optimized")
    st.dataframe(pd.DataFrame([
        ["1 Cutting data", "40 designed experiments + tool-wear curves, real C45 turning (Gombár 2026)",
         "Calibrated roughness + Taylor model; Bayesian optimization vs DOE vs random with 20 noisy trial cuts"],
        ["2 Schedule", "Published benchmark: 54 jobs, 5 unrelated machines, sequence-dependent setups (Jun 2025)",
         "Iterated greedy local search vs published CP, MILP, GA, SA, dispatch rules, LLMs"],
        ["3 Maintenance", f"{R['maintenance']['vehicles']:,} real trucks, {R['maintenance']['repairs']:,} repairs, "
                          f"{R['maintenance']['censored']:,} censored (SCANIA Component X)",
         "Censored Weibull fit, age-replacement optimum, hold-out check with Kaplan-Meier"],
        ["4 Spare parts", f"{R['inventory']['parts']} real parts, 51 months of sales (carparts)",
         "(s, Q) policies optimized by simulation, tested on 12 unseen months"],
    ], columns=["Stage", "Real data", "Method"]), hide_index=True, width="stretch")
    st.markdown(f"""
**How the stages connect**

- The recommended cutting data makes processing times **{100 * (1 - H['cycle_factor']):.0f} % shorter**; this week's
  order book is rescheduled with the new times.
- Machines carry a wearing component; components that reach the optimal replacement age
  (**{R['maintenance']['T_star']:.0f}** time units) get a maintenance window in the schedule.
- The new cutting data changes insert use by **{H['insert_use_change_pct']:+.0f} %** (fewer passes, and tool life
  barely drops with speed in the real data), so spare-part rules must be reviewed when the process changes.
""")

# ---------------------------------------------------------------------------------------------------------------------
with tab_p:
    cal = R["calibration"]
    st.subheader("Cutting data: model calibrated on real turning experiments")
    c = st.columns(3)
    c[0].metric("Roughness model R²", f"{cal['ra']['r2']:.2f}", help="40 measurements, 2^3 design with 5 replicates")
    c[1].metric("Replicate scatter of Ra", f"±{100 * cal['ra']['replicate_sd_log']:.0f} %",
                help="Used as the noise of every simulated trial cut")
    c[2].metric("Taylor exponent k (T = C · vc^-k)", f"{cal['tool_life']['k']:.2f}",
                f"C = {cal['tool_life']['C_T']:.0f} min", delta_color="off",
                help="Time to flank wear VB = 0.3 mm at f = 0.1 mm, ap = 0.2 mm, fitted on 70-240 m/min")
    l, r = st.columns(2)
    meas = pd.DataFrame(cal["ra"]["measured"])
    meas["cv"] = cal["ra"]["cv_pred"]
    fig = go.Figure(go.Scatter(x=meas["Ra"], y=meas["cv"], mode="markers", marker=dict(color=GREEN, size=8),
                               text=[f"vc {a:.0f}, f {b}, ap {c_}" for a, b, c_ in zip(meas.vc, meas.f, meas.ap)]))
    mx = float(max(meas.Ra.max(), meas.cv.max())) * 1.05
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dot", color=GREY), showlegend=False)
    fig.update_layout(height=320, xaxis_title="Measured Ra (µm)", yaxis_title="Predicted, run left out (µm)",
                      margin=dict(t=30), title="Roughness: leave-one-run-out check", showlegend=False)
    l.plotly_chart(fig, width="stretch")
    life = cal["tool_life"]["life_min_at_speed"]
    sp = sorted(int(k) for k in life)
    xs = list(range(60, 251, 5))
    fig = go.Figure(go.Scatter(x=sp, y=[life[str(s)] if str(s) in life else life[s] for s in sp], mode="markers",
                               marker=dict(size=11, color=ORANGE), name="from measured wear curves"))
    fig.add_scatter(x=xs, y=[cal["tool_life"]["C_T"] * v ** -cal["tool_life"]["k"] for v in xs], mode="lines",
                    line_color=GREEN, name="Taylor fit")
    fig.update_layout(height=320, xaxis_title="Cutting speed (m/min)", yaxis_title="Tool life to VB 0.3 mm (min)",
                      margin=dict(t=60), title="Tool life (f 0.1 mm, ap 0.2 mm)", legend=dict(orientation="h", y=1.08, x=0))
    r.plotly_chart(fig, width="stretch")
    st.caption("The roughness check leaves out a whole corner of the 2-level design, so it is an extrapolation and "
               "the error is larger than inside the range. Tool life falls only slowly with speed in this data.")

    rows = []
    for f, p in R["parts"].items():
        for label, x in (("Handbook", p["handbook"]), ("Recommended (BO)", p["recommended"]),
                         ("Model optimum", p["optimum"])):
            rows.append({"Part": p["name"], "Setting": label, "v [m/min]": round(x["v"]), "f [mm/rev]": round(x["f"], 3),
                         "Passes": x["passes"], "Ra [µm]": round(x["ra_um"], 2), "Ra limit": p["ra_max"],
                         "Tool life [min]": round(x["tool_life_min"]), "Power [kW]": round(x["power_kw"], 1),
                         "Cycle [min]": round(x["cycle_min"], 2), "Cost / part [€]": round(x["cost_eur"], 2)})
    st.markdown("#### Recommended cutting data")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("#### Which search strategy finds the best setting with 20 trial cuts?")
    st.caption("Every strategy was repeated with different (real-size) measurement noise. Gap = extra cost of the "
               "recommended setting compared with the optimum of the calibrated model.")
    bm = pd.DataFrame([{"Part": R["parts"][f]["name"], "Strategy": s["strategy"], "Median gap %": s["median_gap_pct"],
                        "90th pct gap %": s["p90_gap_pct"], "Meets spec %": s["feasible_pct"]}
                       for f, b in R["benchmark"].items() for s in b["summary"].values()])
    fig = go.Figure()
    for strat, col in zip(bm["Strategy"].unique(), [GREY, ORANGE, GREEN]):
        d = bm[bm["Strategy"] == strat]
        fig.add_bar(x=d["Part"], y=d["Median gap %"], name=strat, marker_color=col)
    fig.update_layout(barmode="group", height=330, yaxis_title="Median gap (%)", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(bm.round(1), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------------------------------------
with tab_s:
    sb = R["schedule_benchmark"]
    mc = R["schedule_model_check"]
    st.subheader("Scheduling: unrelated machines, sequence-dependent setups, weighted tardiness")
    st.caption(f"Model check: on {mc['proven_optimal']} small benchmark instances solved to proven optimality, "
               f"{mc['equal_to_published']} give exactly the published best value, so our model and objective match "
               "the benchmark.")
    rk = pd.DataFrame(sb["ranking"])
    fig = go.Figure(go.Bar(y=rk["method"], x=rk["mean_gap_pct"], orientation="h",
                           marker_color=[GREEN if "ours" in m else GREY for m in rk["method"]]))
    fig.update_layout(height=40 + 26 * len(rk), xaxis_title="Mean gap to best known solution (%)",
                      yaxis=dict(autorange="reversed"), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{sb['instances']} medium instances, {sb['time_limit_s']:.0f} s per instance for our iterated greedy. "
               "Grey = results published with the benchmark (CP = IBM CP Optimizer, HGA = hybrid genetic algorithm, "
               "SA = simulated annealing, LLM rows = schedules written by language models).")

    st.markdown(f"#### This week's order book (benchmark instance {R['week']['instance']})")
    wk = R["week"]["results"]
    st.caption("The benchmark instances are deliberately overloaded (tight due dates), so most jobs finish late in "
               "any schedule. That is why the objective is weighted tardiness: how late, weighted by customer importance.")
    st.dataframe(pd.DataFrame([{"Cycle times": k.split("|")[0], "Method": k.split("|")[1],
                                "Weighted tardiness": v["kpis"]["weighted_tardiness"], "Late jobs": v["kpis"]["late_jobs"],
                                "Setup time": v["kpis"]["setup_total"], "Makespan": v["kpis"]["makespan"]}
                               for k, v in wk.items()]), hide_index=True, width="stretch")
    pick = st.radio("Show schedule", list(wk), index=list(wk).index("optimized|Iterated greedy"), horizontal=True)
    due, rows = R["week"]["due"], wk[pick]["rows"]
    fig = go.Figure()
    for x in rows:
        late = x["end"] > due[x["job"]]
        if x["setup"]:
            fig.add_bar(y=[f"M{x['machine'] + 1}"], x=[x["setup"]], base=[x["setup_start"]], orientation="h",
                        marker_color="#d9d2c7", showlegend=False, hovertext=f"setup {x['setup']}")
        fig.add_bar(y=[f"M{x['machine'] + 1}"], x=[x["end"] - x["start"]], base=[x["start"]], orientation="h",
                    marker_color=PAL[x["job"] % len(PAL)], showlegend=False, text=str(x["job"]),
                    marker_line=dict(color=RED if late else "white", width=3 if late else 1),
                    hovertext=f"job {x['job']}, due {due[x['job']]}" + (" LATE" if late else ""))
    for k, wins in R["week"]["windows"].items():
        for a, d in wins:
            fig.add_bar(y=[f"M{int(k) + 1}"], x=[d], base=[a], orientation="h", marker_color=RED, opacity=0.55,
                        text="maint.", showlegend=False)
    fig.update_layout(barmode="overlay", height=300, xaxis_title="Time units", margin=dict(t=10),
                      yaxis=dict(categoryorder="category descending"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Grey = setup, red block = maintenance window, red outline = late job. 'optimized' = processing "
               f"times × {H['cycle_factor']:.2f} from the new cutting data.")

# ---------------------------------------------------------------------------------------------------------------------
with tab_m:
    M = R["maintenance"]
    st.subheader("When to replace a wearing component? (real fleet data)")
    c = st.columns(4)
    c[0].metric("Best replacement age", f"{M['T_star']:.0f}",
                f"95 % CI {M['ci95']['T_star'][0]:.0f}–{M['ci95']['T_star'][1]:.0f}", delta_color="off")
    c[1].metric("Weibull shape β", f"{M['fit']['beta']:.2f}", "wear-out (β > 1)", delta_color="off")
    c[2].metric("Characteristic life η", f"{M['fit']['eta']:.0f}")
    c[3].metric("Hold-out optimum (Kaplan-Meier)", f"{M['T_holdout_km']:.0f}")
    l, r = st.columns(2)
    sv = M["survival"]
    fig = go.Figure(go.Scatter(x=sv["t"], y=sv["weibull"], mode="lines", line_color=GREEN, name="Weibull fit (70 %)"))
    fig.add_scatter(x=sv["t"], y=sv["km_holdout"], mode="lines", line=dict(color=ORANGE, shape="hv"),
                    name="Kaplan-Meier, hold-out 30 %")
    fig.update_layout(height=320, xaxis_title=f"Age ({M['time_unit']})", yaxis_title="Share still working",
                      margin=dict(t=60), title="Does the fitted model match unseen vehicles?",
                      legend=dict(orientation="h", y=1.08, x=0))
    l.plotly_chart(fig, width="stretch")
    cv = pd.DataFrame(M["curve"])
    fig = go.Figure(go.Scatter(x=cv["age"], y=cv["weibull"] * 1000, mode="lines", line_color=GREEN, name="Weibull model"))
    fig.add_scatter(x=cv["age"], y=cv["km_holdout"] * 1000, mode="lines", line_color=ORANGE, name="hold-out vehicles")
    fig.add_vline(x=M["T_star"], line_dash="dash", annotation_text="optimum")
    fig.update_layout(height=320, xaxis_title="Replacement age", yaxis_title="€ per 1000 time units", margin=dict(t=60),
                      title="Cost of the age-replacement policy", legend=dict(orientation="h", y=1.08, x=0))
    r.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame([{"Policy": k, "€ / 1000 time units (model)": round(v["eur_per_1000_time_units"]),
                                "€ / 1000 time units (hold-out)": None if v["holdout_eur_per_1000_time_units"] is None
                                else round(v["holdout_eur_per_1000_time_units"]),
                                "Ends in breakdown %": round(v["share_ending_in_breakdown_pct"])}
                               for k, v in M["compare"].items()]), hide_index=True, width="stretch")
    st.markdown("**Sensitivity: the best age depends on how expensive a breakdown is**")
    st.dataframe(pd.DataFrame([{"Breakdown cost / planned repair cost": s["breakdown_vs_planned"],
                                "Best replacement age": round(s["best_age"])} for s in M["sensitivity"]]),
                 hide_index=True)
    pl = M["planning"]
    st.info(f"Planning list: {pl['past_T']:,} components in service are already past the replacement age, "
            f"{pl['due_within_soon']:,} more reach it within the next {pl['soon_window']:.0f} time units.")
    st.caption(f"Costs assumed: planned repair {M['costs']['planned_eur']:.0f} €, breakdown {M['costs']['breakdown_eur']:.0f} €. "
               "Time is in the dataset's anonymised units.")

# ---------------------------------------------------------------------------------------------------------------------
with tab_i:
    I = R["inventory"]
    st.subheader(f"Spare parts: reorder point s and order quantity Q for {I['parts']} real parts")
    df = pd.DataFrame([{"Policy": k, "€ / month (all parts)": round(v["eur_per_month_all_parts"]),
                        "Fill rate %": round(v["fill_rate_pct"], 1), "Average stock (units)": round(v["avg_stock_units"]),
                        "Orders / month": round(v["orders_per_month"], 1)} for k, v in I["summary"].items()])
    st.dataframe(df, hide_index=True, width="stretch")
    co = I["costs"]
    st.caption(f"Scored only on the last {I['test_months']} months (Apr 2001 – Mar 2002), which no policy saw. Lead "
               f"time {I['lead_time_months']} month, lost sales. Assumed costs: part value {co['unit_value_eur']:.0f} €, "
               f"holding {co['holding_pct_per_month']:.0f} % per month, {co['order_eur']:.0f} € per order, "
               f"{co['lost_sale_eur']:.0f} € per lost sale.")
    ex = I["example_part"]
    fig = go.Figure(go.Bar(x=ex["months"], y=ex["demand"],
                           marker_color=[GREEN if i < I["train_months"] else ORANGE for i in range(len(ex["demand"]))]))
    fig.add_vline(x=I["train_months"] - 0.5, line_dash="dash", annotation_text="test months →")
    fig.update_xaxes(type="category", tickangle=-45, nticks=18)
    fig.update_layout(height=300, yaxis_title="Units sold", margin=dict(t=30),
                      title=f"Example part {ex['id']}: green = training months, orange = test months")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(ex["policies"]).rename(columns={"policy": "Policy", "eur_per_month": "€ / month",
                                                              "fill_rate_pct": "Fill rate %"}).round(2), hide_index=True)
    st.caption("Demand of these parts falls over their life cycle (about 2.0 units per month in 1998, 0.6 in the test "
               "year). Policies built on the full history overstock; planning with recent demand matters more than "
               "the optimization method.")

# ---------------------------------------------------------------------------------------------------------------------
with tab_a:
    st.subheader("AI advisor")
    llm = LLM()
    st.caption(f"Model: {llm.name}. The advisor only explains the optimizer results; every number in its answer is "
               "checked against the fact sheet.")
    q = st.text_input("Ask a question (or leave empty for the weekly recommendation)",
                      placeholder="Why should we replace the component before it fails?")
    if st.button("Ask the advisor", type="primary"):
        with st.spinner("Thinking ..."):
            a = advise(R, q or None, llm)
        st.markdown(a.text)
        st.caption(f"Answered by {a.generator}")
        for w in a.warnings:
            st.warning(w)
    with st.expander("Fact sheet given to the AI"):
        st.text("\n".join(fact_sheet(R)))
