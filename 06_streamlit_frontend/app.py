"""ICU Transfer Prediction -> Explainable Risk Stratification -> ICU Crisis Capacity Management (research demo).
Run from repository root: streamlit run 06_streamlit_frontend/app.py
Uses the frozen Step 53 XGBoost model; it never trains or modifies it.
Risk order = predicted ICU-transfer risk, NOT clinical urgency. Simulations are hypothetical.
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from xgboost import XGBClassifier, DMatrix

APP_DIR = Path(__file__).resolve().parent
NAVY, TEAL, LIGHT, AMBER, RED, GREEN = "#0B2545", "#0E9AA7", "#9FD8DD", "#E0A030", "#C0392B", "#2E9E6B"
FILES = ["final_xgboost_75_model.json", "final_inference_bundle.joblib", "final_model_metadata.json"]
STAT = {"first": "first value", "last": "latest value", "mean": "average", "minimum": "lowest value", "maximum": "highest value",
        "std": "variability", "count": "number of tests", "change": "change (latest minus first)", "slope": "trend over time"}


@st.cache_resource(show_spinner="Loading Step 53 model...")
def load_model():
    options = [os.environ.get("ICU_MODEL_DIR"), APP_DIR / "model",
               APP_DIR.parent / "model training figures" / "expanded_lab_xgboost" / "step53_final_75_model"]
    folder = next((Path(p) for p in options if p and all((Path(p) / f).exists() for f in FILES)), None)
    if folder is None:
        raise FileNotFoundError("Put the 3 Step 53 model files in a folder named 'model' next to this file.")
    model = XGBClassifier()
    model.load_model(str(folder / FILES[0]))
    bundle = joblib.load(folder / FILES[1])
    metadata = json.loads((folder / FILES[2]).read_text(encoding="utf-8"))
    return model, bundle, metadata


def preprocess(df, bundle):
    required = list(bundle["raw_columns"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required input columns: {missing[:12]}")
    raw = df[required].copy()
    cat = list(bundle["categorical_columns"])
    for c in cat:
        raw[c] = raw[c].fillna("Unknown").astype(str)
    for c in required:
        if c not in cat:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
            raw[c] = raw[c].replace([np.inf, -np.inf], np.nan)
    matrix = np.asarray(bundle["preprocessor"].transform(raw), dtype=np.float32)
    selected = matrix[:, np.asarray(bundle["selected_indices"], dtype=int)]
    if selected.shape[1] != 75:
        raise ValueError(f"Expected 75 selected predictors, got {selected.shape[1]}")
    return selected


def lab_names():
    f = APP_DIR / "data" / "lab_names_verified.csv"           # built from MIMIC d_labitems; unverified IDs stay as numbers
    if not f.exists():
        return {}
    d = pd.read_csv(f, dtype={"itemid": str})
    return dict(zip(d.itemid, d.name))


def readable(feature, names):
    if feature == "age":
        return "Age"
    if feature.startswith("lab_"):
        _, item, stat = feature.split("_", 2)
        return f"{names.get(item, 'Lab test ' + item)}: {STAT.get(stat, stat)}"
    for col, nice in (("admission_type_", "Admission type"), ("admission_location_", "Admission source"), ("gender_", "Sex")):
        if feature.startswith(col):
            return f"{nice}: {feature[len(col):].title()}"
    return feature


def stratify(risk, rank, threshold):
    """Tiers use only the model's validation-selected threshold and the cohort rank (no invented cut-offs)."""
    if rank <= 3:
        return "Highest in cohort"
    return "Above review threshold" if risk >= threshold else "Below threshold"


def replicate(risks_sorted, factor):
    """Surge scenario: `factor` times as many patients with the same risk profile (keeps probabilities calibrated)."""
    return np.sort(np.tile(risks_sorted, int(factor)))[::-1]


def load_stay_hours():
    """Observed ICU stay lengths (hours) from data/icu_stay_hours.csv, made by tools/make_stay_lengths.py from MIMIC-IV icustays."""
    f = APP_DIR / "data" / "icu_stay_hours.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)["los_hours"].dropna()
    d = d[d > 0]
    return d.to_numpy(float) if len(d) >= 30 else None


def draw_residual_stays(rng, stay_hours, weights, k):
    """Remaining stay of patients ALREADY in the ICU: longer stays are more likely to be in the unit at any moment, so pick a stay
    with probability proportional to its length, then a random point inside it (standard residual-life correction)."""
    idx = rng.choice(len(stay_hours), size=k, p=weights)
    return np.maximum(1.0, stay_hours[idx] * rng.random(k))


def weighted_summary(r):
    """Count-based vs risk-weighted shortfall for one scenario (weight of an unmet request = predicted risk of that patient)."""
    req, um = float(r["need_chance"].sum()), float(r["unmet"].mean())
    nr, ur = float(r["need_risk"].mean()), float(r["unmet_risk"].mean())
    return {"requests": req, "unmet": um, "pct_requests_unmet": 100 * um / max(req, 1e-9),
            "weighted_unmet": ur, "pct_risk_unserved": 100 * ur / max(nr, 1e-9)}


def crisis_level(p_any_unmet):
    """Illustrative alert levels (adjustable defaults, not validated): Green < 10%, Amber < 40%, Red >= 40%."""
    return ("GREEN", GREEN) if p_any_unmet < 0.10 else (("AMBER", AMBER) if p_any_unmet < 0.40 else ("RED", RED))


def simulate_priority(risks, beds, occupied, mean_stay_days, max_wait_hours, runs, seed=42, need_prob=None, no_discharge=False, stay_hours=None):
    """Hypothetical ICU simulation in HOURS. Patients are given in priority order (index 0 = highest predicted risk).
    Assumptions (all hypothetical):
      * every patient is admitted to hospital at hour 0 and needs the ICU with probability = predicted risk
        (or with probability need_prob if given, e.g. 1 for flagged patients in the precautionary mode; the order still follows risk);
      * the request is made at a random hour inside the model's 24-72 h prediction window;
      * beds already occupied free up after a random remaining stay (exponential with mean = average stay, or, if stay_hours is
        given, the real observed ICU stay lengths: new patients draw a full stay, current patients a residual stay);
      * each hour, free beds go to waiting patients with the highest predicted risk first;
      * if no_discharge is True (conservative), nobody leaves an ICU bed during the run and all requests are treated as made
        together, so only the beds free at the start are available and they go to the highest-risk requesters first;
      * a request that waits longer than max_wait_hours is UNMET.
    Every request is followed until it is served or exceeds the wait limit, so none is cut off by the end of the run."""
    rng = np.random.default_rng(seed)
    n, mean_h = len(risks), mean_stay_days * 24.0
    if stay_hours is not None:
        stay_hours = np.asarray(stay_hours, float)
        residual_w = stay_hours / stay_hours.sum()
    last_hour = 72 + int(max_wait_hours) + 1
    risk_w = np.asarray(risks, float)                      # weight of a request = the patient's predicted ICU-transfer risk
    needs_n, admitted_n, unmet_n = np.zeros(n), np.zeros(n), np.zeros(n)
    immediate, delayed, unmet, waits, peaks = [], [], [], [], []
    unmet_risk, need_risk = [], []
    for _ in range(runs):
        needs = rng.random(n) < (risks if need_prob is None else need_prob)
        ask_hour = rng.integers(24, 73, n)
        if no_discharge:
            ask_hour = np.full(n, 24)
        k_occ = min(occupied, beds)
        if no_discharge:
            busy_until = [float('inf')] * k_occ
        elif stay_hours is not None:
            busy_until = list(draw_residual_stays(rng, stay_hours, residual_w, k_occ))
        else:
            busy_until = [max(1.0, rng.exponential(mean_h)) for _ in range(k_occ)]
        arrivals = {}
        for i in np.flatnonzero(needs):
            arrivals.setdefault(int(ask_hour[i]), []).append(int(i))
        waiting, now_i, now_d, now_u, peak, now_ur = [], 0, 0, 0, 0, 0.0
        for hour in range(last_hour + 1):
            busy_until = [t for t in busy_until if t > hour]                 # discharges
            waiting += arrivals.get(hour, [])
            still = []
            for i in waiting:                                                # waited too long -> unmet
                if hour - ask_hour[i] > max_wait_hours:
                    now_u += 1; unmet_n[i] += 1; now_ur += float(risk_w[i])
                else:
                    still.append(i)
            waiting = sorted(still)                                          # lowest index = highest risk
            while waiting and len(busy_until) < beds:
                i = waiting.pop(0)
                wait = hour - ask_hour[i]
                if no_discharge:
                    busy_until.append(float('inf'))
                elif stay_hours is not None:
                    busy_until.append(hour + max(1.0, float(stay_hours[rng.integers(len(stay_hours))])))
                else:
                    busy_until.append(hour + max(1.0, rng.exponential(mean_h)))
                admitted_n[i] += 1
                waits.append(wait)
                if wait == 0: now_i += 1
                else: now_d += 1
            peak = max(peak, len(busy_until))
        assert not waiting
        immediate.append(now_i); delayed.append(now_d); unmet.append(now_u); peaks.append(peak)
        unmet_risk.append(now_ur); need_risk.append(float(risk_w[needs].sum()))
        needs_n += needs
    return {"need_chance": needs_n / runs, "admit_chance": admitted_n / runs, "unmet_chance": unmet_n / runs,
            "immediate": np.array(immediate), "delayed": np.array(delayed), "unmet": np.array(unmet),
            "unmet_risk": np.array(unmet_risk), "need_risk": np.array(need_risk),
            "avg_wait_hours": float(np.mean(waits)) if waits else 0.0, "peak": np.array(peaks)}


# ------------------------------------------------------------------ app
st.set_page_config(page_title="ICU Crisis Management", page_icon="🩺", layout="wide")
st.markdown(f"""<style>.hero{{background:{NAVY};color:#fff;padding:18px 26px;border-radius:10px;border-bottom:4px solid {TEAL};margin-bottom:10px}}
.hero h2{{color:#fff;margin:0;padding:0}} .hero p{{color:#CFE3EA;margin:2px 0 0 0;font-size:.9rem}}
.lvl{{border-radius:10px;padding:14px 16px;color:#fff;text-align:center}} .lvl b{{font-size:1.5rem}} .lvl span{{font-size:.85rem}}</style>
<div class="hero"><h2>ICU Transfer Risk & Crisis Capacity Management</h2>
<p>Kidney-disease admissions · ICU transfer at 24-72 h · patient risk → explainable stratification → ICU capacity planning</p></div>""",
            unsafe_allow_html=True)
st.caption("Research demonstration. Risk order is not clinical urgency, alert levels are illustrative, and no real beds are allocated.")

try:
    model, bundle, metadata = load_model()
except Exception as e:
    st.error(str(e)); st.stop()
threshold = float(metadata["selected_threshold"])
tab1, tab2, tab3 = st.tabs(["1  Risk & stratification", "2  Why this risk? (SHAP)", "3  ICU crisis management"])

with tab1:
    file = st.file_uploader("Upload patient features (CSV or Excel)", type=["csv", "xlsx"])
    if file is not None and st.button("Predict ICU-transfer risk", type="primary"):
        try:
            df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
            matrix = preprocess(df, bundle)
            probs = model.predict_proba(matrix)[:, 1]
            ids = df["sample_id"].astype(str).values if "sample_id" in df else [f"Patient {i+1}" for i in range(len(df))]
            out = pd.DataFrame({"patient": ids, "risk": probs, "row": np.arange(len(df))})
            out = out.sort_values("risk", ascending=False, kind="stable").reset_index(drop=True)
            out.insert(0, "rank", np.arange(1, len(out) + 1))
            out["stratum"] = [stratify(r, k, threshold) for r, k in zip(out.risk, out["rank"])]
            st.session_state.update(ranked=out, matrix=matrix, sim=None)
        except Exception as e:
            st.error(str(e))
    ranked = st.session_state.get("ranked")
    if ranked is None:
        st.info("Upload a patient file and press Predict.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Patients", len(ranked))
        b.metric("Highest predicted risk", f"{ranked.risk.max():.1%}")
        c.metric(f"Above review threshold ({threshold:.2%})", int((ranked.risk >= threshold).sum()))
        d.metric("Expected ICU transfers", f"{ranked.risk.sum():.2f}")
        colors = [NAVY if k <= 3 else (TEAL if r >= threshold else LIGHT) for r, k in zip(ranked.risk, ranked["rank"])]
        fig = go.Figure(go.Bar(x=ranked.patient, y=ranked.risk * 100, marker_color=colors))
        fig.add_hline(y=threshold * 100, line_dash="dash", line_color=AMBER, annotation_text="review threshold")
        fig.update_layout(template="plotly_white", height=330, yaxis_title="Risk (%)",
                          title="Navy = highest three · teal = above threshold · light = below threshold")
        st.plotly_chart(fig)
        view = ranked[["rank", "patient", "risk", "stratum"]].copy()
        view["risk"] = (view.risk * 100).round(2)
        view.columns = ["Rank", "Patient", "Predicted risk (%)", "Risk stratum"]
        st.dataframe(view, hide_index=True)
        st.download_button("Download ranking", ranked.drop(columns="row").to_csv(index=False), "ranked_icu_risk.csv", "text/csv")

with tab2:
    ranked = st.session_state.get("ranked")
    if ranked is None:
        st.info("Predict risks on tab 1 first.")
    else:
        who = st.selectbox("Patient (in rank order)", list(ranked.patient))
        r = ranked[ranked.patient == who].iloc[0]
        x = st.session_state["matrix"][int(r.row):int(r.row) + 1]
        shap = model.get_booster().predict(DMatrix(x), pred_contribs=True)[0][:-1]
        names = lab_names()
        ex = pd.DataFrame({"f": bundle["selected_feature_names"], "c": shap, "v": x[0]})
        ex["Feature"] = [readable(f, names) for f in ex.f]
        ex = ex.reindex(ex.c.abs().sort_values(ascending=False).index).head(10).sort_values("c")
        a, b, c = st.columns(3)
        a.metric("Patient", who); b.metric("Predicted risk", f"{r.risk:.1%}"); c.metric("Stratum", r.stratum)
        fig = go.Figure(go.Bar(x=ex.c, y=ex.Feature, orientation="h", marker_color=[TEAL if v > 0 else NAVY for v in ex.c],
                               customdata=ex.v.map(lambda v: "missing" if np.isnan(v) else f"{v:g}"),
                               hovertemplate="%{y}<br>effect %{x:.3f}<br>value %{customdata}<extra></extra>"))
        fig.update_layout(template="plotly_white", height=430, xaxis_title="Effect on model output (log-odds)",
                          title="Top 10 factors (teal raises predicted risk, navy lowers it)")
        st.plotly_chart(fig)
        st.caption("Model contributions, not clinical causes. Lab tests show as numbers until data/lab_names_verified.csv exists.")

with tab3:
    ranked = st.session_state.get("ranked")
    real_stays = load_stay_hours()
    if ranked is None:
        st.info("Predict risks on tab 1 first.")
    elif real_stays is None:
        st.error("Real ICU stay lengths not found. Create data/icu_stay_hours.csv with tools/make_stay_lengths.py, then reload this page.")
    else:
        st.markdown("**ICU capacity stress test:** how does the unit cope if more kidney patients than expected need the ICU? "
                    "Every patient at or above the review threshold is assumed to request an ICU bed (precautionary demand), "
                    "and beds free up according to real MIMIC-IV ICU stay lengths.")
        a, b, c = st.columns(3)
        beds = a.number_input("Total ICU beds", 1, 500, 10)
        out_of_service = a.number_input("Beds out of service (staff / equipment)", 0, int(beds) - 1, 0)
        occupied = b.number_input("Currently occupied beds", 0, int(beds), min(int(beds), 10))
        max_wait = b.number_input("Maximum acceptable wait (hours)", 0, 72, 12)
        top_surge = c.slider("Highest demand surge to test (x)", 2, 5, 3)
        st.caption(f"Uses {len(real_stays):,} observed ICU stays (median {np.median(real_stays) / 24:.1f} days, "
                   f"mean {real_stays.mean() / 24:.1f} days).")
        if st.button("Run crisis stress test", type="primary"):
            eff = int(beds) - int(out_of_service)
            base = ranked.risk.to_numpy(float)
            res, rand = {}, {}
            for k in range(1, int(top_surge) + 1):
                rk = replicate(base, k)
                need = (rk >= threshold).astype(float)                                  # precautionary: every flagged patient requests a bed
                res[k] = simulate_priority(rk, eff, min(int(occupied), eff), 3.0, int(max_wait), 300,
                                           need_prob=need, stay_hours=real_stays)
                perm = np.random.default_rng(7).permutation(len(rk))                    # same patients, random priority order
                rand[k] = simulate_priority(rk[perm], eff, min(int(occupied), eff), 3.0, int(max_wait), 300,
                                            need_prob=need[perm], stay_hours=real_stays)
            used = (f"{int(beds)} beds ({int(out_of_service)} out of service), {int(occupied)} occupied, "
                    f"{int(max_wait)} h wait limit, real ICU stay lengths (n={len(real_stays):,}), "
                    "precautionary demand (all flagged patients request a bed)")
            st.session_state["sim"] = dict(res=res, eff=eff, used=used, rand=rand)
        sim = st.session_state.get("sim")
        if sim and not (all(k in sim for k in ("res", "eff", "used", "rand"))
                        and all("unmet_risk" in v for v in sim["res"].values())):
            st.session_state["sim"] = sim = None              # left over from an older version of the app
        if sim:
            res, eff = sim["res"], sim["eff"]
            st.markdown(f"Usable beds: **{eff}** · cohort size: **{len(ranked)}** patients per 1x")
            st.caption("Showing results from the last run: " + sim.get("used", "settings not recorded") + ". Change a setting, then press Run again.")
            cols = st.columns(len(res))
            for col, (k, s) in zip(cols, res.items()):
                lvl, colr = crisis_level(float((s["unmet"] > 0).mean()))
                col.markdown(f"<div class='lvl' style='background:{colr}'><span>{'Flagged cohort (1x)' if k == 1 else f'Surge {k}x'}</span><br>"
                             f"<b>{lvl}</b><br><span>{(s['unmet'] > 0).mean():.0%} chance of unmet request<br>"
                             f"{s['need_chance'].sum():.1f} requests · wait {s['avg_wait_hours']:.1f} h</span></div>", unsafe_allow_html=True)
            ks = [("1x (current)" if k == 1 else f"{k}x surge") for k in res]
            fig = go.Figure([go.Bar(name="Admitted immediately", x=ks, y=[s["immediate"].mean() for s in res.values()], marker_color=TEAL),
                             go.Bar(name="Admitted after waiting", x=ks, y=[s["delayed"].mean() for s in res.values()], marker_color=LIGHT),
                             go.Bar(name="Unmet (waited too long)", x=ks, y=[s["unmet"].mean() for s in res.values()], marker_color=RED)])
            fig.update_layout(barmode="stack", template="plotly_white", height=340, yaxis_title="Average ICU requests",
                              title="What happens to ICU requests as demand rises")
            st.plotly_chart(fig)
            st.markdown("##### Risk-weighted view: is missing a high-risk patient worse?")
            rows = []
            for k in res:
                w, wr = weighted_summary(res[k]), weighted_summary(sim["rand"][k])
                rows.append({"Scenario": "1x (current)" if k == 1 else f"{k}x surge", "Unmet requests": w["unmet"],
                             "% of requests unmet": w["pct_requests_unmet"], "Risk-weighted unmet score": w["weighted_unmet"],
                             "% of predicted risk unserved (risk priority)": w["pct_risk_unserved"],
                             "% of predicted risk unserved (random order)": wr["pct_risk_unserved"]})
            rw = pd.DataFrame(rows)
            fig3 = go.Figure([go.Bar(name="% of requests unmet", x=rw["Scenario"], y=rw["% of requests unmet"], marker_color=LIGHT),
                              go.Bar(name="% of predicted risk unserved (risk priority)", x=rw["Scenario"],
                                     y=rw["% of predicted risk unserved (risk priority)"], marker_color=TEAL),
                              go.Bar(name="% of predicted risk unserved (random order)", x=rw["Scenario"],
                                     y=rw["% of predicted risk unserved (random order)"], marker_color=RED)])
            fig3.update_layout(barmode="group", template="plotly_white", height=340, yaxis_title="%",
                               title="Each unmet request is weighted by the patient's predicted risk")
            st.plotly_chart(fig3)
            st.dataframe(rw.round(2), hide_index=True)
            st.caption("Weight of an unmet request = that patient's predicted ICU-transfer risk. 'Risk-weighted unmet score' = sum of those weights "
                       "(averaged over runs). If '% of predicted risk unserved' is lower than '% of requests unmet', the patients left without a bed "
                       "had lower-than-average risk, i.e. priority protected the higher-risk ones. Random order shows the same patients served "
                       "without using the risk ranking. Risk is the model's transfer probability, not clinical urgency.")
            s = res[max(res)]; n = len(s["need_chance"]); q = max(1, n // 4)
            groups = {"Highest-risk 25%": slice(0, q), "Middle 50%": slice(q, n - q), "Lowest-risk 25%": slice(n - q, n)}
            served = [float(s["admit_chance"][g].sum() / max(s["need_chance"][g].sum(), 1e-9)) * 100 for g in groups.values()]
            fig2 = go.Figure(go.Bar(x=list(groups), y=served, marker_color=[NAVY, TEAL, LIGHT]))
            fig2.update_layout(template="plotly_white", height=300, yaxis_title="% of requests served in time", yaxis_range=[0, 100],
                               title=f"Under the {max(res)}x surge, highest predicted risk is served first")
            st.plotly_chart(fig2)
            worst = crisis_level(float((s["unmet"] > 0).mean()))[0]
            advice = {"GREEN": "Capacity looks adequate in this scenario; keep monitoring flagged patients.",
                      "AMBER": "Some requests may go unmet; review flagged patients early and prepare surge / step-down options.",
                      "RED": "Unmet requests are likely; escalate to a capacity-management review and protect beds for the highest-risk patients."}[worst]
            st.info(f"**Discussion point ({worst} at {max(res)}x surge):** {advice}")
            st.caption("Hypothetical assumptions: all patients admitted at hour 0; every patient at or above the review threshold requests a bed "
                       "(an upper bound: only about 1 in 16 flagged patients was actually transferred in testing); requests come at a random hour in the "
                       "model's 24-72 h window; beds free up using real MIMIC-IV ICU stay lengths (new patients draw a full stay, current patients a "
                       "remaining stay); beds go to the highest predicted risk first; a wait above the limit is unmet; surge = more patients with the "
                       "same risk profile. Alert levels (Green < 10%, Amber < 40%, Red >= 40% chance of any unmet request) are illustrative, not "
                       "validated. Not clinical guidance.")
