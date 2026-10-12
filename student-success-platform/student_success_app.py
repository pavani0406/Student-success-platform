"""
Smart Campus Analytics - AI-Powered Student Success Platform  (ONE FILE)

Contains: synthetic data + Student Success Score + risk flags + segments,
          Streamlit dashboard, Groq LLM advisor, and a batch-plan CLI.

Install:   pip install streamlit pandas numpy plotly groq
Dashboard: streamlit run student_success_app.py
Batch:     python student_success_app.py --batch --level High --limit 10
           python student_success_app.py --batch --dry-run          (no API key)

API key:   set GROQ_API_KEY (env var, .env file, .streamlit/secrets.toml, or sidebar box).
           Never commit the key to GitHub.
"""
import argparse
import functools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. INDICATOR / SCORE DEFINITION
#    key: (label, pillar, share inside pillar, min, max, inverted?)
# ---------------------------------------------------------------------------
IND = {
    "cgpa":                  ("CGPA", "Academic", 0.45, 0, 10, False),
    "internal_marks":        ("Internal marks %", "Academic", 0.35, 0, 100, False),
    "backlogs":              ("Backlogs", "Academic", 0.20, 0, 6, True),
    "attendance":            ("Attendance %", "Attendance", 1.00, 0, 100, False),
    "lms_logins_per_week":   ("LMS logins / week", "LMS", 0.40, 0, 15, False),
    "assignment_completion": ("Assignment completion %", "LMS", 0.60, 0, 100, False),
    "events":                ("Events attended", "Engagement", 0.25, 0, 12, False),
    "clubs":                 ("Clubs joined", "Engagement", 0.15, 0, 4, False),
    "hackathons":            ("Hackathons", "Engagement", 0.30, 0, 5, False),
    "certifications":        ("Certifications", "Engagement", 0.30, 0, 8, False),
    "aptitude":              ("Aptitude score", "Placement", 0.30, 0, 100, False),
    "coding":                ("Coding score", "Placement", 0.40, 0, 100, False),
    "mock_interview":        ("Mock interview score", "Placement", 0.30, 0, 100, False),
    "tech_skill":            ("Technical skill", "Skills", 0.60, 0, 100, False),
    "soft_skill":            ("Soft skill", "Skills", 0.40, 0, 100, False),
    "student_satisfaction":  ("Student satisfaction (1-5)", "Feedback", 0.50, 1, 5, False),
    "faculty_feedback":      ("Faculty feedback (1-5)", "Feedback", 0.50, 1, 5, False),
}
PILLARS = ["Academic", "Attendance", "LMS", "Engagement", "Placement", "Skills", "Feedback"]
DEFAULT_W = {"Academic": 30, "Attendance": 15, "LMS": 12, "Engagement": 8,
             "Placement": 22, "Skills": 8, "Feedback": 5}
ACAD_P = ["Academic", "Attendance", "LMS"]
PLACE_P = ["Placement", "Skills", "Engagement"]

PILLAR_ACTIONS = {
    "Academic": "Remedial classes / peer tutoring and a backlog-clearing plan",
    "Attendance": "Mentor meeting + attendance recovery plan; alert guardian if below 65%",
    "LMS": "Nudge on pending assignments; short LMS onboarding and weekly check-ins",
    "Engagement": "Invite to a club, hackathon team or technical event",
    "Placement": "Coding/aptitude drills, mock-interview slots, placement-cell mentoring",
    "Skills": "Skill certification track and soft-skill / communication workshop",
    "Feedback": "Counselling check-in; collect qualitative feedback on teaching and campus life",
}
SEG_ACTIONS = {
    "Star Performers": "Fast-track: stretch projects, top-tier recruiter exposure, peer-mentor roles.",
    "Academic Strong, Placement Gap": "Targeted placement prep: coding bootcamp, aptitude drills, mock interviews.",
    "Practical Achievers": "Academic recovery: tutoring and backlog plan; leverage their practical strengths.",
    "Needs Holistic Support": "Mentor assignment, early counselling, attendance + academic + skills plan.",
}
SEG_ORDER = list(SEG_ACTIONS)


# ---------------------------------------------------------------------------
# 2. SYNTHETIC DATA (stand-in for the integrated college data sources)
# ---------------------------------------------------------------------------
@functools.cache
def make_data(n=600, seed=7):
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)
    eng = rng.normal(0, 1, n)
    prep = 0.25 * ability + rng.normal(0, 1, n)  # placement prep only loosely tied to grades
    z = lambda s=1: rng.normal(0, s, n)
    clip = np.clip
    df = pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "department": rng.choice(["CSE", "ECE", "EEE", "MECH", "CIVIL", "IT"], n, p=[.28, .17, .12, .13, .12, .18]),
        "year": rng.choice([1, 2, 3, 4], n),
        "cgpa": clip(6.8 + 0.9 * ability + 0.3 * eng + z(.4), 4, 10),
        "internal_marks": clip(62 + 10 * ability + 3 * eng + z(6), 20, 100),
        "backlogs": clip(rng.poisson(np.maximum(0.05, 1.1 - 0.8 * ability)), 0, 6),
        "attendance": clip(78 + 8 * eng + 3 * ability + z(7), 40, 100),
        "lms_logins_per_week": clip(rng.poisson(np.maximum(0.5, 5 + 3 * eng)), 0, 15),
        "assignment_completion": clip(70 + 14 * eng + 4 * ability + z(10), 0, 100),
        "events": clip(rng.poisson(np.maximum(0.2, 3 + 2 * eng)), 0, 12),
        "clubs": clip(rng.poisson(np.maximum(0.1, 1 + 0.5 * eng)), 0, 4),
        "hackathons": clip(rng.poisson(np.maximum(0.1, 1 + 0.8 * eng + 0.3 * prep)), 0, 5),
        "certifications": clip(rng.poisson(np.maximum(0.1, 2 + 1.2 * prep + 0.3 * eng)), 0, 8),
        "aptitude": clip(55 + 8 * ability + 12 * prep + z(8), 10, 100),
        "coding": clip(48 + 5 * ability + 16 * prep + z(9), 5, 100),
        "mock_interview": clip(52 + 4 * ability + 10 * prep + 4 * eng + z(8), 10, 100),
        "tech_skill": clip(55 + 5 * ability + 14 * prep + z(8), 10, 100),
        "soft_skill": clip(58 + 8 * eng + 6 * prep + z(8), 10, 100),
        "student_satisfaction": clip(3.5 + 0.4 * eng + z(.5), 1, 5),
        "faculty_feedback": clip(3.4 + 0.4 * ability + 0.3 * eng + z(.5), 1, 5),
    }).round(2)
    # monthly trend table (attendance + LMS) used for the "key trends" charts
    months = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
    slope = rng.normal(0.4 * eng, 1.4)
    rows = []
    for m, mon in enumerate(months):
        rows.append(pd.DataFrame({
            "student_id": df.student_id, "month": mon,
            "attendance": clip(df.attendance + slope * (m - 2.5) + rng.normal(0, 2, n), 30, 100),
            "lms_logins": clip(df.lms_logins_per_week + 0.3 * slope * (m - 2.5) + rng.normal(0, 1, n), 0, 20),
        }))
    return df, pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. SCORING, RISK, SEGMENTS, EXPLAINABILITY
# ---------------------------------------------------------------------------
def compute(df, pillar_w):
    w = pd.Series(pillar_w, dtype=float)
    w = (w / w.sum()) if w.sum() > 0 else pd.Series(DEFAULT_W, dtype=float) / 100
    norm = pd.DataFrame(index=df.index)
    for k, (_, _, _, lo, hi, inv) in IND.items():
        x = ((df[k] - lo) / (hi - lo)).clip(0, 1)
        norm[k] = 1 - x if inv else x
    pillars = pd.DataFrame({p: sum(IND[k][2] * norm[k] for k in IND if IND[k][1] == p) * 100
                            for p in PILLARS})
    contrib = pd.DataFrame({k: norm[k] * IND[k][2] * w[IND[k][1]] * 100 for k in IND})
    gap = contrib - contrib.mean()                      # + helps vs cohort, - drags vs cohort

    def idx(names):
        ww = w[names]
        return (pillars[names] * ww).sum(axis=1) / max(ww.sum(), 1e-9)

    out = df.copy()
    out["success_score"] = (pillars * w[PILLARS]).sum(axis=1).round(1)
    out["academic_index"] = idx(ACAD_P).round(1)
    out["placement_index"] = idx(PLACE_P).round(1)
    for p in PILLARS:
        out[f"P_{p}"] = pillars[p].round(1)

    a_reasons, p_reasons = [], []
    for r in out.itertuples():
        a, p = [], []
        if r.attendance < 70: a.append(f"attendance {r.attendance:.0f}%")
        if r.cgpa < 5.5: a.append(f"CGPA {r.cgpa:.1f}")
        if r.backlogs >= 3: a.append(f"{int(r.backlogs)} backlogs")
        if r.assignment_completion < 45: a.append(f"assignments {r.assignment_completion:.0f}%")
        if r.academic_index < 45: a.append(f"academic index {r.academic_index:.0f}")
        if r.placement_index < 35: p.append(f"placement index {r.placement_index:.0f}")
        if r.coding < 30: p.append(f"coding {r.coding:.0f}")
        if r.aptitude < 35: p.append(f"aptitude {r.aptitude:.0f}")
        if r.mock_interview < 35: p.append(f"mock interview {r.mock_interview:.0f}")
        a_reasons.append("; ".join(a)); p_reasons.append("; ".join(p))
    out["academic_risk"] = [bool(x) for x in a_reasons]
    out["placement_risk"] = [bool(x) for x in p_reasons]
    out["academic_risk_reasons"], out["placement_risk_reasons"] = a_reasons, p_reasons
    both = out.academic_risk & out.placement_risk
    either = out.academic_risk | out.placement_risk
    out["risk_level"] = np.where(both | (out.success_score < 40), "High",
                                 np.where(either, "Medium", "Low"))

    A_CUT, P_CUT = out.academic_index.median(), out.placement_index.median()
    out.attrs["cuts"] = (float(A_CUT), float(P_CUT))
    hi_a, hi_p = out.academic_index >= A_CUT, out.placement_index >= P_CUT
    out["segment"] = np.select(
        [hi_a & hi_p, hi_a & ~hi_p, ~hi_a & hi_p],
        ["Star Performers", "Academic Strong, Placement Gap", "Practical Achievers"],
        default="Needs Holistic Support")
    return out, gap, pillars


def pillar_gap(gap):
    return pd.DataFrame({p: gap[[k for k in IND if IND[k][1] == p]].sum(axis=1) for p in PILLARS})


# ---------------------------------------------------------------------------
# LLM ADVISOR (Groq)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "llama-3.3-70b-versatile"   # change in the sidebar if Groq retires it

PLAN_SYSTEM = """You are an academic success advisor at an engineering college.
You receive structured facts about ONE student (scores, risk flags, and which indicators
help or hurt their Success Score versus the cohort).
Rules:
- Use ONLY the facts provided. Never invent grades, events or personal circumstances.
- Be specific, practical and kind. No diagnosis, no blame.
Return markdown with exactly these sections:
### Summary (2 sentences)
### Root causes (max 3 bullets, each tied to a number from the facts)
### 30-day action plan (4-5 bullets: action - owner (faculty / mentor / placement cell / student) - week)
### Message to the student (3-4 supportive sentences, encouraging, no jargon)"""

QA_SYSTEM = """You are a data analyst assistant for college administrators.
Answer ONLY from the cohort summary provided. If the answer is not in the summary, say so
and suggest which dashboard tab or filter could answer it. Be concise, use numbers, and end
with one recommended action when relevant."""


def get_client(api_key):
    """Return a Groq client, or None if no key / package is available."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except Exception:
        return None


def _chat(client, model, system, user, temperature=0.3, history=None):
    msgs = [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": user}]
    resp = client.chat.completions.create(model=model, messages=msgs,
                                          temperature=temperature, max_tokens=900)
    return resp.choices[0].message.content.strip()


# ----------------------------- student plans -------------------------------
def student_facts(row, gap_row, pillar_gap_row, ind_labels):
    """Compact fact sheet for one student (row = scored DataFrame row)."""
    g = gap_row.rename(index=ind_labels).sort_values()
    return (
        f"Student {row['student_id']} | {row['department']} year {int(row['year'])}\n"
        f"Success Score {row['success_score']:.1f}/100 | risk level {row['risk_level']} | segment {row['segment']}\n"
        f"Academic index {row['academic_index']:.0f}, placement index {row['placement_index']:.0f}\n"
        f"Key numbers: CGPA {row['cgpa']:.2f}, backlogs {int(row['backlogs'])}, attendance {row['attendance']:.0f}%, "
        f"assignments {row['assignment_completion']:.0f}%, LMS logins/wk {row['lms_logins_per_week']:.0f}, "
        f"coding {row['coding']:.0f}, aptitude {row['aptitude']:.0f}, mock interview {row['mock_interview']:.0f}, "
        f"certifications {int(row['certifications'])}, hackathons {int(row['hackathons'])}\n"
        f"Academic risk reasons: {row['academic_risk_reasons'] or 'none'}\n"
        f"Placement risk reasons: {row['placement_risk_reasons'] or 'none'}\n"
        "Biggest drags vs cohort (score points): "
        + ", ".join(f"{k} {v:+.1f}" for k, v in g.head(4).items()) + "\n"
        "Biggest strengths vs cohort: "
        + ", ".join(f"{k} {v:+.1f}" for k, v in g.tail(3)[::-1].items()) + "\n"
        "Weakest pillars vs cohort: "
        + ", ".join(f"{k} {v:+.1f}" for k, v in pillar_gap_row.sort_values().head(2).items())
    )


def fallback_plan(row, pillar_gap_row, pillar_actions):
    weakest = pillar_gap_row.sort_values().head(2).index
    lines = [f"### Summary\n{row['student_id']} has a Success Score of {row['success_score']:.1f} "
             f"({row['risk_level']} risk, segment: {row['segment']}).",
             "### Root causes"]
    if row["academic_risk_reasons"]:
        lines.append(f"- Academic: {row['academic_risk_reasons']}")
    if row["placement_risk_reasons"]:
        lines.append(f"- Placement: {row['placement_risk_reasons']}")
    lines.append("### 30-day action plan")
    for i, p in enumerate(weakest, 1):
        lines.append(f"- {pillar_actions[p]} - mentor - week {i}")
    lines.append("\n*(Rule-based plan - add a Groq API key in the sidebar for an AI-written plan.)*")
    return "\n".join(lines)


def intervention_plan(client, model, facts):
    return _chat(client, model, PLAN_SYSTEM, facts, temperature=0.3)


# ----------------------------- cohort Q&A ----------------------------------
def cohort_summary(df, gap, ind_labels, max_watch=15):
    """Aggregated, privacy-light cohort text the LLM may answer from."""
    parts = [
        f"Students: {len(df)}. Avg Success Score {df.success_score.mean():.1f}. "
        f"Risk levels: {df.risk_level.value_counts().to_dict()}. "
        f"Academic risk {df.academic_risk.mean():.0%}, placement risk {df.placement_risk.mean():.0%}.",
        "By department (avg score | % high risk): " + "; ".join(
            f"{n}: {x.success_score.mean():.1f} | {(x.risk_level == 'High').mean():.0%}"
            for n, x in df.groupby("department")),
        "By year (avg score | % high risk): " + "; ".join(
            f"Y{n}: {x.success_score.mean():.1f} | {(x.risk_level == 'High').mean():.0%}"
            for n, x in df.groupby("year")),
        f"Segments: {df.segment.value_counts().to_dict()}",
        "Avg indicators: " + ", ".join(
            f"{lab} {df[k].mean():.1f}" for k, lab in ind_labels.items()),
        "Biggest drags for HIGH-risk students vs cohort (points): " + ", ".join(
            f"{ind_labels[k]} {v:+.1f}" for k, v in gap[df.risk_level == "High"].mean().sort_values().head(5).items()),
        f"Attendance < 75%: {(df.attendance < 75).sum()} students (avg score "
        f"{df[df.attendance < 75].success_score.mean():.1f} vs {df[df.attendance >= 75].success_score.mean():.1f}).",
        f"Backlogs >= 2: {(df.backlogs >= 2).sum()} students.",
        "Lowest-scoring students: " + "; ".join(
            f"{r.student_id} ({r.department}, Y{r.year}, {r.success_score:.0f}, {r.academic_risk_reasons or r.placement_risk_reasons})"
            for r in df.nsmallest(max_watch, "success_score").itertuples()),
    ]
    return "\n\n".join(parts)


def ask_cohort(client, model, question, summary, history=None):
    user = f"COHORT SUMMARY:\n{summary}\n\nQUESTION: {question}"
    return _chat(client, model, QA_SYSTEM, user, temperature=0.2, history=history)


# ---------------------------------------------------------------------------
# BATCH MODE (python student_success_app.py --batch ...)
# ---------------------------------------------------------------------------
def load_env(path=".env"):
    """Tiny .env reader (KEY=value lines) so no extra package is needed."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def call_with_retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                      # rate limit / network
            if i == tries - 1:
                raise
            wait = 2 ** (i + 1)
            print(f"  retrying in {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)


def batch_main():
    ap = argparse.ArgumentParser(description="Batch LLM intervention plans")
    ap.add_argument("--batch", action="store_true", help="run batch mode")
    ap.add_argument("--csv", help="student CSV (same columns as the template); default = built-in sample data")
    ap.add_argument("--level", default="High", choices=["High", "Medium", "Low"], help="risk level to plan for")
    ap.add_argument("--limit", type=int, default=10, help="max students (lowest scores first)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="plans", help="output file prefix -> <prefix>.md and <prefix>.csv")
    ap.add_argument("--dry-run", action="store_true", help="skip the LLM; write rule-based plans")
    a = ap.parse_args()

    load_env()
    client = None
    if not a.dry_run:
        client = get_client(os.getenv("GROQ_API_KEY"))
        if client is None:
            sys.exit("No usable GROQ_API_KEY (or `groq` not installed). Put it in .env / the environment, "
                     "run `pip install groq`, or use --dry-run.")

    raw = pd.read_csv(a.csv) if a.csv else make_data()[0]
    data, gap, _ = compute(raw, DEFAULT_W)
    pg = pillar_gap(gap)
    labels = {k: IND[k][0] for k in IND}

    todo = data[data.risk_level == a.level].nsmallest(a.limit, "success_score")
    print(f"{len(todo)} students at {a.level} risk -> {'rule-based' if client is None else a.model}")

    rows = []
    for n, (i, r) in enumerate(todo.iterrows(), 1):
        print(f"[{n}/{len(todo)}] {r.student_id} (score {r.success_score:.1f})")
        if client is None:
            plan = fallback_plan(r, pg.loc[i], PILLAR_ACTIONS)
        else:
            facts = student_facts(r, gap.loc[i], pg.loc[i], labels)
            plan = call_with_retry(lambda: intervention_plan(client, a.model, facts))
            time.sleep(0.5)                          # be gentle with rate limits
        rows.append({"student_id": r.student_id, "department": r.department, "year": r.year,
                     "success_score": r.success_score, "risk_level": r.risk_level, "plan": plan})

    out = pd.DataFrame(rows)
    out.to_csv(f"{a.out}.csv", index=False)
    with open(f"{a.out}.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Intervention plans ({a.level} risk)\n\n")
        for r in rows:
            fh.write(f"## {r['student_id']} - {r['department']} Y{r['year']} - score {r['success_score']:.1f}\n\n{r['plan']}\n\n---\n\n")
    print(f"Saved {a.out}.md and {a.out}.csv")


def run_dashboard():
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="Smart Campus Analytics", page_icon="🎓", layout="wide")
    make_data_cached = st.cache_data(make_data.__wrapped__)

    # ---------------------------------------------------------------------------
    # 4. SIDEBAR: data source, filters, what-if weights
    # ---------------------------------------------------------------------------
    st.sidebar.title("🎓 Smart Campus Analytics")
    raw, trend = make_data_cached()
    st.sidebar.download_button("Download sample/template CSV", raw.to_csv(index=False), "student_data_template.csv")
    up = st.sidebar.file_uploader("Upload your own student CSV (same columns)", type="csv")
    if up is not None:
        try:
            up_df = pd.read_csv(up)
            missing = [c for c in ["student_id", "department", "year", *IND] if c not in up_df.columns]
            if missing:
                st.sidebar.error(f"Missing columns: {missing}")
            else:
                raw, trend = up_df, None
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")

    with st.sidebar.expander("⚖️ Score weights (what-if)", expanded=False):
        weights = {p: st.slider(p, 0, 50, DEFAULT_W[p], key=f"w_{p}") for p in PILLARS}
        st.caption(f"Weights are re-normalised to 100 (current sum: {sum(weights.values())}).")

    with st.sidebar.expander("🤖 AI advisor (LLM)", expanded=False):
        try:
            secret_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            secret_key = ""
        api_key = st.text_input("Groq API key", value=secret_key, type="password",
                                help="Or set GROQ_API_KEY in env / .streamlit/secrets.toml")
        model = st.text_input("Model", value=DEFAULT_MODEL)
    client = get_client(api_key)

    data, gap, pillars = compute(raw, weights)
    pgap = pillar_gap(gap)

    st.sidebar.markdown("### Filters")
    depts = st.sidebar.multiselect("Department", sorted(data.department.unique()), default=sorted(data.department.unique()))
    years = st.sidebar.multiselect("Year", sorted(data.year.unique()), default=sorted(data.year.unique()))
    levels = st.sidebar.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    segs = st.sidebar.multiselect("Segment", SEG_ORDER, default=SEG_ORDER)
    mask = (data.department.isin(depts) & data.year.isin(years) & data.risk_level.isin(levels) & data.segment.isin(segs))
    f = data[mask]

    st.title("AI-Powered Student Success Platform")
    st.caption("Predict · Optimize · Improve student success — unified view of academic, attendance, LMS, engagement, placement, skills and feedback data.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🚨 Risk watchlist", "🧑‍🎓 Student 360°", "🧩 Segments", "💡 Insights", "🤖 AI Advisor"])

    # ------------------------------- OVERVIEW ----------------------------------
    with tab1:
        if f.empty:
            st.warning("No students match the current filters.")
        else:
            c = st.columns(5)
            c[0].metric("Students", len(f))
            c[1].metric("Avg Success Score", f"{f.success_score.mean():.1f}")
            c[2].metric("High risk", f"{(f.risk_level == 'High').mean():.0%}")
            c[3].metric("Academic risk", f"{f.academic_risk.mean():.0%}")
            c[4].metric("Placement risk", f"{f.placement_risk.mean():.0%}")

            l, r = st.columns(2)
            fig = px.histogram(f, x="success_score", color="risk_level", nbins=30,
                               category_orders={"risk_level": ["High", "Medium", "Low"]},
                               color_discrete_map={"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"},
                               title="Success Score distribution")
            l.plotly_chart(fig, use_container_width=True)
            dep = f.groupby(["department", "risk_level"]).size().reset_index(name="students")
            fig = px.bar(dep, x="department", y="students", color="risk_level", barmode="stack",
                         category_orders={"risk_level": ["High", "Medium", "Low"]},
                         color_discrete_map={"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"},
                         title="Risk level by department")
            r.plotly_chart(fig, use_container_width=True)

            l, r = st.columns(2)
            pm = f[[f"P_{p}" for p in PILLARS]].mean()
            pm.index = PILLARS
            fig = px.bar(pm.reset_index(), x="index", y=0, title="Average pillar score (0-100)",
                         labels={"index": "", "0": "score"})
            fig.update_layout(yaxis_title="score")
            l.plotly_chart(fig, use_container_width=True)
            if trend is not None:
                t = trend[trend.student_id.isin(f.student_id)].merge(f[["student_id", "risk_level"]], on="student_id")
                t = t.groupby(["month", "risk_level"], sort=False)[["attendance", "lms_logins"]].mean().reset_index()
                fig = px.line(t, x="month", y="attendance", color="risk_level", markers=True,
                              category_orders={"risk_level": ["High", "Medium", "Low"]},
                              color_discrete_map={"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"},
                              title="Monthly attendance trend by risk level")
                r.plotly_chart(fig, use_container_width=True)
            else:
                r.info("Monthly trends are available for the built-in sample data only.")

    # ------------------------------- WATCHLIST ---------------------------------
    with tab2:
        st.subheader("Students needing intervention")
        wl = f[f.risk_level != "Low"].sort_values("success_score")
        show = wl[["student_id", "department", "year", "success_score", "risk_level", "segment",
                   "academic_risk_reasons", "placement_risk_reasons"]]
        st.dataframe(show, use_container_width=True, hide_index=True, height=480)
        st.download_button("⬇️ Download watchlist CSV", show.to_csv(index=False), "watchlist.csv")

    # ------------------------------- STUDENT 360 -------------------------------
    with tab3:
        if f.empty:
            st.warning("No students match the current filters.")
        else:
            sid = st.selectbox("Select student", f.sort_values("success_score").student_id)
            i = data.index[data.student_id == sid][0]
            s = data.loc[i]
            c = st.columns(4)
            c[0].metric("Success Score", f"{s.success_score:.1f}", f"{s.success_score - data.success_score.mean():+.1f} vs cohort")
            c[1].metric("Risk level", s.risk_level)
            c[2].metric("Academic / Placement idx", f"{s.academic_index:.0f} / {s.placement_index:.0f}")
            c[3].metric("Segment", s.segment)

            l, r = st.columns(2)
            cohort = [data[f"P_{p}"].mean() for p in PILLARS]
            mine = [s[f"P_{p}"] for p in PILLARS]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=mine + mine[:1], theta=PILLARS + PILLARS[:1], fill="toself", name=sid))
            fig.add_trace(go.Scatterpolar(r=cohort + cohort[:1], theta=PILLARS + PILLARS[:1], name="Cohort avg"))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), title="Pillar profile vs cohort")
            l.plotly_chart(fig, use_container_width=True)

            g = gap.loc[i].rename(index={k: IND[k][0] for k in IND}).sort_values()
            top = pd.concat([g.head(5), g.tail(5)])
            fig = px.bar(top, x=top.values, y=top.index, orientation="h",
                         color=top.values > 0, color_discrete_map={True: "#2ca02c", False: "#d62728"},
                         title="Why this score? Score points gained / lost vs cohort average",
                         labels={"x": "points vs cohort avg", "y": ""})
            fig.update_layout(showlegend=False)
            r.plotly_chart(fig, use_container_width=True)

            st.markdown("**Risk explanation**")
            if s.academic_risk: st.error(f"Academic risk: {s.academic_risk_reasons}")
            if s.placement_risk: st.warning(f"Placement risk: {s.placement_risk_reasons}")
            if not (s.academic_risk or s.placement_risk): st.success("No risk flags triggered.")

            st.markdown("**Suggested interventions**")
            weakest = pgap.loc[i].sort_values().head(2).index
            for p in weakest:
                st.write(f"- **{p}** ({pgap.loc[i, p]:+.1f} pts vs cohort): {PILLAR_ACTIONS[p]}")
            st.write(f"- **Segment plan:** {SEG_ACTIONS[s.segment]}")

    # ------------------------------- SEGMENTS ----------------------------------
    with tab4:
        if f.empty:
            st.warning("No students match the current filters.")
        else:
            fig = px.scatter(f, x="academic_index", y="placement_index", color="segment", hover_data=["student_id", "success_score"],
                             category_orders={"segment": SEG_ORDER}, title="Academic strength vs placement readiness")
            a_cut, p_cut = data.attrs["cuts"]
            fig.add_vline(x=a_cut, line_dash="dot"); fig.add_hline(y=p_cut, line_dash="dot")
            st.plotly_chart(fig, use_container_width=True)
            prof = f.groupby("segment").agg(students=("student_id", "count"), avg_score=("success_score", "mean"),
                                            cgpa=("cgpa", "mean"), attendance=("attendance", "mean"),
                                            coding=("coding", "mean"), aptitude=("aptitude", "mean")).round(1).reindex(SEG_ORDER).dropna()
            prof["recommended_action"] = [SEG_ACTIONS[s] for s in prof.index]
            st.dataframe(prof, use_container_width=True)

    # ------------------------------- INSIGHTS ----------------------------------
    with tab5:
        st.subheader("Auto-generated insights")
        d = data
        low_att = d[d.attendance < 75]
        gapseg = d[(d.cgpa >= 7.5) & (d.placement_index < d.placement_index.median())]
        hi = d.risk_level == "High"
        drivers = gap[hi].mean().sort_values().head(3)
        worst_dept = d.groupby("department").apply(lambda x: (x.risk_level == "High").mean()).sort_values(ascending=False)
        ins = [
            f"**{len(low_att)} students ({len(low_att)/len(d):.0%})** have attendance below 75%; their average Success Score is "
            f"**{low_att.success_score.mean():.1f}** vs **{d[d.attendance >= 75].success_score.mean():.1f}** for the rest.",
            f"**{len(gapseg)} students** have CGPA ≥ 7.5 but below-median placement readiness — good grades, weak placement readiness. "
            f"Top candidates for coding/aptitude bootcamps.",
            f"**{worst_dept.index[0]}** has the highest share of high-risk students ({worst_dept.iloc[0]:.0%}).",
            "Biggest drags for high-risk students (vs cohort): " + ", ".join(f"**{IND[k][0]}** ({v:.1f} pts)" for k, v in drivers.items()) + ".",
            f"Assignment completion correlates with CGPA at **r = {d.assignment_completion.corr(d.cgpa):.2f}**; "
            f"coding score correlates with CGPA at only **r = {d.coding.corr(d.cgpa):.2f}** — grades alone do not predict placement readiness.",
        ]
        for t in ins:
            st.markdown("- " + t)
        st.subheader("How the Success Score works")
        wdf = pd.DataFrame([(k, IND[k][0], IND[k][1], IND[k][2] * weights[IND[k][1]] / max(sum(weights.values()), 1) * 100)
                            for k in IND], columns=["key", "Indicator", "Pillar", "Effective weight (%)"]).drop(columns="key")
        st.dataframe(wdf.round(1), use_container_width=True, hide_index=True)
        st.caption("Score = Σ (normalised indicator × effective weight). Backlogs are inverted (fewer is better).")

    # ------------------------------- AI ADVISOR (LLM) --------------------------
    with tab6:
        labels = {k: IND[k][0] for k in IND}
        st.subheader("AI Advisor")
        if client is None:
            st.info("No Groq API key set - plans fall back to rule-based text. Add a key in the sidebar for LLM output.")

        st.markdown("#### 1. Personalised intervention plan")
        if f.empty:
            st.warning("No students match the current filters.")
        else:
            sid2 = st.selectbox("Student", f.sort_values("success_score").student_id, key="llm_student")
            j = data.index[data.student_id == sid2][0]
            facts = student_facts(data.loc[j], gap.loc[j], pgap.loc[j], labels)
            with st.expander("Facts sent to the LLM"):
                st.text(facts)
            if st.button("Generate plan", type="primary"):
                with st.spinner("Writing plan..."):
                    try:
                        if client:
                            st.session_state["plan"] = intervention_plan(client, model, facts)
                        else:
                            st.session_state["plan"] = fallback_plan(data.loc[j], pgap.loc[j], PILLAR_ACTIONS)
                    except Exception as e:
                        st.error(f"LLM call failed ({e}). Showing rule-based plan.")
                        st.session_state["plan"] = fallback_plan(data.loc[j], pgap.loc[j], PILLAR_ACTIONS)
                    st.session_state["plan_for"] = sid2
            if st.session_state.get("plan_for") == sid2:
                st.markdown(st.session_state["plan"])
                st.download_button("Download plan", st.session_state["plan"], f"plan_{sid2}.md")

        st.markdown("#### 2. Ask the data")
        st.caption("Answers come only from an aggregated summary of the current dataset (all students, ignoring filters).")
        st.session_state.setdefault("chat", [])
        for m in st.session_state["chat"]:
            st.chat_message(m["role"]).write(m["content"])
        q = st.chat_input("e.g. Which department needs attention first and why?")
        if q:
            st.chat_message("user").write(q)
            if client is None:
                ans = "Add a Groq API key in the sidebar to enable this. Meanwhile, see the Insights tab for auto-generated findings."
            else:
                try:
                    ans = ask_cohort(client, model, q, cohort_summary(data, gap, labels),
                                         history=st.session_state["chat"][-6:])
                except Exception as e:
                    ans = f"LLM call failed: {e}"
            st.chat_message("assistant").write(ans)
            st.session_state["chat"] += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if "--batch" in sys.argv:
    batch_main()
else:
    run_dashboard()
