# Smart Campus Analytics - AI-Powered Student Success Platform

Everything lives in ONE file: `student_success_app.py` (data, scoring, dashboard, LLM advisor, batch CLI).

## Run
```
pip install -r requirements.txt
streamlit run student_success_app.py
```
Batch LLM plans: `python student_success_app.py --batch --level High --limit 10` (add `--dry-run` for no API key).
Deploy: push this folder to GitHub -> Streamlit Community Cloud -> main file `student_success_app.py`.
API key: `GROQ_API_KEY` via env var, `.env` (copy `.env.example`), Streamlit Secrets, or the sidebar box. Never commit it.

## Student Success Score (0-100)
Each indicator is min-max normalised to 0-1 (backlogs inverted), averaged inside its pillar,
then pillars are combined with weights (adjustable live in the sidebar).

| Pillar (weight) | Indicators (share inside pillar) |
|---|---|
| Academic (30) | CGPA 45%, internal marks 35%, backlogs 20% |
| Attendance (15) | Overall attendance 100% |
| LMS (12) | Assignment completion 60%, logins/week 40% |
| Engagement (8) | Hackathons 30%, certifications 30%, events 25%, clubs 15% |
| Placement (22) | Coding 40%, aptitude 30%, mock interview 30% |
| Skills (8) | Technical 60%, soft 40% |
| Feedback (5) | Student satisfaction 50%, faculty feedback 50% |

`Score = sum(normalised indicator x indicator share x pillar weight) x 100`

Two sub-indices support the risk logic and segmentation:
- **Academic Index** = Academic + Attendance + LMS pillars
- **Placement Index** = Placement + Skills + Engagement pillars

## Risk flags (rule-based, fully transparent)
- **Academic risk** if any: attendance < 70%, CGPA < 5.5, backlogs >= 3, assignment completion < 45%, Academic Index < 45
- **Placement risk** if any: Placement Index < 35, coding < 30, aptitude < 35, mock interview < 35
- **Risk level**: High = both flags or score < 40; Medium = one flag; Low = none
- Each flag lists the exact triggering reasons per student.

## Explainability
For every student, each indicator's contribution is compared with the cohort average;
the Student 360 view shows the 5 biggest drags and 5 biggest boosts (points vs cohort),
a pillar radar vs cohort, and interventions for the two weakest pillars.

## Segments (cut-offs at cohort medians)
Star Performers · Academic Strong, Placement Gap · Practical Achievers (placement-ready, academically weak) · Needs Holistic Support

## AI Advisor (LLM) - (in-file)
- **Personalised intervention plan:** the student's score, risk reasons and top drags/strengths vs cohort are sent to an LLM (Groq, `llama-3.3-70b-versatile` by default). It returns a summary, root causes, a 30-day action plan and a supportive message for the student.
- **Ask the data:** chat over an aggregated cohort summary (departments, years, segments, drivers, lowest scorers). Answers are limited to that summary.
- **Guardrails:** the prompt forbids inventing facts; only student ID and numeric indicators are sent (no names/contacts); falls back to a rule-based plan if there is no key or the call fails.
- **Setup:** set `GROQ_API_KEY` (env var, `.streamlit/secrets.toml`, or the sidebar box). On Streamlit Cloud add it under App settings -> Secrets: `GROQ_API_KEY = "gsk_..."`.

## API key safety
- Never hard-code the key or commit it. `.gitignore` already excludes `.env` and `.streamlit/secrets.toml`; only `.env.example` (placeholder) goes to GitHub.
- On Streamlit Cloud, put the key in App settings -> Secrets.
- If a key is ever pushed to GitHub, revoke it in the Groq console and create a new one.

## Suggested demo flow (3 min)
1. Overview: KPIs, score distribution, risk by department, attendance trend by risk level
2. Watchlist: filter High risk, download CSV
3. Student 360: pick a high-risk student, show "why" chart + interventions
4. Segments: academic vs placement scatter, segment actions
5. Insights + what-if weight sliders
6. AI Advisor: generate a plan for the high-risk student, then ask which department needs attention first
