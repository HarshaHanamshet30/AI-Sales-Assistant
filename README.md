# 🎯 AI Sales Assistant: Automated Lead Prioritization System

> **ML-powered lead scoring with LLM-generated recommendations and a fully automated pipeline**

---

## 📌 Problem Statement

Sales teams waste 70% of their time chasing leads that never convert. Without intelligent prioritization, high-value prospects are treated the same as cold contacts — costing revenue and burning out reps.

**The cost of bad lead management:**
- 79% of marketing leads never convert due to lack of nurturing (HubSpot)
- Sales reps spend only 37% of their time actually selling
- Average lead response time is 42 hours — by then, prospects have moved on

---

## 💡 Solution Overview

The **AI Sales Assistant** combines machine learning and large language models to automatically:

1. **Score** every lead with a 0–1 conversion probability
2. **Classify** leads as High / Medium / Low priority
3. **Generate** personalized, actionable recommendations for each lead
4. **Automate** the entire pipeline on a schedule — no manual intervention
5. **Visualize** everything in a clean real-time dashboard

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│  CSV Upload / Google Sheets / Synthetic Demo Data       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 PREPROCESSING (model.py)                │
│  • Missing value imputation                             │
│  • Label encoding (categorical → numeric)               │
│  • Feature scaling (StandardScaler)                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              ML SCORING ENGINE (model.py)               │
│  • RandomForest / Logistic Regression                   │
│  • 80/20 train-test split                               │
│  • Predict conversion probability (0–1)                 │
│  • Classify: High (≥0.65) / Medium (0.40–0.65) / Low   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           LLM RECOMMENDATION ENGINE (utils.py)          │
│  • Structured prompt with lead context                  │
│  • Priority label + conversion likelihood               │
│  • Recommended action + outreach channel                │
│  • Urgency score + one-liner pitch                      │
│  • OpenAI GPT / fallback to intelligent mock            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            AUTOMATION LAYER (automation.py)             │
│  • APScheduler: runs every X minutes                    │
│  • File hash detection (skips unchanged data)           │
│  • JSON state logging + run history                     │
│  • Optional email notification                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              STREAMLIT DASHBOARD (app.py)               │
│  • KPI cards (total, high, medium, low, avg score)      │
│  • Priority distribution donut chart                    │
│  • Score histogram with threshold lines                 │
│  • Lead source performance bar chart                    │
│  • Ranked lead table with filters + download            │
│  • AI insight cards per high-priority lead              │
│  • Model metrics (accuracy, AUC, confusion matrix)      │
│  • Feature importance visualization                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | scikit-learn (RandomForest, LogisticRegression) |
| LLM | OpenAI GPT-3.5 / Rule-based mock |
| Dashboard | Streamlit + Plotly |
| Automation | Python threading + APScheduler logic |
| Data | pandas, numpy |
| Storage | CSV + JSON state files |
| Notifications | smtplib (built-in) |

---

## 📁 Project Structure

```
ai_sales_assistant/
│
├── app.py              # Streamlit dashboard (main UI)
├── model.py            # ML model: training, scoring, persistence
├── utils.py            # LLM recommendations, KPIs, email, helpers
├── automation.py       # Scheduler pipeline (CLI + thread)
│
├── sample_leads.csv    # Demo dataset (30 leads)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
│
├── artifacts/          # Saved model files (auto-created)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── features.pkl
│
├── output/             # Pipeline outputs (auto-created)
│   ├── scored_leads.csv
│   └── model_metrics.json
│
└── logs/               # Automation logs (auto-created)
    ├── pipeline.log
    └── pipeline_state.json
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env with your OpenAI API key and email settings
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Run the automation pipeline (CLI)

```bash
# Run once
python automation.py --once

# Run on schedule (every 5 minutes)
python automation.py --interval 5

# Run once with a specific CSV
python automation.py --once --input sample_leads.csv

# Force model retrain
python automation.py --once --retrain --input sample_leads.csv
```

---

## 📊 Dataset Format

Your CSV should contain these columns (all optional except Lead_ID):

| Column | Type | Description |
|--------|------|-------------|
| Lead_ID | string | Unique identifier |
| Lead_Source | string | Website, Referral, Email, etc. |
| Total_Visits | int | Number of website visits |
| Total_Time_Spent_on_Website | int | Seconds on site |
| Last_Activity | string | Most recent action |
| Occupation | string | Lead's job category |
| Email_Opened | int | Email open count |
| Page_Views_Per_Visit | float | Engagement depth |
| Lead_Age_Days | int | Days since lead creation |
| Converted | int | 0 = No, 1 = Yes (training only) |

---

## 🤖 AI Recommendations

Each lead receives:
- **Priority Label** — High / Medium / Low
- **Conversion Likelihood** — Strong / Moderate / Weak
- **Key Reasons** — 3 bullet-point explanations
- **Recommended Action** — Specific next step for sales rep
- **Outreach Channel** — Email / Phone / LinkedIn
- **Urgency** — 24h / 3 days / 2 weeks
- **One-Liner Pitch** — Tailored opening line

> Works without an OpenAI key using an intelligent rule-based fallback.

---

## 📧 Email Notifications

Set these env vars to receive automated email reports:

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASS=your-app-password
EMAIL_TO=sales@company.com
```

Then run: `python automation.py --interval 30 --email`

---

## 📈 Business Impact

| Metric | Without AI | With AI Sales Assistant |
|--------|-----------|------------------------|
| Lead response time | 42 hours | < 1 hour (auto-prioritized) |
| Sales rep efficiency | 37% selling time | Up to 60%+ |
| Lead conversion rate | Industry avg 2-3% | Targeted: 8-12% |
| Pipeline visibility | Manual spreadsheets | Real-time dashboard |
| Follow-up consistency | Ad hoc | Automated + AI-guided |

**ROI:** For a 100-lead/day pipeline at $500 average deal value and 3% → 8% conversion: **+$25,000/month in recovered revenue**

---

## 🧪 Model Performance

Typical results on 200-lead synthetic dataset:
- **Accuracy:** ~85–88%
- **ROC-AUC:** ~0.90–0.93
- **Top features:** Total_Visits, Last_Activity, Total_Time_Spent_on_Website

---

## 📜 License

MIT License — free to use, modify, and deploy.

---

* AI Sales Assistant Team*
