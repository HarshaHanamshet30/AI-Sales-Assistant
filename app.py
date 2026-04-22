"""
app.py - Streamlit Dashboard for AI Sales Assistant
Clean, production-grade UI with KPIs, charts, and AI insights
"""

import os
import json
import time
import threading
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from model import (
    generate_sample_data, train_model, score_leads,
    save_model, load_model, model_exists
)
from utils import (
    generate_bulk_insights, compute_kpis, save_results,
    load_results, timestamp_now
)
from automation import run_pipeline, start_scheduler, load_state

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sales Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main { background: #0e1117; }

  .metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  .metric-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
  }
  .metric-card .label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #8892a4;
    font-weight: 500;
  }
  .high { color: #00d4aa; }
  .medium { color: #f59e0b; }
  .low { color: #6b7280; }
  .total { color: #60a5fa; }
  .score { color: #a78bfa; }

  .badge-high   { background:#0d3b31; color:#00d4aa; border:1px solid #00d4aa33; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .badge-medium { background:#3b2d0d; color:#f59e0b; border:1px solid #f59e0b33; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .badge-low    { background:#1a1f2e; color:#6b7280; border:1px solid #6b728033; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

  .insight-card {
    background: #1a1f2e;
    border-left: 3px solid #00d4aa;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 8px 0;
  }
  .insight-card h4 { margin: 0 0 8px 0; color: #e2e8f0; font-size: 0.9rem; }
  .insight-card p  { margin: 0; color: #94a3b8; font-size: 0.85rem; }

  .status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:6px; }
  .dot-active { background:#00d4aa; box-shadow:0 0 6px #00d4aa; animation: pulse 2s infinite; }
  .dot-idle   { background:#6b7280; }

  @keyframes pulse {
    0%,100% { opacity:1; }
    50% { opacity:0.4; }
  }

  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
  .stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0891b2);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
  }
  .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(0,212,170,0.4); }

  h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
for key, default in {
    "df_scored": None,
    "metrics": None,
    "ai_generated": False,
    "scheduler_running": False,
    "last_run": None,
    "run_count": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def priority_badge(p):
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(p, "badge-low")
    return f'<span class="{cls}">{p}</span>'


def score_bar(s):
    pct = int(float(s) * 100)
    color = "#00d4aa" if pct >= 65 else "#f59e0b" if pct >= 40 else "#6b7280"
    return f"""
    <div style="background:#1a1f2e;border-radius:4px;height:8px;width:100%;overflow:hidden;">
      <div style="background:{color};height:100%;width:{pct}%;border-radius:4px;
                  transition:width 0.5s ease;"></div>
    </div>
    <span style="font-size:0.75rem;color:#94a3b8;">{pct}%</span>"""


def run_full_pipeline(df_input=None):
    with st.spinner("🧠 Training model & scoring leads…"):
        df = df_input if df_input is not None else generate_sample_data(200)
        model, scaler, metrics, feature_names, encoders = train_model(df)
        save_model(model, scaler, feature_names, encoders)
        scored = score_leads(df, model, scaler, feature_names, encoders)
        st.session_state.df_scored = scored
        st.session_state.metrics = metrics
        st.session_state.ai_generated = False
        st.session_state.last_run = timestamp_now()
        st.session_state.run_count += 1
        save_results(scored)
    st.success(f"✅ Pipeline complete! {len(scored)} leads scored.")


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 AI Sales Assistant")
    st.markdown("---")

    # Data source
    st.markdown("### 📂 Data Source")
    data_source = st.radio("", ["Use Demo Data", "Upload CSV"], label_visibility="collapsed")

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload leads CSV", type=["csv"])

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    model_type = st.selectbox("Algorithm", ["random_forest", "logistic_regression"])
    st.markdown("---")

    # Pipeline trigger
    if st.button("🚀 Run Pipeline", use_container_width=True):
        df_input = None
        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)
        run_full_pipeline(df_input)

    st.markdown("---")

    # Automation
    st.markdown("### ⏰ Automation")
    interval = st.slider("Run every (minutes)", 1, 60, 5)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True, disabled=st.session_state.scheduler_running):
            st.session_state.scheduler_running = True
            t = start_scheduler(interval_minutes=interval)
            st.success(f"Scheduler started ({interval}m)")
    with col2:
        if st.button("⏹ Stop", use_container_width=True, disabled=not st.session_state.scheduler_running):
            st.session_state.scheduler_running = False
            st.info("Scheduler stopped")

    # Status
    dot = "dot-active" if st.session_state.scheduler_running else "dot-idle"
    status = "Running" if st.session_state.scheduler_running else "Idle"
    st.markdown(
        f'<span class="status-dot {dot}"></span>**{status}**',
        unsafe_allow_html=True
    )
    if st.session_state.last_run:
        st.caption(f"Last run: {st.session_state.last_run}")

    st.markdown("---")
    state = load_state()
    st.caption(f"Total pipeline runs: **{state.get('runs', 0)}**")


# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 24px 0;">
  <h1 style="font-size:2rem; margin:0; background:linear-gradient(135deg,#00d4aa,#60a5fa);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    🎯 AI Sales Lead Prioritization
  </h1>
  <p style="color:#8892a4; margin:6px 0 0 0; font-size:0.9rem;">
    ML-powered lead scoring with automated AI recommendations
  </p>
</div>
""", unsafe_allow_html=True)

# Load existing results if session is fresh
if st.session_state.df_scored is None:
    existing = load_results()
    if not existing.empty:
        st.session_state.df_scored = existing
        st.info("📦 Showing results from last pipeline run. Click **Run Pipeline** to refresh.")

# ─────────────────────────────────────────────
# No data state
# ─────────────────────────────────────────────
if st.session_state.df_scored is None:
    st.markdown("""
    <div style="text-align:center; padding:80px 20px; background:#1a1f2e;
                border-radius:16px; border:1px dashed #2d3748; margin-top:20px;">
      <div style="font-size:3rem; margin-bottom:16px;">🚀</div>
      <h2 style="color:#e2e8f0;">Ready to score your leads</h2>
      <p style="color:#8892a4;">Click <strong>Run Pipeline</strong> in the sidebar to start</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


df = st.session_state.df_scored
kpis = compute_kpis(df)

# ─────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

cards = [
    (c1, kpis["total_leads"], "Total Leads", "total"),
    (c2, kpis["high_priority"], f"High Priority ({kpis['high_priority_pct']}%)", "high"),
    (c3, kpis["medium_priority"], "Medium Priority", "medium"),
    (c4, kpis["low_priority"], "Low Priority", "low"),
    (c5, f"{kpis['avg_lead_score']:.3f}", "Avg Lead Score", "score"),
]

for col, val, label, cls in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="value {cls}">{val}</div>
          <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Charts row
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔥 Lead Table", "🤖 AI Insights", "📈 Model Metrics"])

with tab1:
    col_l, col_r = st.columns(2)

    with col_l:
        # Priority distribution
        priority_counts = df["Priority"].value_counts().reset_index()
        priority_counts.columns = ["Priority", "Count"]
        color_map = {"High": "#00d4aa", "Medium": "#f59e0b", "Low": "#6b7280"}
        fig_pie = px.pie(
            priority_counts, names="Priority", values="Count",
            color="Priority", color_discrete_map=color_map,
            title="Lead Priority Distribution",
            hole=0.55,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            title_font_color="#e2e8f0",
            legend=dict(font=dict(color="#94a3b8")),
        )
        fig_pie.update_traces(textfont_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        # Score histogram
        fig_hist = px.histogram(
            df, x="Lead_Score", nbins=20,
            title="Lead Score Distribution",
            color_discrete_sequence=["#60a5fa"],
        )
        fig_hist.add_vline(x=0.65, line_dash="dash", line_color="#00d4aa",
                           annotation_text="High threshold", annotation_font_color="#00d4aa")
        fig_hist.add_vline(x=0.40, line_dash="dash", line_color="#f59e0b",
                           annotation_text="Medium threshold", annotation_font_color="#f59e0b")
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            title_font_color="#e2e8f0",
            bargap=0.1,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Lead source breakdown
    if "Lead_Source" in df.columns:
        source_df = df.groupby("Lead_Source")["Lead_Score"].mean().reset_index()
        source_df.columns = ["Source", "Avg Score"]
        source_df = source_df.sort_values("Avg Score", ascending=True)
        fig_bar = px.bar(
            source_df, x="Avg Score", y="Source", orientation="h",
            title="Average Lead Score by Source",
            color="Avg Score",
            color_continuous_scale=[[0, "#6b7280"], [0.5, "#f59e0b"], [1, "#00d4aa"]],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            title_font_color="#e2e8f0",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────────────
with tab2:
    st.markdown("### 🔥 Ranked Lead Table")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_priority = st.multiselect(
            "Priority", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    with col_f2:
        min_score = st.slider("Min Lead Score", 0.0, 1.0, 0.0, 0.05)
    with col_f3:
        n_rows = st.selectbox("Show rows", [25, 50, 100, 200], index=1)

    filtered = df[
        (df["Priority"].isin(filter_priority)) &
        (df["Lead_Score"] >= min_score)
    ].head(n_rows)

    # Display columns
    display_cols = [c for c in [
        "Rank", "Lead_ID", "Lead_Source", "Occupation",
        "Total_Visits", "Last_Activity", "Lead_Score", "Priority"
    ] if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].style.background_gradient(
            subset=["Lead_Score"], cmap="RdYlGn", vmin=0, vmax=1
        ).format({"Lead_Score": "{:.3f}"}),
        use_container_width=True,
        height=420,
    )

    # Download
    csv = filtered.to_csv(index=False)
    st.download_button(
        "⬇️ Download Results CSV",
        data=csv,
        file_name=f"scored_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
with tab3:
    st.markdown("### 🤖 AI-Powered Lead Insights")

    if not st.session_state.ai_generated:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.info("Click the button to generate AI recommendations for all leads.")
        with col_b:
            if st.button("✨ Generate AI Insights", use_container_width=True):
                progress = st.progress(0, "Generating insights…")
                df_ai = generate_bulk_insights(df.copy(), use_mock=True)
                st.session_state.df_scored = df_ai
                st.session_state.ai_generated = True
                progress.progress(100, "Done!")
                save_results(df_ai)
                st.rerun()
    else:
        df_ai = st.session_state.df_scored
        high_leads = df_ai[df_ai["Priority"] == "High"].head(5)

        st.markdown("#### 🔥 Top High-Priority Leads")
        for _, row in high_leads.iterrows():
            with st.expander(
                f"🎯 {row.get('Lead_ID', 'Lead')} | Score: {row['Lead_Score']:.2%} | {row.get('Lead_Source', '')}",
                expanded=False,
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**🎯 Priority:** {row.get('Priority', 'N/A')}")
                    st.markdown(f"**⚡ Urgency:** {row.get('AI_Urgency', 'N/A')}")
                    st.markdown(f"**📞 Channel:** {row.get('AI_Channel', 'N/A')}")
                with col2:
                    st.markdown(f"**🧠 One-liner:** _{row.get('AI_OneLiner', '')}_")
                st.markdown(f"**✅ Recommended Action:**")
                st.info(row.get("AI_Action", "No recommendation available"))
                st.markdown("**💡 Key Reasons:**")
                for reason in str(row.get("AI_Reasons", "")).split(" | "):
                    st.markdown(f"- {reason}")

        st.markdown("---")
        if "AI_Action" in df_ai.columns:
            csv_ai = df_ai.to_csv(index=False)
            st.download_button(
                "⬇️ Download Full AI Report",
                data=csv_ai,
                file_name=f"ai_lead_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────
with tab4:
    st.markdown("### 📈 Model Performance Metrics")

    metrics = st.session_state.metrics
    if not metrics:
        # Try loading from file
        if os.path.exists("output/model_metrics.json"):
            with open("output/model_metrics.json") as f:
                metrics = json.load(f)

    if metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{float(metrics.get('accuracy', 0)):.2%}")
        with col2:
            st.metric("ROC-AUC Score", f"{float(metrics.get('roc_auc', 0)):.4f}")
        with col3:
            st.metric("Algorithm", metrics.get("model_type", "random_forest").replace("_", " ").title())

        col4, col5 = st.columns(2)
        with col4:
            st.metric("Training Samples", metrics.get("train_samples", "—"))
        with col5:
            st.metric("Test Samples", metrics.get("test_samples", "—"))

        # Feature importances
        if "feature_importances" in metrics:
            fi = pd.DataFrame(
                list(metrics["feature_importances"].items()),
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True).tail(10)

            fig_fi = px.bar(
                fi, x="Importance", y="Feature", orientation="h",
                title="Feature Importances (Top 10)",
                color="Importance",
                color_continuous_scale=[[0, "#2d3748"], [1, "#00d4aa"]],
            )
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                title_font_color="#e2e8f0",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # Confusion matrix
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                colorscale=[[0, "#1a1f2e"], [1, "#00d4aa"]],
                text=cm, texttemplate="%{text}",
                showscale=False,
            ))
            fig_cm.update_layout(
                title="Confusion Matrix",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                title_font_color="#e2e8f0",
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Run the pipeline first to see model metrics.")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5568;font-size:0.8rem;'>"
    "🎯 AI Sales Assistant | ML + LLM Pipeline | Built for Hackathon 2024"
    "</p>",
    unsafe_allow_html=True
)
