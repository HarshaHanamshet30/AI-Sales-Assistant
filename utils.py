"""
utils.py - AI recommendations + helper utilities
Generates LLM-powered insights for each lead
"""

import os
import json
import time
import hashlib
import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────────
# LLM Integration (OpenAI / fallback mock)
# ─────────────────────────────────────────────

def get_llm_recommendation(lead_row: dict, use_mock: bool = False) -> dict:
    """
    Generate AI-powered recommendation for a single lead.
    Falls back to a rule-based mock if no API key is set.
    """
    if use_mock or not os.getenv("OPENAI_API_KEY"):
        return _mock_recommendation(lead_row)
    return _openai_recommendation(lead_row)


def _build_prompt(lead: dict) -> str:
    return f"""You are an expert sales strategist. Analyze this lead and provide recommendations.

LEAD DATA:
- Lead ID: {lead.get('Lead_ID', 'N/A')}
- Lead Source: {lead.get('Lead_Source', 'N/A')}
- Occupation: {lead.get('Occupation', 'N/A')}
- Total Website Visits: {lead.get('Total_Visits', 0)}
- Time Spent on Website (seconds): {lead.get('Total_Time_Spent_on_Website', 0)}
- Last Activity: {lead.get('Last_Activity', 'N/A')}
- Email Opens: {lead.get('Email_Opened', 0)}
- Lead Age (days): {lead.get('Lead_Age_Days', 0)}
- ML Conversion Probability: {lead.get('Lead_Score', 0):.2%}
- Priority Tier: {lead.get('Priority', 'N/A')}

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{
  "priority_label": "High|Medium|Low",
  "conversion_likelihood": "Strong|Moderate|Weak",
  "key_reasons": ["reason 1", "reason 2", "reason 3"],
  "recommended_action": "Specific, actionable next step for the sales team",
  "outreach_channel": "Email|Phone|LinkedIn|WhatsApp",
  "urgency": "Contact within 24h|Contact within 3 days|Nurture over 2 weeks",
  "one_liner": "One-sentence pitch tailored to this lead"
}}"""


def _openai_recommendation(lead: dict) -> dict:
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert B2B sales strategist. Always respond in valid JSON only.",
                },
                {"role": "user", "content": _build_prompt(lead)},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        return {**_mock_recommendation(lead), "error": str(e)}


def _mock_recommendation(lead: dict) -> dict:
    """Rule-based mock recommendation (no API needed for demo)."""
    score = float(lead.get("Lead_Score", 0.5))
    visits = int(lead.get("Total_Visits", 0))
    activity = str(lead.get("Last_Activity", ""))
    source = str(lead.get("Lead_Source", ""))
    time_spent = int(lead.get("Total_Time_Spent_on_Website", 0))

    # Determine tier
    if score >= 0.65:
        tier = "High"
        likelihood = "Strong"
        urgency = "Contact within 24h"
        channel = "Phone" if activity == "Demo Requested" else "Email"
        reasons = [
            f"High engagement: {visits} website visits",
            f"Activity signal: '{activity}' indicates intent",
            f"Source '{source}' historically converts well",
        ]
        action = (
            f"Call immediately — this lead has shown strong buying intent. "
            f"Reference their {activity.lower()} and offer a personalized demo."
        )
        one_liner = "Ready to buy — prioritize a personal outreach today."
    elif score >= 0.40:
        tier = "Medium"
        likelihood = "Moderate"
        urgency = "Contact within 3 days"
        channel = "Email"
        reasons = [
            f"Moderate engagement ({visits} visits, {time_spent}s on site)",
            "Has shown some interest but not yet committed",
            "Needs nurturing to move down the funnel",
        ]
        action = (
            "Send a tailored case study or success story relevant to their occupation. "
            "Follow up with a soft call-to-action within 48 hours."
        )
        one_liner = "Interested but needs nurturing — send a targeted value email."
    else:
        tier = "Low"
        likelihood = "Weak"
        urgency = "Nurture over 2 weeks"
        channel = "Email"
        reasons = [
            f"Low website engagement ({visits} visits)",
            "No strong activity signal detected",
            "May need education-first content",
        ]
        action = (
            "Add to a drip email campaign. Share educational content (blog posts, "
            "webinars). Re-evaluate in 2 weeks."
        )
        one_liner = "Early-stage lead — enrol in nurture sequence and revisit later."

    return {
        "priority_label": tier,
        "conversion_likelihood": likelihood,
        "key_reasons": reasons,
        "recommended_action": action,
        "outreach_channel": channel,
        "urgency": urgency,
        "one_liner": one_liner,
    }


# ─────────────────────────────────────────────
# Batch AI insights
# ─────────────────────────────────────────────

def generate_bulk_insights(df: pd.DataFrame, use_mock: bool = True, delay: float = 0.1) -> pd.DataFrame:
    """
    Run LLM recommendations for all leads.
    Adds columns: AI_Action, AI_Reasons, AI_Urgency, AI_Channel, AI_OneLiner
    """
    results = []
    for _, row in df.iterrows():
        rec = get_llm_recommendation(row.to_dict(), use_mock=use_mock)
        results.append(rec)
        time.sleep(delay)

    rec_df = pd.DataFrame(results)

    df = df.reset_index(drop=True)
    df["AI_Action"] = rec_df.get("recommended_action", "")
    df["AI_Reasons"] = rec_df.get("key_reasons", "").apply(
        lambda x: " | ".join(x) if isinstance(x, list) else str(x)
    )
    df["AI_Urgency"] = rec_df.get("urgency", "")
    df["AI_Channel"] = rec_df.get("outreach_channel", "")
    df["AI_OneLiner"] = rec_df.get("one_liner", "")
    return df


# ─────────────────────────────────────────────
# KPI helpers
# ─────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    high = int((df["Priority"] == "High").sum())
    medium = int((df["Priority"] == "Medium").sum())
    low = int((df["Priority"] == "Low").sum())
    avg_score = float(df["Lead_Score"].mean())
    conversion_rate = (
        float(df["Converted"].mean()) if "Converted" in df.columns else None
    )
    return {
        "total_leads": total,
        "high_priority": high,
        "medium_priority": medium,
        "low_priority": low,
        "avg_lead_score": round(avg_score, 3),
        "high_priority_pct": round(high / total * 100, 1) if total else 0,
        "conversion_rate": round(conversion_rate * 100, 1) if conversion_rate is not None else None,
    }


# ─────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────

def save_results(df: pd.DataFrame, path: str = "output/scored_leads.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_results(path: str = "output/scored_leads.csv") -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def get_file_hash(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────
# Email notification (optional)
# ─────────────────────────────────────────────

def send_email_notification(df: pd.DataFrame, recipient: str = None):
    """
    Sends a summary email. Requires SMTP env vars.
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    recipient = recipient or os.getenv("EMAIL_TO")
    if not recipient:
        return False, "No recipient configured"

    kpis = compute_kpis(df)
    high_leads = df[df["Priority"] == "High"][["Lead_ID", "Lead_Score", "Lead_Source"]].head(5)

    body = f"""
    <h2>🎯 Lead Scoring Pipeline - Automated Report</h2>
    <p><strong>Run at:</strong> {timestamp_now()}</p>
    <h3>📊 Summary</h3>
    <ul>
      <li>Total Leads Processed: <strong>{kpis['total_leads']}</strong></li>
      <li>High Priority: <strong>{kpis['high_priority']} ({kpis['high_priority_pct']}%)</strong></li>
      <li>Average Lead Score: <strong>{kpis['avg_lead_score']}</strong></li>
    </ul>
    <h3>🔥 Top High-Priority Leads</h3>
    {high_leads.to_html(index=False)}
    <p><em>AI Sales Assistant — Automated Pipeline</em></p>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[AI Sales Assistant] {kpis['high_priority']} Hot Leads Ready – {timestamp_now()}"
    msg["From"] = os.getenv("SMTP_USER", "noreply@salesai.com")
    msg["To"] = recipient
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(
            os.getenv("SMTP_HOST", "smtp.gmail.com"),
            int(os.getenv("SMTP_PORT", 587)),
        ) as server:
            server.starttls()
            server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
            server.sendmail(msg["From"], recipient, msg.as_string())
        return True, "Email sent"
    except Exception as e:
        return False, str(e)
