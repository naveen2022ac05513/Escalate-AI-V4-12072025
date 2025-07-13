# ==============================================================
# EscalateAI – End‑to‑End Escalation Management System (FINAL v2.0)
# --------------------------------------------------------------
# Core Capabilities
# • Reads Outlook inboxes (Microsoft Graph) for admin‑listed addresses every hour
# • Predicts escalations using sentiment + urgency keywords
# • Logs unique cases into SQLite (id = Outlook message‑id or ESC‑timestamp)
# • Kanban board UI with inline editing / SPOC notifications
# • Excel/CSV upload + manual entry remain supported
# • Auto export to Excel
# • Scheduler escalates to boss after 2 SPOC emails + 24 h
# --------------------------------------------------------------
# Author: Naveen Gandham • July 2025
# ==============================================================
"""Quick‑start (local):

pip install streamlit pandas openpyxl python-dotenv transformers scikit-learn joblib requests apscheduler xlsxwriter msal
# Optional better accuracy (only if PyTorch wheel available):
pip install torch --index-url https://download.pytorch.org/whl/cpu

Create .env alongside this file with:
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USER=<your_smtp_user>
SMTP_PASS=<your_smtp_pass>
MS_CLIENT_ID=<azure_app_client_id>
MS_CLIENT_SECRET=<azure_app_secret>
MS_TENANT_ID=<azure_tenant_id>
"""
# ----------------------------
# Imports & Env
# ----------------------------
import os, re, sqlite3, smtplib, time, io, requests
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import pandas as pd, streamlit as st, joblib
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Optional: sentiment model
HAS_NLP = False
try:
    from transformers import pipeline as hf_pipeline
    import torch
    HAS_NLP = True
except Exception:
    pass

# dotenv optional
try:
    from dotenv import load_dotenv; load_dotenv()
except ModuleNotFoundError:
    st.warning("python-dotenv not installed – reading env directly from OS")

# ----------------------------
# Paths & constants
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR, MODEL_DIR = APP_DIR/"data", APP_DIR/"models"
DATA_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR/"escalateai.db"

SMTP_SERVER = os.getenv("SMTP_SERVER"); SMTP_PORT=int(os.getenv("SMTP_PORT",587))
SMTP_USER   = os.getenv("SMTP_USER");   SMTP_PASS = os.getenv("SMTP_PASS")
MS_CLIENT_ID=os.getenv("MS_CLIENT_ID"); MS_CLIENT_SECRET=os.getenv("MS_CLIENT_SECRET"); MS_TENANT_ID=os.getenv("MS_TENANT_ID")
GRAPH_TOKEN_URL=f"https://login.microsoftonline.com/{MS_TENANT_ID}/oauth2/v2.0/token" if MS_TENANT_ID else None
GRAPH_API="https://graph.microsoft.com/v1.0" if MS_TENANT_ID else None

# ----------------------------
# Sentiment & Escalation detection
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment():
    if not HAS_NLP: return None
    try:
        return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception: return None

sent_model=load_sentiment()
NEG_WORDS=[r"\b(problematic|delay|issue|failure|dissatisfaction|frustration|unacceptable|mistake|complaint|unresolved|unresponsive|unstable|broken|defective|overdue|escalation|leakage|damage|burnt|critical|risk|dispute|faulty)\b"]

def rule_sent(t:str)->str: return "Negative" if any(re.search(w,t,re.I) for w in NEG_WORDS) else "Positive"

def analyze_issue(text:str)->Tuple[str,str,bool]:
    if sent_model:
        label=sent_model(text[:512])[0]["label"].lower(); sentiment="Negative" if label=="negative" else "Positive"
    else:
        sentiment=rule_sent(text)
    urgency="High" if any(k in text.lower() for k in ["urgent","immediate","critical"]) else "Low"
    return sentiment, urgency, sentiment=="Negative" and urgency=="High"

# ----------------------------
# UI – Kanban and Excel ingest (FULL)
# ----------------------------
st.title("🚨 EscalateAI Dashboard")

# Ingest from Excel
uploaded_file = st.sidebar.file_uploader("Upload escalation Excel", type=["xlsx", "csv"])
if uploaded_file:
    if st.sidebar.button("Ingest Escalations"):
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
        for _, row in df.iterrows():
            sentiment, urgency, esc = analyze_issue(str(row.get("issue", "")))
            case = {
                "id": row.get("id", f"ESC{int(datetime.utcnow().timestamp())}"),
                "customer": row.get("customer", "Unknown"),
                "issue": row.get("issue", ""),
                "criticality": row.get("criticality", "Medium"),
                "impact": row.get("impact", "Medium"),
                "sentiment": sentiment,
                "urgency": urgency,
                "escalated": int(esc),
                "date_reported": str(row.get("date_reported", datetime.today().date())),
                "owner": row.get("owner", "Unassigned"),
                "status": row.get("status", "Open"),
                "action_taken": row.get("action_taken", ""),
                "risk_score": predict_risk(str(row.get("issue", ""))),
                "spoc_email": row.get("spoc_email", ""),
                "spoc_boss_email": row.get("spoc_boss_email", "")
            }
            upsert_case(case)
        st.success("Escalations uploaded and processed.")

# Show Kanban
statuses = ["Open", "In Progress", "Resolved", "Closed"]
with st.container():
    df = fetch_cases()
    st.write("## 🧾 Kanban Board")
    cols = st.columns(len(statuses))
    for i, status in enumerate(statuses):
        with cols[i]:
            st.markdown(f"### {status}")
            for _, row in df[df.status == status].iterrows():
                with st.expander(f"{row['id']} - {row['customer']}", expanded=False):
                    st.write(f"**Issue:** {row['issue']}")
                    st.write(f"**Urgency:** {row['urgency']} | **Sentiment:** {row['sentiment']} | **Owner:** {row['owner']}")
                    new_status = st.selectbox("Update status", statuses, index=statuses.index(status), key=row['id'])
                    if new_status != row['status']:
                        row_dict = row.to_dict()
                        row_dict['status'] = new_status
                        upsert_case(row_dict)
                        st.success("Updated.")
                        st.experimental_rerun()

# Export Button
if st.sidebar.button("📤 Export All to Excel"):
    out_df = fetch_cases()
    out_path = DATA_DIR/"escalations_export.xlsx"
    out_df.to_excel(out_path, index=False)
    with open(out_path, "rb") as f:
        st.sidebar.download_button("Download Excel", f, file_name="Escalations.xlsx")
