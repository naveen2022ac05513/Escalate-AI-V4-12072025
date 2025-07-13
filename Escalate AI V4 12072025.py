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
import os, re, sqlite3, atexit, smtplib, time, io, requests
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
try:
    from transformers import pipeline as hf_pipeline; import torch; HAS_NLP=True
except Exception:
    HAS_NLP=False

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
# SQLite init & helpers
# ----------------------------

def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS escalations (
          id TEXT PRIMARY KEY,
          customer TEXT, issue TEXT, criticality TEXT, impact TEXT,
          sentiment TEXT, urgency TEXT, escalated INTEGER,
          date_reported TEXT, owner TEXT, status TEXT, action_taken TEXT,
          risk_score REAL,
          spoc_email TEXT, spoc_boss_email TEXT,
          spoc_notify_count INTEGER DEFAULT 0,
          spoc_last_notified TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS notification_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          escalation_id TEXT, recipient_email TEXT, subject TEXT, body TEXT, sent_at TEXT
        );
        CREATE TABLE IF NOT EXISTS monitored_emails (
          email TEXT PRIMARY KEY, added_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """)
init_db()

ESC_COLS=[c[1] for c in sqlite3.connect(DB_PATH).execute("PRAGMA table_info(escalations)").fetchall()]

def upsert_case(case:dict):
    data={k:case.get(k) for k in ESC_COLS};
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"REPLACE INTO escalations ({','.join(data.keys())}) VALUES ({','.join('?'*len(data))})", tuple(data.values())); conn.commit()

def fetch_cases()->pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM escalations ORDER BY created_at DESC", sqlite3.connect(DB_PATH))

# ----------------------------
# Risk model (optional)
# ----------------------------
MODEL_FILE=MODEL_DIR/"risk_model.joblib"
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
risk_model=load_model()

def predict_risk(txt:str)->float: return float(risk_model.predict_proba([txt])[0][1]) if risk_model else 0.0

# ----------------------------
# SMTP mail helper
# ----------------------------

def send_email(to:str,subj:str,body:str,eid:str):
    if not all([SMTP_SERVER,SMTP_USER,SMTP_PASS]): return False
    msg=MIMEText(body); msg["Subject"]=subj; msg["From"]=SMTP_USER; msg["To"]=to
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO notification_log (escalation_id,recipient_email,subject,body,sent_at) VALUES (?,?,?,?,?)",(eid,to,subj,body,datetime.now().isoformat())); conn.commit()
        return True
    except Exception as e:
        st.error(f"SMTP failed: {e}"); return False

# ----------------------------
# Outlook helpers (Graph)
# ----------------------------
if MS_CLIENT_ID and MS_CLIENT_SECRET and MS_TENANT_ID:
    def get_token():
        payload={"grant_type":"client_credentials","client_id":MS_CLIENT_ID,"client_secret":MS_CLIENT_SECRET,"scope":"https://graph.microsoft.com/.default"}
        r=requests.post(GRAPH_TOKEN_URL,data=payload)
        r.raise_for_status(); return r.json()["access_token"]

    def fetch_emails(address:str,token:str):
        hdr={"Authorization":f"Bearer {token}"}
        since=(datetime.utcnow()-timedelta(hours=1)).isoformat()+"Z"
        url=f"{GRAPH_API}/users/{address}/mailFolders/inbox/messages"
        params={"$top":25,"$filter":f"receivedDateTime ge {since}","$orderby":"receivedDateTime desc"}
        r=requests.get(url,headers=hdr,params=params); r.raise_for_status(); return r.json().get("value",[])

    def monitor_emails_job():
        token=get_token()
        emails=[row[0] for row in sqlite3.connect(DB_PATH).execute("SELECT email FROM monitored_emails").fetchall()]
        for addr in emails:
            for msg in fetch_emails(addr,token):
                mid=msg["id"]; preview=msg.get("bodyPreview","")
                if not preview: continue
                if sqlite3.connect(DB_PATH).execute("SELECT 1 FROM escalations WHERE id=?",(mid,)).fetchone(): continue
                sentiment,urgency,esc=analyze_issue(preview)
                if esc:
                    case={"id":mid,"customer":msg.get("from",{}).get("emailAddress",{}).get("name","Unknown"),"issue":preview,"criticality":"High","impact":"High","sentiment":sentiment,"urgency":urgency,"escalated":1,"date_reported":msg.get("receivedDateTime","")[:10],"owner":"Auto","status":"Open","action_taken":"","risk_score":predict_risk(preview)}; upsert_case(case)

    if "outlook_sched" not in st.session_state:
        BackgroundScheduler().add_job(monitor_emails_job,"interval",hours=1).start(); st.session_state["outlook_sched"]="on"
else:
    st.sidebar.warning("Outlook integration disabled – set MS_CLIENT_* env vars")

# ----------------------------
# Boss escalation job
# ----------------------------

def boss_check():
    df=fetch_cases();
    for _,r in df.iterrows():
        if r.get("spoc_notify_count",0)>=2 and r.get("spoc_boss_email") and r.get("spoc_last_notified"):
            if datetime.now()-datetime.fromisoformat(r["spoc_last_notified"])>timedelta(hours=24):
                send_email(r["spoc_boss_email"],f"⚠️ Escalation {r['id']} unattended",f"Please review escalation {r['id']}",r["id"])

if "boss_sched" not in st.session_state:
    BackgroundScheduler().add_job(boss_check,"interval",hours=1).start(); st.session_state["boss_sched"]="on"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="EscalateAI",layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking")

# -- Sidebar: email monitor
with st.sidebar.expander("📧 Monitor Outlook Emails"):
    new_email=st.text_input("Add email address")
    if st.button("Add") and new_email and "@" in new_email:
        sqlite3.connect(DB_PATH).execute("INSERT OR IGNORE INTO monitored_emails (email) VALUES (?)",(new_email,)); st.success("Added"); st.experimental_rerun()
    for em in sqlite3.connect(DB_PATH).execute("SELECT email FROM monitored_emails").fetchall():
        em=em[0]; st.write(em)
        if st.button("Remove",key=f"rem_{em}"): sqlite3.connect(DB_PATH).execute("DELETE FROM monitored_emails WHERE email=?",(em,)); st.experimental_rerun()

# -- Sidebar: upload & manual
with st.sidebar:
    st.header("📥 Upload Escalations")
    uploaded=st.file_uploader("Excel/CSV",type=["xlsx","csv"])
    if uploaded and st.button("Ingest"):
        df=pd.read_excel(uploaded) if uploaded.name.endswith("xlsx") else pd.read_csv(uploaded)
        for _, row in df.iterrows():
            sentiment,urgency,esc=analyze_issue(str(row.get("issue","")))
            case={"id":row.get("id",f"ESC{int(datetime.utcnow().timestamp())}"),"customer":row.get("customer","Unknown"),"issue":row.get("issue",""),"criticality":row.get("criticality","Medium"),"impact":row.get("impact","Medium"),"sentiment":sentiment,"urgency":urgency,"escalated":int(esc),"date_reported":str(row.get("date_reported",datetime.today().date())),"owner":row.get("owner","Unassigned"),"status":row.get("status","Open"),"action_taken":row.get("action_taken",""),"risk
