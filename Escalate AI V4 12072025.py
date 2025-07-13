# ==============================================================
# EscalateAI – Escalation Management with Outlook Integration
# --------------------------------------------------------------
# • Outlook inbox + sent‑items polling every hour (configurable)
# • Sentiment detection (HF transformers if available, else rule‑based)
# • Negative mails auto‑tagged as potential escalations
# • SPOC notifications + 24‑hour manager escalation
# • Streamlit Kanban dashboard, CSV/Excel upload, manual entry, export
# --------------------------------------------------------------
# Author: Naveen Gandham • v1.2.1 • July 2025
# ==============================================================

"""Setup
pip install streamlit pandas openpyxl python-dotenv apscheduler scikit-learn joblib xlsxwriter O365
# Optional for better NLP
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
"""

# ========== Standard Lib ==========
import os, re, sqlite3, atexit, smtplib, time, io
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List

# ========== Third‑party ==========
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Optional sentiment NLP ------------------------------------------------------
try:
    from transformers import pipeline as hf_pipeline  # type: ignore
    import torch  # noqa: F401 (CPU wheel)
    HAS_NLP = True
except Exception:
    HAS_NLP = False

# Optional Outlook / MS Graph --------------------------------------------------
try:
    from O365 import Account, FileSystemTokenBackend, MSGraphProtocol  # type: ignore
    HAS_O365 = True
except ModuleNotFoundError:
    HAS_O365 = False

# ========== Paths & ENV ==========
APP_DIR   = Path(__file__).resolve().parent
DATA_DIR  = APP_DIR / "data";  DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = APP_DIR / "models"; MODEL_DIR.mkdir(exist_ok=True)
DB_PATH   = DATA_DIR / "escalateai.db"
TOKEN_DIR = DATA_DIR / "o365_tokens"; TOKEN_DIR.mkdir(exist_ok=True)

load_dotenv()
SMTP_SERVER   = os.getenv("SMTP_SERVER")
SMTP_PORT     = int(os.getenv("SMTP_PORT", 587))
SMTP_USER     = os.getenv("SMTP_USER")
SMTP_PASS     = os.getenv("SMTP_PASS")
O365_CLIENT_ID     = os.getenv("O365_CLIENT_ID")
O365_CLIENT_SECRET = os.getenv("O365_CLIENT_SECRET")
O365_TENANT_ID     = os.getenv("O365_TENANT_ID")
POLL_INTERVAL_MIN  = int(os.getenv("POLL_INTERVAL_MINUTES", 60))
SENDER_FILTER      = [s.strip().lower() for s in os.getenv("SENDER_FILTER", "").split(',') if s.strip()]

# ========== Sentiment Model ==========
@st.cache_resource(show_spinner=False)
def load_sentiment():
    if not HAS_NLP:
        return None
    try:
        return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception:
        return None

sent_model = load_sentiment()
NEG_WORDS = [r"\b(delay|issue|failure|dissatisfaction|unacceptable|complaint|escalation|critical|risk|faulty)\b"]

def rule_sent(text: str) -> str:
    return "Negative" if any(re.search(p, text, re.I) for p in NEG_WORDS) else "Positive"

def analyze_issue(text: str) -> Tuple[str, str, bool]:
    if sent_model:
        lbl = sent_model(text[:512])[0]["label"].lower()
        sentiment = "Negative" if lbl == "negative" else "Positive"
    else:
        sentiment = rule_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical", "asap"]) else "Low"
    return sentiment, urgency, sentiment == "Negative" and urgency == "High"

# ========== Outlook Account (if available) ==========
account = None
if HAS_O365 and O365_CLIENT_ID and O365_CLIENT_SECRET and O365_TENANT_ID:
    creds = (O365_CLIENT_ID, O365_CLIENT_SECRET)
    token_backend = FileSystemTokenBackend(token_path=TOKEN_DIR, token_filename="o365_token.txt")
    protocol = MSGraphProtocol(default_resource="me")
    account = Account(creds, auth_flow_type="credentials", tenant_id=O365_TENANT_ID, token_backend=token_backend, protocol=protocol)
    if not account.is_authenticated:
        account.authenticate(scopes=["https://graph.microsoft.com/.default"])
    inbox_folder = account.mailbox().inbox_folder() if account else None

# ========== DB Init ==========

def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            customer TEXT,
            issue TEXT,
            criticality TEXT,
            impact TEXT,
            sentiment TEXT,
            urgency TEXT,
            escalated INTEGER,
            date_reported TEXT,
            owner TEXT,
            status TEXT,
            action_taken TEXT,
            risk_score REAL,
            spoc_email TEXT,
            spoc_boss_email TEXT,
            spoc_notify_count INTEGER DEFAULT 0,
            spoc_last_notified TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS notification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            escalation_id TEXT,
            recipient_email TEXT,
            subject TEXT,
            body TEXT,
            sent_at TEXT
        )""")
        c.commit()

init_db()
ESC_COLS = [r[1] for r in sqlite3.connect(DB_PATH).execute("PRAGMA table_info(escalations)").fetchall()]

# ========== DB helpers ==========

def upsert_case(d: dict):
    rec = {k: d.get(k) for k in ESC_COLS}
    with sqlite3.connect(DB_PATH) as c:
        c.execute(f"REPLACE INTO escalations ({','.join(rec.keys())}) VALUES ({','.join('?'*len(rec))})", tuple(rec.values()))
        c.commit()

def fetch_cases() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql("SELECT * FROM escalations ORDER BY datetime(created_at) DESC", c)

def fetch_logs() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql("SELECT * FROM notification_log ORDER BY datetime(sent_at) DESC", c)

# ========== Risk model (optional) ==========
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
@st.cache_resource(show_spinner=False)
def load_risk_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

risk_model = load_risk_model()

def predict_risk(txt: str) -> float:
    return float(risk_model.predict_proba([txt])[0][1]) if risk_model else 0.0

# ========== Email send ==========

def send_email(to_: str, sub: str, body: str, esc_id: str, retries: int = 3):
    if not (SMTP_SERVER and SMTP_USER and SMTP_PASS):
        st.error("SMTP not configured")
        return False
    for att in range(retries):
        try:
            msg = MIMEText(body)
            msg["Subject"], msg["From"], msg["To"] = sub, SMTP_USER, to_
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as s:
                s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
            with sqlite3.connect(DB_PATH) as c:
                c.execute("INSERT INTO notification_log (escalation_id, recipient_email, subject, body, sent_at) VALUES (?,?,?,?,?)", (esc_id, to_, sub, body, datetime.utcnow().isoformat()))
                c.commit()
            return True
        except Exception as e:
            if att == retries-1:
                st.error(f"Email error: {e}")
            time.sleep(1)
    return False

# ========== Outlook Polling ==========

def poll_outlook():
    if not inbox_folder:
        return
    q = inbox_folder.new_query().on_attribute("is_read").equals(False)
    for msg in inbox_folder.get_messages(limit=50, query=q):
        try:
            body = msg.body or msg.body_preview or msg.subject or ""
            if not body:
                continue
            sender = (msg.sender.address if msg.sender else "unknown").lower()
            if SENDER_FILTER and sender not in SENDER_FILTER:
                continue
            sentiment, urgency, esc = analyze_issue(body)
            if sentiment == "Negative":
                eid = msg.message_id
                case = {
                    "id": eid,
                    "customer": sender,
                    "issue": (msg.subject or "") + "\n" + body[:500],
                    "criticality": "High",
                    "impact": "High",
                    "sentiment": sentiment,
                    "urgency": urgency,
                    "escalated": int(esc),
                    "date_reported": str(msg.received.date()),
                    "owner": "Unassigned",
                    "status": "Open",
                    "action_taken": "",
                    "risk_score": predict_risk(body),
                    "spoc_email": sender,
                    "spoc_boss_email": "",
                }
                upsert_case(case)
            msg.mark_as_read()
        except Exception as e:
            print("[Outlook Poll]", e)

# ========== Reminder & Escalation job ==========

def reminder_job():
    df = fetch_cases()
    now = datetime.utcnow()
    for _, r in df.iterrows():
        if r.status != "Open":
            continue
        hrs = (now - datetime.fromisoformat(r.date_reported)).total_seconds() / 3600
        cnt = r.get("spoc_notify_count", 0) or 0
        last = r.get("spoc_last_notified")
        # SPOC reminders every 6h (max 2)
        if cnt < 2 and hrs >= (cnt+1)*6:
            if send_email(r.spoc_email, f"Escalation {r.id} Requires Action", r.issue[:400], r.id):
                upd = r.to_dict(); upd["spoc_notify_count"] = cnt+1; upd["spoc_last_notified"] = now.isoformat(); upsert_case(upd)
        # Manager escalation after 24h
        elif cnt >=2 and hrs>=24 and r.spoc_boss_email and not r.escalated:
            if send_email(r.spoc_boss_email, f"⚠️ Escalation {r.id} Pending >24h", r.issue[:400], r.id):
                upd = r.to_dict(); upd["escalated"] = 1; upsert_case(upd)

# ========== Scheduler ==========
if "sched" not in st.session_state:
    sched = BackgroundScheduler()
    if account: sched.add_job(poll_outlook, "interval", minutes=POLL_INTERVAL_MIN)
    sched.add_job(reminder_job, "interval", hours=1)
    sched.start(); atexit.register(lambda: sched.shutdown(wait=False))
    st.session_state.sched = True

# ========== Utility ==========

def df_to_excel_bytes(df: pd.DataFrame):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="Escalations")
    return buf.getvalue()

# ========== Streamlit UI ==========
st.set_page_config("EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Dashboard")

# Sidebar file upload & manual form ---------------------------
with st.sidebar:
    st.header("📥 Upload Escalations")
    f = st.file_uploader("Excel / CSV", type=["xlsx","csv"])
    if f and st.button("Ingest File"):
        df_u = pd.read_excel(f) if f.name.endswith("xlsx") else pd.read_csv(f)
        for _, row in df_u.iterrows():
            s, u, e = analyze_issue(str(row.get("issue","")))
            case = {
                "id": row.get("id", f"ESC{int(time.time()*1000)}"),
                "customer": row.get("customer","Unknown"),
                "issue": row.get("issue",""),
                "criticality": row.get("criticality","Medium"),
                "impact": row.get("impact","Medium"),
                "sentiment": s,
                "urgency": u,
                "
