import os, re, sqlite3, smtplib, requests, atexit, io
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

# ── Load ENV ──
load_dotenv()
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"; DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# ── DB Setup ──
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        customer TEXT, issue TEXT,
        sentiment TEXT, urgency TEXT,
        risk_score REAL, status TEXT,
        action_taken TEXT, owner TEXT,
        spoc_email TEXT, spoc_manager_email TEXT,
        spoc_notify_count INTEGER DEFAULT 0,
        spoc_last_notified TEXT,
        escalated INTEGER DEFAULT 0,
        date_reported TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS spoc_directory (
        spoc_email TEXT PRIMARY KEY,
        spoc_name TEXT,
        spoc_manager_email TEXT,
        teams_webhook TEXT
    )""")
    conn.commit(); conn.close()

def get_escalation_columns():
    conn = sqlite3.connect(DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(escalations)")]
    conn.close()
    return cols

init_db()

# ── Helpers ──
NEG_WORDS = re.compile(r"\b(delay|problem|issue|complaint|failure|critical|risk|unresponsive|defect)\b", re.I)

def analyze_issue(text):
    sentiment = "Negative" if NEG_WORDS.search(text) else "Positive"
    urgency = "High" if any(w in text.lower() for w in ["urgent", "immediate", "critical"]) else "Low"
    return sentiment, urgency, sentiment == "Negative" and urgency == "High"

def predict_risk(text):
    return round(min(len(text.split())/50, 1.0), 2)

def upsert_case(case: dict):
    valid = get_escalation_columns()
    clean = {k: case[k] for k in case if k in valid}
    cols = ",".join(clean.keys())
    qms = ",".join("?" for _ in clean)
    upd = ",".join(f"{k}=excluded.{k}" for k in clean if k != "id")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
        INSERT INTO escalations ({cols}) VALUES ({qms})
        ON CONFLICT(id) DO UPDATE SET {upd}
        """, tuple(clean.values()))

def fetch_cases():
    conn = sqlite3.connect(DB_PATH)
    schema = pd.read_sql("PRAGMA table_info(escalations)", conn)
    if "created_at" in schema["name"].values:
        df = pd.read_sql("SELECT * FROM escalations ORDER BY datetime(created_at) DESC", conn)
    else:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    conn.close()
    return df

def notify_spoc(esc_id, spoc_email):
    df = pd.read_sql("SELECT teams_webhook FROM spoc_directory WHERE spoc_email=?",
                     sqlite3.connect(DB_PATH), params=(spoc_email,))
    webhook = df.teams_webhook.iloc[0] if not df.empty else None
    body = f"🔔 Notification for escalation {esc_id}"
    try:
        msg = MIMEText(body)
        msg["From"], msg["To"], msg["Subject"] = SMTP_USER, spoc_email, "Escalation Alert"
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as s:
            s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
        requests.post(webhook, json={"text": body}) if webhook else None
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
            UPDATE escalations SET
              spoc_notify_count = spoc_notify_count + 1,
              spoc_last_notified = ?
            WHERE id = ?
            """, (datetime.utcnow().isoformat(), esc_id))
    except: pass

def monitor_reminders():
    df = fetch_cases()
    now = datetime.utcnow()
    for _, r in df[df["status"] == "Open"].iterrows():
        try:
            reported = datetime.fromisoformat(r["date_reported"])
            hours = (now - reported).total_seconds() / 3600
            if r["spoc_notify_count"] < 2 and hours > (r["spoc_notify_count"] + 1)*6:
                notify_spoc(r["id"], r["spoc_email"])
            elif r["spoc_notify_count"] >= 2 and hours > 24 and not r["escalated"]:
                notify_spoc(r["id"], r["spoc_manager_email"])
                upsert_case({**r, "escalated": 1})
        except: pass

sched = BackgroundScheduler()
sched.add_job(monitor_reminders, "interval", hours=1)
sched.start()
atexit.register(lambda: sched.shutdown(wait=False))

# ── UI ──
st.set_page_config("EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

with st.sidebar:
    st.header("📥 Upload Escalations")
    f = st.file_uploader("Excel/CSV", type=["xlsx", "csv"])
    if f and st.button("Ingest"):
        df = pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)
        for _, row in df.iterrows():
            sent, urg, flag = analyze_issue(str(row.get("issue", "")))
            case = {
                "id": row.get("id", f"ESC{int(datetime.utcnow().timestamp())}"),
                "customer": row.get("customer", "Unknown"),
                "issue": row.get("issue", ""),
                "sentiment": sent,
                "urgency": urg,
                "risk_score": predict_risk(str(row.get("issue", ""))),
                "status": row.get("status", "Open"),
                "action_taken": row.get("action_taken", ""),
                "owner": row.get("owner", "Unassigned"),
                "spoc_email": row.get("spoc_email", ""),
                "spoc_manager_email": row.get("spoc_manager_email", ""),
                "spoc_notify_count": 0,
                "spoc_last_notified": "",
                "escalated": 0,
                "date_reported": str(row.get("date_reported", datetime.utcnow().isoformat()))
            }
            upsert_case(case)
        st.success("Escalations ingested.")

    st.header("✏️ Manual Entry")
    with st.form("manual"):
        cname = st.text_input("Customer")
        issue = st.text_area("Issue")
        owner = st.text_input("Owner", value="Unassigned")
        spoc  = st.text_input("SPOC Email")
        mgr   = st.text_input("Manager Email")
        if st.form_submit_button("Log Escalation"):
            s, u, esc = analyze_issue(issue)
            case = {
                "id": f"ESC{int(datetime.utcnow().timestamp())}",
                "customer": cname,
                "issue": issue,
                "sentiment": s,
                "urgency": u,
                "risk_score": predict_risk(issue),
                "status": "Open",
                "action_taken": "",
                "owner": owner,
                "spoc_email": spoc,
                "spoc_manager_email": mgr,
                "spoc_notify_count": 0,
                "spoc_last_notified": "",
                "escalated": 0,
                "date_reported": datetime.utcnow().isoformat()
            }
            upsert_case(case)
            notify_spoc(case["id"], case["spoc_email"])
            st.success(f"Logged {case['id']}")

    st.header("📋 SPOC Directory")
    f_spoc = st.file_uploader("Upload SPOC Excel", type="xlsx", key="spocdir")
    if f_spoc and st.button("Ingest SPOC"):
        df_spoc = pd.read_excel(f_spoc)
        with sqlite3.connect(DB_PATH) as conn:
            for _, r in df_spoc.iterrows():
                conn.execute("""
                INSERT OR REPLACE INTO spoc_directory
                (spoc_email, spoc_name, spoc_manager_email, teams_webhook)
                VALUES (?, ?, ?, ?)
                """, (r["spoc_email"], r["spoc_name"], r["spoc_manager_email"], r["teams_webhook"]))
        st.success("SPOC directory updated.")

# ── Board ──
df = fetch_cases()
df
