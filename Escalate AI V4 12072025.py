import os, re, sqlite3, smtplib, requests, atexit, io
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

# ─── Load Config ─────────────────────────────────────────────────
load_dotenv()
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"; DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
SMTP_USER   = os.getenv("SMTP_USER")
SMTP_PASS   = os.getenv("SMTP_PASS")

# ─── Initialize DB ───────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Main escalation table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            customer TEXT, issue TEXT,
            sentiment TEXT, urgency TEXT,
            risk_score REAL, status TEXT,
            action_taken TEXT, owner TEXT,
            spoc_email TEXT, spoc_manager_email TEXT,
            spoc_notify_count INTEGER DEFAULT 0,
            spoc_last_notified TEXT, escalated INTEGER DEFAULT 0,
            date_reported TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # SPOC directory
    cur.execute("""
        CREATE TABLE IF NOT EXISTS spoc_directory (
            spoc_email TEXT PRIMARY KEY,
            spoc_name TEXT,
            spoc_manager_email TEXT,
            teams_webhook TEXT
        )
    """)

    # Patch schema for legacy tables
    cur.execute("PRAGMA table_info(escalations)")
    cols = [r[1] for r in cur.fetchall()]
    if "created_at" not in cols:
        try:
            cur.execute("ALTER TABLE escalations ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")
        except: pass

    conn.commit(); conn.close()

init_db()

# ─── Utility functions ───────────────────────────────────────────
NEG_WORDS = re.compile(r"\b(delay|problem|issue|complaint|failure|critical|risk|unresponsive|defect)\b", re.I)

def analyze_issue(text):
    sentiment = "Negative" if NEG_WORDS.search(text) else "Positive"
    urgency   = "High" if any(w in text.lower() for w in ["urgent","immediate","critical"]) else "Low"
    return sentiment, urgency, sentiment=="Negative" and urgency=="High"

def predict_risk(text):
    return round(min(len(text.split())/50,1.0), 2)

def upsert_case(case):
    cols = ",".join(case.keys())
    qms  = ",".join("?" for _ in case)
    update = ",".join(f"{k}=excluded.{k}" for k in case.keys() if k != "id")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            INSERT INTO escalations ({cols}) VALUES ({qms})
            ON CONFLICT(id) DO UPDATE SET {update}
        """, tuple(case.values()))

def fetch_cases():
    conn = sqlite3.connect(DB_PATH)
    tbl  = pd.read_sql("PRAGMA table_info(escalations)", conn)
    if "created_at" in tbl["name"].values:
        df = pd.read_sql("SELECT * FROM escalations ORDER BY datetime(created_at) DESC", conn)
    else:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    conn.close()
    return df

# ─── Notifications ───────────────────────────────────────────────
def send_email(to, msg, subject="Escalation Notification"):
    email = MIMEText(msg)
    email["From"], email["To"], email["Subject"] = SMTP_USER, to, subject
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as s:
        s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.send_message(email)

def send_teams(webhook, msg):
    if webhook:
        try: requests.post(webhook, json={"text": msg})
        except: pass

def notify_spoc(esc_id, spoc_email):
    df = pd.read_sql("SELECT teams_webhook FROM spoc_directory WHERE spoc_email=?", sqlite3.connect(DB_PATH), params=(spoc_email,))
    webhook = df.teams_webhook.iloc[0] if not df.empty else None
    body = f"🔔 Notification for escalation {esc_id}"
    try:
        send_email(spoc_email, body)
        send_teams(webhook, body)
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE escalations SET
                  spoc_notify_count = spoc_notify_count + 1,
                  spoc_last_notified = ?
                WHERE id = ?
            """, (now, esc_id))
    except: pass

# ─── Reminder Scheduler ─────────────────────────────────────────
def monitor_reminders():
    df  = fetch_cases()
    now = datetime.utcnow()
    for _, r in df[df["status"] == "Open"].iterrows():
        reported = datetime.fromisoformat(r["date_reported"])
        hours    = (now - reported).total_seconds()/3600
        if r["spoc_notify_count"] < 2 and hours > (r["spoc_notify_count"]+1)*6:
            notify_spoc(r["id"], r["spoc_email"])
        elif r["spoc_notify_count"] >= 2 and hours > 24 and not r["escalated"]:
            send_email(r["spoc_manager_email"], f"⚠️ Escalation {r['id']} unattended")
            upsert_case({**r, "escalated":1})

sched = BackgroundScheduler()
sched.add_job(monitor_reminders, "interval", hours=1)
sched.start()
atexit.register(lambda: sched.shutdown(wait=False))

# ─── Streamlit UI ────────────────────────────────────────────────
st.set_page_config("EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

# Sidebar: Upload + Manual Entry + SPOC
with st.sidebar:
    st.header("📥 Upload Escalations")
    f = st.file_uploader("Excel/CSV", type=["xlsx","csv"])
    if f and st.button("Ingest File"):
        df_u = pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)
        for _, row in df_u.iterrows():
            s, u, esc = analyze_issue(str(row.get("issue","")))
            case = {
                "id":               str(row.get("id", f"ESC{int(datetime.utcnow().timestamp())}")),
                "customer":         row.get("customer", "Unknown"),
                "issue":            row.get("issue", ""),
                "sentiment":        s,
                "urgency":          u,
                "risk_score":       predict_risk(str(row.get("issue",""))),
                "status":           row.get("status", "Open"),
                "action_taken":     row.get("action_taken", ""),
                "owner":            row.get("owner", "Unassigned"),
                "spoc_email":       row.get("spoc_email", ""),
                "spoc_manager_email": row.get("spoc_manager_email", ""),
                "spoc_notify_count":0,
                "spoc_last_notified":"",
                "escalated":        0,
                "date_reported":    str(row.get("date_reported", datetime.utcnow().isoformat()))
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
            s,u,esc = analyze_issue(issue)
            case = {
                "id":               f"ESC{int(datetime.utcnow().timestamp())}",
                "customer":         cname,
                "issue":            issue,
                "sentiment":        s,
                "urgency":          u,
                "risk_score":       predict_risk(issue),
                "status":           "Open",
                "action_taken":     "",
                "owner":            owner,
                "spoc_email":       spoc,
                "spoc_manager_email": mgr,
                "spoc_notify_count":0,
                "spoc_last_notified":"",
                "escalated":        0,
                "date_reported":    datetime.utcnow().isoformat()
            }
            upsert_case(case)
            notify_spoc(case["id"], case["spoc_email"])
            st
