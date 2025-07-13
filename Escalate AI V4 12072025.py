# streamlit_app.py

import os, re, sqlite3, smtplib, requests, atexit, io
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

# ── ENV & PATH SETUP ─────────────────────────────────────────────────────────
load_dotenv()
APP_DIR    = Path(__file__).parent
DATA_DIR   = APP_DIR / "data"; DATA_DIR.mkdir(exist_ok=True)
DB_PATH    = DATA_DIR / "escalateai.db"

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
SMTP_USER   = os.getenv("SMTP_USER")
SMTP_PASS   = os.getenv("SMTP_PASS")

# ── DB INITIALIZATION ────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY,
                customer TEXT,
                issue TEXT,
                sentiment TEXT,
                urgency TEXT,
                risk_score REAL,
                status TEXT,
                action_taken TEXT,
                owner TEXT,
                spoc_email TEXT,
                spoc_manager_email TEXT,
                spoc_notify_count INTEGER DEFAULT 0,
                spoc_last_notified TEXT,
                date_reported TEXT,
                escalated INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS spoc_directory (
                spoc_email TEXT PRIMARY KEY,
                spoc_name TEXT,
                spoc_manager_email TEXT,
                teams_webhook TEXT
            )
        """)
        conn.commit()

init_db()

# ── ESCALATION HELPERS ───────────────────────────────────────────────────────
NEG_WORDS = re.compile(
    r"\b(problematic|delay|issue|failure|complaint|unresolved|unresponsive|broken|defect|critical|risk)\b",
    re.IGNORECASE
)

def analyze_issue(text):
    sentiment = "Negative" if NEG_WORDS.search(text) else "Positive"
    urgency   = "High" if any(w in text.lower() for w in ("urgent","immediate","critical")) else "Low"
    return sentiment, urgency, (sentiment=="Negative" and urgency=="High")

def predict_risk(text):
    return round(min(len(text.split())/50,1.0),2)

def upsert_case(case):
    cols   = ",".join(case.keys())
    qmarks = ",".join("?" for _ in case)
    upd    = ",".join(f"{k}=excluded.{k}" for k in case.keys() if k!="id")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            INSERT INTO escalations ({cols}) VALUES ({qmarks})
            ON CONFLICT(id) DO UPDATE SET {upd}
        """, tuple(case.values()))

def fetch_cases():
    return pd.read_sql_query(
        "SELECT * FROM escalations ORDER BY created_at DESC",
        sqlite3.connect(DB_PATH)
    )

# ── NOTIFICATIONS ────────────────────────────────────────────────────────────
def send_email(to, body, subject="Escalation Notification"):
    msg = MIMEText(body)
    msg["From"], msg["To"], msg["Subject"] = SMTP_USER, to, subject
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as s:
        s.starttls(); s.login(SMTP_USER,SMTP_PASS); s.send_message(msg)

def send_teams(webhook, msg):
    if webhook:
        requests.post(webhook, json={"text":msg})

def notify_spoc(escalation_id, spoc_email):
    row = pd.read_sql_query(
        "SELECT teams_webhook FROM spoc_directory WHERE spoc_email=?",
        sqlite3.connect(DB_PATH),
        params=(spoc_email,)
    )
    webhook = row.teams_webhook.iloc[0] if not row.empty else None
    body    = f"🔔 Notification for escalation {escalation_id}"
    try:
        send_email(spoc_email, body)
        send_teams(webhook, body)
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE escalations
                   SET spoc_notify_count = spoc_notify_count + 1,
                       spoc_last_notified = ?
                 WHERE id = ?
            """, (now, escalation_id))
    except Exception:
        pass

# ── SCHEDULER ────────────────────────────────────────────────────────────────
def monitor_reminders():
    df  = fetch_cases()
    now = datetime.utcnow()
    for _, r in df[df.status=="Open"].iterrows():
        reported = datetime.fromisoformat(r.date_reported)
        hours    = (now - reported).total_seconds() / 3600
        # up to 2 reminders every 6h
        if r.spoc_notify_count < 2 and hours > (r.spoc_notify_count + 1) * 6:
            notify_spoc(r.id, r.spoc_email)
        # escalate to manager after 24h
        elif r.spoc_notify_count >= 2 and hours > 24 and not r.escalated:
            body = f"⚠️ Escalation {r.id} unattended"
            send_email(r.spoc_manager_email, body)
            upsert_case({**r.to_dict(), "escalated": 1})

sched = BackgroundScheduler()
sched.add_job(monitor_reminders, "interval", hours=1)
sched.start()
atexit.register(lambda: sched.shutdown(wait=False))

# ── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

# Sidebar: Upload + Manual + SPOC List
with st.sidebar:
    st.header("📥 Upload Escalations")
    f = st.file_uploader("Excel / CSV", type=["xlsx","csv"])
    if f and st.button("Ingest File"):
        df_u = pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)
        for _, row in df_u.iterrows():
            s,u,esc = analyze_issue(str(row.get("issue","")))
            case = {
                "id":               row.get("id", f"ESC{int(datetime.utcnow().timestamp())}"),
                "customer":         row.get("customer","Unknown"),
                "issue":            row.get("issue",""),
                "sentiment":        s,
                "urgency":          u,
                "risk_score":       predict_risk(str(row.get("issue",""))),
                "status":           row.get("status","Open"),
                "action_taken":     row.get("action_taken",""),
                "owner":            row.get("owner","Unassigned"),
                "spoc_email":       row.get("spoc_email",""),
                "spoc_manager_email": row.get("spoc_manager_email",""),
                "date_reported":    row.get("date_reported", datetime.utcnow().isoformat())
            }
            upsert_case(case)
        st.success("Cases ingested.")

    st.header("✏️ Manual Entry")
    with st.form("manual"):
        cname = st.text_input("Customer")
        issue = st.text_area("Issue")
        owner = st.text_input("Owner", "Unassigned")
        spoc  = st.text_input("SPOC Email")
        mgr   = st.text_input("Manager Email")
        if st.form_submit_button("Log Escalation"):
            s,u,esc = analyze_issue(issue)
            case = {
                "id":                f"ESC{int(datetime.utcnow().timestamp())}",
                "customer":          cname,
                "issue":             issue,
                "sentiment":         s,
                "urgency":           u,
                "risk_score":        predict_risk(issue),
                "status":            "Open",
                "action_taken":      "",
                "owner":             owner,
                "spoc_email":        spoc,
                "spoc_manager_email":mgr,
                "date_reported":     datetime.utcnow().isoformat()
            }
            upsert_case(case)
            notify_spoc(case["id"], case["spoc_email"])
            st.success(f"Logged {case['id']}")

    st.header("📋 SPOC Directory")
    spoc_file = st.file_uploader("Upload SPOC Excel", type="xlsx", key="sd")
    if spoc_file and st.button("Ingest SPOC"):
        df_s = pd.read_excel(spoc_file)
        with sqlite3.connect(DB_PATH) as conn:
            for _, r in df_s.iterrows():
                conn.execute("""
                  INSERT OR REPLACE INTO spoc_directory
                  (spoc_email, spoc_name, spoc_manager_email, teams_webhook)
                  VALUES (?,?,?,?)
                """, (r.spoc_email, r.spoc_name, r.spoc_manager_email, r.teams_webhook))
        st.success("SPOC list updated.")

# Main: Kanban — 3 columns
df = fetch_cases()
cols = st.columns(3)
statuses = ["Open","In Progress","Resolved"]

for idx, status in enumerate(statuses):
    with cols[idx]:
        st.subheader(status)
        for _, r in df[df.status==status].iterrows():
            with st.expander(f"{r.id} – {r.customer}", expanded=False):
                st.markdown(f"""
                **Issue:** {r.issue}  
                **Sentiment/Urgency:** {r.sentiment} / {r.urgency}  
                **Owner:** {r.owner}  
                **Risk Score:** {r.risk_score:.2f}  
                **Status:** {r.status}  
                **Action Taken:** {r.action_taken}  
                **SPOC Email:** {r.spoc_email}  
                **Manager Email:** {r.spoc_manager_email}  
                """)
                if st.button("Notify SPOC", key=f"ntf-{r.id}"):
                    notify_spoc(r.id, r.spoc_email)
                    st.success("SPOC notified.")

# Download Excel
buffer = io.BytesIO()
df.to_excel(buffer, index=False, sheet_name="Escalations")
buffer.seek(0)
st.download_button(
    "📥 Download Board as Excel",
    buffer,
    file_name="escalations_board.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
