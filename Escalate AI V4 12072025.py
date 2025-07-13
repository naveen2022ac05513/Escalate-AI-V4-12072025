import os, re, sqlite3, smtplib, requests
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
import pandas as pd, streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler

# Load env
from dotenv import load_dotenv; load_dotenv()

# Constants
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "escalateai.db"
DATA_DIR.mkdir(exist_ok=True)

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# ------------------ DB Setup ------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY, issue TEXT, customer TEXT,
            sentiment TEXT, urgency TEXT, status TEXT,
            date_reported TEXT, spoc_email TEXT,
            spoc_boss_email TEXT, reminders_sent INTEGER,
            escalated INTEGER, priority TEXT
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS spoc_directory (
            spoc_email TEXT PRIMARY KEY,
            spoc_name TEXT,
            spoc_manager_email TEXT,
            teams_webhook TEXT
        )""")

init_db()

# ------------------ Utils ------------------
def analyze_priority(email):
    return "High" if not email.endswith(".se.com") else "Low"

def send_email(to, message, subject="Escalation Notification"):
    msg = MIMEText(message)
    msg["From"] = SMTP_USER
    msg["To"] = to
    msg["Subject"] = subject
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls(); s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_USER, [to], msg.as_string())

def send_teams(webhook, message):
    if webhook:
        requests.post(webhook, json={"text": message})

def notify(spoc_email, escalation_id, level="Initial"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT teams_webhook FROM spoc_directory WHERE spoc_email=?", (spoc_email,))
    row = cur.fetchone()
    webhook = row[0] if row else None
    message = f"{level} notification for escalation: {escalation_id}"
    send_email(spoc_email, message)
    send_teams(webhook, message)
    conn.close()

def upsert_escalation(case):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""INSERT INTO escalations (
        id, issue, customer, sentiment, urgency, status,
        date_reported, spoc_email, spoc_boss_email,
        reminders_sent, escalated, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
        status=excluded.status,
        reminders_sent=excluded.reminders_sent,
        escalated=excluded.escalated""",
        tuple(case[k] for k in [
            "id", "issue", "customer", "sentiment", "urgency", "status",
            "date_reported", "spoc_email", "spoc_boss_email",
            "reminders_sent", "escalated", "priority"]))
    conn.commit()
    conn.close()

def fetch_cases():
    return pd.read_sql("SELECT * FROM escalations", sqlite3.connect(DB_PATH))

# ------------------ Reminder Engine ------------------
def monitor_reminders():
    df = fetch_cases()
    now = datetime.utcnow()
    for _, row in df[df.status == "Open"].iterrows():
        reported = datetime.fromisoformat(row["date_reported"])
        hours_passed = (now - reported).total_seconds() / 3600
        if row["reminders_sent"] < 2 and hours_passed > (row["reminders_sent"] + 1) * 6:
            notify(row["spoc_email"], row["id"], level=f"Reminder {row['reminders_sent'] + 1}")
            row["reminders_sent"] += 1
        elif hours_passed > 24 and row["reminders_sent"] >= 2 and not row["escalated"]:
            send_email(row["spoc_boss_email"], f"Escalation {row['id']} has not been resolved in 24h.")
            row["escalated"] = 1
        upsert_escalation(row.to_dict())

sched = BackgroundScheduler()
sched.add_job(monitor_reminders, "interval", hours=1)
sched.start()

# ------------------ UI ------------------
st.title("🚨 EscalateAI v3.0")

# SPOC Upload
spoc_file = st.sidebar.file_uploader("Upload SPOC Directory", type="xlsx")
if spoc_file and st.sidebar.button("Ingest SPOC Data"):
    df = pd.read_excel(spoc_file)
    conn = sqlite3.connect(DB_PATH)
    for _, row in df.iterrows():
        conn.execute("""INSERT OR REPLACE INTO spoc_directory (spoc_email, spoc_name, spoc_manager_email, teams_webhook)
            VALUES (?, ?, ?, ?)""", (row["spoc_email"], row["spoc_name"], row["spoc_manager_email"], row["teams_webhook"]))
    conn.commit(); conn.close()
    st.success("SPOC directory updated.")

# Escalation Upload
upload = st.sidebar.file_uploader("Upload Escalation File", type=["xlsx"])
if upload and st.sidebar.button("Ingest Escalations"):
    df = pd.read_excel(upload)
    for _, row in df.iterrows():
        case = {
            "id": str(row.get("id", f"ESC{int(datetime.utcnow().timestamp())}")),
            "issue": row.get("issue", ""),
            "customer": row.get("customer", "Unknown"),
            "sentiment": "Negative" if "issue" in row and len(row["issue"]) > 20 else "Positive",
            "urgency": "High" if "urgent" in str(row.get("issue", "")).lower() else "Low",
            "status": row.get("status", "Open"),
            "date_reported": str(row.get("date_reported", datetime.utcnow().isoformat())),
            "spoc_email": row.get("spoc_email", ""),
            "spoc_boss_email": row.get("spoc_boss_email", ""),
            "reminders_sent": 0,
            "escalated": 0,
            "priority": analyze_priority(row.get("customer_email", "unknown@abc.com"))
        }
        upsert_escalation(case)
        notify(case["spoc_email"], case["id"], level="Initial")
    st.success("Escalations ingested and SPOC notified.")

# Kanban View
st.subheader("📋 Kanban Board")
statuses = ["Open", "In Progress", "Resolved", "Closed"]
df = fetch_cases()
cols = st.columns(len(statuses))
for i, status in enumerate(statuses):
    with cols[i]:
        st.markdown(f"### {status}")
        for _, row in df[df.status == status].iterrows():
            with st.expander(f"{row['id']} – {row['customer']}"):
                st.write(row["issue"])
                st.write(f"Sentiment: {row['sentiment']} | Urgency: {row['urgency']}")
                new_status = st.selectbox("Update Status", statuses, index=statuses.index(status), key=row["id"])
                if new_status != row["status"]:
                    row["status"] = new_status
                    upsert_escalation(row.to_dict())
                    st.success("Status updated.")
                    st.experimental_rerun()
