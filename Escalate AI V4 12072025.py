import os, re, sqlite3, smtplib, requests
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
import pandas as pd, streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

# Load .env values
load_dotenv()
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "escalateai.db"
DATA_DIR.mkdir(exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS spoc_directory (
        spoc_email TEXT PRIMARY KEY,
        spoc_name TEXT,
        spoc_manager_email TEXT,
        teams_webhook TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        issue TEXT, customer TEXT, sentiment TEXT, urgency TEXT,
        status TEXT, date_reported TEXT, spoc_email TEXT,
        spoc_boss_email TEXT, reminders_sent INTEGER,
        escalated INTEGER, priority TEXT)""")
    conn.commit(); conn.close()

init_db()

def analyze_priority(email: str):
    return "High" if not email.endswith(".se.com") else "Low"

def send_email(to, message, subject="Escalation Notification"):
    try:
        msg = MIMEText(message)
        msg["From"] = SMTP_USER
        msg["To"] = to
        msg["Subject"] = subject
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, [to], msg.as_string())
    except Exception as e:
        print(f"SMTP error: {e}")

def send_teams(webhook, message):
    if webhook:
        try:
            requests.post(webhook, json={"text": message})
        except Exception as e:
            print(f"Teams webhook error: {e}")

def notify_spoc(spoc_email, escalation_id, level="Initial"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT teams_webhook FROM spoc_directory WHERE spoc_email=?", (spoc_email,))
    row = cur.fetchone()
    webhook = row[0] if row else None
    msg = f"{level} notification for escalation {escalation_id}"
    send_email(spoc_email, msg)
    send_teams(webhook, msg)
    conn.close()

def upsert_escalation(case):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""INSERT INTO escalations (
        id, issue, customer, sentiment, urgency,
        status, date_reported, spoc_email, spoc_boss_email,
        reminders_sent, escalated, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
        status=excluded.status,
        reminders_sent=excluded.reminders_sent,
        escalated=excluded.escalated""",
        tuple(case[k] for k in [
            "id", "issue", "customer", "sentiment", "urgency",
            "status", "date_reported", "spoc_email", "spoc_boss_email",
            "reminders_sent", "escalated", "priority"]))
    conn.commit(); conn.close()

def fetch_cases():
    return pd.read_sql("SELECT * FROM escalations", sqlite3.connect(DB_PATH))

def monitor_reminders():
    df = fetch_cases()
    now = datetime.utcnow()
    for _, row in df[df.status == "Open"].iterrows():
        reported = datetime.fromisoformat(row["date_reported"])
        hours = (now - reported).total_seconds() / 3600
        if row["reminders_sent"] < 2 and hours > (row["reminders_sent"] + 1) * 6:
            notify_spoc(row["spoc_email"], row["id"], f"Reminder {row['reminders_sent'] + 1}")
            row["reminders_sent"] += 1
        elif hours > 24 and row["reminders_sent"] >= 2 and not row["escalated"]:
            send_email(row["spoc_boss_email"], f"Escalation {row['id']} was not resolved in time.")
            row["escalated"] = 1
        upsert_escalation(row.to_dict())

sched = BackgroundScheduler()
sched.add_job(monitor_reminders, "interval", hours=1)
sched.start()

# ------------------ UI ------------------
st.title("🚨 EscalateAI v3.1")

spoc_file = st.sidebar.file_uploader("Upload SPOC Directory", type="xlsx")
if spoc_file and st.sidebar.button("Ingest SPOC"):
    df = pd.read_excel(spoc_file)
    conn = sqlite3.connect(DB_PATH)
    for _, r in df.iterrows():
        conn.execute("INSERT OR REPLACE INTO spoc_directory VALUES (?, ?, ?, ?)", (
            r["spoc_email"], r["spoc_name"], r["spoc_manager_email"], r["teams_webhook"]
        ))
    conn.commit(); conn.close()
    st.success("SPOC directory updated.")

upload_file = st.sidebar.file_uploader("Upload Escalation File", type="xlsx")
if upload_file and st.sidebar.button("Ingest Escalations"):
    df = pd.read_excel(upload_file)
    for _, r in df.iterrows():
        case = {
            "id": str(r.get("id", f"ESC{int(datetime.utcnow().timestamp())}")),
            "issue": r.get("issue", ""),
            "customer": r.get("customer", "Unknown"),
            "sentiment": "Negative" if "issue" in r and len(str(r["issue"])) > 15 else "Positive",
            "urgency": "High" if "urgent" in str(r.get("issue", "")).lower() else "Low",
            "status": r.get("status", "Open"),
            "date_reported": str(r.get("date_reported", datetime.utcnow().isoformat())),
            "spoc_email": r.get("spoc_email", ""),
            "spoc_boss_email": r.get("spoc_boss_email", ""),
            "reminders_sent": 0,
            "escalated": 0,
            "priority": analyze_priority(r.get("customer_email", "abc@xyz.com"))
        }
        upsert_escalation(case)
        notify_spoc(case["spoc_email"], case["id"], "Initial")
    st.success("Escalations processed and SPOC notified.")

st.subheader("📋 Kanban Board")
statuses = ["Open", "In Progress", "Resolved", "Closed"]
df = fetch_cases()
cols = st.columns(len(statuses))
for i, status in enumerate(statuses):
    with cols[i]:
        st.markdown(f"### {status}")
        for _, r in df[df.status == status].iterrows():
            with st.expander(f"{r['id']} — {r['customer']}"):
                st.write(f"**Issue:** {r['issue']}")
                st.write(f"**Urgency:** {r['urgency']} | Sentiment: {r['sentiment']} | Priority: {r['priority']}")
                new_status = st.selectbox("Update Status", statuses, index=statuses.index(status), key=r['id'])
                if new_status != r["status"]:
                    r["status"] = new_status
                    upsert_escalation(r.to_dict())
                    st.success("Updated!")
                    st.experimental_rerun()
