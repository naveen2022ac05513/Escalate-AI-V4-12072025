"""
Quick-start:
  pip install streamlit pandas openpyxl python-dotenv transformers scikit-learn joblib requests apscheduler msal

Create a `.env` alongside this file with:
  SMTP_SERVER=smtp.mail.yahoo.com
  SMTP_PORT=587
  SMTP_USER=<your_yahoo_email>
  SMTP_PASS=<your_yahoo_app_password>
  MS_CLIENT_ID=<azure_app_client_id>
  MS_CLIENT_SECRET=<azure_app_secret>
  MS_TENANT_ID=<azure_tenant_id>
"""

import os
import sqlite3
import smtplib
import requests
import io
import atexit
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from transformers import pipeline as hf_pipeline
import torch
import msal

# ----------------------------
# Paths & ENV
# ----------------------------
load_dotenv()
APP_DIR            = Path(__file__).resolve().parent
DATA_DIR           = APP_DIR / "data"
MODEL_DIR          = APP_DIR / "models"
DB_PATH            = DATA_DIR / "escalateai.db"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SMTP_SERVER        = os.getenv("SMTP_SERVER")
SMTP_PORT          = int(os.getenv("SMTP_PORT", 587))
SMTP_USER          = os.getenv("SMTP_USER")
SMTP_PASS          = os.getenv("SMTP_PASS")

MS_CLIENT_ID       = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET   = os.getenv("MS_CLIENT_SECRET")
MS_TENANT_ID       = os.getenv("MS_TENANT_ID")
GRAPH_API          = "https://graph.microsoft.com/v1.0" if MS_TENANT_ID else None
AUTHORITY_URL      = f"https://login.microsoftonline.com/{MS_TENANT_ID}" if MS_TENANT_ID else None
SCOPE              = ["https://graph.microsoft.com/.default"] if MS_TENANT_ID else []

# ----------------------------
# Sentiment & Escalation Detection
# ----------------------------
NEG_WORDS = [
  r"\b(problematic|delay|issue|failure|dissatisfaction|frustration|unacceptable|mistake|complaint|unresolved|unresponsive|unstable|broken|defective|overdue|escalation|leakage|damage|burnt|critical|risk|dispute|faulty)\b"
]

@st.cache_resource(show_spinner=False)
def load_sentiment():
    try:
        return hf_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
    except:
        return None

sent_model = load_sentiment()

def rule_sent(text: str) -> str:
    return "Negative" if any(re.search(w, text, re.I) for w in NEG_WORDS) else "Positive"

def analyze_issue(text: str):
    if sent_model:
        label = sent_model(text[:512])[0]["label"].lower()
        sentiment = "Negative" if label == "negative" else "Positive"
    else:
        sentiment = rule_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical"]) else "Low"
    return sentiment, urgency, sentiment == "Negative" and urgency == "High"

def analyze_priority(email: str) -> str:
    return "High" if not email.endswith(".se.com") else "Low"

# ----------------------------
# Database Init & Helpers
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # 1) Create escalations table if missing (without ordering column)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            customer TEXT, issue TEXT, sentiment TEXT, urgency TEXT,
            priority TEXT, status TEXT, date_reported TEXT,
            spoc_email TEXT, spoc_boss_email TEXT,
            reminders_sent INTEGER DEFAULT 0,
            escalated INTEGER DEFAULT 0
        )
    """)

    # 2) Add created_at column if it doesn’t already exist
    cur.execute("PRAGMA table_info(escalations)")
    cols = [c[1] for c in cur.fetchall()]
    if "created_at" not in cols:
        try:
            cur.execute("""
                ALTER TABLE escalations
                ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP
            """)
        except sqlite3.OperationalError:
            # Older SQLite might not support ALTER―ignore if it fails
            pass

    conn.commit()
    conn.close()

init_db()

def upsert_case(case: dict):
    cols = ",".join(case.keys())
    qms  = ",".join("?" for _ in case)
    upd  = ",".join(f"{c}=excluded.{c}" for c in case.keys() if c not in ("id","created_at"))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            INSERT INTO escalations ({cols}) VALUES ({qms})
            ON CONFLICT(id) DO UPDATE SET {upd}
        """, tuple(case.values()))

def fetch_cases() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    # Inspect schema to see if created_at is present
    tbl = pd.read_sql_query("PRAGMA table_info(escalations)", conn)
    if "created_at" in tbl["name"].values:
        df = pd.read_sql_query(
            "SELECT * FROM escalations ORDER BY created_at DESC",
            conn
        )
    else:
        df = pd.read_sql_query("SELECT * FROM escalations", conn)

    conn.close()
    return df


def fetch_spocs() -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM spoc_directory",
        sqlite3.connect(DB_PATH)
    )

# ----------------------------
# Email & Teams Notifications
# ----------------------------
def send_email(to_email, subject, body, esc_id):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = to_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        # log
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO notification_log
                (escalation_id, recipient_email, subject, body, sent_at)
                VALUES (?, ?, ?, ?, ?)
            """, (esc_id, to_email, subject, body, datetime.utcnow().isoformat()))
        return True
    except Exception as e:
        print("Email send failed:", e)
        return False

def send_teams(webhook, message):
    if not webhook:
        return
    try:
        requests.post(webhook, json={"text": message})
    except Exception as e:
        print("Teams webhook failed:", e)

def notify_spoc(spoc_email, esc_id, level="Initial"):
    df = fetch_spocs()
    row = df[df.spoc_email == spoc_email]
    webhook = row.teams_webhook.values[0] if not row.empty else None
    subject = f"{level} notification for escalation {esc_id}"
    body    = f"{level} notification for escalation {esc_id}"
    send_email(spoc_email, subject, body, esc_id)
    send_teams(webhook, body)

# ----------------------------
# Outlook Ingestion via Graph
# ----------------------------
def get_graph_token():
    app = msal.ConfidentialClientApplication(
        client_id=MS_CLIENT_ID,
        client_credential=MS_CLIENT_SECRET,
        authority=AUTHORITY_URL
    )
    result = app.acquire_token_for_client(scopes=SCOPE)
    return result.get("access_token")

def ingest_outlook():
    if not GRAPH_API:
        return
    token = get_graph_token()
    if not token:
        return

    headers = {"Authorization": f"Bearer {token}"}
    # fetch unread messages
    url = f"{GRAPH_API}/me/mailFolders/Inbox/messages?$top=20&$filter=isRead eq false"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print("Graph fetch error:", resp.status_code, resp.text)
        return

    existing = fetch_cases().id.tolist()
    for msg in resp.json().get("value", []):
        mid     = msg["id"]
        if mid in existing:
            continue

        subj    = msg.get("subject", "")
        body    = msg.get("bodyPreview", "")
        sender  = msg.get("from", {}).get("emailAddress", {}).get("address", "Unknown")
        sent, urg, esc = analyze_issue(subj + " " + body)
        prio    = analyze_priority(sender)

        # pick first SPOC mapping if available
        spocs = fetch_spocs()
        if not spocs.empty:
            default = spocs.iloc[0]
            spoc_email      = default.spoc_email
            boss_email      = default.spoc_manager_email
        else:
            spoc_email = boss_email = ""

        case = {
            "id":             mid,
            "customer":       sender,
            "issue":          subj + "\n" + body,
            "sentiment":      sent,
            "urgency":        urg,
            "priority":       prio,
            "status":         "Open",
            "date_reported":  msg.get("receivedDateTime", datetime.utcnow().isoformat()),
            "spoc_email":     spoc_email,
            "spoc_boss_email":boss_email,
            "reminders_sent": 0,
            "escalated":      int(esc)
        }
        upsert_case(case)
        notify_spoc(spoc_email, mid, "Initial")

        # mark as read
        patch_url = f"{GRAPH_API}/me/messages/{mid}"
        requests.patch(patch_url, headers=headers, json={"isRead": True})

# ----------------------------
# Reminder Scheduler
# ----------------------------
def monitor_reminders():
    df  = fetch_cases()
    now = datetime.utcnow()
    for _, row in df[df.status == "Open"].iterrows():
        reported = datetime.fromisoformat(row["date_reported"])
        hours    = (now - reported).total_seconds() / 3600
        # reminders every 6h, up to 2 times
        if row.reminders_sent < 2 and hours > (row.reminders_sent + 1) * 6:
            notify_spoc(row.spoc_email, row.id, f"Reminder {row.reminders_sent + 1}")
            row.reminders_sent += 1
        # escalate to boss after 24h
        elif hours > 24 and row.reminders_sent >= 2 and not row.escalated:
            send_email(
                row.spoc_boss_email,
                f"⚠️ Escalation {row.id} unattended",
                f"Escalation {row.id} requires your attention.",
                row.id
            )
            row.escalated = 1
        upsert_case(row.to_dict())

sched = BackgroundScheduler()
sched.add_job(ingest_outlook, "interval", minutes=30)
sched.add_job(monitor_reminders, "interval", hours=1)
sched.start()
atexit.register(lambda: sched.shutdown(wait=False))

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

# Admin: SPOC Directory
with st.sidebar:
    st.header("📋 SPOC Directory")
    spoc_file = st.file_uploader("Upload SPOC Excel", type="xlsx")
    if spoc_file and st.button("Ingest SPOC"):
        df_spoc = pd.read_excel(spoc_file)
        conn    = sqlite3.connect(DB_PATH)
        for _, r in df_spoc.iterrows():
            conn.execute(
                "INSERT OR REPLACE INTO spoc_directory VALUES (?, ?, ?, ?)",
                (r["spoc_email"], r["spoc_name"], r["spoc_manager_email"], r["teams_webhook"])
            )
        conn.commit(); conn.close()
        st.success("SPOC directory updated.")

# Manual & File Ingest
with st.sidebar:
    st.header("📥 Ingest Escalations")
    upload = st.file_uploader("Excel/CSV", type=["xlsx", "csv"])
    if upload and st.button("Ingest File"):
        df_u = (
            pd.read_excel(upload) if upload.name.endswith(".xlsx")
            else pd.read_csv(upload)
        )
        for _, row in df_u.iterrows():
            sent, urg, esc = analyze_issue(str(row.get("issue", "")))
            case = {
                "id":              row.get("id", f"ESC{int(datetime.utcnow().timestamp())}"),
                "customer":        row.get("customer", "Unknown"),
                "issue":           row.get("issue", ""),
                "sentiment":       sent,
                "urgency":         urg,
                "priority":        analyze_priority(row.get("customer_email", "")),
                "status":          row.get("status", "Open"),
                "date_reported":   str(row.get("date_reported", datetime.utcnow().isoformat())),
                "spoc_email":      row.get("spoc_email", ""),
                "spoc_boss_email": row.get("spoc_boss_email", ""),
                "reminders_sent":  0,
                "escalated":       int(esc)
            }
            upsert_case(case)
            notify_spoc(case["spoc_email"], case["id"], "Initial")
        st.success("Uploaded & processed escalations.")

    st.header("✏️ Manual Entry")
    with st.form("manual_entry"):
        cname = st.text_input("Customer")
        issue = st.text_area("Issue")
        spoc  = st.text_input("SPOC Email")
        boss  = st.text_input("Boss Email")
        if st.form_submit_button("Log Escalation"):
            sent, urg, esc = analyze_issue(issue)
            case = {
                "id":              f"ESC{int(datetime.utcnow().timestamp())}",
                "customer":        cname,
                "issue":           issue,
                "sentiment":       sent,
                "urgency":         urg,
                "priority":        analyze_priority(""),
                "status":          "Open",
                "date_reported":   datetime.utcnow().isoformat(),
                "spoc_email":      spoc,
                "spoc_boss_email": boss,
                "reminders_sent":  0,
                "escalated":       int(esc)
            }
            upsert_case(case)
            notify_spoc(spoc, case["id"], "Initial")
            st.success(f"Logged Escalation {case['id']}")

# Main View: Filter & Kanban
show_escalated = st.checkbox("🔍 Show only escalated cases", value=False)
df = fetch_cases()
if show_escalated:
    df = df[df.escalated == 1]

statuses = ["Open", "In Progress", "Resolved", "Closed"]
cols = st.columns(len(statuses))
for i, status in enumerate(statuses):
    with cols[i]:
        st.subheader(status)
        for _, r in df[df.status == status].iterrows():
            with st.expander(f"{r.id} — {r.customer}"):
                st.write(r.issue)
                st.markdown(f"**Sentiment:** {r.sentiment} | **Urgency:** {r.urgency} | **Priority:** {r.priority}")
                new_status = st.selectbox("Update status", statuses, index=statuses.index(status), key=r.id)
                if new_status != r.status:
                    up = r.to_dict()
                    up["status"] = new_status
                    upsert_case(up)
                    st.success("Status updated.")
                    st.experimental_rerun()

# Download board as Excel
buffer = io.BytesIO()
df.to_excel(buffer, index=False, sheet_name="Escalations")
buffer.seek(0)
st.download_button(
    "📥 Download Board as Excel",
    data=buffer,
    file_name="escalations_board.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
