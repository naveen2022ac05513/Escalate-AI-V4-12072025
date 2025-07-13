
# ========== EscalateAI – Escalation Management with Outlook Integration ==========
# Author: Naveen Gandham | Version: 1.2.0 | July 2025
# Description: Adds full support for Outlook inbox/outbox parsing with sentiment detection,
# automatic tagging, and escalation notification logic. Scheduled hourly polling.

import os, re, sqlite3, atexit, smtplib, time
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import joblib, pandas as pd, streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Outlook dependencies
from O365 import Account, FileSystemTokenBackend

# ========== Paths & ENV ==========
APP_DIR   = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
DATA_DIR  = APP_DIR / "data"
DB_PATH   = DATA_DIR / "escalateai.db"
TOKEN_DIR = DATA_DIR / "o365_tokens"
TOKEN_DIR.mkdir(exist_ok=True)

load_dotenv()
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
SMTP_USER   = os.getenv("SMTP_USER")
SMTP_PASS   = os.getenv("SMTP_PASS")
OUTLOOK_CLIENT_ID = os.getenv("O365_CLIENT_ID")
OUTLOOK_CLIENT_SECRET = os.getenv("O365_CLIENT_SECRET")
TENANT_ID = os.getenv("O365_TENANT_ID")
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", 60))

# ========== Sentiment Analysis Setup ==========
@st.cache_resource(show_spinner=False)
def load_sentiment():
    if not HAS_NLP:
        return None
    try:
        return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception:
        return None

sent_model = load_sentiment()

NEG_WORDS = [
    r"\b(problematic|delay|issue|failure|dissatisfaction|frustration|unacceptable|mistake|complaint|unresolved|unresponsive|unstable|broken|defective|overdue|escalation|leakage|damage|burnt|critical|risk|dispute|faulty)\b"
]

def rule_sent(text: str) -> str:
    return "Negative" if any(re.search(p, text, re.I) for p in NEG_WORDS) else "Positive"

def analyze_issue(text: str) -> Tuple[str, str, bool]:
    if sent_model:
        label = sent_model(text[:512])[0]["label"].lower()
        sentiment = "Negative" if label == "negative" else "Positive"
    else:
        sentiment = rule_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical", "asap"]) else "Low"
    return sentiment, urgency, sentiment == "Negative" and urgency == "High"

# ========== Outlook Account Setup ==========
credentials = (OUTLOOK_CLIENT_ID, OUTLOOK_CLIENT_SECRET)
token_backend = FileSystemTokenBackend(token_path=TOKEN_DIR, token_filename='o365_token.txt')
account = Account(credentials, auth_flow_type='credentials', tenant_id=TENANT_ID, token_backend=token_backend)
account.authenticate()

# ========== Database Initialization & Helpers ==========

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS escalations (
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
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS notification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                escalation_id TEXT,
                recipient_email TEXT,
                subject TEXT,
                body TEXT,
                sent_at TEXT
            )"""
        )
        conn.commit()

        # Auto‑add any new columns (future‑proof)
        cur.execute("PRAGMA table_info(escalations)")
        cols = [c[1] for c in cur.fetchall()]
        need = {
            "spoc_notify_count": "INTEGER DEFAULT 0",
            "spoc_last_notified": "TEXT",
            "owner": "TEXT",
        }
        for c, t in need.items():
            if c not in cols:
                try:
                    cur.execute(f"ALTER TABLE escalations ADD COLUMN {c} {t}")
                except Exception:
                    pass
        conn.commit()

init_db()

ESC_COLS = [c[1] for c in sqlite3.connect(DB_PATH).execute("PRAGMA table_info(escalations)").fetchall()]

def upsert_case(case: dict):
    data = {k: case.get(k) for k in ESC_COLS}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"REPLACE INTO escalations ({','.join(data.keys())}) VALUES ({','.join('?'*len(data))})",
            tuple(data.values()),
        )
        conn.commit()

def fetch_cases() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM escalations ORDER BY datetime(created_at) DESC", conn)

def fetch_logs() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM notification_log ORDER BY datetime(sent_at) DESC", conn)

# ========== Outlook Email Parsing ==========
def parse_outlook_emails():
    mailbox = account.mailbox()
    folder = mailbox.inbox_folder()
    q = folder.new_query().on_attribute("is_read").equals(False)
    messages = folder.get_messages(limit=30, query=q, download_attachments=False)
    for msg in messages:
        try:
            subj = msg.subject or "(No Subject)"
            body = msg.body or ""
            sender = msg.sender.address
            if not body.strip():
                continue
            sentiment, urgency, escalate = analyze_issue(body)
            if sentiment == "Negative":
                esc_id = f"AUTOESC{int(datetime.utcnow().timestamp())}"
                case = {
                    "id": esc_id,
                    "customer": sender,
                    "issue": subj + "\n" + body[:500],
                    "criticality": "High",
                    "impact": "High",
                    "sentiment": sentiment,
                    "urgency": urgency,
                    "escalated": int(escalate),
                    "date_reported": str(datetime.today().date()),
                    "owner": "Unassigned",
                    "status": "Open",
                    "action_taken": "",
                    "risk_score": predict_risk(body),
                    "spoc_email": sender,
                    "spoc_boss_email": "",  # You can update via directory or config
                }
                upsert_case(case)
                msg.mark_as_read()
        except Exception as e:
            print("[Outlook Poll Error]", e)

# ========== Notification Logic ==========
def notify_and_escalate():
    df = fetch_cases()
    for _, r in df.iterrows():
        try:
            if r.status != "Open":
                continue
            hours_since = (datetime.now() - datetime.fromisoformat(r.date_reported)).total_seconds() / 3600
            if r.spoc_notify_count < 2 and hours_since >= (r.spoc_notify_count + 1) * 6:
                subj = f"Escalation {r.id} Requires Your Attention"
                body = f"Dear SPOC,\n\nPlease review the issue:\n{r.issue[:300]}..."
                if send_email(r.spoc_email, subj, body, r.id):
                    r.spoc_notify_count += 1
                    r.spoc_last_notified = datetime.now().isoformat()
                    upsert_case(r.to_dict())
            elif r.spoc_notify_count >= 2 and hours_since >= 24 and not r.escalated:
                boss = r.spoc_boss_email
                if boss:
                    subj = f"⚠️ Escalation {r.id} Not Addressed in 24h"
                    body = f"Manager,\n\nEscalation {r.id} from {r.customer} remains unaddressed.\nPlease take action."
                    if send_email(boss, subj, body, r.id):
                        r.spoc_notify_count += 1
                        r.spoc_last_notified = datetime.now().isoformat()
                        r.escalated = 1
                        upsert_case(r.to_dict())
        except Exception as e:
            print("[Notify Error]", e)

# ========== Scheduler ==========
if "scheduler" not in st.session_state:
    scheduler = BackgroundScheduler()
    scheduler.add_job(parse_outlook_emails, "interval", minutes=POLL_INTERVAL_MINUTES)
    scheduler.add_job(notify_and_escalate, "interval", hours=1)
    scheduler.start()
    atexit.register(scheduler.shutdown)
    st.session_state.scheduler = True

# ========== Email Sending with Retries ==========

def send_email(to_email: str, subject: str, body: str, esc_id: str, retries: int = 3) -> bool:
    if not (SMTP_SERVER and SMTP_USER and SMTP_PASS):
        st.error("SMTP not configured properly")
        return False
    attempt = 0
    while attempt < retries:
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = f"EscalateAI <{SMTP_USER}>"
            msg["To"] = to_email
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as s:
                s.starttls()
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
            # Log
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "INSERT INTO notification_log (escalation_id, recipient_email, subject, body, sent_at) VALUES (?,?,?,?,?)",
                    (esc_id, to_email, subject, body, datetime.utcnow().isoformat()),
                )
                conn.commit()
            return True
        except Exception as e:  # noqa: BLE001
            attempt += 1
            time.sleep(2)
            if attempt == retries:
                st.error(f"Email failed: {e}")
                return False

# ========== ML Risk Prediction Model ==========
MODEL_FILE = MODEL_DIR / "risk_model.joblib"

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None

risk_model = load_model()

def predict_risk(issue: str) -> float:
    return float(risk_model.predict_proba([issue])[0][1]) if risk_model else 0.0

# ========== Outlook / Microsoft Graph Email Ingestion ==========
if HAS_O365 and O365_CLIENT_ID and O365_CLIENT_SECRET and O365_TENANT_ID:
    credentials = (O365_CLIENT_ID, O365_CLIENT_SECRET)
    protocol = MSGraphProtocol(default_resource="me")
    account = Account(credentials, auth_flow_type="credentials", tenant_id=O365_TENANT_ID, protocol=protocol)
    if not account.is_authenticated:
        account.authenticate(scopes=["https://graph.microsoft.com/.default"])
    mailbox = account.mailbox()
    inbox_folder = mailbox.inbox_folder()
else:
    account = None


def fetch_unread_emails() -> List[dict]:
    """Return list of unread email metadata dicts."""
    if not account:
        return []
    messages = inbox_folder.get_messages(limit=50, query="isRead eq false")
    res: List[dict] = []
    for msg in messages:
        try:
            sender = msg.sender.address.lower() if msg.sender else "unknown"
            if SENDER_FILTER and sender not in SENDER_FILTER:
                continue  # skip unwanted senders
            res.append({
                "id": msg.message_id,
                "sender": sender,
                "subject": msg.subject or "",
                "body_preview": msg.body_preview or msg.subject or "",
                "body": msg.body or msg.body_preview or msg.subject or "",
                "received": msg.received,
            })
            msg.mark_as_read()
        except Exception:
            continue
    return res


def process_emails_to_cases():
    new_cnt = 0
    for em in fetch_unread_emails():
        sentiment, urgency, esc = analyze_issue(em["body"])
        case = {
            "id": em["id"],  # Graph message_id is globally unique
            "customer": em["sender"],
            "issue": em["body"][:2000],  # truncate long bodies
            "criticality": "Medium",
            "impact": "Medium",
            "sentiment": sentiment,
            "urgency": urgency,
            "escalated": int(esc),
            "date_reported": em["received"].strftime("%Y-%m-%d"),
            "owner": "Unassigned",
            "status": "Open",
            "action_taken": "",
            "risk_score": predict_risk(em["body"]),
            "spoc_email": "",
            "spoc_boss_email": "",
        }
        upsert_case(case)
        new_cnt += 1
    if new_cnt:
        st.toast(f"📧 {new_cnt} new escalation(s) ingested from Outlook", icon="✉️")

# ========== Scheduler Jobs ==========

def boss_escalation_job():
    try:
        df = fetch_cases()
        now = datetime.utcnow()
        for _, r in df.iterrows():
            count = r.get("spoc_notify_count", 0) or 0
            last = r.get("spoc_last_notified")
            boss = r.get("spoc_boss_email")
            if count >= 2 and boss and last:
                last_dt = datetime.fromisoformat(last)
                if now - last_dt > timedelta(hours=24):
                    subj = f"⚠️ Escalation {r['id']} unattended"
                    body = f"Dear Manager,\n\nEscalation {r['id']} has received no response for 24 hours after two notifications.\n\nIssue excerpt:\n{r['issue'][:400]}..."
                    if send_email(boss, subj, body, r["id"]):
                        upd = r.to_dict()
                        upd["spoc_notify_count"] = count + 1
                        upsert_case(upd)
    except Exception as e:
        st.warning(f"Boss escalation job error: {e}")


def outlook_poll_job():
    try:
        process_emails_to_cases()
    except Exception as e:
        st.warning(f"Outlook poll error: {e}")

# START scheduler once (per Streamlit session)
if "sched_main" not in st.session_state:
    sched = BackgroundScheduler()
    sched.add_job(boss_escalation_job, "interval", hours=1)
    if account:
        sched.add_job(outlook_poll_job, "interval", minutes=POLL_INTERVAL_MIN)
    sched.start()
    atexit.register(lambda: sched.shutdown(wait=False))
    st.session_state["sched_main"] = True

# ========== Utility ==========

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Escalations")
    return output.getvalue()

# ========== Streamlit UI ==========

st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

# Sidebar upload & manual entry
with st.sidebar:
    st.header("📥 Upload Escalations")
    upl = st.file_uploader("Excel / CSV", type=["xlsx", "csv"])
    if upl and st.button("Ingest File"):
        df_u = pd.read_excel(upl) if upl.name.endswith("xlsx") else pd.read_csv(upl)
        for _, row in df_u.iterrows():
            sentiment, urgency, esc = analyze_issue(str(row.get("issue", "")))
            case = {
                "id": row.get("id", f"ESC{int(datetime.utcnow().timestamp()*1000)}"),
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
                "risk_score": predict_risk(row.get("issue", "")),
                "spoc_email": row.get("spoc_email", ""),
                "spoc_boss_email": row.get("spoc_boss_email", ""),
            }
            upsert_case(case)
        st.success("Uploaded cases ingested successfully.")

    st.header("✏️ Manual Entry")
    with st.form("manual_entry"):
        cname = st.text_input("Customer")
        issue = st.text_area("Issue")
        crit = st.selectbox("Criticality", ["Low", "Medium", "High"], index=1)
        imp = st.selectbox("Impact", ["Low", "Medium", "High"], index=1)
        owner = st.text_input("Owner", value="Unassigned")
        spoc = st.text_input("SPOC Email")
        boss = st.text_input("Boss Email")
        if st.form_submit_button("Log Escalation"):
            sentiment, urgency, esc = analyze_issue(issue)
            case = {
                "id": f"ESC{int(datetime.utcnow().timestamp()*1000)}",
                "customer": cname,
                "issue": issue,
                "criticality": crit,
                "impact": imp,
                "sentiment": sentiment,
                "urgency": urgency,
                "escalated": int(esc),
                "date_reported": str(datetime.today().date()),
                "owner": owner,
                "status": "Open",
                "action_taken": "",
                "risk_score": predict_risk(issue),
                "spoc_email": spoc,
                "spoc_boss_email": boss,
            }
            upsert_case(case)
            st.success(f"Escalation {case['id']} logged")

# Main Kanban & Filters

df = fetch_cases()
show_filter = st.radio("📌 Show", ["All", "Only Escalated"], horizontal=True)
if show_filter == "Only Escalated":
    df = df[df.escalated == 1]

status_summary = df.status.value_counts().to_dict()
st.markdown(
    f"### 🔢 Summary: "
    f"Open: {status_summary.get('Open',0)}, "
    f"In Progress: {status_summary.get('In Progress',0)}, "
    f"Resolved: {status_summary.get('Resolved',0)}"
)

cols = st.columns(3)
for status, col in zip(["Open", "In Progress", "Resolved"], cols):
    subset = df[df.status == status]
    with col:
        st.markdown(f"### {status}")
        if subset.empty:
            st.caption("— No cases —")
        for _, row in subset.iterrows():
            icon = "🔺" if row.escalated else ""
            with st.expander(f"{icon} {row['id']} – {row['issue'][:50]}…"):
                st.markdown(f"**Customer:** {row['customer']}")
                st.markdown(f"**Sentiment / Urgency:** {row['sentiment']} / {row['urgency']}")
                st.markdown(f"**Owner:** {row['owner']}")
                st.markdown(f"**Risk Score:** {row['risk_score']:.2f}")

                new_status = st.selectbox(
                    "Status",
                    ["Open", "In Progress", "Resolved"],
                    index=["Open", "In Progress", "Resolved"].index(row.status),
                    key=f"st_{row.id}",
                )
                new_action = st.text_area("Action Taken", value=row.action_taken or "", key=f"act_{row.id}")
                new_spoc = st.text_input("SPOC Email", value=row.spoc_email or "", key=f"spoc_{row.id}")
                new_boss = st.text_input("Boss Email", value=row.spoc_boss_email or "", key=f"boss_{row.id}")

                if st.button("Notify SPOC", key=f"notify_{row.id}") and new_spoc:
                    subj = f"Escalation {row.id} Notification"
                    body = (
                        f"Dear SPOC,\n\nPlease attend to escalation {row.id}.\n\nIssue (excerpt):\n{row.issue[:400]}…"
                    )
                    if send_email(new_spoc, subj, body, row.id):
                        updated = row.to_dict()
                        updated["spoc_notify_count"] = (row.get("spoc_notify_count") or 0) + 1
                        updated["spoc_last_notified"] = datetime.utcnow().isoformat()
                        updated["spoc_email"] = new_spoc
                        updated["spoc_boss_email"] = new_boss
                        updated["status"] = new_status
                        updated["action_taken"] = new_action
                        upsert_case(updated)
                        st.success("Notification sent & saved")
                        st.experimental_rerun()
                elif any([
                    new_status != row.status,
                    new_action != row.action_taken,
                    new_spoc != row.spoc_email,
                    new_boss != row.spoc_boss_email,
                ]):
                    updated = row.to_dict()
                    updated.update({
                        "status": new_status,
                        "action_taken": new_action,
                        "spoc_email": new_spoc,
                        "spoc_boss_email": new_boss,
                    })
                    upsert_case(updated)
                    st.success("Changes saved")

# Logs & Export
st.subheader("⬇️ Download Escalation Data")
excel_data = df_to_excel_bytes(fetch_cases())
st.download_button(
    "Download All as Excel",
    data=excel_data,
    file_name="escalations_export.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("📜 Notification History"):
    logs_df = fetch_logs()
    st.dataframe(logs_df, use_container_width=True)
