# streamlit_app.py
import os, re, smtplib, requests, atexit, io
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import sqlalchemy
import msal
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Integer,
    DateTime, Text, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ── Env & Paths ─────────────────────────────────────────────────────────────
load_dotenv()
BASE_DIR   = Path(__file__).parent
DB_URL     = (
    f"mssql+pyodbc://{os.getenv('AZ_SQL_USER')}:{os.getenv('AZ_SQL_PASS')}@"
    f"{os.getenv('AZ_SQL_SERVER')}/{os.getenv('AZ_SQL_DB')}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)
RISK_URL   = os.getenv("RISK_API_URL")

SMTP_SRV   = os.getenv("SMTP_SERVER")
SMTP_PORT  = int(os.getenv("SMTP_PORT", 587))
SMTP_USER  = os.getenv("SMTP_USER")
SMTP_PASS  = os.getenv("SMTP_PASS")

MS_CLIENT_ID     = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
MS_TENANT_ID     = os.getenv("MS_TENANT_ID")
AUTHORITY_URL    = f"https://login.microsoftonline.com/{MS_TENANT_ID}"
SCOPE            = ["User.Read"]  # Graph scope if needed

# ── Database Setup (SQLAlchemy) ──────────────────────────────────────────────
Base = declarative_base()
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

class SPOC(Base):
    __tablename__ = "spoc_directory"
    spoc_email       = Column(String, primary_key=True)
    spoc_name        = Column(String)
    spoc_manager_email = Column(String)
    teams_webhook    = Column(String)

class Escalation(Base):
    __tablename__ = "escalations"
    id               = Column(String, primary_key=True)
    customer         = Column(String)
    issue            = Column(Text)
    sentiment        = Column(String)
    urgency          = Column(String)
    priority         = Column(String)
    status           = Column(String, default="Open")
    date_reported    = Column(DateTime, default=func.now())
    spoc_email       = Column(String)
    spoc_boss_email  = Column(String)
    reminders_sent   = Column(Integer, default=0)
    escalated        = Column(Integer, default=0)
    created_at       = Column(DateTime, default=func.now())

class NotificationLog(Base):
    __tablename__ = "notification_log"
    id               = Column(Integer, primary_key=True, autoincrement=True)
    escalation_id    = Column(String)
    recipient_email  = Column(String)
    subject          = Column(String)
    body             = Column(Text)
    sent_at          = Column(DateTime, default=func.now())

class Feedback(Base):
    __tablename__ = "feedback"
    id               = Column(Integer, primary_key=True, autoincrement=True)
    escalation_id    = Column(String)
    rating           = Column(Integer)
    comments         = Column(Text)
    submitted_at     = Column(DateTime, default=func.now())

Base.metadata.create_all(engine)

# ── Azure AD Device-Code Login ───────────────────────────────────────────────
def azure_ad_login():
    app = msal.PublicClientApplication(
        MS_CLIENT_ID,
        authority=AUTHORITY_URL
    )
    flow = app.initiate_device_flow(scopes=SCOPE)
    st.info(flow["message"])
    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        st.error("Azure AD login failed")
        st.stop()
    return result["id_token_claims"]["preferred_username"]

# ── Sentiment/Urgency Analysis ──────────────────────────────────────────────
NEG_PAT = re.compile(
    r"\b(problematic|delay|issue|failure|complaint|unresolved|unresponsive|broken|defect|critical|risk)\b",
    re.IGNORECASE
)

def analyze_issue(text: str):
    sent = "Negative" if NEG_PAT.search(text) else "Positive"
    urg  = "High" if any(k in text.lower() for k in ["urgent","immediate","critical"]) else "Low"
    return sent, urg, (sent=="Negative" and urg=="High")

def analyze_priority(email: str):
    return "High" if not email.endswith(".se.com") else "Low"

def predict_risk(text: str):
    try:
        r = requests.post(RISK_URL, json={"text": text}, timeout=5)
        return r.json().get("risk_score", 0.0)
    except:
        return 0.0

# ── Notifications ────────────────────────────────────────────────────────────
def send_email(to, sub, body, esc_id):
    msg = MIMEText(body)
    msg["Subject"], msg["From"], msg["To"] = sub, SMTP_USER, to
    with smtplib.SMTP(SMTP_SRV, SMTP_PORT, timeout=10) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
    # log
    db = SessionLocal()
    db.add(NotificationLog(
        escalation_id=esc_id,
        recipient_email=to,
        subject=sub,
        body=body
    ))
    db.commit()
    db.close()

def send_teams(webhook, msg):
    if webhook:
        requests.post(webhook, json={"text": msg})

def notify_spoc(spoc_email, esc_id, level="Initial"):
    db  = SessionLocal()
    sp  = db.query(SPOC).get(spoc_email)
    webhook = sp.teams_webhook if sp else None
    subj    = f"{level} Notification: {esc_id}"
    body    = f"{level} notification for escalation {esc_id}"
    send_email(spoc_email, subj, body, esc_id)
    send_teams(webhook, body)
    db.close()

# ── Outlook Ingestion (Microsoft Graph) ──────────────────────────────────────
def get_graph_token():
    app = msal.ConfidentialClientApplication(
        MS_CLIENT_ID, client_credential=MS_CLIENT_SECRET, authority=AUTHORITY_URL
    )
    res = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    return res.get("access_token", "")

def ingest_outlook():
    token = get_graph_token()
    if not token: return
    hdr = {"Authorization": f"Bearer {token}"}
    url = "https://graph.microsoft.com/v1.0/me/mailFolders/Inbox/messages?$top=20&$filter=isRead eq false"
    r   = requests.get(url, headers=hdr).json().get("value", [])
    db  = SessionLocal()
    existing = {e.id for e in db.query(Escalation).all()}
    for m in r:
        mid   = m["id"]
        if mid in existing: continue
        subj  = m.get("subject","")
        body  = m.get("bodyPreview","")
        sender= m["from"]["emailAddress"]["address"]
        sent, urg, esc_flag = analyze_issue(subj+" "+body)
        prio  = analyze_priority(sender)
        # pick default SPOC
        spoc  = db.query(SPOC).first()
        case = Escalation(
            id=mid,
            customer=sender,
            issue=subj+"\n"+body,
            sentiment=sent,
            urgency=urg,
            priority=prio,
            status="Open",
            date_reported=datetime.fromisoformat(m["receivedDateTime"].replace("Z","+00:00")),
            spoc_email=spoc.spoc_email if spoc else "",
            spoc_boss_email=spoc.spoc_manager_email if spoc else "",
            escalated=int(esc_flag)
        )
        db.add(case)
        db.commit()
        notify_spoc(case.spoc_email, case.id, "Initial")
        # mark read
        requests.patch(f"https://graph.microsoft.com/v1.0/me/messages/{mid}", headers=hdr, json={"isRead": True})
    db.close()

# ── Reminder Scheduler ────────────────────────────────────────────────────────
def monitor_reminders():
    db  = SessionLocal()
    now = datetime.utcnow()
    for e in db.query(Escalation).filter(Escalation.status=="Open"):
        hours = (now - e.date_reported).total_seconds()/3600
        if e.reminders_sent < 2 and hours > (e.reminders_sent+1)*6:
            notify_spoc(e.spoc_email, e.id, f"Reminder {e.reminders_sent+1}")
            e.reminders_sent += 1
        elif hours>24 and e.reminders_sent>=2 and e.escalated==0:
            send_email(e.spoc_boss_email, f"⚠️ Escalation {e.id} Unattended",
                       f"Escalation {e.id} requires your attention.", e.id)
            e.escalated = 1
        db.commit()
    db.close()

sched = BackgroundScheduler()
sched.add_job(ingest_outlook, "interval", minutes=30)
sched.add_job(monitor_reminders, "interval", hours=1)
sched.start()
atexit.register(lambda: sched.shutdown(wait=False))

# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="EscalateAI v4.0", layout="wide")
st.title("🚨 EscalateAI v4.0")

# 1) Azure AD Login
if "user" not in st.session_state:
    user = azure_ad_login()
    st.session_state["user"] = user
st.sidebar.markdown(f"**Logged in as:** {st.session_state['user']}")

# 2) Admin: SPOC Directory
with st.sidebar.expander("📋 SPOC Directory", expanded=True):
    spoc_file = st.file_uploader("Upload SPOC Excel", type="xlsx")
    if spoc_file and st.button("Ingest SPOC"):
        df = pd.read_excel(spoc_file)
        db = SessionLocal()
        for _, r in df.iterrows():
            db.merge(SPOC(
                spoc_email=r.spoc_email,
                spoc_name=r.spoc_name,
                spoc_manager_email=r.spoc_manager_email,
                teams_webhook=r.teams_webhook
            ))
        db.commit(); db.close()
        st.success("SPOC directory updated.")

# 3) Ingest via File or Manual
with st.sidebar.expander("📥 Ingest Escalations", expanded=True):
    f = st.file_uploader("Excel/CSV", type=["xlsx","csv"])
    if f and st.button("Ingest File"):
        df = (pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f))
        db = SessionLocal()
        for _, r in df.iterrows():
            sent, urg, escf = analyze_issue(str(r.issue))
            rec = Escalation(
                id=str(r.get("id", f"ESC{int(datetime.utcnow().timestamp())}")),
                customer=r.get("customer","Unknown"),
                issue=r.issue,
                sentiment=sent,
                urgency=urg,
                priority=analyze_priority(r.get("customer_email","")),
                status=r.get("status","Open"),
                date_reported=r.get("date_reported", datetime.utcnow()),
                spoc_email=r.get("spoc_email",""),
                spoc_boss_email=r.get("spoc_boss_email",""),
                escalated=int(escf)
            )
            db.merge(rec); db.commit()
            notify_spoc(rec.spoc_email, rec.id, "Initial")
        db.close()
        st.success("Escalations ingested.")

    with st.form("manual"):
        cname = st.text_input("Customer")
        issue = st.text_area("Issue")
        spoc  = st.text_input("SPOC Email")
        boss  = st.text_input("Boss Email")
        if st.form_submit_button("Log Escalation"):
            sent, urg, escf = analyze_issue(issue)
            rec = Escalation(
                id=f"ESC{int(datetime.utcnow().timestamp())}",
                customer=cname,
                issue=issue,
                sentiment=sent,
                urgency=urg,
                priority=analyze_priority(""),
                status="Open",
                date_reported=datetime.utcnow(),
                spoc_email=spoc,
                spoc_boss_email=boss,
                escalated=int(escf)
            )
            db = SessionLocal(); db.add(rec); db.commit(); db.close()
            notify_spoc(spoc, rec.id, "Initial")
            st.success(f"Logged {rec.id}")

# 4) Main Board + Filter + Download
show_escalated = st.checkbox("🔍 Show only escalated cases")
df = pd.read_sql_table("escalations", con=engine)
if show_escalated:
    df = df[df.escalated == 1]

cols = st.columns(4)
for idx, status in enumerate(["Open","In Progress","Resolved","Closed"]):
    with cols[idx]:
        st.subheader(status)
        for _, r in df[df.status==status].iterrows():
            with st.expander(f"{r.id} — {r.customer}", expanded=False):
                st.write(r.issue)
                st.markdown(f"**Sent:** {r.sentiment} | **Urg:** {r.urgency} | **Prio:** {r.priority}")
                new = st.selectbox("Status", ["Open","In Progress","Resolved","Closed"],
                                   index=["Open","In Progress","Resolved","Closed"].index(r.status), key=r.id)
                if new != r.status:
                    db = SessionLocal()
                    rec= db.query(Escalation).get(r.id)
                    rec.status=new
                    db.commit(); db.close()
                    st.experimental_rerun()

# Feedback on resolved
st.markdown("## 📝 Feedback")
fb_esc = st.selectbox("Select Resolved Escalation", df[df.status=="Resolved"]["id"].tolist())
if fb_esc:
    rating  = st.slider("Rate resolution", 1, 5)
    comments= st.text_area("Comments")
    if st.button("Submit Feedback"):
        db = SessionLocal()
        db.add(Feedback(escalation_id=fb_esc, rating=rating, comments=comments))
        db.commit(); db.close()
        st.success("Thanks for your feedback!")

# Download Excel
buf = io.BytesIO()
df.to_excel(buf, index=False, sheet_name="Escalations")
buf.seek(0)
st.download_button("📥 Download Board as Excel", buf,
                   file_name="escalations_board.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
