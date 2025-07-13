import os
import re
import sqlite3
import smtplib
import requests
import atexit
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

# ────────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()
DB_PATH = Path("data/escalateai.db")
DB_PATH.parent.mkdir(exist_ok=True)

SMTP_SERVER = os.getenv("SMTP_SERVER") or "smtp.example.com"
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "no‑reply@example.com")
SMTP_PASS = os.getenv("SMTP_PASS", "changeme")

# ────────────────────────────────────────────────────────────────────────────────
# DATABASE INITIALISATION
# ────────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they do not yet exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
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
            escalated INTEGER DEFAULT 0,
            date_reported TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS spoc_directory (
            spoc_email TEXT PRIMARY KEY,
            spoc_name TEXT,
            spoc_manager_email TEXT,
            teams_webhook TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()

# ────────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────

def get_columns() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(escalations)")]
    conn.close()
    return cols


def analyze_issue(text: str) -> tuple[str, str, bool]:
    """Return (sentiment, urgency, needs_escalation)."""
    negative_pat = r"(dissatisfaction|leakage|failure|issue|critical)"
    urgent_pat = r"(urgent|immediate|impact)"
    sentiment = "Negative" if re.search(negative_pat, text, re.I) else "Positive"
    urgency = "High" if re.search(urgent_pat, text, re.I) else "Low"
    return sentiment, urgency, sentiment == "Negative" and urgency == "High"


def predict_risk(text: str) -> float:
    """A simple heuristic risk score between 0 and 1."""
    return round(min(len(text.split()) / 50, 1.0), 2)


def upsert_case(case: dict) -> None:
    """Insert or update an escalation."""
    cols = get_columns()
    clean = {k: case[k] for k in case if k in cols}
    keys = ",".join(clean.keys())
    qms = ",".join(["?"] * len(clean))
    upd = ",".join(f"{k}=excluded.{k}" for k in clean if k != "id")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"""
            INSERT INTO escalations ({keys}) VALUES ({qms})
            ON CONFLICT(id) DO UPDATE SET {upd}
            """,
            tuple(clean.values()),
        )


def fetch_cases() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    schema = pd.read_sql("PRAGMA table_info(escalations)", conn)
    cols = schema["name"].tolist()
    order = " ORDER BY datetime(created_at) DESC" if "created_at" in cols else ""
    df = pd.read_sql("SELECT * FROM escalations" + order, conn)
    conn.close()
    return df


def notify_spoc(esc_id: str, email: str) -> None:
    if not email:
        return  # Nothing to notify

    df = pd.read_sql(
        "SELECT teams_webhook FROM spoc_directory WHERE spoc_email=?",
        sqlite3.connect(DB_PATH),
        params=(email,),
    )
    webhook = df.teams_webhook.iloc[0] if not df.empty else None

    try:
        body = f"🔔 Notification for escalation {esc_id}"
        msg = MIMEText(body)
        msg["From"], msg["To"], msg["Subject"] = SMTP_USER, email, "Escalation Alert"
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)

        if webhook:
            requests.post(webhook, json={"text": body}, timeout=5)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                UPDATE escalations SET
                  spoc_notify_count = spoc_notify_count + 1,
                  spoc_last_notified = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), esc_id),
            )
    except Exception as exc:  # noqa: BLE001
        # Log exception for debug; in production consider using proper logging
        print("[notify_spoc]", exc)


# ────────────────────────────────────────────────────────────────────────────────
# BACKGROUND REMINDER SCHEDULER
# ────────────────────────────────────────────────────────────────────────────────

def monitor_reminders() -> None:
    df = fetch_cases()
    now = datetime.utcnow()
    for _, r in df[df["status"].str.lower() == "open"].iterrows():
        try:
            reported = datetime.fromisoformat(r["date_reported"])
            hours_open = (now - reported).total_seconds() / 3600

            # Notify SPOC every 6 h (max twice) and manager after 24 h
            if r["spoc_notify_count"] < 2 and hours_open > (r["spoc_notify_count"] + 1) * 6:
                notify_spoc(r["id"], r["spoc_email"])

            elif (
                r["spoc_notify_count"] >= 2
                and hours_open > 24
                and not r["escalated"]
            ):
                notify_spoc(r["id"], r["spoc_manager_email"])
                upsert_case({**r.to_dict(), "escalated": 1})
        except Exception as exc:  # noqa: BLE001
            print("[monitor_reminders]", exc)


sched = BackgroundScheduler()
sched.add_job(monitor_reminders, "interval", hours=1)
sched.start()
atexit.register(sched.shutdown)

# ────────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config("EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracker")

# ───────────── Sidebar: Upload & Manual Entry ─────────────
with st.sidebar:
    st.header("📥 Upload Escalations")
    uploaded = st.file_uploader("Excel/CSV", type=["xlsx", "csv"])
    if uploaded and st.button("Ingest File"):
        df_u = (
            pd.read_excel(uploaded)
            if uploaded.name.lower().endswith(".xlsx")
            else pd.read_csv(uploaded)
        )

        base_id = int(datetime.utcnow().timestamp() * 1000)
        for idx, row in df_u.iterrows():
            issue = str(row.get("Brief Issue", "") or row.get("issue", ""))
            sentiment, urgency, _ = analyze_issue(issue)

            case = {
                "id": f"ESC{base_id + idx}",
                "customer": row.get("Customer", "Unknown"),
                "issue": issue,
                "sentiment": sentiment,
                "urgency": urgency,
                "risk_score": predict_risk(issue),
                "status": str(row.get("Status", "Open")).strip().title(),
                "action_taken": row.get("Action taken", ""),
                "owner": row.get("Owner", ""),
                "spoc_email": row.get("SPOC Email", ""),
                "spoc_manager_email": row.get("Manager Email", ""),
                "spoc_notify_count": 0,
                "spoc_last_notified": "",
                "escalated": 0,
                "date_reported": str(
                    row.get("Issue reported date", datetime.utcnow().isoformat())
                ),
            }
            upsert_case(case)
        st.success("Escalations ingested.")

    st.header("✏️ Manual Entry")
    with st.form("manual-entry"):
        cname = st.text_input("Customer")
        issue = st.text_area("Issue")
        owner = st.text_input("Owner", "Unassigned")
        spoc = st.text_input("SPOC Email")
        mgr = st.text_input("Manager Email")
        if st.form_submit_button("Log"):
            sentiment, urgency, _ = analyze_issue(issue)
            esc_id = f"ESC{int(datetime.utcnow().timestamp() * 1000)}"
            case = {
                "id": esc_id,
                "customer": cname,
                "issue": issue,
                "sentiment": sentiment,
                "urgency": urgency,
                "risk_score": predict_risk(issue),
                "status": "Open",
                "action_taken": "",
                "owner": owner,
                "spoc_email": spoc,
                "spoc_manager_email": mgr,
                "spoc_notify_count": 0,
                "spoc_last_notified": "",
                "escalated": 0,
                "date_reported": datetime.utcnow().isoformat(),
            }
            upsert_case(case)
            notify_spoc(esc_id, spoc)
            st.success(f"Escalation {esc_id} logged.")

# ───────────── Kanban Board ─────────────
df = fetch_cases()
if df.empty:
    st.info("No escalations logged yet.")
else:
    df["status"] = df["status"].fillna("Open").astype(str).str.strip().str.title()

    counts = df["status"].value_counts().to_dict()
    emojis = {"Open": "🟥", "In Progress": "🟧", "Resolved": "🟩"}
    summary = " | ".join(
        f"{emojis.get(s, '')} {s}: {counts.get(s, 0)}" for s in ["Open", "In Progress", "Resolved"]
    )

    st.markdown(f"### {summary}")

    # Sort by status then risk score (descending)
    for _, r in (
        df.sort_values(["status", "risk_score"], ascending=[True, False]).iterrows()
    ):
        with st.expander(f"{r['id']} – {r['customer']}", expanded=False):
            st.write(r.get("issue", "No issue provided"))
            st.markdown(
                f"**Sentiment / Urgency:** {r.get('sentiment', '–')} / {r.get('urgency', '–')}  \n"
                f"**Owner:** {r.get('owner', 'Unassigned')}  \n"
                f"**Risk Score:** {float(r.get('risk_score', 0) or 0):.2f}  \n"
                f"**Status:** {r.get('status', '–')}  \n"
                f"**Action Taken:** {r.get('action_taken', '')}  \n"
                f"**SPOC Email:** {r.get('spoc_email', '–')}  \n"
                f"**Manager Email:** {r.get('spoc_manager_email', '–')}"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Notify SPOC", key=f"notify-{r['id']}"):
                    notify_spoc(r["id"], r["spoc_email"])
                    st.success("SPOC notified.")
            with col2:
                new_status = st.selectbox(
                    "Update status:",
                    ["Open", "In Progress", "Resolved"],
                    index=["Open", "In Progress", "Resolved"].index(r["status"]),
                    key=f"status-{r['id']}",
                )
                if new_status != r["status"]:
                    upsert_case({"id": r["id"], "status": new_status})
                    st.experimental_rerun()

# ────────────────────────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────────────────────────

st.caption(
    "© 2025 EscalateAI  |  Built with Streamlit  |  Sends automated notifications every 6 h"
)
