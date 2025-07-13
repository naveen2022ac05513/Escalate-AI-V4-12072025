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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            internal_notes TEXT DEFAULT ""
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

            if r["spoc_notify_count"] < 2 and hours_open > (r["spoc_notify_count"] + 1) * 6:
                notify_spoc(r["id"], r["spoc_email"])

            elif r["spoc_notify_count"] >= 2 and hours_open > 24 and not r["escalated"]:
                notify_spoc(r["id"], r["spoc_manager_email"])
                upsert_case({**r.to_dict(), "escalated": 1})
        except Exception as exc:  # noqa: BLE001
            print("[monitor_reminders]", exc)


sched = BackgroundScheduler()
sched.add_job(monitor_reminders, "interval", hours=1)
sched.start()
atexit.register(sched.shutdown)

# ────────────────────────────────────────────────────────────────────────────────
# UI DASHBOARD (Streamlit)
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config("EscalateAI Dashboard", layout="wide")
st.title("🚨 EscalateAI – Escalation Management")

# Load Data
df = fetch_cases()

# Filters and Search
with st.sidebar:
    st.header("🔍 Filters")
    status_filter = st.multiselect("Status", options=df["status"].unique().tolist())
    owner_filter = st.multiselect("Owner", options=df["owner"].dropna().unique().tolist())
    keyword = st.text_input("Search by keyword (ID/Customer/Issue)")

    df_filtered = df.copy()
    if status_filter:
        df_filtered = df_filtered[df_filtered["status"].isin(status_filter)]
    if owner_filter:
        df_filtered = df_filtered[df_filtered["owner"].isin(owner_filter)]
    if keyword:
        df_filtered = df_filtered[df_filtered.apply(
            lambda row: keyword.lower() in str(row["id"]).lower()
                        or keyword.lower() in str(row["customer"]).lower()
                        or keyword.lower() in str(row["issue"]).lower(),
            axis=1)]

    if st.button("Export to CSV"):
        st.download_button("Download", data=df_filtered.to_csv(index=False),
                           file_name="escalations.csv", mime="text/csv")

# Display Summary
status_counts = df_filtered["status"].value_counts().to_dict()
st.markdown("**Summary**")
st.write(" | ".join(f"{k}: {v}" for k, v in status_counts.items()))

# Display Escalations
for _, row in df_filtered.iterrows():
    highlight = "🔴 " if row["sentiment"] == "Negative" and row["urgency"] == "High" else ""
    with st.expander(f"{highlight}{row['id']} – {row['customer']}", expanded=False):
        st.write(row["issue"])
        st.text(f"Sentiment/Urgency: {row['sentiment']} / {row['urgency']}")
        st.text(f"Risk Score: {row['risk_score']} | Status: {row['status']} | Owner: {row['owner']}")
        st.text_area("Action Taken", value=row["action_taken"] or "", key=f"action-{row['id']}")
        st.text_area("Internal Notes", value=row["internal_notes"] or "", key=f"notes-{row['id']}")
        if st.button("Notify SPOC", key=f"notify-{row['id']}"):
            notify_spoc(row["id"], row["spoc_email"])
            st.success("SPOC Notified!")
        if st.button("Save Changes", key=f"save-{row['id']}"):
            upsert_case({
                "id": row["id"],
                "action_taken": st.session_state[f"action-{row['id']}"][:500],
                "internal_notes": st.session_state[f"notes-{row['id']}"][:1000]
            })
            st.success("Saved.")
