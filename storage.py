"""
storage.py
-----------
Investment Journal persistence. Backend is SQLite for zero-setup simplicity.

IMPORTANT CAVEAT (read this if deploying to Streamlit Community Cloud):
Streamlit Cloud's filesystem is ephemeral across redeploys/reboots of the
app container. A local SQLite file can survive routine reruns within the
same running instance, but is NOT guaranteed to survive a redeploy, app
sleep/wake cycle, or platform migration. Since the Journal is explicitly
the most valuable feature here, this module also provides one-click JSON
export/import so you always have a portable backup outside the database.
Recommended habit: click "Export Journal Backup" after adding notes, and
keep the JSON file somewhere safe (or commit it to your private repo).

If you later want true durable cloud storage, swap this module for a
Google Sheets or Supabase-backed version — the function signatures below
(save_entry, get_entry, get_all_entries) are the only surface app.py talks
to, so the rest of the app won't need to change.
"""

import json
import sqlite3
from datetime import datetime, timezone

DB_PATH = "journal.db"

FIELDS = [
    "reasons_owned",
    "competitive_advantages",
    "long_term_thesis",
    "biggest_risks",
    "fair_value_estimate",
    "ideal_buy_price",
    "ideal_csp_strike",
    "min_covered_call_strike",
    "sell_triggers",
]


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS journal (
            ticker TEXT PRIMARY KEY,
            reasons_owned TEXT,
            competitive_advantages TEXT,
            long_term_thesis TEXT,
            biggest_risks TEXT,
            fair_value_estimate TEXT,
            ideal_buy_price TEXT,
            ideal_csp_strike TEXT,
            min_covered_call_strike TEXT,
            sell_triggers TEXT,
            updated_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS iv_history (
            ticker TEXT,
            snapshot_date TEXT,
            atm_iv REAL,
            PRIMARY KEY (ticker, snapshot_date)
        )
        """
    )
    return conn


def log_iv_snapshot(ticker: str, atm_iv: float):
    """Records today's ATM IV for this ticker (one row per ticker per day).
    Building this up over time is what lets IV Rank/Percentile become a real
    calculation instead of the historical-volatility-based approximation."""
    if atm_iv is None:
        return
    conn = _get_conn()
    today = datetime.now(timezone.utc).date().isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO iv_history (ticker, snapshot_date, atm_iv) VALUES (?, ?, ?)",
        (ticker.upper(), today, float(atm_iv)),
    )
    conn.commit()
    conn.close()


def get_iv_history(ticker: str):
    """Returns a list of (date_str, atm_iv) tuples, oldest first."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT snapshot_date, atm_iv FROM iv_history WHERE ticker = ? ORDER BY snapshot_date ASC",
        (ticker.upper(),),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def save_entry(ticker: str, values: dict):
    conn = _get_conn()
    values = {k: values.get(k, "") for k in FIELDS}
    values["ticker"] = ticker.upper()
    values["updated_at"] = datetime.now(timezone.utc).isoformat()
    placeholders = ", ".join(["?"] * (len(FIELDS) + 2))
    columns = ", ".join(["ticker"] + FIELDS + ["updated_at"])
    conn.execute(
        f"INSERT OR REPLACE INTO journal ({columns}) VALUES ({placeholders})",
        [values["ticker"]] + [values[f] for f in FIELDS] + [values["updated_at"]],
    )
    conn.commit()
    conn.close()


def get_entry(ticker: str) -> dict:
    conn = _get_conn()
    cur = conn.execute("SELECT * FROM journal WHERE ticker = ?", (ticker.upper(),))
    row = cur.fetchone()
    col_names = [d[0] for d in cur.description]
    conn.close()
    if row is None:
        return {f: "" for f in FIELDS}
    return dict(zip(col_names, row))


def get_all_entries() -> list:
    conn = _get_conn()
    cur = conn.execute("SELECT * FROM journal")
    col_names = [d[0] for d in cur.description]
    rows = [dict(zip(col_names, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def export_backup_json() -> str:
    """Returns a JSON string of every journal entry, for download."""
    return json.dumps(get_all_entries(), indent=2)


def import_backup_json(json_text: str):
    """Restores journal entries from a previously exported JSON backup."""
    entries = json.loads(json_text)
    for entry in entries:
        ticker = entry.get("ticker")
        if not ticker:
            continue
        save_entry(ticker, entry)
