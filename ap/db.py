"""SQLite state: PO ledger, invoice history and the run log.

Reference data (vendors, POs, GRNs) is read-only CSV, the source of truth.
The ledger and history change as invoices are approved; reset() reseeds them.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from functools import lru_cache

from .config import DATA_DIR, DB_PATH, SAMPLES_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS po_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    po_number TEXT, line_no INTEGER, qty REAL, amount REAL,
    vendor_id TEXT, invoice_number TEXT, run_id TEXT, posted_at TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS invoice_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor_id TEXT, invoice_number TEXT, invoice_number_norm TEXT, invoice_date TEXT,
    subtotal REAL, total REAL, file_hash TEXT, po_number TEXT, run_id TEXT,
    processed_at TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, started_at TEXT, duration_ms REAL,
    file_name TEXT, file_hash TEXT, file_kind TEXT, engine TEXT,
    vendor_name TEXT, vendor_id TEXT, invoice_number TEXT, invoice_date TEXT,
    total REAL, po_number TEXT, outcome TEXT, with_note INTEGER, priority TEXT,
    codes TEXT, owners TEXT, headline TEXT, stages TEXT, decision TEXT
);
"""


def norm_invoice_no(s: str | None) -> str:
    return re.sub(r"[^0-9A-Z]", "", (s or "").upper())


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    fresh = not DB_PATH.exists()
    with conn() as c:
        c.executescript(SCHEMA)
    if fresh:
        reset()


def reset() -> None:
    """Reseed ledger + history and clear the run log, so every demo starts identically."""
    with conn() as c:
        c.executescript("DROP TABLE IF EXISTS po_ledger; DROP TABLE IF EXISTS invoice_history; DROP TABLE IF EXISTS runs;")
        c.executescript(SCHEMA)
        with open(DATA_DIR / "seed_invoices.csv") as f:
            for r in csv.DictReader(f):
                fh = ""
                if r["sample_file"] and (SAMPLES_DIR / r["sample_file"]).exists():
                    fh = hashlib.sha256((SAMPLES_DIR / r["sample_file"]).read_bytes()).hexdigest()
                c.execute(
                    "INSERT INTO invoice_history (vendor_id, invoice_number, invoice_number_norm, invoice_date, subtotal,"
                    " total, file_hash, po_number, run_id, processed_at, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (r["vendor_id"], r["invoice_number"], norm_invoice_no(r["invoice_number"]), r["invoice_date"],
                     float(r["subtotal"]), float(r["total"]), fh, r["po_number"], None, r["processed_at"], "seed"))
                for ln in json.loads(r["lines"]):
                    c.execute(
                        "INSERT INTO po_ledger (po_number, line_no, qty, amount, vendor_id, invoice_number, run_id,"
                        " posted_at, source) VALUES (?,?,?,?,?,?,?,?,?)",
                        (r["po_number"], ln["line_no"], ln["qty"], ln["amount"], r["vendor_id"], r["invoice_number"],
                         None, r["processed_at"], "seed"))


# ---------------- reference data (read-only CSV) ----------------
def _read(name):
    with open(DATA_DIR / name) as f:
        return list(csv.DictReader(f))


@lru_cache
def vendors() -> list[dict]:
    return _read("vendors.csv")


@lru_cache
def po_lines() -> list[dict]:
    rows = _read("po_register.csv")
    for r in rows:
        r["line_no"] = int(r["line_no"])
        r["qty"] = float(r["qty"])
        r["unit_price"] = float(r["unit_price"])
        r["gst_rate"] = float(r["gst_rate"])
    return rows


def pos() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in po_lines():
        po = out.setdefault(r["po_number"], dict(po_number=r["po_number"], vendor_id=r["vendor_id"], po_date=r["po_date"],
                                                 status=r["status"], requester=r["requester"], lines=[]))
        po["lines"].append(r)
    return out


@lru_cache
def grns() -> list[dict]:
    rows = _read("grn.csv")
    for r in rows:
        r["line_no"] = int(r["line_no"])
        r["qty_received"] = float(r["qty_received"])
    return rows


def received(po_number: str, line_no: int) -> tuple[float, list[dict]]:
    rows = [g for g in grns() if g["po_number"] == po_number and g["line_no"] == line_no]
    return sum(g["qty_received"] for g in rows), rows


# ---------------- ledger ----------------
def invoiced(po_number: str, line_no: int | None = None) -> dict:
    q = "SELECT COALESCE(SUM(qty),0) qty, COALESCE(SUM(amount),0) amount, GROUP_CONCAT(DISTINCT invoice_number) invs" \
        " FROM po_ledger WHERE po_number=?"
    args: list = [po_number]
    if line_no is not None:
        q += " AND line_no=?"
        args.append(line_no)
    with conn() as c:
        r = c.execute(q, args).fetchone()
    return dict(qty=r["qty"], amount=r["amount"], invoices=[i for i in (r["invs"] or "").split(",") if i])


def ledger_rows() -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM po_ledger ORDER BY id")]


def history() -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM invoice_history ORDER BY id")]


def post_approval(run_id: str, vendor_id: str, inv, file_hash: str, po_number: str, postings: list[dict]) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with conn() as c:
        for p in postings:
            c.execute("INSERT INTO po_ledger (po_number, line_no, qty, amount, vendor_id, invoice_number, run_id,"
                      " posted_at, source) VALUES (?,?,?,?,?,?,?,?,?)",
                      (po_number, p["line_no"], p["qty"], p["amount"], vendor_id, inv.invoice_number, run_id, now, "run"))
        c.execute("INSERT INTO invoice_history (vendor_id, invoice_number, invoice_number_norm, invoice_date, subtotal,"
                  " total, file_hash, po_number, run_id, processed_at, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (vendor_id, inv.invoice_number, norm_invoice_no(inv.invoice_number),
                   inv.invoice_date.isoformat() if inv.invoice_date else None, inv.subtotal, inv.total, file_hash,
                   po_number, run_id, now, "run"))


# ---------------- run log ----------------
def save_run(rec: dict) -> None:
    cols = ",".join(rec.keys())
    qs = ",".join("?" for _ in rec)
    with conn() as c:
        c.execute(f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({qs})", list(rec.values()))


def runs() -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY started_at DESC, rowid DESC")]


def run(run_id: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(r) if r else None
