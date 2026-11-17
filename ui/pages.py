"""The four pages: process, dashboard, reference data, how it works."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ap import db
from ap.config import (BUYER, DATA_DIR, DUP_AMOUNT_PCT, DUP_WINDOW_DAYS, LOW_CONFIDENCE, PO_INFERENCE_MIN_CONFIDENCE,
                       PRICE_TOLERANCE_ABS_PER_LINE, PRICE_TOLERANCE_PCT, SAMPLES_DIR, VENDOR_NAME_MATCH_MIN)
from ap.money import fmt_inr
from ap.pipeline import STAGES, run_pipeline
from ap.policy import POLICIES
from ui import render
from ui.style import AMBER, GREEN, INK, MUTED, OUTCOME_COLOR, RED, RULE, e, outcome_chip, stage_card

UPLOADS = DATA_DIR / "uploads"


def manifest() -> list[dict]:
    return json.loads((SAMPLES_DIR / "manifest.json").read_text())


def pdf_for(rec: dict) -> bytes | None:
    p = UPLOADS / f"{rec['file_hash']}.pdf"
    if p.exists():
        return p.read_bytes()
    s = SAMPLES_DIR / rec["file_name"]
    return s.read_bytes() if s.exists() else None


def _when(rec) -> str:
    try:
        return dt.datetime.strptime(rec["started_at"], "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y")
    except Exception:
        return ""


def static_run(stages: list[dict], decision: dict, rec: dict, pdf: bytes | None) -> None:
    left, right = st.columns([5, 7], gap="large")
    with left:
        st.markdown(render.invoice_preview(pdf, decision["outcome"], _when(rec)), unsafe_allow_html=True)
    with right:
        for s in stages:
            with st.container(key=f"stg-{s['key']}-{rec['run_id']}"):
                st.markdown(render.card_html(s), unsafe_allow_html=True)
                if s["status"] != "skip":
                    with st.expander("Evidence"):
                        render.evidence(s)
    render.decision_panel(decision, rec)


# ======================================================================================
def process():
    st.title("Process an invoice")
    st.markdown('<p class="lede">Drop in a vendor invoice, or pick one of the prepared cases. Nine checks run in order; '
                'each shows what it found and why. The LLM only reads the document. Rules make every decision.</p>',
                unsafe_allow_html=True)
    m = manifest()
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        src = st.segmented_control("Source", ["Prepared cases", "Upload a PDF"], default="Prepared cases",
                                   label_visibility="collapsed")
        pdf, name, case = None, None, None
        if src == "Upload a PDF":
            up = st.file_uploader("Invoice PDF", type=["pdf"], label_visibility="collapsed")
            if up:
                pdf, name = up.getvalue(), up.name
        else:
            idx = st.selectbox("Case", range(len(m)), format_func=lambda i: m[i]["title"], label_visibility="collapsed")
            case = m[idx]
            pdf, name = (SAMPLES_DIR / case["file"]).read_bytes(), case["file"]
    with c2:
        if case:
            exp = case["expected"]
            st.markdown(
                f'<div style="font-size:.9rem;color:{INK};line-height:1.5">{e(case["story"])}</div>'
                f'<div style="font-size:.8rem;color:{MUTED};margin-top:6px">{e(case["vendor"])} · {e(case["kind"])} · '
                f'₹{fmt_inr(case["total"])} · expected {outcome_chip(exp)}'
                f'{" " + ", ".join(case["expected_codes"]) if case["expected_codes"] else ""}</div>',
                unsafe_allow_html=True)
        run = st.button("Run checks", type="primary", disabled=pdf is None, use_container_width=True)

    if run and pdf:
        st.session_state.pop("last", None)
        live_run(pdf, name)
    elif st.session_state.get("last"):
        last = st.session_state["last"]
        st.divider()
        static_run(last["stages"], last["decision"], last["rec"], last["pdf"])
    else:
        st.divider()
        left, right = st.columns([5, 7], gap="large")
        with left:
            if pdf:
                st.markdown(render.invoice_preview(pdf), unsafe_allow_html=True)
        with right:
            for i, (k, n, d) in enumerate(STAGES, 1):
                st.markdown(stage_card(i, n, d, "pending", llm=k == "extract"), unsafe_allow_html=True)


def live_run(pdf: bytes, name: str) -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    (UPLOADS / f"{hashlib.sha256(pdf).hexdigest()}.pdf").write_bytes(pdf)
    pace = 0.35 if st.session_state.get("pacing", True) else 0.0
    st.divider()
    left, right = st.columns([5, 7], gap="large")
    with left:
        preview = st.empty()
        preview.markdown(render.invoice_preview(pdf), unsafe_allow_html=True)
    with right:
        phs = {}
        for i, (k, n, d) in enumerate(STAGES, 1):
            phs[k] = st.container(key=f"stg-{k}-live").empty()
            phs[k].markdown(stage_card(i, n, d, "pending", llm=k == "extract"), unsafe_allow_html=True)
    stages = []
    engine = "offline" if st.session_state.get("offline") else "auto"
    for ev in run_pipeline(pdf, name, engine):
        if ev[0] == "start":
            k = ev[1]
            i = render.STAGE_INDEX[k]
            desc = render.STAGE_DESC[k]
            if k == "extract" and engine != "offline":
                desc = "Reading the document with Gemini…"
            phs[k].markdown(stage_card(i, dict((a, b) for a, b, _ in STAGES)[k], desc, "running",
                                       llm=k == "extract"), unsafe_allow_html=True)
        elif ev[0] == "done":
            res = ev[1].model_dump(mode="json")
            stages.append(res)
            with phs[res["key"]].container():
                st.markdown(render.card_html(res), unsafe_allow_html=True)
                if res["status"] != "skip":
                    with st.expander("Evidence"):
                        render.evidence(res)
            if pace and res["status"] != "skip":
                time.sleep(pace)
        elif ev[0] == "decision":
            decision, rec = ev[1].model_dump(mode="json"), ev[2]
            preview.markdown(render.invoice_preview(pdf, decision["outcome"], _when(rec)), unsafe_allow_html=True)
            render.decision_panel(decision, rec)
            st.session_state["last"] = dict(stages=stages, decision=decision, rec=rec, pdf=pdf)


# ======================================================================================
def _kpi(label, value, sub=""):
    return (f'<div class="kpi"><div class="kpi-label">{e(label)}</div><div class="kpi-value">{e(value)}</div>'
            f'<div class="kpi-sub">{e(sub)}</div></div>')


def dashboard():
    st.title("Dashboard")
    st.markdown('<p class="lede">Every run, what it decided and why. Click a row to replay its checks.</p>',
                unsafe_allow_html=True)
    with st.expander("Run the whole sample inbox"):
        st.caption("Resets the demo, then processes every prepared invoice in order, as if they arrived in the AP "
                   "mailbox this morning. Useful for filling the dashboard.")
        if st.button("Reset and process all samples"):
            db.reset()
            st.session_state.pop("last", None)
            m = manifest()
            bar = st.progress(0.0, text="Starting")
            engine = "offline" if st.session_state.get("offline") else "auto"
            for i, c in enumerate(m):
                bar.progress(i / len(m), text=f"Processing {c['file']}")
                for _ in run_pipeline((SAMPLES_DIR / c["file"]).read_bytes(), c["file"], engine):
                    pass
            bar.progress(1.0, text=f"Processed {len(m)} invoices")

    runs = db.runs()
    if not runs:
        st.markdown('<div class="empty">No invoices processed yet. Run one from <b>Process invoice</b>, or process the '
                    'whole sample inbox above.</div>', unsafe_allow_html=True)
        return
    df = pd.DataFrame(runs)
    n = len(df)
    appr = df[df.outcome == "Approve"]
    holds = (df.outcome == "Hold").sum()
    rejects = (df.outcome == "Reject").sum()
    st.markdown('<div class="kpis">' + "".join([
        _kpi("Invoices processed", f"{n}", f"{df.file_name.nunique()} distinct files"),
        _kpi("Approved with no touch", f"{len(appr) / n:.0%}", f"{len(appr)} of {n}"),
        _kpi("Value approved", f"₹{fmt_inr(appr.total.fillna(0).sum(), decimals=0)}", "posted to PO ledger"),
        _kpi("In the exceptions queue", f"{holds}", f"plus {rejects} sent back to vendors"),
        _kpi("Average time per invoice", f"{df.duration_ms.mean() / 1000:.1f} s",
             f"slowest {df.duration_ms.max() / 1000:.1f} s"),
    ]) + "</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("#### Why invoices were stopped")
        codes = df[df.outcome != "Approve"].codes.str.split(",").explode().dropna()
        codes = codes[codes != ""]
        if len(codes):
            from ap.policy import BY_CODE
            vc = codes.value_counts().sort_values()
            colors = [OUTCOME_COLOR[BY_CODE[c].outcome] for c in vc.index]
            fig = go.Figure(go.Bar(x=vc.values, y=[BY_CODE[c].label for c in vc.index], orientation="h",
                                   marker=dict(color=colors, cornerradius=4), text=vc.values, textposition="outside",
                                   hovertemplate="%{y}: %{x} invoice(s)<extra></extra>", width=0.55))
            fig.update_layout(height=max(220, 34 * len(vc) + 60), margin=dict(l=0, r=30, t=6, b=6),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="IBM Plex Sans, Helvetica, Arial, sans-serif", color=INK, size=13),
                              xaxis=dict(visible=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"Bar colour = the outcome that code forces: green approve-with-note, amber hold, red reject.")
        else:
            st.caption("Nothing stopped yet.")
    with c2:
        st.markdown("#### Outcomes")
        order = ["Approve", "Hold", "Reject"]
        cnt = df.outcome.value_counts().reindex(order).fillna(0).astype(int)
        fig = go.Figure()
        for o in order:
            fig.add_trace(go.Bar(x=[cnt[o]], y=[""], orientation="h", name=o, marker=dict(color=OUTCOME_COLOR[o]),
                                 text=[f"{o} {cnt[o]}" if cnt[o] else ""], textposition="inside",
                                 insidetextanchor="middle", hovertemplate=f"{o}: %{{x}}<extra></extra>"))
        fig.update_layout(barmode="stack", height=110, margin=dict(l=0, r=0, t=4, b=4), showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.1,
                          font=dict(family="IBM Plex Sans, Helvetica, Arial, sans-serif", color="#fff", size=13),
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        owners = df[df.outcome != "Approve"].owners.str.split(", ").explode()
        owners = owners[~owners.isin(["", "—"])].value_counts()
        if len(owners):
            st.markdown("#### Who has work waiting")
            st.dataframe(pd.DataFrame({"Owner": owners.index, "Invoices": owners.values}), hide_index=True,
                         use_container_width=True)

    st.markdown("#### Run history")
    oc = st.multiselect("Outcome", ["Approve", "Hold", "Reject"], default=["Approve", "Hold", "Reject"],
                        label_visibility="collapsed")
    view = df[df.outcome.isin(oc)].copy()
    view["Outcome"] = view.apply(lambda r: r.outcome + (" + note" if r.with_note else "")
                                 + (" · high priority" if r.priority == "high" else ""), axis=1)
    for col in ["vendor_name", "invoice_number", "po_number"]:
        view[col] = view[col].fillna("—")
    view["codes"] = view.codes.str.replace(",", ", ")
    view["Total"] = view.total.map(lambda v: "—" if pd.isna(v) else "₹" + fmt_inr(v))
    view["Time"] = (view.duration_ms / 1000).map(lambda v: f"{v:.1f} s")
    table = view[["started_at", "Outcome", "vendor_name", "invoice_number", "Total", "po_number", "codes", "owners",
                  "engine", "Time"]].rename(columns=dict(
        started_at="When", vendor_name="Vendor", invoice_number="Invoice", po_number="PO", codes="Reason codes",
        owners="Routed to", engine="Reader"))
    styled = table.style.map(lambda v: f"color: {OUTCOME_COLOR[v.split(' ')[0]]}; font-weight: 600"
                             if isinstance(v, str) and v.split(" ")[0] in OUTCOME_COLOR else "", subset=["Outcome"])
    sel = st.dataframe(styled, hide_index=True, use_container_width=True, on_select="rerun",
                       selection_mode="single-row", key="runs_table", column_config={
                           "Outcome": st.column_config.TextColumn(width="medium"),
                           "Reason codes": st.column_config.TextColumn(width="large")})
    rows = sel.selection.rows if sel and sel.selection else []
    if rows:
        rec = view.iloc[rows[0]].to_dict()
        rec = db.run(rec["run_id"])
        st.divider()
        st.markdown(f"### Replay · {e(rec['file_name'])} · {rec['started_at']}")
        static_run(json.loads(rec["stages"]), json.loads(rec["decision"]), rec, pdf_for(rec))


# ======================================================================================
def reference():
    st.title("Reference data")
    st.markdown('<p class="lede">The source of truth the rules check against. Vendors, POs and goods receipts are '
                'read-only exports; the ledger and history change as invoices are approved.</p>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.tabs(["Purchase orders", "Vendors", "Goods receipts", "PO ledger", "Invoice history"])
    with t1:
        vname = {v["vendor_id"]: v["legal_name"] for v in db.vendors()}
        summary, lines = [], []
        for po in db.pos().values():
            value = sum(l["qty"] * l["unit_price"] for l in po["lines"])
            billed = db.invoiced(po["po_number"])["amount"]
            rec_val = sum(db.received(po["po_number"], l["line_no"])[0] * l["unit_price"] for l in po["lines"])
            summary.append({"PO": po["po_number"], "Vendor": vname[po["vendor_id"]], "Status": po["status"],
                            "Raised": po["po_date"], "Lines": len(po["lines"]), "Value": "₹" + fmt_inr(value, decimals=0),
                            "Received": rec_val / value, "Billed": billed / value,
                            "Left to bill": "₹" + fmt_inr(value - billed, decimals=0), "Requester": po["requester"]})
            for l in po["lines"]:
                done = db.invoiced(po["po_number"], l["line_no"])["qty"]
                rec, _ = db.received(po["po_number"], l["line_no"])
                lines.append({"PO": po["po_number"], "Line": l["line_no"], "Item": l["item"], "HSN": l["hsn"],
                              "Ordered": f"{fmt_inr(l['qty'], decimals=0)} {l['unit']}",
                              "Rate": "₹" + fmt_inr(l["unit_price"]), "GST": f"{l['gst_rate']:g}%",
                              "Received": f"{fmt_inr(rec, decimals=0)}", "Billed": f"{fmt_inr(done, decimals=0)}",
                              "Left": f"{fmt_inr(l['qty'] - done, decimals=0)}"})
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True, column_config={
            "Received": st.column_config.ProgressColumn("Received (GRN)", min_value=0, max_value=1, format="percent"),
            "Billed": st.column_config.ProgressColumn("Billed so far", min_value=0, max_value=1, format="percent")})
        st.markdown("**PO lines**")
        st.dataframe(pd.DataFrame(lines), hide_index=True, use_container_width=True)
    with t2:
        st.dataframe(pd.DataFrame(db.vendors()), hide_index=True, use_container_width=True)
    with t3:
        st.dataframe(pd.DataFrame(db.grns()), hide_index=True, use_container_width=True)
    with t4:
        led = pd.DataFrame(db.ledger_rows())
        st.dataframe(led.drop(columns=["id"]) if len(led) else led, hide_index=True, use_container_width=True)
        st.caption("Rows marked 'seed' existed before the demo; 'run' rows were posted by approvals.")
    with t5:
        h = pd.DataFrame(db.history())
        st.dataframe(h.drop(columns=["id", "invoice_number_norm"]) if len(h) else h, hide_index=True,
                     use_container_width=True)


# ======================================================================================
PROCESS_MAP = f"""
digraph G {{
  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.22;
  node [shape=box, style="rounded,filled", fillcolor="white", color="{RULE}", fontname="IBM Plex Sans",
        fontsize=11, fontcolor="{INK}", margin="0.14,0.07"];
  edge [color="#9AA3AE", arrowsize=0.6];
  pdf [label="Invoice PDF\\n(digital or scanned)", shape=note, fillcolor="#F6F7F5"];
  s1 [label="1 Ingest\\nhash, text layer"];
  s2 [label="2 Extract\\nGemini → JSON schema", color="#2D4BA0", penwidth=1.6, fontcolor="#2D4BA0"];
  s3 [label="3 Normalise\\n₹1,23,456 · dates"];
  s4 [label="4 Arithmetic & tax\\nlines, GST type"];
  s5 [label="5 Vendor\\nGSTIN, master"];
  s6 [label="6 PO match\\nexplicit or inferred"];
  s7 [label="7 Lines & balance\\nPO ↔ GRN ↔ ledger"];
  s8 [label="8 Duplicates\\nhash, number, look-alike"];
  s9 [label="9 Decide & explain\\npolicy table"];
  ref [label="Vendor master · PO register\\nGRN · PO ledger · history", shape=cylinder, fillcolor="#F6F7F5"];
  ok [label="Approve\\npost to ledger", color="{GREEN}", fontcolor="{GREEN}"];
  hold [label="Hold\\nroute to owner", color="{AMBER}", fontcolor="{AMBER}"];
  rej [label="Reject\\ndraft vendor email", color="{RED}", fontcolor="{RED}"];
  pdf -> s1 -> s2 -> s3 -> s4 -> s5 -> s6 -> s7 -> s8 -> s9;
  ref -> s5 [style=dashed]; ref -> s6 [style=dashed]; ref -> s7 [style=dashed]; ref -> s8 [style=dashed];
  s9 -> ok; s9 -> hold; s9 -> rej;
  {{rank=same; s6; ref}}
}}
"""


def how_it_works():
    st.title("How it works")
    st.markdown('<p class="lede">An AP clerk opens every invoice, hunts for the PO in a spreadsheet, checks the numbers '
                'and decides. This does the same work in under 30 seconds and leaves the clerk only the exceptions. '
                'The LLM reads; rules decide.</p>', unsafe_allow_html=True)
    c0, c1 = st.columns([2, 3], gap="large")
    with c0:
        st.graphviz_chart(PROCESS_MAP, use_container_width=True)
    with c1:
        st.markdown("#### Design choices")
        st.markdown(f"""
- **Only stage 2 uses an LLM.** Gemini transcribes into a strict schema, returns `null` instead of guessing, and
  scores its own confidence per field. Everything that decides money is plain Python, so the same invoice and data
  always give the same answer, traceable to a rule.
- **Misreads can't slip through.** A misread number fails the arithmetic stage. On digital PDFs the reading is also
  cross-checked against the PDF's own text layer.
- **Zero false approvals.** When unsure, hold: a wrong approval costs money, a wrong hold costs a minute.
- **Every reason is shown,** not just the first, and the vendor gets one email listing everything to fix.
- **State across runs.** A PO ledger tracks what's already billed, so split invoices and over-billing are caught.
- **Fails safe.** If Gemini is down, digital PDFs fall back to a text-layer parser and scanned ones are held.
""")
        st.markdown("#### Tolerances")
        st.markdown(f"""
| Setting | Value |
|---|---|
| Unit price variance | ±{PRICE_TOLERANCE_PCT:g}%, capped at ₹{PRICE_TOLERANCE_ABS_PER_LINE} per line |
| Arithmetic rounding | ±₹1 |
| Vendor name fuzzy match | ≥ {VENDOR_NAME_MATCH_MIN}/100 |
| PO inference | confidence ≥ {PO_INFERENCE_MIN_CONFIDENCE}, always held for confirmation |
| Suspected duplicate | same vendor, amount ±{DUP_AMOUNT_PCT:g}%, within {DUP_WINDOW_DAYS} days |
| Extraction confidence | key fields ≥ {LOW_CONFIDENCE} |
""")
    c2, c3 = st.columns(2, gap="large")
    with c2:
        st.markdown("#### Assumptions")
        st.markdown(f"""
- The buyer is **{BUYER['name']}**, {BUYER['state']} (GSTIN `{BUYER['gstin']}`). Its state decides
  intra-state (CGST + SGST) versus inter-state (IGST) tax.
- The vendor master, PO register and goods receipts are exported as CSV and are the source of truth.
- A missing invoice number or date makes it an invalid GST tax invoice, so only the vendor can fix it: **Reject**.
  **Hold** means someone on our side can resolve it.
- One invoice per PDF, in INR.
""")
    with c3:
        st.markdown("#### Next steps")
        st.markdown("""
- Pull invoices straight from the AP mailbox instead of uploading.
- Write approved invoices back to the ERP and schedule payment.
- Let the clerk resolve a hold in the app (confirm PO, accept price) and re-run.
- Multi-currency and multi-page invoices with several POs.
""")
    st.markdown("#### Decision policy")
    st.caption("The outcome is the most severe code that fired: Reject > Hold > Approve.")
    st.dataframe(pd.DataFrame([{"Code": p.code, "Fires when": p.fires_when, "Outcome": p.outcome, "Routed to": p.owner,
                                "What happens next": p.action,
                                "In vendor email": {"always": "Yes", "on_reject": "If rejected", "never": "No"}[p.email]}
                               for p in POLICIES]), hide_index=True, use_container_width=True,
                 height=36 * (len(POLICIES) + 1) + 4,
                 column_config={"Fires when": st.column_config.TextColumn(width="large"),
                                "What happens next": st.column_config.TextColumn(width="large")})
