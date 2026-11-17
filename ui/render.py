"""Renderers shared by the live run view and the dashboard replay."""
from __future__ import annotations

import base64
import datetime as dt
import urllib.parse
from functools import lru_cache

import pandas as pd
import pymupdf
import streamlit as st

from ap.money import fmt_inr
from ap.policy import BY_CODE
from ap.pipeline import STAGES
from ui.style import OUTCOME_COLOR, OUTCOME_STAMP, e, stage_card

STAGE_INDEX = {k: i for i, (k, _, _) in enumerate(STAGES, 1)}
STAGE_DESC = {k: d for k, _, d in STAGES}


@lru_cache(maxsize=64)
def page_png_b64(pdf: bytes, dpi: int = 110) -> str:
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    return base64.b64encode(doc[0].get_pixmap(dpi=dpi).tobytes("png")).decode()


def invoice_preview(pdf: bytes | None, outcome: str | None = None, when: str | None = None) -> str:
    if not pdf:
        return '<div class="empty">Preview not available for this run.</div>'
    stamp = ""
    if outcome:
        date = (when or dt.date.today().strftime("%d %b %Y")).upper()
        stamp = (f'<div class="stamp" style="color:{OUTCOME_COLOR[outcome]}">{OUTCOME_STAMP[outcome]}'
                 f'<small>NORTHWIND AP · {e(date)}</small></div>')
    return f'<div class="paper"><img alt="Invoice page 1" src="data:image/png;base64,{page_png_b64(pdf)}"/>{stamp}</div>'


def card_html(res: dict) -> str:
    return stage_card(STAGE_INDEX[res["key"]], res["name"], STAGE_DESC[res["key"]], res["status"], res["summary"],
                      res.get("ms"), llm=res["key"] == "extract")


def _df(rows, **kw):
    if not rows:
        st.caption("Nothing to show.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, **kw)


def _money(v):
    return "—" if v is None else "₹" + fmt_inr(v)


def evidence(res: dict) -> None:
    ev, key = res.get("evidence") or {}, res["key"]
    if not ev:
        st.caption("No evidence recorded for this stage.")
        return
    if key == "ingest":
        _df([dict(Property=k, Value=str(v)) for k, v in [
            ("File", ev["file"]), ("SHA-256", ev["sha256"]), ("Size", f"{ev['bytes']:,} bytes"), ("Pages", ev["pages"]),
            ("Text-layer characters", f"{ev['text_chars']:,}"), ("Embedded images", ev["images"]),
            ("Classified as", ev["kind"])]])
        st.caption("The hash is used for exact-duplicate detection and to cache the reading of this exact file.")
    elif key == "extract":
        if ev.get("llm_error"):
            st.warning(f"LLM unavailable: {ev['llm_error']}")
        conf = ev.get("confidence") or {}
        if conf:
            st.markdown("**Reader's confidence per field** (below 0.70 on a key field puts the invoice on hold)")
            _df([dict(Field=k.replace("_", " "), Confidence=v) for k, v in conf.items()],
                column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        if ev.get("text_layer_crosscheck"):
            st.markdown("**Cross-check against the PDF's own text layer**")
            _df([dict(Field=c["field"], LLM=str(c["llm"]), **{"Text layer": str(c["text_layer"])},
                      Agree="✓" if c["agree"] else "✕") for c in ev["text_layer_crosscheck"]])
        if ev.get("raw"):
            st.markdown(f"**Raw reading** from `{ev.get('engine')}`, strings exactly as printed")
            st.json(ev["raw"], expanded=False)
    elif key == "normalise":
        _df([dict(Field=f["field"], **{"As printed": "—" if f["raw"] is None else str(f["raw"])},
                  Parsed="—" if f["normalised"] is None else str(f["normalised"])) for f in ev["fields"]])
        if ev.get("lines"):
            _df([dict(Line=l["idx"], Item=l["description"], HSN=l["hsn"], Qty=l["qty"], Unit=l["unit"],
                      Rate=_money(l["unit_price"]), Amount=_money(l["amount"])) for l in ev["lines"]])
    elif key == "arithmetic":
        _df([dict(Check=c["check"], Expected=_money(c["expected"]), Printed=_money(c["found"]),
                  Result="✓" if c["ok"] else "✕", Note=c.get("note") or "") for c in ev["checks"]])
        t = ev["tax"]
        st.caption(f"Vendor state (from GSTIN): {t['vendor_state'] or 'unknown'} · Buyer: {t['buyer_state']} · "
                   f"Supply: {t['supply'] or 'unknown'} · CGST {_money(t['cgst'])} · SGST {_money(t['sgst'])} · "
                   f"IGST {_money(t['igst'])} · effective rate {t['effective_rate']}%")
    elif key == "vendor":
        v = ev.get("vendor")
        c1, c2 = st.columns(2)
        c1.markdown(f"**On the invoice**  \nGSTIN `{ev.get('gstin') or '—'}`  \n"
                    f"{'✓' if ev.get('gstin_valid') else '✕'} {e(ev.get('gstin_check'))}")
        if v:
            c2.markdown(f"**Vendor master: {v['vendor_id']}**  \n{v['legal_name']}  \nGSTIN `{v['gstin']}`  \n"
                        f"{v['state']} · status **{v['status']}**  \nMatched by: {ev.get('method')}")
        else:
            c2.markdown("**Vendor master**  \nNo match")
        st.markdown("**Closest names in the master**")
        _df(ev.get("candidates"))
    elif key == "po":
        st.markdown(f"Method: **{ev.get('method', '—')}**")
        if ev.get("po"):
            p = ev["po"]
            st.markdown(f"{p['po_number']} · {p['status']} · raised {p['po_date']} by {p['requester']}")
        if ev.get("candidates"):
            st.markdown("**Open POs for this vendor, scored** (confidence = 0.75 × line overlap + 0.25 × amount fit)")
            _df([dict(PO=c["po_number"], Items=c["items"], **{"Line overlap": c["line_overlap"],
                      "Amount fit": c["amount_fit"], "Left on PO": _money(c["remaining"])}, Confidence=c["confidence"])
                 for c in ev["candidates"]],
                column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    elif key == "lines":
        rows = []
        for r in ev.get("rows", []):
            if not r.get("po_line"):
                rows.append({"Invoice line": f"{r['inv_line']}. {r['item']}", "PO line": "—", "Result": r["status"]})
                continue
            rows.append({"Invoice line": f"{r['inv_line']}. {r['item']}", "PO line": f"{r['po_line']}. {r['po_item']}",
                         "Match": r["match"], "Billed rate": _money(r["inv_price"]), "PO rate": _money(r["po_price"]),
                         "Var %": f"{r['var_pct']:+.2f}%", "Ordered": fmt_inr(r["ordered"], decimals=0),
                         "Billed before": fmt_inr(r["invoiced_before"], decimals=0),
                         "Left": fmt_inr(r["remaining"], decimals=0), "Received (GRN)": fmt_inr(r["received"], decimals=0),
                         "This invoice": fmt_inr(r["this_qty"], decimals=0), "Result": r["status"]})
        _df(rows)
        st.caption(f"PO value {_money(ev.get('ordered_value'))} · already invoiced {_money(ev.get('invoiced_value'))} · "
                   f"left {_money(ev.get('remaining_value'))} · this invoice at PO prices {_money(ev.get('this_value'))}")
    elif key == "duplicate":
        cands = ev.get("candidates") or []
        if cands:
            _df([{"Earlier invoice": c["invoice_number"], "Dated": c["invoice_date"], "Total": _money(c["total"]),
                  "PO": c["po"], "Approved": c["processed_at"], "Amount diff": f"{c['amount_diff_pct']:.2f}%",
                  "Days apart": c["days_apart"], "Flagged": "⚑" if c["flagged"] else ""} for c in cands])
        st.caption(f"{ev.get('history_checked', 0)} processed invoices searched by file hash, vendor + number, "
                   "and amount ±1% within 15 days.")
    elif key == "decide":
        if ev.get("posted"):
            st.markdown("**Posted to the PO ledger**")
            _df([dict(PO=p["po"], Line=p["line_no"], Qty=p["qty"], Amount=_money(p["amount"])) for p in ev["posted"]])
        else:
            st.caption("Nothing posted: only approved invoices touch the ledger.")


def decision_panel(d: dict, rec: dict) -> None:
    outcome = d["outcome"]
    color = OUTCOME_COLOR[outcome]
    owners = ", ".join(o for o in d["owners"] if o != "—") or "the payment run"
    pri = ' · <b style="color:#B3261E">High priority</b>' if d.get("priority") == "high" else ""
    total = f"₹{fmt_inr(rec['total'])}" if rec.get("total") else "amount unknown"
    st.markdown(
        f'<div class="verdict" style="border-left-color:{color}">'
        f'<div class="outcome" style="color:{color}">{OUTCOME_STAMP[outcome]}{" WITH A NOTE" if d.get("with_note") else ""}</div>'
        f'<div class="headline">{e(d["headline"])}</div>'
        f'<div class="meta">{e(rec.get("vendor_name") or "Unknown vendor")} · invoice {e(rec.get("invoice_number") or "—")}'
        f' · {total} · PO {e(rec.get("po_number") or "—")} · routed to <b>{e(owners)}</b>{pri} · '
        f'{rec.get("duration_ms", 0) / 1000:.1f} s</div></div>', unsafe_allow_html=True)

    left, right = st.columns([7, 5], gap="large")
    with left:
        st.markdown(f"#### Why: {len(d['reasons'])} reason{'s' if len(d['reasons']) != 1 else ''} fired"
                    if d["reasons"] else "#### Why: every check passed")
        if d["reasons"]:
            items = []
            for r in d["reasons"]:
                p = BY_CODE[r["code"]]
                c = OUTCOME_COLOR[p.outcome]
                items.append(
                    f'<div class="reason"><div><span class="code" style="color:{c}">{e(r["code"])}</span>'
                    f'<div class="label">{e(p.label)}</div><div class="who">{e(p.outcome)} · {e(p.owner)}</div></div>'
                    f'<div><div class="msg">{e(r["message"])}</div><div class="stage-from">Found in: {e(r["stage"])}</div>'
                    f'</div></div>')
            st.markdown('<div class="email">' + "".join(items) + "</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="email"><div class="msg">Vendor verified, PO matched, prices and quantities within '
                        'the PO and goods receipt, arithmetic and GST correct, no duplicate found.</div></div>',
                        unsafe_allow_html=True)
        st.markdown("#### Next")
        for a in d["next_actions"]:
            st.markdown(f"- {a}")
    with right:
        em = d.get("vendor_email")
        if em:
            st.markdown("#### Vendor email, drafted")
            st.html(
                f'<div class="email"><div class="hdr">To <b>{e(em["to"])}</b><br/>Subject <b>{e(em["subject"])}</b></div>'
                f'<pre>{e(em["body"])}</pre></div>')
            href = "mailto:" + em["to"] + "?" + urllib.parse.urlencode(
                {"subject": em["subject"], "body": em["body"]}, quote_via=urllib.parse.quote)
            c1, c2 = st.columns(2)
            c1.link_button("Open in mail app", href, use_container_width=True)
            c2.download_button("Download .txt", em["body"], file_name=f"vendor-email-{rec['run_id']}.txt",
                               use_container_width=True, key=f"dl-{rec['run_id']}")
            st.caption(f"Wording: {em.get('wording', 'template')}. Every fact in it comes from the rules above.")
        elif outcome == "Hold":
            st.markdown("#### No vendor email")
            st.caption("This hold is resolved on our side, so nothing goes to the vendor yet."
                       + (" For a suspected fraud, contact the vendor only on a number already on file."
                          if any(r["code"] == "GSTIN_MISMATCH" for r in d["reasons"]) else ""))
        if d.get("posted"):
            st.markdown("#### Posted to ledger")
            _df([dict(PO=p["po"], Line=p["line_no"], Qty=fmt_inr(p["qty"], decimals=0), Amount=_money(p["amount"]))
                 for p in d["posted"]])
