"""The nine-stage invoice pipeline. Only stage 2 (Extract) uses an LLM.

`run_pipeline()` is a generator: it yields ("start", key) before each stage and
("done", StageResult) after it, then ("decision", Decision, run_record). The UI
renders these as they arrive; the batch runner and tests just drain it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

import pymupdf

from . import db
from .config import (BUYER, DUP_AMOUNT_PCT, DUP_WINDOW_DAYS, KEY_FIELDS, LINE_MATCH_MIN, LOW_CONFIDENCE,
                     PO_INFERENCE_MIN_CONFIDENCE, PRICE_TOLERANCE_ABS_PER_LINE, PRICE_TOLERANCE_PCT,
                     ROUNDING_TOLERANCE, VENDOR_NAME_MATCH_MIN)
from .extract import ExtractionUnavailable, cache_get, cache_put, gemini_extract, textlayer_extract
from .gstin import clean as clean_gstin, state_of, validate as validate_gstin
from .matching import name_score, pair_lines
from .models import Decision, Invoice, Line, RawExtraction, Reason, StageResult
from .money import fmt_inr, parse_amount
from .notify import draft_vendor_email
from .policy import BY_CODE, SEVERITY, outcome_of

STAGES = [
    ("ingest", "Ingest", "Fingerprint the file and check for a text layer"),
    ("extract", "Extract", "Read the invoice into a strict schema (Gemini)"),
    ("normalise", "Normalise", "Parse Indian numbers and dates, check required fields"),
    ("arithmetic", "Arithmetic & tax", "Lines, subtotal, GST and total must reconcile"),
    ("vendor", "Vendor verification", "GSTIN checksum, vendor master match, status"),
    ("po", "PO match", "Find the purchase order, or infer it"),
    ("lines", "Line & balance match", "Price, PO balance and goods received, per line"),
    ("duplicate", "Duplicate check", "Same file, same number, or a look-alike"),
    ("decide", "Decide & explain", "Apply the policy, explain, post to ledger"),
]
STAGE_NAMES = {k: n for k, n, _ in STAGES}
GST_SLABS = {0.0, 0.25, 3.0, 5.0, 12.0, 18.0, 28.0, 40.0}


@dataclass
class Ctx:
    pdf: bytes
    file_name: str
    engine_pref: str = "auto"            # auto | offline
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    file_hash: str = ""
    kind: str = ""                       # digital | scanned
    engine: str = ""
    raw: RawExtraction | None = None
    inv: Invoice | None = None
    vendor: dict | None = None
    po: dict | None = None
    pairs: list[dict] = field(default_factory=list)
    postings: list[dict] = field(default_factory=list)
    halted: bool = False                 # nothing readable: later stages skip
    reasons: list[Reason] = field(default_factory=list)


def _d(iso: str | None) -> str:
    try:
        return dt.date.fromisoformat(iso).strftime("%d %b %Y")
    except Exception:
        return iso or "?"


def _brief(reasons, limit=230) -> str:
    """One-line stage summary from the reasons that fired."""
    text = " ".join(f"{BY_CODE[r.code].label}: {r.message.split(': ', 1)[-1]}" for r in reasons)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def R(code: str, message: str, **data) -> Reason:
    return Reason(code=code, message=message, data=data)


# ======================================================================================
# 1. Ingest
# ======================================================================================
def s_ingest(ctx: Ctx) -> StageResult:
    ctx.file_hash = hashlib.sha256(ctx.pdf).hexdigest()
    doc = pymupdf.open(stream=ctx.pdf, filetype="pdf")
    chars = [len(p.get_text("text").strip()) for p in doc]
    images = sum(len(p.get_images()) for p in doc)
    ctx.kind = "digital" if sum(chars) >= 200 else "scanned"
    return StageResult(
        key="ingest", name="Ingest", status="pass",
        summary=f"{ctx.kind.capitalize()} PDF, {len(doc)} page{'s' if len(doc) > 1 else ''}"
                + (f", {sum(chars):,} characters of text" if ctx.kind == "digital" else ", no text layer (image only)"),
        evidence=dict(file=ctx.file_name, sha256=ctx.file_hash, bytes=len(ctx.pdf), pages=len(doc),
                      text_chars=sum(chars), images=images, kind=ctx.kind))


# ======================================================================================
# 2. Extract (the only LLM step)
# ======================================================================================
def s_extract(ctx: Ctx) -> StageResult:
    ev: dict[str, Any] = {}
    status, note = "pass", ""
    cached = cache_get(ctx.file_hash) if ctx.engine_pref != "offline" else None
    if cached:
        ctx.raw = RawExtraction.model_validate(cached["raw"])
        ctx.engine = cached["engine"]
        note = f"read by {cached['engine']} (cached result for this exact file)"
        ev["cache"] = dict(hit=True, created_at=cached.get("created_at"))
    else:
        err = None
        if ctx.engine_pref != "offline":
            try:
                raw, model = gemini_extract(ctx.pdf, ctx.kind == "scanned")
                ctx.raw, ctx.engine = raw, model
                cache_put(ctx.file_hash, dict(engine=model, raw=raw.model_dump(), file=ctx.file_name,
                                              created_at=dt.datetime.now().isoformat(timespec="seconds")))
                note = f"read by {model}" + (" from page images" if ctx.kind == "scanned" else " from the PDF")
            except ExtractionUnavailable as e:
                err = str(e)
        else:
            err = "offline mode selected"
        if ctx.raw is None:
            ev["llm_error"] = err
            if ctx.kind == "digital":
                ctx.raw = textlayer_extract(ctx.pdf, BUYER["name"], BUYER["gstin"])
                ctx.engine = "text-layer parser"
                status = "warn"
                note = f"Gemini unavailable ({err}); used the offline text-layer parser"
            else:
                ctx.halted = True
                ctx.engine = "none"
                return StageResult(
                    key="extract", name="Extract", status="fail",
                    summary="Scanned document and the vision model is unavailable, so nothing can be read",
                    reasons=[R("EXTRACTION_UNAVAILABLE",
                               f"This is an image-only PDF and Gemini could not be reached ({err}). "
                               "Rather than guess, the invoice is held for manual keying.")],
                    evidence=ev)

    # Cross-check: on a digital PDF, compare the LLM's reading to the text layer.
    if ctx.kind == "digital" and ctx.engine != "text-layer parser":
        tl = textlayer_extract(ctx.pdf, BUYER["name"], BUYER["gstin"])
        checks = []
        for f in ["invoice_number", "vendor_gstin", "po_reference", "total"]:
            a, b = getattr(ctx.raw, f), getattr(tl, f)
            if b is None:
                continue
            same = (parse_amount(a) == parse_amount(b)) if f == "total" else \
                (re.sub(r"\W", "", str(a or "")).upper() == re.sub(r"\W", "", str(b)).upper())
            checks.append(dict(field=f, llm=a, text_layer=b, agree=same))
        ev["text_layer_crosscheck"] = checks
        if any(not c["agree"] for c in checks):
            status = "warn"
            note += "; disagrees with the text layer on " + ", ".join(c["field"] for c in checks if not c["agree"])

    ev["engine"] = ctx.engine
    ev["raw"] = ctx.raw.model_dump()
    ev["confidence"] = ctx.raw.confidence.model_dump()
    n = len(ctx.raw.line_items)
    return StageResult(key="extract", name="Extract", status=status,
                       summary=f"{n} line item{'s' if n != 1 else ''}, {note}", evidence=ev)


# ======================================================================================
# 3. Normalise
# ======================================================================================
_DATE_FORMATS = ["%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y", "%d.%m.%y", "%d %b %Y", "%d %B %Y",
                 "%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y", "%b %d, %Y", "%d %b, %Y"]


def parse_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    s = re.sub(r"\s+", " ", s.strip().replace(",", ", ")).replace(" ,", ",")
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    for f in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(s, f).date()
        except ValueError:
            continue
    return None


def norm_po(s: str | None) -> str | None:
    if not s:
        return None
    m = re.search(r"(\d{3,6})", s)
    return f"PO-{m.group(1)}" if m else s.strip().upper()


def _num(s):
    v = parse_amount(s)
    return None if v is None else v


def s_normalise(ctx: Ctx) -> StageResult:
    raw = ctx.raw
    lines = []
    for i, l in enumerate(raw.line_items, 1):
        lines.append(Line(idx=i, description=(l.description or "").strip() or None, hsn=(l.hsn or "").strip() or None,
                          qty=_num(l.quantity), unit=l.unit, unit_price=_num(l.unit_price), amount=_num(l.amount)))
    conf = raw.confidence.model_dump()
    inv = Invoice(
        vendor_name=(raw.vendor_name or "").strip() or None, vendor_gstin=clean_gstin(raw.vendor_gstin),
        invoice_number=(raw.invoice_number or "").strip() or None, invoice_date=parse_date(raw.invoice_date),
        po_reference=norm_po(raw.po_reference), lines=lines, subtotal=_num(raw.subtotal),
        cgst=_num(raw.cgst_amount) or 0.0, sgst=_num(raw.sgst_amount) or 0.0, igst=_num(raw.igst_amount) or 0.0,
        cgst_rate=_num(raw.cgst_rate), sgst_rate=_num(raw.sgst_rate), igst_rate=_num(raw.igst_rate),
        total=_num(raw.total), confidence=conf)
    ctx.inv = inv

    reasons = []
    missing = []
    if not inv.invoice_number:
        missing.append("invoice number")
    if not inv.invoice_date:
        missing.append("invoice date" if not raw.invoice_date else f"a readable invoice date (got '{raw.invoice_date}')")
    if inv.total is None:
        missing.append("invoice total")
    if missing:
        reasons.append(R("MISSING_REQUIRED_FIELD",
                         f"The invoice has no {', no '.join(missing)}. A GST tax invoice must carry "
                         f"{'these' if len(missing) > 1 else 'it'}, so only the vendor can fix this by reissuing.", fields=missing,
                         vendor=f"The invoice does not show {' or '.join(missing)}, which a GST tax invoice must carry."))
    low = []
    for f in KEY_FIELDS:
        present = getattr(raw, f) if f != "line_items" else raw.line_items
        if present and conf.get(f, 1.0) < LOW_CONFIDENCE:
            low.append(f)
    if low:
        pretty = {"vendor_name": "vendor name", "vendor_gstin": "GSTIN", "invoice_number": "invoice number",
                  "invoice_date": "invoice date", "line_items": "line items", "total": "total"}
        reasons.append(R("LOW_EXTRACTION_CONFIDENCE",
                         "Read with low confidence: " + ", ".join(
                             f"{pretty[f]} '{getattr(raw, f) if f != 'line_items' else '…'}' ({conf[f]:.2f})" for f in low)
                         + ". A person should check these against the image.", fields=low))

    table = [
        dict(field="Invoice number", raw=raw.invoice_number, normalised=inv.invoice_number),
        dict(field="Invoice date", raw=raw.invoice_date, normalised=inv.invoice_date.isoformat() if inv.invoice_date else None),
        dict(field="PO reference", raw=raw.po_reference, normalised=inv.po_reference),
        dict(field="Vendor GSTIN", raw=raw.vendor_gstin, normalised=inv.vendor_gstin),
        dict(field="Subtotal", raw=raw.subtotal, normalised=inv.subtotal),
        dict(field="CGST", raw=raw.cgst_amount, normalised=inv.cgst),
        dict(field="SGST", raw=raw.sgst_amount, normalised=inv.sgst),
        dict(field="IGST", raw=raw.igst_amount, normalised=inv.igst),
        dict(field="Total", raw=raw.total, normalised=inv.total),
    ]
    status = "fail" if missing else ("warn" if low else "pass")
    summary = ("Missing " + ", ".join(missing)) if missing else \
        (f"Low confidence on {', '.join(low)}" if low else
         f"All required fields present; dated {inv.invoice_date:%d %b %Y}, total ₹{fmt_inr(inv.total)}")
    return StageResult(key="normalise", name="Normalise", status=status, summary=summary, reasons=reasons,
                       evidence=dict(fields=table, lines=[l.model_dump() for l in lines], low_confidence=low))


# ======================================================================================
# 4. Arithmetic & tax
# ======================================================================================
def _close(a, b, tol=ROUNDING_TOLERANCE):
    return a is not None and b is not None and abs(a - b) <= tol


def s_arithmetic(ctx: Ctx) -> StageResult:
    inv = ctx.inv
    reasons, checks = [], []
    bad_lines = []
    for l in inv.lines:
        if l.qty is None or l.unit_price is None or l.amount is None:
            checks.append(dict(check=f"Line {l.idx}: qty × rate", expected=None, found=l.amount, ok=False,
                               note="a value is unreadable"))
            bad_lines.append(l.idx)
            continue
        exp = round(l.qty * l.unit_price, 2)
        ok = _close(exp, l.amount)
        checks.append(dict(check=f"Line {l.idx}: {fmt_inr(l.qty, decimals=0)} × {fmt_inr(l.unit_price)}",
                           expected=exp, found=l.amount, ok=ok))
        if not ok:
            bad_lines.append(l.idx)
    line_sum = round(sum(l.amount or 0 for l in inv.lines), 2)
    sub = inv.subtotal if inv.subtotal is not None else line_sum
    sub_ok = _close(line_sum, sub)
    checks.append(dict(check="Σ line amounts = subtotal", expected=line_sum, found=inv.subtotal, ok=sub_ok,
                       note=None if inv.subtotal is not None else "no subtotal printed; using Σ lines"))
    tax_total = round(inv.cgst + inv.sgst + inv.igst, 2)
    exp_total = round(sub + tax_total, 2)
    tot_ok = _close(exp_total, inv.total)
    checks.append(dict(check="Subtotal + GST = total", expected=exp_total, found=inv.total, ok=tot_ok))

    msgs = []
    for idx in bad_lines:
        l = inv.lines[idx - 1]
        if None not in (l.qty, l.unit_price, l.amount):
            msgs.append(f"line {idx} shows ₹{fmt_inr(l.amount)} but {fmt_inr(l.qty, decimals=0)} × ₹{fmt_inr(l.unit_price)}"
                        f" = ₹{fmt_inr(l.qty * l.unit_price)}")
        else:
            msgs.append(f"line {idx} has an unreadable quantity, rate or amount")
    if not sub_ok:
        msgs.append(f"the lines add up to ₹{fmt_inr(line_sum)} but the subtotal says ₹{fmt_inr(inv.subtotal)}")
    if not tot_ok and inv.total is not None:
        msgs.append(f"subtotal ₹{fmt_inr(sub)} + GST ₹{fmt_inr(tax_total)} = ₹{fmt_inr(exp_total)}, "
                    f"but the total says ₹{fmt_inr(inv.total)}")
    if msgs:
        reasons.append(R("ARITHMETIC_MISMATCH", "The numbers don't add up: " + "; ".join(msgs) + ".",
                         lines=bad_lines, vendor="The amounts do not reconcile: " + "; ".join(msgs) + "."))

    # ---- GST type: who is supplying to whom ----
    vendor_state = state_of(inv.vendor_gstin)
    buyer_state = BUYER["state"]
    tax_msgs, tax_vendor = [], []
    supply = None
    if vendor_state:
        supply = "intra-state" if vendor_state == buyer_state else "inter-state"
        if supply == "intra-state" and inv.igst > 0:
            tax_msgs.append(f"IGST of ₹{fmt_inr(inv.igst)} was charged, but the vendor ({vendor_state}) and Northwind "
                            f"({buyer_state}) are in the same state, so this supply needs CGST + SGST")
            tax_vendor.append(f"IGST of ₹{fmt_inr(inv.igst)} has been charged, but your registration and ours are both in "
                              f"{buyer_state}, so CGST + SGST applies to this supply, not IGST")
        if supply == "inter-state" and (inv.cgst > 0 or inv.sgst > 0):
            tax_msgs.append(f"CGST/SGST was charged, but the vendor is in {vendor_state} and Northwind is in "
                            f"{buyer_state}, so this inter-state supply needs IGST")
            tax_vendor.append(f"CGST/SGST has been charged, but this is an inter-state supply ({vendor_state} to "
                              f"{buyer_state}), so IGST applies")
        if supply == "intra-state" and inv.cgst > 0 and not _close(inv.cgst, inv.sgst):
            tax_msgs.append(f"CGST (₹{fmt_inr(inv.cgst)}) and SGST (₹{fmt_inr(inv.sgst)}) should be equal")
    eff_rate = round(tax_total / sub * 100, 2) if sub else None
    if eff_rate is not None and tax_total > 0 and not any(abs(eff_rate - s) < 0.06 for s in GST_SLABS):
        tax_msgs.append(f"the effective GST rate is {eff_rate}%, which is not a valid GST slab")
    if tax_total == 0 and sub:
        tax_msgs.append("no GST is charged on a taxable supply")
    if tax_msgs:
        reasons.append(R("TAX_TYPE_ERROR", "GST is wrong: " + "; ".join(tax_msgs) + ".", supply=supply,
                         vendor=("; ".join(tax_vendor) if tax_vendor else "The GST shown is incorrect: " + "; ".join(tax_msgs)) + "."))

    tax = dict(vendor_state=vendor_state, buyer_state=buyer_state, supply=supply, effective_rate=eff_rate,
               cgst=inv.cgst, sgst=inv.sgst, igst=inv.igst)
    ok_all = not reasons
    note = ""
    if ok_all and ctx.kind == "scanned":
        note = " The reading is self-consistent, which also validates the extraction."
    summary = (f"Every line, the subtotal and the total reconcile; {supply or 'GST'} tax at {eff_rate:g}% is correct."
               + note) if ok_all else _brief(reasons)
    return StageResult(key="arithmetic", name="Arithmetic & tax", status="pass" if ok_all else "fail",
                       summary=summary, reasons=reasons, evidence=dict(checks=checks, tax=tax))


# ======================================================================================
# 5. Vendor verification
# ======================================================================================
def s_vendor(ctx: Ctx) -> StageResult:
    inv = ctx.inv
    gst_ok, gst_note = validate_gstin(inv.vendor_gstin)
    vendors = db.vendors()
    by_gstin = next((v for v in vendors if v["gstin"] == inv.vendor_gstin), None) if inv.vendor_gstin else None
    scored = sorted(((name_score(inv.vendor_name, v["legal_name"]), v) for v in vendors), key=lambda t: -t[0])
    candidates = [dict(vendor_id=v["vendor_id"], legal_name=v["legal_name"], gstin=v["gstin"], score=round(s))
                  for s, v in scored[:3]]
    reasons, method = [], None
    vendor = None
    if by_gstin:
        vendor, method = by_gstin, "GSTIN exact match"
    elif scored and scored[0][0] >= VENDOR_NAME_MATCH_MIN:
        vendor, method = scored[0][1], f"name match ({scored[0][0]:.0f}/100)"
        if inv.vendor_gstin and inv.vendor_gstin != vendor["gstin"]:
            reasons.append(R("GSTIN_MISMATCH",
                             f"The invoice says it is from {vendor['legal_name']}, but its GSTIN {inv.vendor_gstin} is not "
                             f"the one on file ({vendor['gstin']}). This is how impersonation fraud usually looks: "
                             "verify with the vendor on a known phone number before any payment.",
                             on_invoice=inv.vendor_gstin, on_file=vendor["gstin"]))
        elif not inv.vendor_gstin:
            reasons.append(R("GSTIN_MISMATCH", f"No GSTIN on the invoice; {vendor['legal_name']} is registered as "
                                               f"{vendor['gstin']}.", on_invoice=None, on_file=vendor["gstin"]))
    else:
        reasons.append(R("VENDOR_UNKNOWN",
                         f"'{inv.vendor_name}' (GSTIN {inv.vendor_gstin or 'none'}) is not in the vendor master; "
                         f"closest is {candidates[0]['legal_name']} at {candidates[0]['score']}/100."))
    if vendor and vendor["status"] != "active":
        reasons.append(R("VENDOR_BLOCKED", f"{vendor['legal_name']} is marked '{vendor['status']}' in the vendor master."))
    ctx.vendor = vendor
    status = "fail" if any(BY_CODE[r.code].outcome != "Approve" for r in reasons) else "pass"
    if vendor and not reasons:
        summary = f"{vendor['legal_name']} ({vendor['vendor_id']}), active; matched by {method}"
    elif vendor:
        summary = f"Matched {vendor['legal_name']} by {method}, but: " + BY_CODE[reasons[0].code].label
    else:
        summary = "Vendor not found in the master"
    return StageResult(key="vendor", name="Vendor verification", status=status, summary=summary, reasons=reasons,
                       evidence=dict(gstin=inv.vendor_gstin, gstin_valid=gst_ok, gstin_check=gst_note, method=method,
                                     vendor=vendor, candidates=candidates))


# ======================================================================================
# 6. PO match
# ======================================================================================
def po_remaining_value(po: dict) -> float:
    ordered = sum(l["qty"] * l["unit_price"] for l in po["lines"])
    return round(ordered - db.invoiced(po["po_number"])["amount"], 2)


def s_po(ctx: Ctx) -> StageResult:
    inv, vendor = ctx.inv, ctx.vendor
    pos = db.pos()
    reasons, ev = [], {}
    if inv.po_reference:
        po = pos.get(inv.po_reference)
        ev["method"] = "explicit reference on the invoice"
        if not po:
            reasons.append(R("PO_NOT_FOUND", f"The invoice quotes {inv.po_reference}, which is not in the PO register.",
                             vendor=f"The invoice quotes purchase order {inv.po_reference}, which we cannot find."))
        else:
            ev["po"] = {k: v for k, v in po.items() if k != "lines"}
            if vendor and po["vendor_id"] != vendor["vendor_id"]:
                owner = next((v["legal_name"] for v in db.vendors() if v["vendor_id"] == po["vendor_id"]), "?")
                reasons.append(R("PO_VENDOR_MISMATCH", f"{po['po_number']} was raised on {owner}, not on {vendor['legal_name']}."))
            if po["status"] != "open":
                reasons.append(R("PO_CLOSED", f"{po['po_number']} is {po['status']}; nothing more should be billed against it."))
            ctx.po = po
        status = "fail" if reasons else "pass"
        summary = f"{inv.po_reference} quoted on the invoice" + ("" if reasons else
                                                                 f"; open, raised by {ctx.po['requester']}")
        if reasons:
            summary = f"{inv.po_reference}: " + reasons[0].message
        return StageResult(key="po", name="PO match", status=status, summary=summary, reasons=reasons, evidence=ev)

    # ---- No PO number: infer from the vendor's open POs ----
    ev["method"] = "inferred (no PO number on the invoice)"
    if not vendor:
        reasons.append(R("PO_NOT_FOUND", "No PO number on the invoice and the vendor is unknown, so no PO can be inferred."))
        return StageResult(key="po", name="PO match", status="fail", summary="No PO reference and no vendor to infer from",
                           reasons=reasons, evidence=ev)
    open_pos = [p for p in pos.values() if p["vendor_id"] == vendor["vendor_id"] and p["status"] == "open"]
    sub = inv.subtotal or sum(l.amount or 0 for l in inv.lines)
    cands = []
    for p in open_pos:
        pairs = pair_lines(inv.lines, p["lines"], LINE_MATCH_MIN)
        overlap = sum(x["score"] for x in pairs if x["po_line"]) / (100 * max(1, len(inv.lines)))
        remaining = po_remaining_value(p)
        if remaining <= 0:
            fit = 0.0
        elif sub <= remaining * (1 + PRICE_TOLERANCE_PCT / 100):
            fit = 1.0 - min(0.5, (remaining - sub) / remaining * 0.5)   # closer to the remaining balance = better
        else:
            fit = max(0.0, 1 - (sub - remaining) / remaining)
        conf = round(min(0.95, 0.75 * overlap + 0.25 * fit), 2)   # never certain without a PO number
        cands.append(dict(po_number=p["po_number"], requester=p["requester"], items=", ".join(l["item"] for l in p["lines"]),
                          line_overlap=round(overlap, 2), amount_fit=round(fit, 2), remaining=remaining, confidence=conf))
    cands.sort(key=lambda c: -c["confidence"])
    ev["candidates"] = cands
    if cands and cands[0]["confidence"] >= PO_INFERENCE_MIN_CONFIDENCE and \
            (len(cands) == 1 or cands[0]["confidence"] - cands[1]["confidence"] >= 0.15):
        best = cands[0]
        ctx.po = pos[best["po_number"]]
        ev["po"] = {k: v for k, v in ctx.po.items() if k != "lines"}
        reasons.append(R("PO_INFERRED",
                         f"No PO number on the invoice. Best match is {best['po_number']} (confidence {best['confidence']:.2f}: "
                         f"line overlap {best['line_overlap']:.2f}, amount fit {best['amount_fit']:.2f}). "
                         f"{ctx.po['requester']} should confirm before payment.",
                         po=best["po_number"], confidence=best["confidence"]))
        summary = f"Inferred {best['po_number']} with confidence {best['confidence']:.2f}; needs the requester to confirm"
    else:
        top = f" (best: {cands[0]['po_number']} at {cands[0]['confidence']:.2f})" if cands else ""
        reasons.append(R("PO_NOT_FOUND", f"No PO number on the invoice and no open PO for {vendor['legal_name']} "
                                         f"matches confidently{top}.",
                             vendor="The invoice does not quote our purchase order number."))
        summary = "No PO number and none could be inferred confidently"
    return StageResult(key="po", name="PO match", status="warn" if ctx.po else "fail", summary=summary,
                       reasons=reasons, evidence=ev)


# ======================================================================================
# 7. Line & balance match (3-way: invoice <-> PO <-> GRN)
# ======================================================================================
def s_lines(ctx: Ctx) -> StageResult:
    inv, po = ctx.inv, ctx.po
    if not po:
        return StageResult(key="lines", name="Line & balance match", status="skip",
                           summary="Skipped: no PO to match against")
    pairs = pair_lines(inv.lines, po["lines"], LINE_MATCH_MIN)
    ctx.pairs = pairs
    by_no = {l["line_no"]: l for l in po["lines"]}
    rows, reasons, postings = [], [], []
    price_ok_msgs, price_high_msgs, over_msgs, grn_msgs, rate_msgs = [], [], [], [], []
    v_price, v_over, v_grn = [], [], []
    for pr in pairs:
        il = inv.lines[pr["inv_line"] - 1]
        if not pr["po_line"]:
            reasons.append(R("LINE_NOT_ON_PO", f"Invoice line {il.idx} '{il.description}' matches no line on {po['po_number']}."))
            rows.append(dict(inv_line=il.idx, item=il.description, po_line=None, status="not on PO"))
            continue
        pl = by_no[pr["po_line"]]
        done = db.invoiced(po["po_number"], pl["line_no"])
        rec_qty, grn_rows = db.received(po["po_number"], pl["line_no"])
        remaining = pl["qty"] - done["qty"]
        qty = il.qty or 0
        var_pct = (il.unit_price - pl["unit_price"]) / pl["unit_price"] * 100 if il.unit_price else 0.0
        var_amt = (il.unit_price - pl["unit_price"]) * qty if il.unit_price else 0.0
        flags = []
        if var_pct > 0.005:
            if var_pct <= PRICE_TOLERANCE_PCT and var_amt <= PRICE_TOLERANCE_ABS_PER_LINE:
                price_ok_msgs.append(f"line {il.idx} at ₹{fmt_inr(il.unit_price)} vs ₹{fmt_inr(pl['unit_price'])} "
                                     f"(+{var_pct:.2f}%, ₹{fmt_inr(var_amt, decimals=0)} on the line)")
                flags.append("price +ok")
            else:
                price_high_msgs.append(f"'{pl['item']}' billed at ₹{fmt_inr(il.unit_price)} against ₹{fmt_inr(pl['unit_price'])} "
                                       f"on the PO (+{var_pct:.1f}%, ₹{fmt_inr(var_amt, decimals=0)} over on {fmt_inr(qty, decimals=0)} "
                                       f"{pl['unit']}; tolerance is {PRICE_TOLERANCE_PCT:g}% capped at ₹{PRICE_TOLERANCE_ABS_PER_LINE})")
                flags.append("price high")
                v_price.append(f"'{pl['item']}' is billed at ₹{fmt_inr(il.unit_price)} per {pl['unit']}, but our purchase "
                               f"order {po['po_number']} agrees ₹{fmt_inr(pl['unit_price'])} (+{var_pct:.1f}%)")
        if qty > remaining + 1e-9:
            prev = f" ({', '.join(done['invoices'])})" if done["invoices"] else ""
            over_msgs.append(f"'{pl['item']}': PO ordered {fmt_inr(pl['qty'], decimals=0)} {pl['unit']}, "
                             f"{fmt_inr(done['qty'], decimals=0)} already invoiced{prev}, so {fmt_inr(max(remaining, 0), decimals=0)} "
                             f"left; this invoice bills {fmt_inr(qty, decimals=0)}")
            flags.append("over PO")
            v_over.append(f"'{pl['item']}': our purchase order {po['po_number']} is for {fmt_inr(pl['qty'], decimals=0)} "
                          f"{pl['unit']}, of which {fmt_inr(done['qty'], decimals=0)} has already been invoiced; this invoice "
                          f"bills a further {fmt_inr(qty, decimals=0)}")
        if done["qty"] + qty > rec_qty + 1e-9:
            grn_msgs.append(f"'{pl['item']}': {fmt_inr(rec_qty, decimals=0)} {pl['unit']} received "
                            f"({', '.join(sorted({g['grn_id'] for g in grn_rows})) or 'no GRN'}), "
                            f"{fmt_inr(done['qty'], decimals=0)} already invoiced, this invoice adds {fmt_inr(qty, decimals=0)}")
            flags.append("not received")
            v_grn.append(f"'{pl['item']}': we have received {fmt_inr(rec_qty, decimals=0)} {pl['unit']} so far and "
                         f"{fmt_inr(done['qty'], decimals=0)} has been invoiced already; this invoice bills {fmt_inr(qty, decimals=0)}")
        rate = None
        if inv.subtotal:
            rate = round((inv.cgst + inv.sgst + inv.igst) / inv.subtotal * 100, 2)
            if abs(rate - pl["gst_rate"]) > 0.06:
                rate_msgs.append(f"GST charged at {rate:g}% but {pl['item']} carries {pl['gst_rate']:g}% on the PO")
        rows.append(dict(inv_line=il.idx, item=il.description, po_line=pl["line_no"], po_item=pl["item"],
                         match=pr["score"], inv_price=il.unit_price, po_price=pl["unit_price"],
                         var_pct=round(var_pct, 2), var_amt=round(var_amt, 2), ordered=pl["qty"],
                         invoiced_before=done["qty"], remaining=remaining, received=rec_qty, this_qty=qty,
                         unit=pl["unit"], status=", ".join(flags) or "ok"))
        postings.append(dict(line_no=pl["line_no"], qty=qty, amount=round(il.amount or qty * (il.unit_price or 0), 2)))

    # PO value balance (catches over-billing even when quantities are bundled differently)
    # Valued at PO prices, so a price increase is reported once (as a price variance), not twice.
    ordered_val = sum(l["qty"] * l["unit_price"] for l in po["lines"])
    done_val = db.invoiced(po["po_number"])["amount"]
    remaining_val = ordered_val - done_val
    sub = round(sum((r["this_qty"] or 0) * r["po_price"] for r in rows if r.get("po_line"))
                + sum((inv.lines[r["inv_line"] - 1].amount or 0) for r in rows if not r.get("po_line")), 2)
    tol = min(remaining_val * PRICE_TOLERANCE_PCT / 100, PRICE_TOLERANCE_ABS_PER_LINE * max(1, len(inv.lines))) \
        if remaining_val > 0 else 0
    if sub > remaining_val + tol + ROUNDING_TOLERANCE and not over_msgs:
        over_msgs.append(f"{po['po_number']} has ₹{fmt_inr(max(remaining_val, 0))} left to bill; this invoice is "
                         f"worth ₹{fmt_inr(sub)} at PO prices")

    if price_ok_msgs:
        reasons.append(R("PRICE_VARIANCE_OK", "Small price increase within tolerance: " + "; ".join(price_ok_msgs) + "."))
    if price_high_msgs:
        reasons.append(R("PRICE_VARIANCE_HIGH", "Price above the PO: " + "; ".join(price_high_msgs) + ".",
                         vendor="; ".join(v_price) + "."))
    if over_msgs:
        reasons.append(R("OVER_PO_BALANCE", "More than is left on the PO: " + "; ".join(over_msgs) + ".",
                         vendor=("; ".join(v_over) if v_over else "; ".join(over_msgs)) + "."))
    if grn_msgs:
        reasons.append(R("QTY_NOT_RECEIVED", "Billed for more than has been received: " + "; ".join(grn_msgs) + ".",
                         vendor="; ".join(v_grn) + "."))
    if rate_msgs:
        reasons.append(R("TAX_TYPE_ERROR", "Wrong GST rate: " + "; ".join(rate_msgs) + "."))
    ctx.postings = postings

    blocking = [r for r in reasons if BY_CODE[r.code].outcome != "Approve"]
    status = "fail" if blocking else ("warn" if reasons else "pass")
    if not reasons:
        summary = (f"{len(rows)} line{'s match' if len(rows) != 1 else ' matches'} {po['po_number']} on price, PO balance "
                   f"and goods received (₹{fmt_inr(max(remaining_val - sub, 0))} left on the PO after this)")
    else:
        summary = _brief(reasons)
    return StageResult(key="lines", name="Line & balance match", status=status, summary=summary, reasons=reasons,
                       evidence=dict(po=po["po_number"], rows=rows, ordered_value=ordered_val, invoiced_value=done_val,
                                     remaining_value=remaining_val, this_value=sub))


# ======================================================================================
# 8. Duplicate check
# ======================================================================================
def s_duplicate(ctx: Ctx) -> StageResult:
    inv, vendor = ctx.inv, ctx.vendor
    hist = db.history()
    reasons, cands = [], []
    exact = next((h for h in hist if h["file_hash"] == ctx.file_hash), None)
    how = "same file (SHA-256)"
    if not exact and vendor and inv.invoice_number:
        exact = next((h for h in hist if h["vendor_id"] == vendor["vendor_id"]
                      and h["invoice_number_norm"] == db.norm_invoice_no(inv.invoice_number)), None)
        how = "same vendor and invoice number"
    if exact:
        reasons.append(R("DUPLICATE_EXACT", f"Already processed: {exact['invoice_number']} dated {_d(exact['invoice_date'])}, "
                                            f"₹{fmt_inr(exact['total'])}, approved {exact['processed_at']} ({how}).",
                         original=exact))
    if vendor and not exact and inv.total:
        for h in hist:
            if h["vendor_id"] != vendor["vendor_id"] or h["invoice_number_norm"] == db.norm_invoice_no(inv.invoice_number):
                continue
            amt_diff = abs(h["total"] - inv.total) / max(h["total"], inv.total) * 100
            day_diff = abs((dt.date.fromisoformat(h["invoice_date"]) - inv.invoice_date).days) \
                if (h["invoice_date"] and inv.invoice_date) else None
            hit = amt_diff <= DUP_AMOUNT_PCT and day_diff is not None and day_diff <= DUP_WINDOW_DAYS
            cands.append(dict(invoice_number=h["invoice_number"], invoice_date=h["invoice_date"], total=h["total"],
                              po=h["po_number"], processed_at=h["processed_at"], amount_diff_pct=round(amt_diff, 2),
                              days_apart=day_diff, flagged=hit))
            if hit:
                reasons.append(R("DUPLICATE_SUSPECTED",
                                 f"Looks like a re-issue of {h['invoice_number']} (₹{fmt_inr(h['total'])}, dated "
                                 f"{_d(h['invoice_date'])}, approved {h['processed_at']}): same vendor, amount within "
                                 f"{amt_diff:.2f}%, {day_diff} days apart, different number.", original=h))
    cands.sort(key=lambda c: (not c["flagged"], c["amount_diff_pct"]))
    status = "fail" if reasons else "pass"
    n_hist = len([h for h in hist if vendor and h["vendor_id"] == vendor["vendor_id"]])
    summary = reasons[0].message if reasons else \
        f"No match by file hash, number, or amount ± {DUP_AMOUNT_PCT:g}% within {DUP_WINDOW_DAYS} days " \
        f"({n_hist} earlier invoice{'s' if n_hist != 1 else ''} from this vendor checked)"
    return StageResult(key="duplicate", name="Duplicate check", status=status, summary=summary, reasons=reasons,
                       evidence=dict(candidates=cands[:5], history_checked=len(hist)))


# ======================================================================================
# 9. Decide & explain
# ======================================================================================
def decide(ctx: Ctx, results: list[StageResult]) -> Decision:
    reasons: list[Reason] = []
    for r in results:
        for rs in r.reasons:
            rs.stage = r.name
            reasons.append(rs)
    codes = [r.code for r in reasons]
    outcome = outcome_of(codes)
    reasons.sort(key=lambda r: -SEVERITY[BY_CODE[r.code].outcome])
    top = [r for r in reasons if BY_CODE[r.code].outcome == outcome]
    owners = list(dict.fromkeys(BY_CODE[r.code].owner for r in top if BY_CODE[r.code].owner != "—"))
    actions = list(dict.fromkeys(BY_CODE[r.code].action for r in reasons if BY_CODE[r.code].outcome != "Approve"))
    priority = "high" if any(BY_CODE[c].priority == "high" for c in codes) else "normal"
    with_note = outcome == "Approve" and bool(codes)
    inv = ctx.inv
    vname = ctx.vendor["legal_name"] if ctx.vendor else (inv.vendor_name if inv else None) or "unknown vendor"
    if outcome == "Approve":
        headline = f"Approved for payment: ₹{fmt_inr(inv.total)} to {vname}" + (", with a note" if with_note else "")
        actions = ["None. Posted to the PO ledger and queued for payment."]
        owners = owners or ["—"]
    elif outcome == "Hold":
        headline = f"On hold: {BY_CODE[top[0].code].label}" + \
                   (f" (+{len(set(codes)) - 1} more)" if len(set(codes)) > 1 else "")
    else:
        headline = f"Rejected: {BY_CODE[top[0].code].label}" + \
                   (f" (+{len(set(codes)) - 1} more)" if len(set(codes)) > 1 else "")
    email = draft_vendor_email(ctx, outcome, reasons) if inv else None
    return Decision(outcome=outcome, headline=headline, with_note=with_note, priority=priority, reasons=reasons,
                    owners=owners, next_actions=actions, vendor_email=email)


def s_decide(ctx: Ctx, results: list[StageResult]) -> tuple[StageResult, Decision]:
    d = decide(ctx, results)
    posted = []
    if d.outcome == "Approve" and ctx.po and ctx.vendor:
        db.post_approval(ctx.run_id, ctx.vendor["vendor_id"], ctx.inv, ctx.file_hash, ctx.po["po_number"], ctx.postings)
        posted = [dict(po=ctx.po["po_number"], **p) for p in ctx.postings]
        d.posted = posted
    codes = list(dict.fromkeys(r.code for r in d.reasons))
    summary = f"{d.outcome}" + (f": {', '.join(codes)}" if codes else ": no exceptions") + \
              (f". Posted {len(posted)} line{'s' if len(posted) != 1 else ''} to {ctx.po['po_number']}." if posted else "")
    status = {"Approve": "pass", "Hold": "warn", "Reject": "fail"}[d.outcome]
    return (StageResult(key="decide", name="Decide & explain", status=status, summary=summary,
                        evidence=dict(codes=codes, outcome=d.outcome, posted=posted)), d)


# ======================================================================================
# Runner
# ======================================================================================
STAGE_FNS = [("ingest", s_ingest), ("extract", s_extract), ("normalise", s_normalise), ("arithmetic", s_arithmetic),
             ("vendor", s_vendor), ("po", s_po), ("lines", s_lines), ("duplicate", s_duplicate)]


def run_pipeline(pdf: bytes, file_name: str, engine_pref: str = "auto") -> Iterator[tuple]:
    db.init()
    ctx = Ctx(pdf=pdf, file_name=file_name, engine_pref=engine_pref)
    started = dt.datetime.now()
    t_all = time.perf_counter()
    results: list[StageResult] = []
    for key, fn in STAGE_FNS:
        yield ("start", key)
        t0 = time.perf_counter()
        if ctx.halted:
            res = StageResult(key=key, name=STAGE_NAMES[key], status="skip", summary="Skipped: nothing could be read")
        else:
            try:
                res = fn(ctx)
            except Exception as e:  # a stage bug must never crash the run: hold for a human
                res = StageResult(key=key, name=STAGE_NAMES[key], status="fail",
                                  summary=f"Stage error: {e}",
                                  reasons=[R("STAGE_ERROR",
                                             f"Internal error in {STAGE_NAMES[key]}: {e}. Held for manual review.")])
                if key in ("ingest", "extract", "normalise"):
                    ctx.halted = True
        res.ms = round((time.perf_counter() - t0) * 1000, 1)
        results.append(res)
        yield ("done", res)
    yield ("start", "decide")
    t0 = time.perf_counter()
    res, decision = s_decide(ctx, results)
    res.ms = round((time.perf_counter() - t0) * 1000, 1)
    results.append(res)
    yield ("done", res)

    inv = ctx.inv
    rec = dict(
        run_id=ctx.run_id, started_at=started.strftime("%Y-%m-%d %H:%M:%S"),
        duration_ms=round((time.perf_counter() - t_all) * 1000, 1), file_name=file_name, file_hash=ctx.file_hash,
        file_kind=ctx.kind, engine=ctx.engine,
        vendor_name=(ctx.vendor or {}).get("legal_name") or (inv.vendor_name if inv else None),
        vendor_id=(ctx.vendor or {}).get("vendor_id"),
        invoice_number=inv.invoice_number if inv else None,
        invoice_date=inv.invoice_date.isoformat() if inv and inv.invoice_date else None,
        total=inv.total if inv else None, po_number=(ctx.po or {}).get("po_number"),
        outcome=decision.outcome, with_note=int(decision.with_note), priority=decision.priority,
        codes=",".join(dict.fromkeys(r.code for r in decision.reasons)), owners=", ".join(decision.owners),
        headline=decision.headline,
        stages=_json([r.model_dump(mode="json") for r in results]),
        decision=_json(decision.model_dump(mode="json")),
    )
    db.save_run(rec)
    yield ("decision", decision, rec)


def _json(o) -> str:
    import json
    return json.dumps(o, ensure_ascii=False, default=str)


def run_to_end(pdf: bytes, file_name: str, engine_pref: str = "auto") -> tuple[Decision, dict, list[StageResult]]:
    results, decision, rec = [], None, None
    for ev in run_pipeline(pdf, file_name, engine_pref):
        if ev[0] == "done":
            results.append(ev[1])
        elif ev[0] == "decision":
            decision, rec = ev[1], ev[2]
    return decision, rec, results
