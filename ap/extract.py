"""Stage 2 readers: Gemini (vision + JSON schema) and an offline text-layer parser.

The reader's only job is to transcribe. It returns strings as printed, `null`
for anything absent, and a confidence per field. It never decides anything.
"""
from __future__ import annotations

import io
import json
import os
import re
import time

import pdfplumber
import pymupdf

from .config import CACHE_DIR, GEMINI_MODEL_CANDIDATES
from .models import FieldConfidence, RawExtraction, RawLine

PROMPT = """You are transcribing an Indian GST invoice for an accounts-payable system.
Return ONLY what is written on the document, in the JSON schema provided.

Rules:
- Copy numbers and dates exactly as printed (keep commas, e.g. "1,23,456.00"; keep the date format).
- If a field is not on the document, return null. Never guess, infer or compute a missing value.
  In particular: if there is no purchase order number printed, po_reference must be null.
- vendor = the seller who issued the invoice; buyer = the party billed.
- line_items: one entry per billed item row, in order. amount = the row's amount before tax.
- subtotal = taxable value before GST. total = final amount payable.
- Tax: fill cgst/sgst/igst rate and amount only if printed.
- confidence: for each field, 0.0-1.0, how sure you are that the value you returned is exactly what is
  written (or, for null, that it is really absent). Use < 0.7 when handwriting, blur, smudges or
  overlapping marks make a value genuinely uncertain. Be honest; low confidence is useful.
- notes: briefly mention anything hard to read or unusual.
"""


class ExtractionUnavailable(Exception):
    pass


# ---------------------------------------------------------------- cache
def cache_get(file_hash: str) -> dict | None:
    p = CACHE_DIR / f"{file_hash}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def cache_put(file_hash: str, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{file_hash}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- Gemini
def api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        return None


def configured_model() -> str | None:
    m = os.environ.get("GEMINI_MODEL")
    if m:
        return m
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_MODEL")  # type: ignore[attr-defined]
    except Exception:
        return None


_working_model: str | None = None
_skip_until: dict[str, float] = {}
_skip_reason: dict[str, str] = {}


def _skip(model: str, seconds: float, why: str) -> None:
    _skip_until[model] = time.time() + seconds
    _skip_reason[model] = why


def gemini_extract(pdf: bytes, is_scanned: bool) -> tuple[RawExtraction, str]:
    """Returns (extraction, model_name). Raises ExtractionUnavailable on any failure."""
    global _working_model
    key = api_key()
    if not key:
        raise ExtractionUnavailable("no GEMINI_API_KEY configured")
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise ExtractionUnavailable(f"google-genai not installed: {e}")

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=45_000))
    if is_scanned:
        # Scanned pages go in as images (rendered at 200 dpi).
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        parts = [types.Part.from_bytes(data=pg.get_pixmap(dpi=200).tobytes("png"), mime_type="image/png")
                 for pg in doc]
    else:
        parts = [types.Part.from_bytes(data=pdf, mime_type="application/pdf")]
    candidates = [m for m in [configured_model(), _working_model, *GEMINI_MODEL_CANDIDATES] if m]
    seen, last_err, errors = set(), None, []
    for model in candidates:
        if model in seen:
            continue
        seen.add(model)
        if _skip_until.get(model, 0) > time.time():
            errors.append(f"{model}: skipped ({_skip_reason.get(model)})")
            continue
        use_thinking = True
        attempt = 0
        while attempt < 3:
            attempt += 1
            config = types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=RawExtraction, temperature=0,
                thinking_config=_thinking(types, model) if use_thinking else None,
            )
            try:
                resp = client.models.generate_content(model=model, contents=[*parts, PROMPT], config=config)
                parsed = resp.parsed
                if parsed is None:
                    parsed = RawExtraction.model_validate_json(resp.text)
                _working_model = model
                return parsed, model
            except Exception as e:  # noqa: BLE001 - any API failure -> retry / next model / fall back
                last_err = e
                msg = str(e)
                if use_thinking and "thinking" in msg.lower():
                    use_thinking = False           # model doesn't accept this thinking setting
                    attempt -= 1
                    continue
                if any(t in msg for t in ("404", "NOT_FOUND", "not found", "not supported")):
                    _skip(model, 3600, "not available for this key")
                    errors.append(f"{model}: not available")
                    break
                if any(t in msg for t in ("429", "RESOURCE_EXHAUSTED")):
                    if "per minute" not in msg.lower() and "PerMinute" not in msg:
                        _skip(model, 900, "quota used up")   # daily/plan quota: waiting won't help
                        errors.append(f"{model}: quota used up")
                        break
                    if attempt < 2:
                        time.sleep(min(_retry_delay(msg), 20))
                        continue
                if any(t in msg for t in ("503", "UNAVAILABLE", "504", "DEADLINE", "500", "INTERNAL",
                                          "timed out", "Timeout")):
                    if attempt < 2:
                        time.sleep(3)
                        continue
                    _skip(model, 120, "overloaded")  # try other models first for the next 2 minutes
                errors.append(f"{model}: {msg[:160]}")
                break
    raise ExtractionUnavailable("Gemini call failed. " + " | ".join(errors or [str(last_err)[:300]]))


def _thinking(types, model: str):
    """Transcription needs no deliberation: keep thinking minimal for speed."""
    if model.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=0)
    return types.ThinkingConfig(thinking_level="low")


def _retry_delay(msg: str) -> float:
    m = re.search(r"retry(?:Delay)?[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\s*s", msg, re.I)
    return min(45.0, float(m.group(1)) + 1.0) if m else 12.0


# ---------------------------------------------------------------- Text-layer fallback
_AMT = r"(?:₹\s*)?(\d[\d,]*\.\d{2})"
_LINE_RE = re.compile(
    r"^(?:\d{1,2}\s+)?(?P<desc>.+?)\s+(?P<hsn>\d{4,8})\s+(?P<qty>\d[\d,]*(?:\.\d+)?)\s+(?P<unit>[A-Za-z]{1,6})\s+"
    r"(?P<rate>\d[\d,]*\.\d{2})\s+(?P<amt>\d[\d,]*\.\d{2})$")
_GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z])\b")
_SUFFIX = re.compile(r"\b(Pvt\.? Ltd|Private Limited|LLP|Ltd|Limited|Industries|Solutions|Supplies|Electricals|Polymers)\b", re.I)


def _find(pattern, text, flags=re.I | re.M):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def textlayer_extract(pdf: bytes, buyer_name: str, buyer_gstin: str) -> RawExtraction:
    with pdfplumber.open(io.BytesIO(pdf)) as doc:
        text = "\n".join((p.extract_text() or "") for p in doc.pages)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    vendor_name = None
    for l in lines[:12]:
        cand = re.split(r"\s{2,}|\s+(?:Invoice|Inv\.|TAX INVOICE)", l)[0].strip()
        if _SUFFIX.search(cand) and buyer_name.lower() not in cand.lower() and "invoice" not in cand.lower():
            vendor_name = cand
            break
    gstins = [g for g in _GSTIN_RE.findall(text) if g != buyer_gstin]
    inv_no = _find(r"(?:Invoice[ \t]*(?:No\.?|Number|#)|Inv\.?[ \t]*No\.?|Bill[ \t]*No\.?)[ \t]*[:\-]?[ \t]*([A-Z0-9/\-]*\d[A-Z0-9/\-]*)", text)
    date = _find(r"(?:Invoice[ \t]*Date|Dated|Date)[ \t]*[:\-]?[ \t]*(\d{1,2}[\-/. ](?:\d{1,2}|[A-Za-z]{3,9})[\-/. ]\d{2,4})", text)
    po = _find(r"(?:PO[ \t]*(?:No\.?|Number|Ref\.?)|Purchase[ \t]*order|Buyer'?s[ \t]*Order[ \t]*No\.?)[ \t]*[:\-]?[ \t]*(PO[\- ]?\d{3,6})", text)

    items = []
    for l in lines:
        m = _LINE_RE.match(l)
        if m:
            items.append(RawLine(description=m["desc"].strip(), hsn=m["hsn"], quantity=m["qty"], unit=m["unit"],
                                 unit_price=m["rate"], amount=m["amt"]))

    def tax(kind):
        m = re.search(rf"{kind}\s*(?:@\s*)?(\d+(?:\.\d+)?)\s*%\s*{_AMT}", text, re.I)
        return (m.group(1), m.group(2)) if m else (None, None)

    cg, sg, ig = tax("CGST"), tax("SGST"), tax("IGST")
    subtotal = _find(rf"^(?:Sub\s*Total|Taxable\s*(?:Value|Amount))\s*:?\s*{_AMT}", text)
    total = _find(rf"^(?:Grand\s*Total|Amount\s*payable|Invoice\s*Total|Total)\s*:?\s*{_AMT}", text)

    def conf(v):
        return 0.98 if v else 1.0  # sure of what it found; equally sure of what isn't printed

    return RawExtraction(
        vendor_name=vendor_name, vendor_gstin=gstins[0] if gstins else None,
        buyer_name=buyer_name if buyer_name.lower() in text.lower() else None,
        buyer_gstin=buyer_gstin if buyer_gstin in text else None,
        invoice_number=inv_no, invoice_date=date, po_reference=po, line_items=items,
        subtotal=subtotal, cgst_rate=cg[0], cgst_amount=cg[1], sgst_rate=sg[0], sgst_amount=sg[1],
        igst_rate=ig[0], igst_amount=ig[1], total=total, currency="INR",
        confidence=FieldConfidence(vendor_name=conf(vendor_name), vendor_gstin=conf(gstins), invoice_number=conf(inv_no),
                                   invoice_date=conf(date), po_reference=conf(po), line_items=0.98 if items else 0.3,
                                   total=conf(total), tax=0.98),
        notes="Parsed from the PDF text layer with fixed patterns (offline fallback, no LLM).",
    )
