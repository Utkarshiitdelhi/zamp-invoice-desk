"""Draft the vendor email: one message listing every issue the vendor must fix.

The facts come from the rules. The wording is a deterministic template; if
Gemini is available it may rephrase it, but only if every figure survives.
"""
from __future__ import annotations

import os
import re

from .config import BUYER
from .money import fmt_inr
from .policy import BY_CODE

ASK = {
    "MISSING_REQUIRED_FIELD": "reissue the invoice with all mandatory GST fields (invoice number, date and total)",
    "ARITHMETIC_MISMATCH": "send a corrected invoice where the line amounts, subtotal, GST and total reconcile",
    "TAX_TYPE_ERROR": "reissue the invoice with the correct GST type and rate",
    "PO_NOT_FOUND": "share the purchase order number this invoice is billed against",
    "PO_VENDOR_MISMATCH": "confirm the purchase order number; the one quoted was not raised on your company",
    "PRICE_VARIANCE_HIGH": "bill at the unit price agreed on our purchase order, or share the approved price revision",
    "OVER_PO_BALANCE": "limit the invoice to the quantity still open on our purchase order",
    "QTY_NOT_RECEIVED": "invoice only the quantity we have received; the balance can be billed after delivery",
}


def _include(code: str, outcome: str) -> bool:
    mode = BY_CODE[code].email
    return mode == "always" or (mode == "on_reject" and outcome == "Reject")


def draft_vendor_email(ctx, outcome: str, reasons) -> dict | None:
    issues = [r for r in reasons if _include(r.code, outcome)]
    if not issues:
        return None
    inv, vendor = ctx.inv, ctx.vendor
    to = vendor["email"] if vendor else ""
    name = vendor["legal_name"] if vendor else (inv.vendor_name or "Vendor")
    ref = inv.invoice_number or "(no invoice number)"
    date = f" dated {inv.invoice_date:%d %b %Y}" if inv.invoice_date else ""
    amount = f" for ₹{fmt_inr(inv.total)}" if inv.total else ""
    if outcome == "Reject":
        subject = f"Invoice {ref}: please reissue ({len(issues)} issue{'s' if len(issues) > 1 else ''})"
        opening = (f"We have received your invoice {ref}{date}{amount}. We are unable to process it in its current form, "
                   f"for the following reason{'s' if len(issues) > 1 else ''}:")
        close = "Please send a corrected invoice that addresses all of the points above, and we will process it promptly."
    else:
        subject = f"Invoice {ref}: information needed"
        opening = f"We have received your invoice {ref}{date}{amount}. Before we can process it, we need the following:"
        close = "Once we have this, we will continue processing the invoice."
    seen, bullets = set(), []
    for i, r in enumerate(issues, 1):
        detail = r.data.get("vendor") or r.message
        ask = ASK.get(r.code)
        bullets.append(f"{len(bullets) + 1}. {detail}" + (f"\n   Please {ask}." if ask and r.code not in seen else ""))
        seen.add(r.code)
    body = (f"Dear {name} team,\n\n{opening}\n\n" + "\n\n".join(bullets) + f"\n\n{close}\n\n"
            f"Regards,\nAccounts Payable\n{BUYER['name']}\nGSTIN {BUYER['gstin']}")
    email = dict(to=to, subject=subject, body=body, wording="template")
    polished = _polish(email)
    return polished or email


def _figures(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,]*(?:\.\d+)?", text))


def _polish(email: dict) -> dict | None:
    """Optional: let Gemini tighten the wording. Rejected if any figure is lost or invented."""
    if os.environ.get("AP_EMAIL_POLISH", "0") != "1":   # off by default: saves API quota, keeps wording fixed
        return None
    try:
        from .extract import api_key, configured_model
        from .config import GEMINI_MODEL_CANDIDATES
        key = api_key()
        if not key:
            return None
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30_000))
        prompt = ("Rewrite this accounts-payable email to a vendor so it is clear, courteous and concise. Keep it as a "
                  "numbered list of issues. Keep EVERY number, amount, date, invoice and PO number exactly as written. "
                  "Do not add any new facts, numbers or promises. Return only the email body.\n\n" + email["body"])
        from . import extract as _x
        for model in [configured_model() or _x._working_model or GEMINI_MODEL_CANDIDATES[0]]:
            try:
                resp = client.models.generate_content(model=model, contents=prompt,
                                                      config=types.GenerateContentConfig(temperature=0))
                text = (resp.text or "").strip()
                if text and _figures(email["body"]) <= _figures(text) and _figures(text) <= _figures(email["body"]):
                    return {**email, "body": text, "wording": f"{model} (figures verified)"}
                return None
            except Exception:
                continue
    except Exception:
        return None
    return None
