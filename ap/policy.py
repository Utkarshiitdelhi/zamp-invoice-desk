"""The decision policy: every reason code, the outcome it forces, and who acts on it.

This table *is* the business logic. The outcome of an invoice is the most severe
outcome among the codes that fired (Reject > Hold > Approve); all fired codes are shown.
"""
from dataclasses import dataclass

SEVERITY = {"Approve": 0, "Hold": 1, "Reject": 2}


@dataclass(frozen=True)
class Policy:
    code: str
    label: str
    fires_when: str
    outcome: str            # Approve | Hold | Reject
    owner: str
    action: str
    email: str = "never"    # always | on_reject | never  -> include in the vendor email?
    priority: str = "normal"


POLICIES = [
    Policy("MISSING_REQUIRED_FIELD", "Required field missing", "No invoice number, date or total",
           "Reject", "Vendor", "Ask the vendor to reissue a complete tax invoice", "always"),
    Policy("ARITHMETIC_MISMATCH", "Numbers don't add up", "Lines, subtotal or total don't reconcile (±₹1)",
           "Reject", "Vendor", "Ask the vendor for a corrected invoice", "always"),
    Policy("TAX_TYPE_ERROR", "Wrong GST type or rate", "IGST on an intra-state supply (or the reverse), or a wrong rate",
           "Reject", "Vendor", "Ask the vendor to reissue with the correct GST", "always"),
    Policy("VENDOR_UNKNOWN", "Vendor not in master", "No GSTIN or name match in the vendor master",
           "Hold", "Procurement", "Onboard the vendor or confirm who this is"),
    Policy("VENDOR_BLOCKED", "Vendor is blocked", "Vendor status is blocked in the master",
           "Reject", "Procurement", "Do not pay; procurement to confirm the block"),
    Policy("GSTIN_MISMATCH", "GSTIN differs from file", "Invoice GSTIN ≠ GSTIN on file for this vendor",
           "Hold", "AP manager", "Verify with the vendor on a known contact before paying (possible fraud)",
           "never", "high"),
    Policy("PO_NOT_FOUND", "No usable PO", "No PO reference and nothing inferable, or the reference isn't in the register",
           "Hold", "AP clerk", "Find the PO or ask the vendor for the PO number", "always"),
    Policy("PO_INFERRED", "PO inferred, needs confirmation", "No PO reference; matched to a PO with confidence ≥ 0.75",
           "Hold", "Requester", "Requester confirms the inferred PO"),
    Policy("PO_VENDOR_MISMATCH", "PO belongs to another vendor", "Referenced PO is for a different vendor",
           "Hold", "AP clerk", "Check the PO reference with the vendor", "always"),
    Policy("PO_CLOSED", "PO is closed", "Referenced PO is closed",
           "Hold", "AP clerk", "Check whether the PO should be reopened"),
    Policy("LINE_NOT_ON_PO", "Item not on the PO", "An invoice line matches no PO line",
           "Hold", "Procurement", "Confirm the item was ordered, or amend the PO"),
    Policy("PRICE_VARIANCE_OK", "Price within tolerance", "Unit price up by ≤ 2% and ≤ ₹500 on the line",
           "Approve", "—", "No action; variance noted for audit"),
    Policy("PRICE_VARIANCE_HIGH", "Price above tolerance", "Unit price up by more than 2% or more than ₹500 on a line",
           "Hold", "Procurement", "Procurement to accept the new price or ask for a credit note", "on_reject"),
    Policy("OVER_PO_BALANCE", "Exceeds what's left on the PO", "Quantity or value exceeds the PO's remaining balance",
           "Hold", "AP clerk + buyer", "Buyer to confirm extra quantity or raise a PO amendment", "on_reject"),
    Policy("QTY_NOT_RECEIVED", "Billed more than received", "Billed quantity > quantity received on the GRN",
           "Hold", "Warehouse", "Warehouse to confirm receipt; pay only what has arrived", "on_reject"),
    Policy("DUPLICATE_EXACT", "Already processed", "Same file, or same vendor + invoice number",
           "Reject", "—", "None: this invoice was already processed"),
    Policy("DUPLICATE_SUSPECTED", "Possible duplicate", "Same vendor, amount within 1%, dated within 15 days, new number",
           "Hold", "AP clerk", "Compare with the linked invoice before paying"),
    Policy("LOW_EXTRACTION_CONFIDENCE", "Couldn't read a field reliably", "A key field read with confidence below 0.7",
           "Hold", "AP clerk", "Check the highlighted field against the image"),
    Policy("EXTRACTION_UNAVAILABLE", "Couldn't read the document", "Scanned PDF and the vision model is unavailable",
           "Hold", "AP clerk", "Retry later or key the invoice in manually"),
    Policy("STAGE_ERROR", "Processing error", "A check could not complete (never silently passes)",
           "Hold", "AP clerk", "Review manually; the error is in the run log"),
]

BY_CODE = {p.code: p for p in POLICIES}


def outcome_of(codes: list[str]) -> str:
    if not codes:
        return "Approve"
    return max((BY_CODE[c].outcome for c in codes), key=SEVERITY.__getitem__)
