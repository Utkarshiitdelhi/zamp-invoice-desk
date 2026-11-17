"""Static configuration: the buyer, tolerances and file locations.

Everything that changes a decision lives here or in the reason-code policy,
so a reviewer can see every threshold in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CACHE_DIR = DATA_DIR / "extraction_cache"
DB_PATH = DATA_DIR / "ap.sqlite3"

# The buyer we process invoices for (fictional).
BUYER = {
    "name": "Northwind Retail Pvt Ltd",
    "gstin": "27AACCN4821K1ZD",
    "state": "Maharashtra",
    "state_code": "27",
    "address": "Plot 14, MIDC Andheri East, Mumbai 400093",
}

# ---- Tolerances (from the design doc, "Decisions" table) ----
PRICE_TOLERANCE_PCT = 2.0          # unit price may differ from PO by up to 2 %
PRICE_TOLERANCE_ABS_PER_LINE = 500  # ... and the line's total variance is capped at Rs 500
ROUNDING_TOLERANCE = 1.0            # Rs 1 for arithmetic checks
VENDOR_NAME_MATCH_MIN = 90          # rapidfuzz score
PO_INFERENCE_MIN_CONFIDENCE = 0.75
LINE_MATCH_MIN = 70                 # fuzzy score to pair an invoice line with a PO line
DUP_AMOUNT_PCT = 1.0                # suspected duplicate: amount within 1 %
DUP_WINDOW_DAYS = 15                # ... and invoice dates within 15 days
LOW_CONFIDENCE = 0.70               # key-field extraction confidence floor

KEY_FIELDS = ["vendor_name", "vendor_gstin", "invoice_number", "invoice_date", "line_items", "total"]

GEMINI_MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.8-flash-lite",
                           "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-flash-lite-latest"]
