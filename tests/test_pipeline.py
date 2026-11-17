"""Unit + scenario tests. Run: python -m pytest -q

Scanned scenarios need the vision model; here we substitute a recorded-style reading
so the deterministic stages (normalise -> decide) are tested offline.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ap import db, extract, pipeline  # noqa: E402
from ap.config import SAMPLES_DIR  # noqa: E402
from ap.gstin import validate, with_checksum  # noqa: E402
from ap.models import FieldConfidence, RawExtraction, RawLine  # noqa: E402
from ap.money import fmt_inr, in_words, parse_amount  # noqa: E402


def test_gstin_checksum():
    assert validate("27AAGCS5521M1Z1")[0]
    assert not validate("27AAGCS5521M1Z2")[0]
    assert with_checksum("27AACCN4821K1Z") == "27AACCN4821K1ZD"


def test_indian_numbers():
    assert fmt_inr(123456.5) == "1,23,456.50"
    assert parse_amount("₹ 1,23,456.00") == 123456.0
    assert parse_amount("35,931/-") == 35931.0
    assert parse_amount("20 nos") == 20.0
    assert in_words(321432) == "Rupees Three Lakh Twenty One Thousand Four Hundred Thirty Two Only"


def _conf(**over):
    base = dict(vendor_name=0.9, vendor_gstin=0.85, invoice_number=0.9, invoice_date=0.9, po_reference=0.95,
                line_items=0.85, total=0.9, tax=0.9)
    base.update(over)
    return FieldConfidence(**base)


HANDWRITTEN = RawExtraction(
    vendor_name="SHREE GANESH ELECTRICALS", vendor_gstin="27ABZPG4417L1Z2", buyer_name="Northwind Retail Pvt. Ltd.",
    buyer_gstin="27AACCN4821K1ZD", invoice_number="247", invoice_date="12/11/26", po_reference=None,
    line_items=[RawLine(description="MCB 32A DP (Havells)", quantity="20 nos", unit_price="640", amount="12,800"),
                RawLine(description="PVC conduit pipe 25mm", quantity="50 pcs", unit_price="95", amount="4,750"),
                RawLine(description="Cu wire 2.5 sqmm 90m", quantity="6 coil", unit_price="2150", amount="12,900")],
    subtotal="30,450", cgst_rate="9%", cgst_amount="2,740.50", sgst_rate="9%", sgst_amount="2,740.50",
    total="35,931/-", currency="INR", confidence=_conf(invoice_date=0.55),
    notes="Handwritten; the day in the date is partly smudged.")

SCANNED_GPL = RawExtraction(
    vendor_name="Gujarat Polymers Ltd", vendor_gstin="24AABCG6675R1ZX", invoice_number="GPL/INV/26/3381",
    invoice_date="12 Nov 2026", po_reference="PO-1006",
    line_items=[RawLine(description="HDPE granules, injection grade", hsn="3901", quantity="500", unit="Kg",
                        unit_price="118.00", amount="59,000.00")],
    subtotal="59,000.00", igst_rate="18%", igst_amount="10,620.00", total="69,620.00", confidence=_conf())


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.sqlite3")
    monkeypatch.setattr(extract, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("ap.pipeline.cache_get", lambda h: None)
    monkeypatch.setattr("ap.pipeline.cache_put", lambda h, p: None)
    db.init()
    db.reset()


def _run(name, reading=None, monkeypatch=None):
    if reading is not None:
        monkeypatch.setattr("ap.pipeline.gemini_extract", lambda pdf, scanned: (reading, "test-reader"))
        pref = "auto"
    else:
        pref = "offline"
    d, rec, results = pipeline.run_to_end((SAMPLES_DIR / name).read_bytes(), name, pref)
    return d, [r.code for r in d.reasons], results


def test_e2_handwritten_infers_po_and_holds(monkeypatch):
    d, codes, results = _run("05_e2_handwritten_no_po.pdf", HANDWRITTEN, monkeypatch)
    assert d.outcome == "Hold"
    assert "PO_INFERRED" in codes and "LOW_EXTRACTION_CONFIDENCE" in codes
    po = next(r for r in results if r.key == "po")
    assert po.evidence["candidates"][0]["po_number"] == "PO-1005"
    assert po.evidence["candidates"][0]["confidence"] >= 0.75
    arith = next(r for r in results if r.key == "arithmetic")
    assert arith.status == "pass"


def test_scanned_billed_before_received(monkeypatch):
    d, codes, _ = _run("09_backup_billed_before_received.pdf", SCANNED_GPL, monkeypatch)
    assert d.outcome == "Hold" and codes == ["QTY_NOT_RECEIVED"]


def test_misread_number_is_caught_by_arithmetic(monkeypatch):
    bad = HANDWRITTEN.model_copy(deep=True)
    bad.line_items[2].amount = "12,800"   # model misreads 12,900 as 12,800
    d, codes, _ = _run("05_e2_handwritten_no_po.pdf", bad, monkeypatch)
    assert "ARITHMETIC_MISMATCH" in codes and d.outcome == "Reject"


def test_all_digital_scenarios_offline():
    manifest = json.loads((SAMPLES_DIR / "manifest.json").read_text())
    for m in manifest:
        if m["kind"] != "digital":
            continue
        d, rec, _ = pipeline.run_to_end((SAMPLES_DIR / m["file"]).read_bytes(), m["file"], "offline")
        codes = {r.code for r in d.reasons}
        assert d.outcome == m["expected"], m["file"]
        assert set(m["expected_codes"]) <= codes, m["file"]


def test_reproducible():
    name = "08_e4_price_creep_wrong_gst.pdf"
    a, _, _ = _run(name)
    b, _, _ = _run(name)
    assert [r.message for r in a.reasons] == [r.message for r in b.reasons]
    assert a.vendor_email["body"] == b.vendor_email["body"]
    assert "TAX_TYPE_ERROR" in a.vendor_email["body"] or "IGST" in a.vendor_email["body"]
