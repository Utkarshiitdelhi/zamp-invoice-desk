"""Generate all synthetic reference data and the test invoice PDFs.

Run:  python scripts/make_data.py
Writes data/*.csv, data/samples/*.pdf and data/samples/manifest.json.
Everything is deterministic (fixed random seed) so file hashes are stable.
"""
import csv
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from ap.config import BUYER
from ap.money import fmt_inr, in_words

DATA = ROOT / "data"
SAMPLES = DATA / "samples"
FONTS = ROOT / "assets" / "fonts"

for name, file in [("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                   ("Serif", "LiberationSerif-Regular.ttf"), ("Serif-Bold", "LiberationSerif-Bold.ttf"),
                   ("Pop", "Poppins-Regular.ttf"), ("Pop-Bold", "Poppins-Bold.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))

# --------------------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------------------
VENDORS = [
    dict(vendor_id="V001", legal_name="Sharma Steel Industries Pvt Ltd", gstin="27AAGCS5521M1Z1",
         state="Maharashtra", state_code="27", city="Nashik", status="active",
         email="accounts@sharmasteel.example.in", address="Gat No. 112, Ambad MIDC, Nashik 422010"),
    dict(vendor_id="V002", legal_name="Deccan Packaging Solutions LLP", gstin="27AAJFD7364Q1ZM",
         state="Maharashtra", state_code="27", city="Pune", status="active",
         email="billing@deccanpack.example.in", address="S. No. 41, Chakan Industrial Area, Pune 410501"),
    dict(vendor_id="V003", legal_name="Shree Ganesh Electricals", gstin="27ABZPG4417L1Z2",
         state="Maharashtra", state_code="27", city="Pune", status="active",
         email="shreeganesh.elec@example.in", address="Shop No. 7, Laxmi Road, Pune 411002"),
    dict(vendor_id="V004", legal_name="Kaveri Office Supplies Pvt Ltd", gstin="29AADCK8812H1ZB",
         state="Karnataka", state_code="29", city="Bengaluru", status="active",
         email="ar@kaverioffice.example.in", address="No. 18, 2nd Cross, Peenya Industrial Area, Bengaluru 560058"),
    dict(vendor_id="V005", legal_name="Apex Industrial Lubricants Pvt Ltd", gstin="27AAECA3390B1Z1",
         state="Maharashtra", state_code="27", city="Thane", status="active",
         email="finance@apexlubes.example.in", address="Unit 9, Wagle Estate, Thane 400604"),
    dict(vendor_id="V006", legal_name="Gujarat Polymers Ltd", gstin="24AABCG6675R1ZX",
         state="Gujarat", state_code="24", city="Ahmedabad", status="active",
         email="receivables@gujpoly.example.in", address="Plot 220, Vatva GIDC, Ahmedabad 382445"),
    dict(vendor_id="V007", legal_name="Rapid Freight Carriers", gstin="07AAKFR2209N1ZX",
         state="Delhi", state_code="07", city="New Delhi", status="blocked",
         email="ops@rapidfreight.example.in", address="Transport Nagar, New Delhi 110033"),
]
V = {v["vendor_id"]: v for v in VENDORS}

# po_number, vendor, date, status, requester, lines: (item, hsn, qty, unit, unit_price, gst_rate)
POS = [
    ("PO-1001", "V001", "2026-10-20", "open", "Ravi Kulkarni (Projects)", [
        ("MS Angle 50x50x5 mm", "7216", 2400, "Kg", 68.50, 18),
        ("MS Plate 6 mm", "7208", 1500, "Kg", 72.00, 18)]),
    ("PO-1002", "V001", "2026-10-28", "open", "Ravi Kulkarni (Projects)", [
        ("MS Round Bar 20 mm", "7214", 1000, "Kg", 70.00, 18)]),
    ("PO-1003", "V002", "2026-10-15", "open", "Neha Joshi (Warehouse)", [
        ("Corrugated shipper carton 5-ply (bundle of 25)", "4819", 100, "Bdl", 1450.00, 5)]),
    ("PO-1004", "V004", "2026-10-25", "open", "Anita Desai (Admin)", [
        ("A4 copier paper 75 GSM (ream)", "4802", 400, "Ream", 245.00, 18),
        ("Box file, laminated", "4820", 200, "Nos", 85.00, 18)]),
    ("PO-1005", "V003", "2026-11-02", "open", "Sunil Pawar (Facilities)", [
        ("MCB 32A double pole", "8536", 20, "Nos", 640.00, 18),
        ("PVC conduit pipe 25 mm (3 m)", "3917", 50, "Nos", 95.00, 18),
        ("Copper wire 2.5 sqmm (90 m coil)", "8544", 6, "Coil", 2150.00, 18)]),
    ("PO-1010", "V003", "2026-11-05", "open", "Sunil Pawar (Facilities)", [
        ("LED batten 20W", "9405", 40, "Nos", 310.00, 18),
        ("Ceiling fan 1200 mm", "8414", 10, "Nos", 2350.00, 18)]),
    ("PO-1006", "V006", "2026-10-30", "open", "Neha Joshi (Warehouse)", [
        ("HDPE granules, injection grade", "3901", 500, "Kg", 118.00, 18)]),
    ("PO-1007", "V006", "2026-10-30", "open", "Neha Joshi (Warehouse)", [
        ("LDPE stretch film 23 micron (roll)", "3920", 200, "Roll", 540.00, 18)]),
    ("PO-1008", "V005", "2026-10-22", "open", "Ravi Kulkarni (Projects)", [
        ("Hydraulic oil ISO VG 68 (210 L drum)", "2710", 4, "Drum", 28500.00, 18),
        ("Lithium grease EP2 (18 kg bucket)", "2710", 10, "Bkt", 3200.00, 18)]),
    ("PO-1009", "V002", "2026-11-01", "open", "Neha Joshi (Warehouse)", [
        ("Bubble wrap roll 1 m x 100 m", "3921", 30, "Roll", 1180.00, 18)]),
    ("PO-0998", "V005", "2026-08-11", "closed", "Ravi Kulkarni (Projects)", [
        ("Cutting oil, soluble (50 L can)", "3403", 8, "Can", 6200.00, 18)]),
]

# grn_id, po, line_no, qty_received, date
GRNS = [
    ("GRN-5001", "PO-1001", 1, 2400, "2026-11-04"), ("GRN-5001", "PO-1001", 2, 1500, "2026-11-04"),
    ("GRN-5002", "PO-1002", 1, 1000, "2026-11-08"),
    ("GRN-5003", "PO-1003", 1, 60, "2026-11-01"), ("GRN-5004", "PO-1003", 1, 40, "2026-11-09"),
    ("GRN-5005", "PO-1004", 1, 200, "2026-11-04"), ("GRN-5005", "PO-1004", 2, 100, "2026-11-04"),
    ("GRN-5011", "PO-1004", 1, 200, "2026-11-12"), ("GRN-5011", "PO-1004", 2, 100, "2026-11-12"),
    ("GRN-5006", "PO-1005", 1, 20, "2026-11-11"), ("GRN-5006", "PO-1005", 2, 50, "2026-11-11"),
    ("GRN-5006", "PO-1005", 3, 6, "2026-11-11"),
    ("GRN-5007", "PO-1006", 1, 300, "2026-11-10"),
    ("GRN-5008", "PO-1007", 1, 50, "2026-11-10"),
    ("GRN-5009", "PO-1008", 1, 4, "2026-11-06"), ("GRN-5009", "PO-1008", 2, 10, "2026-11-06"),
    ("GRN-5010", "PO-1009", 1, 30, "2026-11-12"),
    ("GRN-4990", "PO-0998", 1, 8, "2026-08-20"),
]

# Invoices already processed before the demo starts (seeded into history + PO ledger).
SEED_INVOICES = [
    # The original Kaveri invoice, approved on 7 Nov. E3's re-issue duplicates it.
    dict(vendor_id="V004", invoice_number="KOS/2526/0871", invoice_date="2026-11-06", po="PO-1004",
         lines=[(1, 200, 245.00), (2, 100, 85.00)], tax="inter", rate=18,
         sample_file="07_e3_original_resent.pdf", processed_at="2026-11-07 11:42"),
    # Old, fully billed PO.
    dict(vendor_id="V005", invoice_number="AIL/2526/0198", invoice_date="2026-08-21", po="PO-0998",
         lines=[(1, 8, 6200.00)], tax="intra", rate=18, sample_file=None, processed_at="2026-08-25 15:10"),
]

# --------------------------------------------------------------------------------------
# Invoices to generate
# --------------------------------------------------------------------------------------
def line(desc, hsn, qty, unit, rate):
    return dict(desc=desc, hsn=hsn, qty=qty, unit=unit, rate=rate)


INVOICES = [
    dict(id="01_happy_sharma_steel", vendor="V001", template="classic", number="SSI/25-26/0412",
         date="2026-11-14", po="PO-1001", tax="intra", rate=18,
         lines=[line("MS Angle 50x50x5 mm", "7216", 2400, "Kg", 68.50),
                line("MS Plate 6 mm", "7208", 1500, "Kg", 72.00)],
         title="Happy path: clean steel invoice",
         story="Digital PDF, explicit PO-1001, goods fully received on GRN-5001. Every check passes.",
         expected="Approve", codes=[]),
    dict(id="02_e1_split_1_of_3", vendor="V002", template="compact", number="DPS/1125/118",
         date="2026-11-03", po="PO-1003", tax="intra", rate=5,
         lines=[line("Corrugated shipper carton 5-ply (bundle of 25)", "4819", 60, "Bdl", 1450.00)],
         title="E1 · Split invoice 1 of 3 (60 bundles)",
         story="PO-1003 is for 100 bundles, all received. First partial bill of 60.",
         expected="Approve", codes=[]),
    dict(id="03_e1_split_2_of_3", vendor="V002", template="compact", number="DPS/1125/131",
         date="2026-11-10", po="PO-1003", tax="intra", rate=5,
         lines=[line("Corrugated shipper carton 5-ply (bundle of 25)", "4819", 40, "Bdl", 1450.00)],
         title="E1 · Split invoice 2 of 3 (40 bundles)",
         story="Second partial bill of 40. Together with the first, the PO is now fully billed.",
         expected="Approve", codes=[]),
    dict(id="04_e1_split_3_of_3", vendor="V002", template="compact", number="DPS/1125/142",
         date="2026-11-15", po="PO-1003", tax="intra", rate=5,
         lines=[line("Corrugated shipper carton 5-ply (bundle of 25)", "4819", 15, "Bdl", 1450.00)],
         title="E1 · Split invoice 3 of 3 (15 more bundles)",
         story="On its own this invoice looks fine. Against the ledger, the PO has nothing left to bill.",
         expected="Hold", codes=["OVER_PO_BALANCE", "QTY_NOT_RECEIVED"]),
    dict(id="05_e2_handwritten_no_po", vendor="V003", template="handwritten", number="247",
         date="2026-11-12", po=None, tax="intra", rate=18,
         lines=[line("MCB 32A double pole", "8536", 20, "Nos", 640.00),
                line("PVC conduit pipe 25 mm (3 m)", "3917", 50, "Nos", 95.00),
                line("Copper wire 2.5 sqmm (90 m coil)", "8544", 6, "Coil", 2150.00)],
         title="E2 · Handwritten bill, badly scanned, no PO",
         story="A local electrical shop's bill-book page, photographed skewed and low-contrast. No PO number. "
               "Needs vision extraction, an arithmetic cross-check, and PO inference.",
         expected="Hold", codes=["PO_INFERRED"]),
    dict(id="06_e3_reissued_duplicate", vendor="V004", template="modern", number="KOS/2526/0894",
         date="2026-11-11", po="PO-1004", tax="inter", rate=18,
         lines=[line("A4 copier paper 75 GSM (ream)", "4802", 200, "Ream", 245.00),
                line("Box file, laminated", "4820", 100, "Nos", 85.00)],
         title="E3 · Re-issued duplicate",
         story="Same vendor, items and amount as KOS/2526/0871 (approved 7 Nov), new number, dated 5 days later. "
               "The PO still has room and goods were received, so only the duplicate check can catch it.",
         expected="Hold", codes=["DUPLICATE_SUSPECTED"]),
    dict(id="07_e3_original_resent", vendor="V004", template="modern", number="KOS/2526/0871",
         date="2026-11-06", po="PO-1004", tax="inter", rate=18,
         lines=[line("A4 copier paper 75 GSM (ream)", "4802", 200, "Ream", 245.00),
                line("Box file, laminated", "4820", 100, "Nos", 85.00)],
         title="E3 bonus · The original, emailed again",
         story="The exact file that was already approved on 7 Nov. Same hash, same number.",
         expected="Reject", codes=["DUPLICATE_EXACT"]),
    dict(id="08_e4_price_creep_wrong_gst", vendor="V005", template="classic", number="AIL/2526/0466",
         date="2026-11-13", po="PO-1008", tax="inter_wrong", rate=18,
         lines=[line("Hydraulic oil ISO VG 68 (210 L drum)", "2710", 4, "Drum", 30210.00),
                line("Lithium grease EP2 (18 kg bucket)", "2710", 10, "Bkt", 3200.00)],
         title="E4 · Price creep + wrong GST type",
         story="Hydraulic oil billed 6% above PO-1008, and IGST charged although vendor and buyer are both in "
               "Maharashtra. Two unrelated problems, one vendor email.",
         expected="Reject", codes=["TAX_TYPE_ERROR", "PRICE_VARIANCE_HIGH"]),
    dict(id="09_backup_billed_before_received", vendor="V006", template="modern", number="GPL/INV/26/3381",
         date="2026-11-12", po="PO-1006", tax="inter", rate=18, scanned=True,
         lines=[line("HDPE granules, injection grade", "3901", 500, "Kg", 118.00)],
         title="Backup · Billed before goods arrived (scanned)",
         story="Scanned invoice for 500 kg; the warehouse has received only 300 kg (GRN-5007).",
         expected="Hold", codes=["QTY_NOT_RECEIVED"]),
    dict(id="10_backup_gstin_mismatch", vendor="V001", template="classic", number="SSI/25-26/0419",
         date="2026-11-15", po="PO-1002", tax="intra", rate=18, gstin_override="27AAGCS9910P1ZM",
         lines=[line("MS Round Bar 20 mm", "7214", 1000, "Kg", 70.00)],
         title="Backup · GSTIN mismatch (fraud signal)",
         story="Looks like Sharma Steel, but the GSTIN printed on the invoice is not the one on file.",
         expected="Hold", codes=["GSTIN_MISMATCH"]),
    dict(id="11_backup_price_within_tolerance", vendor="V006", template="modern", number="GPL/INV/26/3392",
         date="2026-11-14", po="PO-1007", tax="inter", rate=18,
         lines=[line("LDPE stretch film 23 micron (roll)", "3920", 50, "Roll", 548.00)],
         title="Backup · Price variance within tolerance",
         story="Unit price ₹548 vs ₹540 on the PO (+1.48%, ₹400 on the line). Inside the ±2% / ₹500 band.",
         expected="Approve", codes=["PRICE_VARIANCE_OK"]),
    dict(id="12_backup_missing_invoice_number", vendor="V002", template="compact", number=None,
         date="2026-11-16", po="PO-1009", tax="intra", rate=18,
         lines=[line("Bubble wrap roll 1 m x 100 m", "3921", 30, "Roll", 1180.00)],
         title="Backup · Missing invoice number",
         story="No invoice number printed. Under GST that's not a valid tax invoice; only the vendor can fix it.",
         expected="Reject", codes=["MISSING_REQUIRED_FIELD"]),
]


def totals(inv):
    sub = round(sum(l["qty"] * l["rate"] for l in inv["lines"]), 2)
    tax = round(sub * inv["rate"] / 100, 2)
    if inv["tax"] == "intra":
        half = round(tax / 2, 2)
        split = dict(cgst=half, sgst=half, igst=0.0)
    else:
        split = dict(cgst=0.0, sgst=0.0, igst=tax)
    total = round(sub + split["cgst"] + split["sgst"] + split["igst"], 2)
    return sub, split, total


def nice_date(iso, style):
    y, m, d = iso.split("-")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if style == "dmy_slash":
        return f"{d}/{m}/{y}"
    if style == "dmy_dash":
        return f"{d}-{m}-{y}"
    return f"{int(d):02d} {months[int(m) - 1]} {y}"


# --------------------------------------------------------------------------------------
# Typed templates
# --------------------------------------------------------------------------------------
W, H = A4


def _right(c, x, y, text, font, size):
    c.setFont(font, size)
    c.drawRightString(x, y, text)


def template_classic(c, inv, v, gstin):
    """Boxed, gridded, old-school invoice (Sharma Steel, Apex)."""
    sub, split, total = totals(inv)
    navy = HexColor("#1f3a5f")
    m = 36
    c.setStrokeColor(navy)
    c.setLineWidth(1.2)
    c.rect(m, m, W - 2 * m, H - 2 * m)
    # Header
    c.setFillColor(navy)
    c.setFont("Serif-Bold", 20)
    c.drawString(m + 14, H - m - 34, v["legal_name"])
    c.setFillColor(black)
    c.setFont("Serif", 9.5)
    c.drawString(m + 14, H - m - 50, v["address"])
    c.drawString(m + 14, H - m - 63, f"GSTIN: {gstin}    State: {v['state']} ({v['state_code']})")
    c.drawString(m + 14, H - m - 76, f"Email: {v['email']}")
    c.setFillColor(navy)
    c.rect(W - m - 150, H - m - 60, 136, 34, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Serif-Bold", 15)
    c.drawCentredString(W - m - 82, H - m - 48, "TAX INVOICE")
    c.setFillColor(black)
    y = H - m - 92
    c.line(m, y, W - m, y)
    # Meta box
    c.setFont("Serif-Bold", 10)
    c.drawString(m + 14, y - 18, "Invoice No:")
    c.drawString(m + 14, y - 34, "Invoice Date:")
    c.drawString(m + 14, y - 50, "PO No:")
    c.setFont("Serif", 10)
    c.drawString(m + 100, y - 18, inv["number"] or "")
    c.drawString(m + 100, y - 34, nice_date(inv["date"], "dmy_dash"))
    c.drawString(m + 100, y - 50, inv["po"] or "")
    c.setFont("Serif-Bold", 10)
    c.drawString(W / 2 + 10, y - 18, "Bill To:")
    c.setFont("Serif", 10)
    c.drawString(W / 2 + 10, y - 32, BUYER["name"])
    c.drawString(W / 2 + 10, y - 45, BUYER["address"])
    c.drawString(W / 2 + 10, y - 58, f"GSTIN: {BUYER['gstin']}")
    c.drawString(W / 2 + 10, y - 71, f"Place of Supply: {BUYER['state']} ({BUYER['state_code']})")
    y -= 84
    c.line(m, y, W - m, y)
    # Table
    cols = [m, m + 30, m + 250, m + 300, m + 360, m + 400, m + 465, W - m]
    heads = ["Sr", "Description of Goods", "HSN", "Qty", "Unit", "Rate", "Amount"]
    c.setFillColor(HexColor("#e8edf4"))
    c.rect(m, y - 20, W - 2 * m, 20, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Serif-Bold", 9.5)
    for i, h in enumerate(heads):
        c.drawString(cols[i] + 4, y - 14, h)
    ty = y - 20
    c.setFont("Serif", 9.5)
    for i, l in enumerate(inv["lines"], 1):
        ty -= 20
        c.drawString(cols[0] + 6, ty + 6, str(i))
        c.drawString(cols[1] + 4, ty + 6, l["desc"])
        c.drawString(cols[2] + 4, ty + 6, l["hsn"])
        _right(c, cols[4] - 6, ty + 6, fmt_inr(l["qty"], decimals=0), "Serif", 9.5)
        c.drawString(cols[4] + 4, ty + 6, l["unit"])
        _right(c, cols[6] - 6, ty + 6, fmt_inr(l["rate"]), "Serif", 9.5)
        _right(c, cols[7] - 6, ty + 6, fmt_inr(l["qty"] * l["rate"]), "Serif", 9.5)
    ty -= 40
    for x in cols[1:-1]:
        c.line(x, y, x, ty)
    c.line(m, ty, W - m, ty)
    # Totals
    rows = [("Sub Total", sub)]
    if inv["tax"] == "intra":
        rows += [(f"CGST @ {inv['rate'] / 2:g}%", split["cgst"]), (f"SGST @ {inv['rate'] / 2:g}%", split["sgst"])]
    else:
        rows += [(f"IGST @ {inv['rate']:g}%", split["igst"])]
    ry = ty - 18
    for label, val in rows:
        c.setFont("Serif", 10)
        c.drawString(cols[5] - 40, ry, label)
        _right(c, cols[7] - 6, ry, fmt_inr(val), "Serif", 10)
        ry -= 16
    c.setFont("Serif-Bold", 11.5)
    c.drawString(cols[5] - 40, ry - 4, "Total")
    _right(c, cols[7] - 6, ry - 4, "₹ " + fmt_inr(total), "Sans-Bold", 11)
    c.setFont("Serif", 9.5)
    c.drawString(m + 14, ry - 30, f"Amount in words: {in_words(total)}")
    c.drawString(m + 14, H / 4, "Bank: HDFC Bank, Current A/c (details on request) · Payment terms: 30 days")
    c.setFont("Serif-Bold", 10)
    c.drawRightString(W - m - 14, m + 60, f"For {v['legal_name']}")
    c.setFont("Serif", 9)
    c.drawRightString(W - m - 14, m + 22, "Authorised Signatory")


def template_modern(c, inv, v, gstin):
    """Minimal, accent-band invoice (Kaveri, Gujarat Polymers)."""
    sub, split, total = totals(inv)
    accent = HexColor("#0f766e") if v["vendor_id"] == "V004" else HexColor("#7c2d12")
    c.setFillColor(accent)
    c.rect(0, 0, 14, H, fill=1, stroke=0)
    c.setFont("Pop-Bold", 26)
    c.drawString(48, H - 70, "Invoice")
    c.setFillColor(black)
    c.setFont("Pop-Bold", 12)
    c.drawRightString(W - 40, H - 54, v["legal_name"])
    c.setFont("Pop", 8.5)
    c.drawRightString(W - 40, H - 68, v["address"])
    c.drawRightString(W - 40, H - 80, f"GSTIN {gstin}")
    c.drawRightString(W - 40, H - 92, v["email"])
    y = H - 140
    c.setFont("Pop", 9)
    c.setFillColor(HexColor("#555555"))
    meta = [("Invoice number", inv["number"] or ""), ("Invoice date", nice_date(inv["date"], "long")),
            ("Purchase order", inv["po"] or "—"), ("Place of supply", f"{BUYER['state']} (27)")]
    for i, (k, val) in enumerate(meta):
        c.setFillColor(HexColor("#666666"))
        c.setFont("Pop", 8.5)
        c.drawString(48, y - i * 16, k)
        c.setFillColor(black)
        c.setFont("Pop-Bold", 9)
        c.drawString(140, y - i * 16, val)
    c.setFillColor(HexColor("#666666"))
    c.setFont("Pop", 8.5)
    c.drawString(330, y, "Billed to")
    c.setFillColor(black)
    c.setFont("Pop-Bold", 9.5)
    c.drawString(330, y - 16, BUYER["name"])
    c.setFont("Pop", 8.5)
    c.drawString(330, y - 30, BUYER["address"])
    c.drawString(330, y - 44, f"GSTIN {BUYER['gstin']}")
    y -= 90
    cols = [48, 280, 335, 390, 440, 500]
    c.setFont("Pop-Bold", 8.5)
    c.setFillColor(accent)
    for x, h in zip(cols, ["Item", "HSN", "Qty", "Unit", "Rate"]):
        c.drawString(x, y, h)
    c.drawRightString(W - 40, y, "Amount")
    c.setStrokeColor(accent)
    c.line(48, y - 6, W - 40, y - 6)
    c.setFillColor(black)
    for i, l in enumerate(inv["lines"]):
        ry = y - 26 - i * 22
        if i % 2 == 0:
            c.setFillColor(HexColor("#f3f4f4"))
            c.rect(44, ry - 7, W - 80, 20, fill=1, stroke=0)
            c.setFillColor(black)
        c.setFont("Pop", 9)
        c.drawString(cols[0], ry, l["desc"])
        c.drawString(cols[1], ry, l["hsn"])
        c.drawString(cols[2], ry, fmt_inr(l["qty"], decimals=0))
        c.drawString(cols[3], ry, l["unit"])
        c.drawString(cols[4], ry, fmt_inr(l["rate"]))
        c.drawRightString(W - 40, ry, fmt_inr(l["qty"] * l["rate"]))
    ry = y - 26 - len(inv["lines"]) * 22 - 26
    rows = [("Taxable value", sub)]
    if inv["tax"] == "intra":
        rows += [(f"CGST {inv['rate'] / 2:g}%", split["cgst"]), (f"SGST {inv['rate'] / 2:g}%", split["sgst"])]
    else:
        rows += [(f"IGST {inv['rate']:g}%", split["igst"])]
    for label, val in rows:
        c.setFont("Pop", 9)
        c.drawString(390, ry, label)
        c.drawRightString(W - 40, ry, fmt_inr(val))
        ry -= 16
    c.setStrokeColor(black)
    c.line(390, ry + 8, W - 40, ry + 8)
    c.setFont("Pop-Bold", 11)
    c.drawString(390, ry - 10, "Amount payable")
    c.drawRightString(W - 40, ry - 10, fmt_inr(total))
    c.setFont("Pop", 8)
    c.setFillColor(HexColor("#666666"))
    c.drawString(48, ry - 44, in_words(total))
    c.drawString(48, 70, "Thank you for your business. Payment due within 45 days of invoice date.")
    c.drawString(48, 58, "This is a computer generated invoice.")


def template_compact(c, inv, v, gstin):
    """Tally-style compact invoice (Deccan Packaging)."""
    sub, split, total = totals(inv)
    m = 40
    c.setFont("Sans-Bold", 12)
    c.drawCentredString(W / 2, H - 50, "TAX INVOICE")
    c.setFont("Sans", 7)
    c.drawRightString(W - m, H - 50, "(ORIGINAL FOR RECIPIENT)")
    c.setLineWidth(0.6)
    top = H - 60
    c.rect(m, top - 110, W - 2 * m, 110)
    c.line(W / 2, top, W / 2, top - 110)
    c.setFont("Sans-Bold", 10)
    c.drawString(m + 6, top - 16, v["legal_name"])
    c.setFont("Sans", 8)
    c.drawString(m + 6, top - 29, v["address"])
    c.drawString(m + 6, top - 41, f"GSTIN/UIN: {gstin}")
    c.drawString(m + 6, top - 53, f"State Name: {v['state']}, Code: {v['state_code']}")
    c.drawString(m + 6, top - 72, "Buyer (Bill to)")
    c.setFont("Sans-Bold", 8.5)
    c.drawString(m + 6, top - 84, BUYER["name"])
    c.setFont("Sans", 8)
    c.drawString(m + 6, top - 95, f"GSTIN/UIN: {BUYER['gstin']}  State Code: 27")
    labels = [("Inv. No.:", inv["number"] or ""), ("Dated:", nice_date(inv["date"], "dmy_dash")),
              ("Buyer's Order No.:", inv["po"] or ""), ("Mode/Terms of Payment:", "30 days")]
    for i, (k, val) in enumerate(labels):
        c.setFont("Sans", 8)
        c.drawString(W / 2 + 6, top - 16 - i * 22, k)
        c.setFont("Sans-Bold", 8.5)
        c.drawString(W / 2 + 110, top - 16 - i * 22, val)
    y = top - 110
    cols = [m, m + 22, m + 250, m + 300, m + 350, m + 385, m + 445, W - m]
    heads = ["Sl", "Description of Goods", "HSN/SAC", "Quantity", "per", "Rate", "Amount"]
    c.rect(m, y - 18, W - 2 * m, 18)
    c.setFont("Sans-Bold", 7.5)
    for i, h in enumerate(heads):
        c.drawString(cols[i] + 3, y - 12, h)
    body_bottom = y - 18 - 220
    c.rect(m, body_bottom, W - 2 * m, 220)
    for x in cols[1:-1]:
        c.line(x, y, x, body_bottom)
    c.setFont("Sans", 8)
    for i, l in enumerate(inv["lines"], 1):
        ry = y - 32 - (i - 1) * 16
        c.drawString(cols[0] + 4, ry, str(i))
        c.setFont("Sans-Bold", 8)
        c.drawString(cols[1] + 3, ry, l["desc"])
        c.setFont("Sans", 8)
        c.drawString(cols[2] + 3, ry, l["hsn"])
        c.drawRightString(cols[4] - 4, ry, fmt_inr(l["qty"], decimals=0))
        c.drawString(cols[4] + 3, ry, l["unit"])
        c.drawRightString(cols[6] - 4, ry, fmt_inr(l["rate"]))
        c.drawRightString(cols[7] - 4, ry, fmt_inr(l["qty"] * l["rate"]))
    ry = body_bottom + 70
    rows = [("Taxable Amount", sub)]
    if inv["tax"] == "intra":
        rows += [(f"Add: CGST {inv['rate'] / 2:g}%", split["cgst"]), (f"Add: SGST {inv['rate'] / 2:g}%", split["sgst"])]
    else:
        rows += [(f"Add: IGST {inv['rate']:g}%", split["igst"])]
    for label, val in rows:
        c.setFont("Sans", 8)
        c.drawString(cols[1] + 120, ry, label)
        c.drawRightString(cols[7] - 4, ry, fmt_inr(val))
        ry -= 14
    c.rect(m, body_bottom - 20, W - 2 * m, 20)
    c.setFont("Sans-Bold", 9)
    c.drawString(cols[1] + 120, body_bottom - 14, "Grand Total")
    c.drawRightString(cols[7] - 4, body_bottom - 14, "₹ " + fmt_inr(total))
    c.setFont("Sans", 7.5)
    c.drawString(m + 4, body_bottom - 36, "Amount Chargeable (in words)")
    c.setFont("Sans-Bold", 8)
    c.drawString(m + 4, body_bottom - 48, "INR " + in_words(total).replace("Rupees ", ""))
    c.setFont("Sans", 7.5)
    c.drawString(m + 4, body_bottom - 80, "Declaration: We declare that this invoice shows the actual price of the goods")
    c.drawString(m + 4, body_bottom - 90, "described and that all particulars are true and correct.")
    c.setFont("Sans-Bold", 8)
    c.drawRightString(W - m - 4, body_bottom - 80, f"for {v['legal_name']}")
    c.setFont("Sans", 7.5)
    c.drawRightString(W - m - 4, body_bottom - 120, "Authorised Signatory")
    c.drawCentredString(W / 2, 40, "This is a Computer Generated Invoice")


TEMPLATES = dict(classic=template_classic, modern=template_modern, compact=template_compact)


def render_typed(inv) -> bytes:
    v = V[inv["vendor"]]
    gstin = inv.get("gstin_override") or v["gstin"]
    tax_mode = inv["tax"]
    inv = dict(inv)
    if tax_mode == "inter_wrong":
        inv["tax"] = "inter"
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, invariant=1)  # invariant => byte-stable output
    c.setTitle(f"Invoice {inv['number'] or ''}")
    c.setAuthor(v["legal_name"])
    TEMPLATES[inv["template"]](c, inv, v, gstin)
    c.showPage()
    c.save()
    return buf.getvalue()


# --------------------------------------------------------------------------------------
# Scans
# --------------------------------------------------------------------------------------
def degrade(img: Image.Image, rng: random.Random, angle: float, contrast: float, blur: float,
            quality: int) -> Image.Image:
    img = img.convert("RGB")
    # paper tone + noise
    w, h = img.size
    noise = Image.effect_noise((w, h), 22).convert("L")
    tone = Image.new("RGB", (w, h), (236, 232, 222))
    img = Image.blend(img, tone, 0.18)
    img = Image.composite(img, Image.new("RGB", (w, h), (205, 200, 190)), noise.point(lambda p: 255 if p > 40 else 225))
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(70, 66, 60))
    # uneven lighting: darker toward one corner
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    g = ((xx / w) * 0.55 + (yy / h) * 0.45) ** 2.2          # smooth falloff toward one corner
    mask = Image.fromarray((g * 95).astype("uint8"), "L")
    shade = Image.new("RGB", (w, h), (40, 36, 30))
    img = Image.composite(shade, img, mask)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue()))


def image_pdf(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    doc = fitz.open()
    page = doc.new_page(width=A4[0], height=A4[1])
    page.insert_image(page.rect, stream=buf.getvalue())
    out = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    return out


def render_scanned(inv, rng) -> bytes:
    pdf = fitz.open(stream=render_typed(inv), filetype="pdf")
    pix = pdf[0].get_pixmap(dpi=130)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = degrade(img, rng, angle=-1.6, contrast=0.85, blur=0.6, quality=55)
    return image_pdf(img)


def render_handwritten(inv, rng) -> bytes:
    """A carbon bill-book page from a small shop, filled in by hand, photographed on a phone."""
    v = V[inv["vendor"]]
    sub, split, total = totals(inv)
    Wp, Hp = 1240, 1754
    img = Image.new("RGB", (Wp, Hp), (250, 247, 238))
    d = ImageDraw.Draw(img)
    maroon = (128, 30, 40)
    f = lambda name, size: ImageFont.truetype(str(FONTS / name), size)
    # --- pre-printed bill book (maroon) ---
    d.text((Wp / 2, 70), "|| Shree Ganeshaya Namah ||", fill=maroon, font=f("LiberationSerif-Regular.ttf", 24), anchor="mm")
    d.text((Wp / 2, 130), "SHREE GANESH ELECTRICALS", fill=maroon, font=f("LiberationSerif-Bold.ttf", 60), anchor="mm")
    d.text((Wp / 2, 185), "Dealers in Electrical Goods, Wires, Switchgear & Fittings", fill=maroon,
           font=f("LiberationSerif-Regular.ttf", 28), anchor="mm")
    d.text((Wp / 2, 222), "Shop No. 7, Laxmi Road, Pune 411002   Mob. 98220 41736", fill=maroon,
           font=f("LiberationSerif-Regular.ttf", 26), anchor="mm")
    d.text((Wp / 2, 262), f"GSTIN : {v['gstin']}", fill=maroon, font=f("LiberationSerif-Bold.ttf", 30), anchor="mm")
    d.rectangle((60, 290, Wp - 60, Hp - 70), outline=maroon, width=3)
    d.text((Wp / 2, 318), "TAX INVOICE", fill=maroon, font=f("LiberationSerif-Bold.ttf", 30), anchor="mm")
    d.line((60, 340, Wp - 60, 340), fill=maroon, width=2)
    lab = f("LiberationSerif-Regular.ttf", 28)
    d.text((85, 365), "No.", fill=maroon, font=lab)
    d.text((Wp - 420, 365), "Date", fill=maroon, font=lab)
    d.text((85, 430), "M/s.", fill=maroon, font=lab)
    d.line((150, 462, Wp - 85, 462), fill=maroon, width=1)
    d.line((85, 522, Wp - 85, 522), fill=maroon, width=1)
    d.text((85, 535), "Customer GSTIN", fill=maroon, font=lab)
    d.line((300, 567, Wp - 85, 567), fill=maroon, width=1)
    top = 590
    cols = [60, 140, 720, 860, 1000, Wp - 60]
    d.line((60, top, Wp - 60, top), fill=maroon, width=2)
    for x, h in zip(cols, ["Sr.", "Particulars", "Qty.", "Rate", "Amount Rs."]):
        d.text((x + 14, top + 12), h, fill=maroon, font=lab)
    d.line((60, top + 55, Wp - 60, top + 55), fill=maroon, width=2)
    bottom = 1290
    for x in cols[1:-1]:
        d.line((x, top, x, bottom), fill=maroon, width=2)
    d.line((60, bottom, Wp - 60, bottom), fill=maroon, width=2)
    d.text((85, Hp - 150), "E. & O.E.", fill=maroon, font=f("LiberationSerif-Regular.ttf", 22))
    d.text((Wp - 470, Hp - 115), "For SHREE GANESH ELECTRICALS", fill=maroon, font=f("LiberationSerif-Bold.ttf", 22))

    # --- handwriting (blue ballpoint, Kalam with per-word jitter) ---
    ink = (28, 44, 120)
    hand = lambda size: ImageFont.truetype(str(FONTS / "Kalam-Regular.ttf"), size)

    def write(x, y, text, size=40, jitter=3, slant=0.0):
        for word in text.split(" "):
            fnt = hand(size + rng.randint(-2, 2))
            bbox = fnt.getbbox(word)
            tw, th = bbox[2] + 8, bbox[3] + 14
            layer = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            ImageDraw.Draw(layer).text((2, 2), word, font=fnt, fill=ink + (235,))
            layer = layer.rotate(slant + rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True)
            img.paste(layer, (int(x), int(y + rng.uniform(-jitter, jitter))), layer)
            x += tw + fnt.getlength(" ") * 0.6
        return x

    write(150, 352, inv["number"], 46)
    write(Wp - 340, 350, "12/11/26", 44)
    write(170, 412, "Northwind Retail Pvt. Ltd.", 44)
    write(100, 474, "Plot 14, MIDC, Andheri (E), Mumbai", 40)
    write(320, 522, BUYER["gstin"], 38)
    rows = [("1", "MCB 32A DP (Havells)", "20 nos", "640", "12,800"),
            ("2", "PVC conduit pipe 25mm", "50 pcs", "95", "4,750"),
            ("3", "Cu wire 2.5 sqmm 90m", "6 coil", "2150", "12,900")]
    y = top + 75
    for sr, part, q, r, a in rows:
        write(cols[0] + 25, y, sr, 42)
        write(cols[1] + 15, y, part, 42)
        write(cols[2] + 12, y, q, 40)
        write(cols[3] + 18, y, r, 42)
        write(cols[4] + 20, y, a, 42)
        y += 92
    y = 1010
    write(cols[1] + 230, y, "Total", 42)
    write(cols[4] + 20, y, "30,450", 42)
    d.line((cols[4] + 10, y - 6, Wp - 75, y - 6), fill=ink, width=2)
    write(cols[1] + 230, y + 70, "CGST 9%", 40)
    write(cols[4] + 20, y + 70, "2,740.50", 40)
    write(cols[1] + 230, y + 135, "SGST 9%", 40)
    write(cols[4] + 20, y + 135, "2,740.50", 40)
    write(cols[1] + 160, bottom + 25, "Grand Total Rs.", 46)
    write(cols[4] + 10, bottom + 22, "35,931/-", 50)
    d.line((cols[4] + 5, bottom + 92, Wp - 75, bottom + 92), fill=ink, width=2)
    d.line((cols[4] + 5, bottom + 100, Wp - 75, bottom + 100), fill=ink, width=2)
    write(85, bottom + 120, "Rs. Thirty five thousand nine hundred", 34)
    write(85, bottom + 165, "thirty one only", 34)
    write(Wp - 330, Hp - 215, "S. Gaikwad", 48, slant=8)

    # smudge across the day of the date (the realistic "can't quite read it" case)
    smudge = Image.new("L", (Wp, Hp), 0)
    ImageDraw.Draw(smudge).ellipse((Wp - 350, 340, Wp - 262, 414), fill=215)
    smudge = smudge.filter(ImageFilter.GaussianBlur(14))
    img = Image.composite(Image.new("RGB", (Wp, Hp), (95, 105, 150)), img, smudge)
    # folded crease
    d = ImageDraw.Draw(img)
    d.line((0, 820, Wp, 836), fill=(222, 216, 204), width=5)

    # phone photo on a desk: rotate, darken, blur, low contrast, JPEG
    img = degrade(img, rng, angle=4.2, contrast=0.72, blur=1.1, quality=42)
    return image_pdf(img)


# --------------------------------------------------------------------------------------
def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def main():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    rng = random.Random(17)
    write_csv(DATA / "vendors.csv", VENDORS,
              ["vendor_id", "legal_name", "gstin", "state", "state_code", "city", "status", "email", "address"])
    po_rows = []
    for po, vid, date, status, req, lines in POS:
        for i, (item, hsn, qty, unit, price, gst) in enumerate(lines, 1):
            po_rows.append(dict(po_number=po, vendor_id=vid, po_date=date, status=status, requester=req,
                                line_no=i, item=item, hsn=hsn, qty=qty, unit=unit, unit_price=price, gst_rate=gst))
    write_csv(DATA / "po_register.csv", po_rows, list(po_rows[0].keys()))
    write_csv(DATA / "grn.csv", [dict(grn_id=g, po_number=p, line_no=l, qty_received=q, received_date=d)
                                 for g, p, l, q, d in GRNS],
              ["grn_id", "po_number", "line_no", "qty_received", "received_date"])
    seed = []
    for s in SEED_INVOICES:
        po_lines = {r["line_no"]: r for r in po_rows if r["po_number"] == s["po"]}
        sub = sum(q * p for _, q, p in s["lines"])
        tax = round(sub * s["rate"] / 100, 2)
        seed.append(dict(vendor_id=s["vendor_id"], invoice_number=s["invoice_number"], invoice_date=s["invoice_date"],
                         po_number=s["po"], subtotal=sub, total=round(sub + tax, 2),
                         lines=json.dumps([dict(line_no=l, qty=q, amount=round(q * p, 2)) for l, q, p in s["lines"]]),
                         sample_file=s["sample_file"] or "", processed_at=s["processed_at"]))
        assert all(l in po_lines for l, _, _ in s["lines"])
    write_csv(DATA / "seed_invoices.csv", seed, list(seed[0].keys()))

    manifest = []
    for inv in INVOICES:
        if inv["template"] == "handwritten":
            pdf = render_handwritten(inv, rng)
        elif inv.get("scanned"):
            pdf = render_scanned(inv, rng)
        else:
            pdf = render_typed(inv)
        fname = f"{inv['id']}.pdf"
        (SAMPLES / fname).write_bytes(pdf)
        sub, split, total = totals(inv if inv["tax"] != "inter_wrong" else {**inv, "tax": "inter"})
        manifest.append(dict(file=fname, title=inv["title"], story=inv["story"], vendor=V[inv["vendor"]]["legal_name"],
                             expected=inv["expected"], expected_codes=inv["codes"], total=total,
                             kind="handwritten scan" if inv["template"] == "handwritten" else
                             ("scan" if inv.get("scanned") else "digital")))
        print(f"  {fname:42s} {manifest[-1]['kind']:17s} ₹{fmt_inr(total):>14s}  -> {inv['expected']}")
    (SAMPLES / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {len(manifest)} invoices, {len(VENDORS)} vendors, {len(POS)} POs, {len(GRNS)} GRN lines.")


if __name__ == "__main__":
    main()
