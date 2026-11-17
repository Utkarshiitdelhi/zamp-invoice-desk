# Invoice desk: from vendor PDF to Approve / Hold / Reject

Zamp ASA case study, PS-1 (Finance / AP). One vendor invoice PDF in; a decision out, with every check, its evidence
and the reason visible.

**The LLM reads, rules decide.** Gemini only transcribes the invoice into a strict schema. Every decision about
money is deterministic Python, so the same invoice and data always give the same outcome, traceable to a rule.

## Run it locally

```bash
./run.sh                           # first run creates .venv and installs everything
# or by hand:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=...          # optional: without it, digital PDFs use the offline parser
streamlit run app.py
```

Or put the key in `.streamlit/secrets.toml` (git-ignored):

```toml
GEMINI_API_KEY = "..."
# GEMINI_MODEL = "gemini-2.5-flash"   # optional; default tries gemini-flash-latest, then fallbacks
```

## What happens to an invoice

| # | Stage | What it does |
|---|---|---|
| 1 | Ingest | SHA-256 of the file; text-layer check to classify digital vs scanned |
| 2 | Extract (**LLM**) | Gemini fills a JSON schema, strings exactly as printed, `null` if absent, confidence per field. Scanned pages go in as images. On digital PDFs the reading is cross-checked against the text layer |
| 3 | Normalise | Indian numbers (₹1,23,456.00), dates in any format, required fields, low-confidence fields |
| 4 | Arithmetic & tax | qty × rate = amount, Σ lines = subtotal, subtotal + GST = total (±₹1); CGST+SGST vs IGST from the two states; valid slab |
| 5 | Vendor | GSTIN format + check digit; match by GSTIN, else fuzzy name ≥ 90; active; GSTIN on invoice must equal GSTIN on file |
| 6 | PO match | Explicit PO, or **inferred** from the vendor's open POs by line overlap + amount fit (always held for confirmation) |
| 7 | Line & balance | Per line: price within ±2% / ₹500, quantity ≤ left on PO (from the ledger), quantity ≤ received on GRN (3-way match), GST rate = PO rate |
| 8 | Duplicates | Exact: same hash, or same vendor + number. Suspected: same vendor, amount ±1%, within 15 days, new number |
| 9 | Decide | Most severe code wins (Reject > Hold > Approve); all codes shown; owner, next action, vendor email; ledger posted only on Approve |

The full policy table is in `ap/policy.py` and on the app's **How it works** page.

## Prepared cases (`data/samples/`)

Run them in this order after **Reset demo** (or use *Dashboard → Run the whole sample inbox*):

| File | Case | Expected |
|---|---|---|
| 01 | Happy path: Sharma Steel vs PO-1001, fully received | Approve, ledger updated |
| 02-04 | **E1** split invoices: PO-1003 for 100, billed 60, 40, then 15 | Approve, Approve, Hold OVER_PO_BALANCE (+ QTY_NOT_RECEIVED) |
| 05 | **E2** handwritten bill-book page, photographed skewed, no PO | Hold PO_INFERRED (+ LOW_EXTRACTION_CONFIDENCE on the smudged date). Needs Gemini |
| 06 | **E3** re-issued duplicate: same as an invoice approved 7 Nov, new number, 5 days later | Hold DUPLICATE_SUSPECTED, links the original |
| 07 | E3 bonus: the original file sent again | Reject DUPLICATE_EXACT |
| 08 | **E4** price 6% over PO + IGST on an intra-state supply | Reject, both issues in one vendor email |
| 09 | Backup: scanned, billed 500 kg, 300 kg received | Hold QTY_NOT_RECEIVED. Needs Gemini |
| 10 | Backup: right name, wrong GSTIN | Hold GSTIN_MISMATCH, high priority, no email to the vendor |
| 11 | Backup: +1.48% price, ₹400 on the line | Approve with note PRICE_VARIANCE_OK |
| 12 | Backup: no invoice number | Reject MISSING_REQUIRED_FIELD |

Check them all from the terminal:

```bash
python scripts/run_scenarios.py            # with GEMINI_API_KEY set: real extraction, and warms the cache
python scripts/run_scenarios.py --offline  # no LLM: 10/12 run, the two scans are held (by design)
python -m pytest -q                        # unit + scenario tests (needs requirements-dev.txt)
```

## Making the live demo bullet-proof

Extraction results are cached by file hash in `data/extraction_cache/`. Run
`python scripts/run_scenarios.py` once with your key, then **commit the cache files**. The deployed app then reads
every prepared case instantly and identically, even if Gemini is rate-limited during the interview. A new upload
still goes to Gemini live.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (public or private).
2. share.streamlit.io → Create app → pick the repo, branch `main`, file `app.py`; under Advanced settings choose
   Python 3.12 and paste `GEMINI_API_KEY = "..."` into Secrets.
3. Open the URL, press **Reset demo**, run the happy path.

## Layout

```
app.py                  Streamlit entry, sidebar, navigation
ui/pages.py             Process invoice (live run view), Dashboard, Reference data, How it works
ui/render.py            Stage cards, evidence panels, decision panel, invoice preview + stamp
ap/pipeline.py          The nine stages and the runner (a generator the UI streams)
ap/policy.py            Reason codes -> outcome, owner, action, vendor-email rule
ap/extract.py           Gemini reader, offline text-layer reader, cache
ap/db.py                SQLite: PO ledger, invoice history, run log; reference CSVs
ap/notify.py            Vendor email (template; optional Gemini wording, kept only if every figure survives)
data/*.csv              Vendor master, PO register, GRN register, seed invoices
scripts/make_data.py    Regenerates all reference data and invoice PDFs (deterministic)
```

## Assumptions

- Buyer: Northwind Retail Pvt Ltd, Maharashtra. Its state decides intra-state (CGST + SGST) vs inter-state (IGST).
- Vendor master, PO register and GRN register are CSV exports and the source of truth.
- A missing invoice number or date is a GST defect: Reject (only the vendor can fix it). Hold = we can resolve it.
- One invoice per PDF, INR. All data is synthetic.

Out of scope, named as next steps: reading invoices from the AP mailbox, ERP write-back, multi-currency.
