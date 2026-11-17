"""Fuzzy matching helpers: vendor names and invoice-line <-> PO-line pairing."""
from __future__ import annotations

import re

from rapidfuzz import fuzz

_CORP = re.compile(r"\b(pvt|private|ltd|limited|llp|co|company|m/s|the|and|&)\b\.?", re.I)
_ABBREV = {
    r"\bcu\b": "copper", r"\bdp\b": "double pole", r"\bsqmm\b": "sq mm", r"\bnos?\b": "", r"\bpcs\b": "",
    r"\bmtr?s?\b": "m", r"\bbdl\b": "bundle", r"\bqty\b": "",
}


def norm_name(s: str | None) -> str:
    s = (s or "").lower().replace(".", " ")
    s = _CORP.sub(" ", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def name_score(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    return float(fuzz.token_sort_ratio(norm_name(a), norm_name(b)))


def norm_item(s: str | None) -> str:
    s = (s or "").lower()
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)       # 25mm -> 25 mm
    s = re.sub(r"[()\[\],/]", " ", s)
    for pat, rep in _ABBREV.items():
        s = re.sub(pat, rep, s)
    return re.sub(r"\s+", " ", s).strip()


def line_pair_score(inv_line, po_line) -> tuple[float, dict]:
    """Score how likely an invoice line is this PO line (0-100), with the parts that made the score."""
    text = fuzz.token_set_ratio(norm_item(inv_line.description), norm_item(po_line["item"]))
    hsn_bonus = 8 if inv_line.hsn and inv_line.hsn.strip()[:4] == str(po_line["hsn"])[:4] else 0
    price_bonus = 0
    if inv_line.unit_price and po_line["unit_price"]:
        dev = abs(inv_line.unit_price - po_line["unit_price"]) / po_line["unit_price"]
        price_bonus = 12 if dev <= 0.02 else (6 if dev <= 0.10 else 0)
    score = min(100.0, text * 0.8 + hsn_bonus + price_bonus + (8 if text >= 90 else 0))
    return score, dict(text=round(text), hsn=hsn_bonus, price=price_bonus)


def pair_lines(inv_lines, po_lines, min_score: float) -> list[dict]:
    """Greedy one-to-one pairing, best scores first."""
    cands = []
    for il in inv_lines:
        for pl in po_lines:
            s, parts = line_pair_score(il, pl)
            cands.append((s, il.idx, pl["line_no"], parts))
    cands.sort(key=lambda c: -c[0])
    used_i, used_p, pairs = set(), set(), {}
    for s, i, p, parts in cands:
        if i in used_i or p in used_p or s < min_score:
            continue
        used_i.add(i)
        used_p.add(p)
        pairs[i] = dict(po_line=p, score=round(s), parts=parts)
    return [dict(inv_line=il.idx, **pairs.get(il.idx, dict(po_line=None, score=0, parts={}))) for il in inv_lines]
