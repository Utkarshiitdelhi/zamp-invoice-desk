"""Indian number formats: Rs 1,23,456.00 <-> 123456.0, and amounts in words."""
import re


def fmt_inr(x: float | None, symbol: bool = False, decimals: int = 2) -> str:
    if x is None:
        return "—"
    neg = x < 0
    x = abs(round(float(x), decimals))
    whole, _, frac = f"{x:.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    s = whole + (f".{frac}" if decimals else "")
    return ("-" if neg else "") + ("₹" if symbol else "") + s


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_amount(raw) -> float | None:
    """Parse '₹1,23,456.00', 'Rs. 35,931/-', '12,800', 2740.5 -> float. None if unreadable."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    s = s.replace("₹", "").replace("Rs.", "").replace("Rs", "").replace("INR", "")
    s = s.replace("/-", "").replace(",", "").replace(" ", "")
    m = _NUM_RE.search(s)
    return float(m.group()) if m else None


_ONES = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
         "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
_TENS = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]


def _two(n):
    return _ONES[n] if n < 20 else (_TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else ""))


def _three(n):
    h, r = divmod(n, 100)
    out = (_ONES[h] + " Hundred") if h else ""
    if r:
        out += (" " if out else "") + _two(r)
    return out


def in_words(amount: float) -> str:
    rupees = int(amount)
    paise = int(round((amount - rupees) * 100))
    parts = []
    for name, size in (("Crore", 10_000_000), ("Lakh", 100_000), ("Thousand", 1000)):
        q, rupees = divmod(rupees, size)
        if q:
            parts.append(f"{_three(q) if q >= 100 else _two(q)} {name}")
    if rupees:
        parts.append(_three(rupees))
    words = "Rupees " + " ".join(parts)
    if paise:
        words += f" and {_two(paise)} Paise"
    return words + " Only"
