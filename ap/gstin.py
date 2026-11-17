"""GSTIN format + checksum validation (the official mod-36 Luhn variant)."""
import re

_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")

STATE_CODES = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "19": "West Bengal", "21": "Odisha", "23": "Madhya Pradesh", "24": "Gujarat",
    "27": "Maharashtra", "29": "Karnataka", "30": "Goa", "32": "Kerala", "33": "Tamil Nadu",
    "36": "Telangana", "37": "Andhra Pradesh",
}


def checksum_char(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        product = _CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += product // 36 + product % 36
    return _CHARS[(36 - total % 36) % 36]


def with_checksum(first14: str) -> str:
    return first14 + checksum_char(first14)


def clean(gstin: str | None) -> str | None:
    if not gstin:
        return None
    return re.sub(r"[^0-9A-Z]", "", gstin.upper())


def validate(gstin: str | None) -> tuple[bool, str]:
    """Return (ok, reason)."""
    g = clean(gstin)
    if not g:
        return False, "GSTIN missing"
    if not GSTIN_RE.match(g):
        return False, f"'{g}' is not in GSTIN format (2-digit state, PAN, entity, Z, check digit)"
    if g[:2] not in STATE_CODES:
        return False, f"unknown state code {g[:2]}"
    expected = checksum_char(g[:14])
    if g[14] != expected:
        return False, f"check digit is '{g[14]}', expected '{expected}'"
    return True, "format and check digit valid"


def state_of(gstin: str | None) -> str | None:
    g = clean(gstin)
    return STATE_CODES.get(g[:2]) if g and len(g) >= 2 else None
