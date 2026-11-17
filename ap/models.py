"""Data shapes passed between stages.

RawExtraction is what the LLM is allowed to produce: strings exactly as printed,
plus its own confidence. Everything numeric is parsed later by deterministic code.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------- What the reader (Gemini or the text-layer parser) returns ----------
class RawLine(BaseModel):
    description: Optional[str] = Field(None, description="Item description exactly as written")
    hsn: Optional[str] = Field(None, description="HSN/SAC code if printed")
    quantity: Optional[str] = Field(None, description="Quantity as printed, e.g. '2,400' or '20 nos'")
    unit: Optional[str] = Field(None, description="Unit of measure, e.g. Kg, Nos, Roll")
    unit_price: Optional[str] = Field(None, description="Rate per unit as printed")
    amount: Optional[str] = Field(None, description="Line amount as printed (before tax)")


class FieldConfidence(BaseModel):
    vendor_name: float = Field(description="0-1: how sure you are of the value returned")
    vendor_gstin: float
    invoice_number: float
    invoice_date: float
    po_reference: float
    line_items: float
    total: float
    tax: float


class RawExtraction(BaseModel):
    vendor_name: Optional[str] = Field(None, description="Seller's legal name as printed")
    vendor_gstin: Optional[str] = Field(None, description="Seller's 15-character GSTIN")
    buyer_name: Optional[str] = None
    buyer_gstin: Optional[str] = None
    invoice_number: Optional[str] = Field(None, description="Invoice / bill number. null if not printed")
    invoice_date: Optional[str] = Field(None, description="Invoice date exactly as printed")
    po_reference: Optional[str] = Field(None, description="Purchase order number if printed, else null")
    line_items: list[RawLine] = Field(default_factory=list)
    subtotal: Optional[str] = Field(None, description="Taxable value / sub total before GST")
    cgst_rate: Optional[str] = None
    cgst_amount: Optional[str] = None
    sgst_rate: Optional[str] = None
    sgst_amount: Optional[str] = None
    igst_rate: Optional[str] = None
    igst_amount: Optional[str] = None
    total: Optional[str] = Field(None, description="Final invoice total including tax")
    currency: Optional[str] = None
    confidence: FieldConfidence
    notes: Optional[str] = Field(None, description="Anything hard to read, ambiguous or unusual")


# ---------- After normalisation ----------
class Line(BaseModel):
    idx: int
    description: Optional[str]
    hsn: Optional[str]
    qty: Optional[float]
    unit: Optional[str]
    unit_price: Optional[float]
    amount: Optional[float]


class Invoice(BaseModel):
    vendor_name: Optional[str]
    vendor_gstin: Optional[str]
    invoice_number: Optional[str]
    invoice_date: Optional[dt.date]
    po_reference: Optional[str]
    lines: list[Line]
    subtotal: Optional[float]
    cgst: float = 0.0
    sgst: float = 0.0
    igst: float = 0.0
    cgst_rate: Optional[float] = None
    sgst_rate: Optional[float] = None
    igst_rate: Optional[float] = None
    total: Optional[float]
    confidence: dict[str, float]


# ---------- Stage + decision records ----------
Status = Literal["pass", "warn", "fail", "skip"]


class Reason(BaseModel):
    code: str
    message: str            # specific, human-readable: what we saw, against what
    stage: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    key: str
    name: str
    status: Status
    summary: str
    reasons: list[Reason] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    ms: float = 0.0


class Decision(BaseModel):
    outcome: Literal["Approve", "Hold", "Reject"]
    headline: str
    with_note: bool = False
    priority: Literal["normal", "high"] = "normal"
    reasons: list[Reason]
    owners: list[str]
    next_actions: list[str]
    vendor_email: Optional[dict[str, str]] = None
    posted: list[dict[str, Any]] = Field(default_factory=list)
