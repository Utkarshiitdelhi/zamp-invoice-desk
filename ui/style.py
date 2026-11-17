"""Visual system: one stylesheet and small HTML helpers.

Palette: ledger paper, blue-black ink, and the three rubber-stamp inks AP desks
actually use (green APPROVED, amber ON HOLD, red REJECTED). The stamp is the one
loud element; everything else stays quiet.
"""
import html

import streamlit as st

INK = "#18212F"
MUTED = "#5C6675"
RULE = "#DFE3E8"
PAPER = "#F6F7F5"
BLUE = "#2D4BA0"
GREEN = "#1E7A4C"
AMBER = "#A3650F"
RED = "#B3261E"
OUTCOME_COLOR = {"Approve": GREEN, "Hold": AMBER, "Reject": RED}
OUTCOME_STAMP = {"Approve": "APPROVED", "Hold": "ON HOLD", "Reject": "REJECTED"}
STATUS_LABEL = {"pass": "Pass", "warn": "Warn", "fail": "Fail", "skip": "Skipped", "running": "Running"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Condensed:wght@600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
.stApp {{ background: {PAPER}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 600; font-size: 1.85rem !important; }}
.num, td, .kpi-value {{ font-variant-numeric: tabular-nums; }}
.lede {{ color: {MUTED}; font-size: 1rem; margin: -0.4rem 0 1.2rem 0; max-width: 72ch; }}

/* ---- stage timeline ---- */
.stage {{ display: grid; grid-template-columns: 34px 1fr; gap: 12px; padding: 12px 14px 12px 10px;
  background: #fff; border: 1px solid {RULE}; border-radius: 6px; margin-bottom: 6px; }}
.stage .dot {{ width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: 0.8rem; font-weight: 600; color: #fff; background: #B8C0CA; }}
.stage.pass .dot {{ background: {GREEN}; }}
.stage.warn .dot {{ background: {AMBER}; }}
.stage.fail .dot {{ background: {RED}; }}
.stage.skip {{ opacity: .55; }}
.stage.running {{ border-color: {BLUE}; box-shadow: 0 0 0 1px {BLUE} inset; }}
.stage.running .dot {{ background: {BLUE}; animation: pulse 1s ease-in-out infinite; }}
.stage.pending {{ background: transparent; border-style: dashed; }}
.stage.pending .name {{ color: {MUTED}; }}
@keyframes pulse {{ 50% {{ box-shadow: 0 0 0 6px rgba(45,75,160,.18); }} }}
.stage .top {{ display: flex; align-items: baseline; gap: 10px; }}
.stage .name {{ font-weight: 600; color: {INK}; }}
.stage .desc {{ color: {MUTED}; font-size: .82rem; }}
.stage .ms {{ margin-left: auto; color: {MUTED}; font-size: .78rem; font-variant-numeric: tabular-nums; }}
.stage .summary {{ color: {INK}; font-size: .9rem; margin-top: 3px; line-height: 1.4; }}
.stage .llm {{ font-size: .7rem; font-weight: 600; color: {BLUE}; border: 1px solid {BLUE}; border-radius: 3px;
  padding: 0 5px; }}
.pill {{ font-size: .72rem; font-weight: 600; padding: 1px 7px; border-radius: 10px; }}
.pill.pass {{ background: #E3F1E9; color: {GREEN}; }}
.pill.warn {{ background: #FBEFD9; color: {AMBER}; }}
.pill.fail {{ background: #F8E1DF; color: {RED}; }}
.pill.skip, .pill.pending {{ background: #ECEEF1; color: {MUTED}; }}
.pill.running {{ background: #E5EAF7; color: {BLUE}; }}

/* ---- invoice preview + stamp ---- */
.paper {{ position: relative; background: #fff; border: 1px solid {RULE}; border-radius: 4px; padding: 6px;
  box-shadow: 0 1px 0 {RULE}; }}
.paper img {{ width: 100%; display: block; }}
.stamp {{ position: absolute; top: 36%; left: 50%; transform: translate(-50%, -50%) rotate(-11deg);
  font-family: 'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif; font-weight: 700; letter-spacing: .08em;
  font-size: 3rem; padding: 4px 20px 2px; border: 5px double currentColor; border-radius: 8px;
  opacity: .86; mix-blend-mode: multiply; white-space: nowrap; text-align: center; line-height: 1.05;
  animation: thunk .28s cubic-bezier(.2,.9,.3,1.3) both; }}
.stamp small {{ display: block; font-size: .78rem; letter-spacing: .12em; font-weight: 600; }}
@keyframes thunk {{ from {{ transform: translate(-50%,-50%) rotate(-11deg) scale(1.6); opacity: 0; }}
  to {{ transform: translate(-50%,-50%) rotate(-11deg) scale(1); opacity: .86; }} }}
@media (prefers-reduced-motion: reduce) {{ .stamp, .stage.running .dot {{ animation: none; }} }}

/* ---- verdict ---- */
.verdict {{ border-left: 6px solid; background: #fff; border-radius: 6px; padding: 16px 20px; margin: 4px 0 14px;
  border-top: 1px solid {RULE}; border-right: 1px solid {RULE}; border-bottom: 1px solid {RULE}; }}
.verdict .outcome {{ font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700; letter-spacing: .06em;
  font-size: .95rem; }}
.verdict .headline {{ font-size: 1.35rem; font-weight: 600; color: {INK}; margin: 2px 0 6px; }}
.verdict .meta {{ color: {MUTED}; font-size: .9rem; }}
.verdict .meta b {{ color: {INK}; font-weight: 600; }}
.reason {{ display: grid; grid-template-columns: 210px 1fr; gap: 14px; padding: 10px 0; border-top: 1px solid {RULE}; }}
.reason:first-child {{ border-top: none; }}
.code {{ font-size: .74rem; font-weight: 600; padding: 2px 7px; border-radius: 3px; display: inline-block;
  border: 1px solid currentColor; }}
.reason .label {{ font-weight: 600; color: {INK}; margin-top: 4px; font-size: .9rem; }}
.reason .who {{ color: {MUTED}; font-size: .8rem; margin-top: 2px; }}
.reason .msg {{ color: {INK}; font-size: .92rem; line-height: 1.5; }}
.reason .stage-from {{ color: {MUTED}; font-size: .78rem; margin-top: 4px; }}
.email {{ background: #fff; border: 1px solid {RULE}; border-radius: 6px; padding: 14px 18px; }}
.email .hdr {{ color: {MUTED}; font-size: .85rem; border-bottom: 1px solid {RULE}; padding-bottom: 8px; margin-bottom: 10px; }}
.email .hdr b {{ color: {INK}; font-weight: 600; }}
.email pre {{ white-space: pre-wrap; font-family: 'IBM Plex Sans', sans-serif; font-size: .9rem; color: {INK};
  margin: 0; line-height: 1.5; background: none; }}

/* ---- dashboard ---- */
.kpis {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 14px; }}
.kpi {{ background: #fff; border: 1px solid {RULE}; border-radius: 6px; padding: 12px 14px; }}
.kpi-label {{ color: {MUTED}; font-size: .82rem; }}
.kpi-value {{ color: {INK}; font-size: 1.6rem; font-weight: 600; margin-top: 2px; }}
.kpi-sub {{ color: {MUTED}; font-size: .78rem; }}
.chip {{ font-size: .75rem; font-weight: 600; padding: 1px 8px; border-radius: 10px; }}
.empty {{ border: 1px dashed {RULE}; border-radius: 6px; padding: 28px; text-align: center; color: {MUTED}; background: #fff; }}
.sidebar-engine {{ font-size: .85rem; line-height: 1.45; padding: 8px 10px; background: #fff; border: 1px solid {RULE};
  border-radius: 6px; }}
.sidebar-engine .d {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
section[data-testid="stSidebar"] {{ background: #EEF0EE; }}
/* stage card + its evidence expander read as one unit */
[class*="st-key-stg-"], [class*="st-key-stg-"] [data-testid="stVerticalBlock"] {{ gap: 0 !important; }}
[data-testid="stColumn"]:has(.paper) {{ position: sticky; top: 3.5rem; align-self: flex-start; }}
[class*="st-key-stg-"] .stage {{ margin-bottom: 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0; }}
[class*="st-key-stg-"] [data-testid="stExpander"] details {{ border-top: none; border-top-left-radius: 0;
  border-top-right-radius: 0; background: #fff; border-color: {RULE}; }}
[class*="st-key-stg-"] [data-testid="stExpander"] summary {{ padding-top: .35rem; padding-bottom: .35rem; }}
[class*="st-key-stg-"] [data-testid="stExpander"] summary p {{ font-size: .8rem; color: {MUTED}; }}
[class*="st-key-stg-"] .stage.skip, [class*="st-key-stg-"] .stage.pending, [class*="st-key-stg-"] .stage.running
  {{ border-radius: 6px; margin-bottom: 0; }}
@media (max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, 1fr); }} .reason {{ grid-template-columns: 1fr; }} }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def stage_card(idx: int, name: str, desc: str, status: str, summary: str = "", ms: float | None = None,
               llm: bool = False) -> str:
    mark = {"pass": "✓", "warn": "!", "fail": "✕", "skip": "–"}.get(status, str(idx))
    pill = "" if status == "pending" else f'<span class="pill {status}">{STATUS_LABEL.get(status, status)}</span>'
    timing = f'<span class="ms">{ms:,.0f} ms</span>' if ms is not None else ""
    llm_tag = '<span class="llm" title="The only stage that uses an LLM">LLM</span>' if llm else ""
    body = f'<div class="summary">{e(summary)}</div>' if summary else f'<div class="desc">{e(desc)}</div>'
    return (f'<div class="stage {status}"><div class="dot">{mark}</div><div>'
            f'<div class="top"><span class="name">{idx}. {e(name)}</span>{llm_tag}{pill}{timing}</div>{body}</div></div>')


def outcome_chip(outcome: str, with_note: bool = False) -> str:
    c = OUTCOME_COLOR[outcome]
    label = outcome + (" · note" if with_note else "")
    return f'<span class="chip" style="color:{c};border:1px solid {c}">{e(label)}</span>'
