"""Reset the demo state and run every sample invoice in order, checking expected outcomes.

    python scripts/run_scenarios.py            # uses Gemini if GEMINI_API_KEY is set (and warms the cache)
    python scripts/run_scenarios.py --offline  # text-layer parser only, no LLM
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ap import db  # noqa: E402
from ap.config import SAMPLES_DIR  # noqa: E402
from ap.pipeline import run_to_end  # noqa: E402

offline = "--offline" in sys.argv
db.init()
db.reset()
manifest = json.loads((SAMPLES_DIR / "manifest.json").read_text())
fails = unread = 0
for m in manifest:
    d, rec, results = run_to_end((SAMPLES_DIR / m["file"]).read_bytes(), m["file"], "offline" if offline else "auto")
    codes = list(dict.fromkeys(r.code for r in d.reasons))
    exp_ok = d.outcome == m["expected"] and set(m["expected_codes"]) <= set(codes)
    unavailable = "EXTRACTION_UNAVAILABLE" in codes
    mark = "✓" if exp_ok else ("~" if unavailable else "✗")
    if not exp_ok and not unavailable:
        fails += 1
    if not exp_ok and unavailable:
        unread += 1
    print(f"{mark} {m['file']:42s} {d.outcome:8s} {','.join(codes) or '-':55s} "
          f"[{rec['engine']}, {rec['duration_ms']:.0f} ms]  expected {m['expected']} {m['expected_codes']}")
    if "-v" in sys.argv:
        for r in results:
            print(f"     {r.status:5s} {r.name:22s} {r.summary}")
        for r in d.reasons:
            print(f"       · {r.code}: {r.message}")
ok = len(manifest) - fails - unread
print(f"\n{ok}/{len(manifest)} as expected"
      + (f", {unread} not read (~): the vision model was unavailable. Run again; cached files are skipped."
         if unread else "")
      + (f", {fails} WRONG" if fails else ""))
sys.exit(1 if fails else 0)
