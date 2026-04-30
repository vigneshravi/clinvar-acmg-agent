"""One-shot helper: copy the SVI knowledge_base markdowns from FinalTermProject_28Apr2026.

Reason this exists: when the SVI module was integrated into PathoMAN 2.0 on
2026-04-30, the agent harness did not have shell-cp permissions, so the
1061-line guideline corpus could not be inlined into the integration commit.
The agent staged `abou_tayoun_2018_pvs1.md` directly via Write, then this
bootstrap script handles the remaining five files at install time.

Usage (one-time):
    python -m svi.bootstrap_kb

Idempotent — skips files that already exist. After running, build_index() in
svi.rag will pick up the full corpus.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# All 6 source files we want in svi/knowledge_base/
KB_FILES = [
    "abou_tayoun_2018_pvs1.md",
    "enigma_vcep_brca12.md",
    "pejaver_2022_pp3_calibration.md",
    "richards_2015_acmg.md",
    "riggs_2020_clingen_dosage.md",
    "tavtigian_2018_bayesian.md",
]

# Default source path — the user's FinalTermProject submission directory.
# Override via FINALTERMPROJECT_KB env var if the path moves.
import os
DEFAULT_SOURCE = (
    Path(os.environ.get("FINALTERMPROJECT_KB", "")) if os.environ.get("FINALTERMPROJECT_KB")
    else Path.home()
    / "Documents/Rutgers_DHI/BINF_5550_GenerativeAIforHealthCare/FinalTermProject_28Apr2026/knowledge_base"
)

DEST = Path(__file__).resolve().parent / "knowledge_base"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    src = DEFAULT_SOURCE
    if not src.exists():
        print(f"[bootstrap_kb] Source directory not found: {src}")
        print("[bootstrap_kb] Set FINALTERMPROJECT_KB env var to override.")
        return 1

    copied = 0
    skipped = 0
    missing = 0
    for fn in KB_FILES:
        s = src / fn
        d = DEST / fn
        if d.exists() and d.stat().st_size > 0:
            print(f"[bootstrap_kb] skip (already present): {fn}")
            skipped += 1
            continue
        if not s.exists():
            print(f"[bootstrap_kb] MISSING in source: {fn}")
            missing += 1
            continue
        shutil.copy2(s, d)
        print(f"[bootstrap_kb] copied: {fn} ({d.stat().st_size} bytes)")
        copied += 1

    print(f"[bootstrap_kb] done — copied={copied} skipped={skipped} missing={missing}")
    return 0 if missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
