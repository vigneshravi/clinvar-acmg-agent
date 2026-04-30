"""Data-leak scanner: PII / hardcoded paths / contact info in source files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # Absolute home-directory paths (macOS or Linux)
    ("home_path",   re.compile(r"/(?:Users|home)/[A-Za-z0-9_\-.]+/")),
    # Email addresses
    ("email",       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # US phone numbers (loose) — matches (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx
    ("phone",       re.compile(r"\b(?:\+?1[\s\-.])?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4}\b")),
]

# Allow-list email/path domains we don't want to flag (clearly non-PII).
_ALLOW_EMAIL_SUBSTRINGS = (
    "noreply@", "no-reply@", "@example.com", "@example.org", "@anthropic.com",
    "@github.com", "@users.noreply",
)
_ALLOW_LINE_SUBSTRINGS = (
    "# noqa: data-leak",
)


def _is_allowed_email(match: str) -> bool:
    m = match.lower()
    return any(needle in m for needle in _ALLOW_EMAIL_SUBSTRINGS)


def _is_allowed_line(line: str) -> bool:
    return any(needle in line for needle in _ALLOW_LINE_SUBSTRINGS)


def scan(project_root: Path, py_files: List[Path]) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []
    for path in py_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(project_root))
        for ln_no, line in enumerate(text.splitlines(), start=1):
            if _is_allowed_line(line):
                continue
            for label, rx in _PATTERNS:
                m = rx.search(line)
                if not m:
                    continue
                if label == "email" and _is_allowed_email(m.group(0)):
                    continue
                findings.append({
                    "file": rel,
                    "line": ln_no,
                    "pattern": label,
                    "message": f"possible {label} in source",
                })
    return findings
