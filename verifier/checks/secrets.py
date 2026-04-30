"""Hardcoded-secret scanner.

Flags suspicious patterns in source files. We only report findings; we do not
print, log, or copy the matched value itself (only its location and pattern
name) so the audit log itself never becomes a key-leak.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------
# Each entry: (label, regex). The regex must NOT capture into groups that we
# would then echo — we only emit the label + line number.

_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("anthropic_key",        re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_proj_key",      re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}")),
    ("openai_legacy_key",    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("ncbi_key_hex32",       re.compile(r"\b[a-f0-9]{32}\b")),
    ("huggingface_pat",      re.compile(r"\bpat_[A-Za-z0-9]{16,}\b")),
    ("langsmith_key_assign", re.compile(r"LANGSMITH_API_KEY[\"'\s]*[:=][\"'\s]*[A-Za-z0-9_\-]{8,}")),
    # Hardcoded os.environ assignment of any sensitive-looking key
    ("os_environ_assign",    re.compile(r"os\.environ\[[\"'][A-Z_]+[\"']\]\s*=\s*[\"'][^\"']{4,}[\"']")),
]


# ---------------------------------------------------------------------------
# False-positive suppression
# ---------------------------------------------------------------------------
# We allow well-known git hash prefixes and example/template values to avoid
# noise when scanning the verifier checks themselves.
_ALLOW_LINE_SUBSTRINGS = (
    "EXAMPLE",
    "your-key-here",
    "<replace-with-",
    "# noqa: secret",  # explicit ignore marker
)


def _is_allowed(line: str) -> bool:
    line_l = line.lower()
    for needle in _ALLOW_LINE_SUBSTRINGS:
        if needle.lower() in line_l:
            return True
    return False


# ---------------------------------------------------------------------------
# .env / .gitignore checks
# ---------------------------------------------------------------------------

def _check_env_in_gitignore(project_root: Path) -> List[Dict[str, object]]:
    """Return findings if a .env file exists but is not listed in .gitignore."""
    findings: List[Dict[str, object]] = []
    env_files = [p for p in project_root.glob(".env*") if p.is_file() and p.name != ".env.example"]
    if not env_files:
        return findings
    gitignore = project_root / ".gitignore"
    if not gitignore.exists():
        for ef in env_files:
            findings.append({
                "file": str(ef.relative_to(project_root)),
                "line": 0,
                "pattern": "env_not_gitignored",
                "message": ".env file present but .gitignore does not exist",
            })
        return findings
    try:
        gi_text = gitignore.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        gi_text = ""
    gi_lines = {ln.strip() for ln in gi_text.splitlines() if ln.strip() and not ln.strip().startswith("#")}
    covered = (".env" in gi_lines) or any(ln.startswith(".env") for ln in gi_lines)
    if not covered:
        for ef in env_files:
            findings.append({
                "file": str(ef.relative_to(project_root)),
                "line": 0,
                "pattern": "env_not_gitignored",
                "message": ".env file is not covered by .gitignore",
            })
    return findings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def scan(project_root: Path, py_files: List[Path]) -> List[Dict[str, object]]:
    """Scan source for hardcoded-secret patterns.

    Returns a list of finding dicts: {file, line, pattern, message}.
    """
    findings: List[Dict[str, object]] = []

    # 1. Pattern scan over Python source.
    for path in py_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for ln_no, line in enumerate(text.splitlines(), start=1):
            if _is_allowed(line):
                continue
            for label, rx in _PATTERNS:
                if rx.search(line):
                    findings.append({
                        "file": str(path.relative_to(project_root)),
                        "line": ln_no,
                        "pattern": label,
                        "message": f"matched secret pattern: {label}",
                    })

    # 2. .env / .gitignore hygiene.
    findings.extend(_check_env_in_gitignore(project_root))

    return findings
