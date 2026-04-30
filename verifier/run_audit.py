"""Top-level audit runner.

Run from the project root:
    python -m verifier.run_audit

Exit code:
    1  if any SECRETS findings are reported (blocking).
    0  otherwise (BUGS and DATA LEAKS are warnings only).
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from typing import Dict, List

from verifier.checks import bugs as _bugs
from verifier.checks import data_leaks as _data_leaks
from verifier.checks import secrets as _secrets


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

# Directories we never want to walk into. Caches and virtualenvs are noisy.
_EXCLUDE_DIRS = {
    ".venv", "venv", "env",
    "__pycache__",
    "cache", "data/cache", "data/faiss_index",
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules",
}


def _project_root() -> Path:
    """Resolve the PathoMAN2.0/ directory (parent of verifier/)."""
    return Path(__file__).resolve().parent.parent


def _is_excluded(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    if parts & _EXCLUDE_DIRS:
        return True
    # Path-prefix exclusions (e.g. data/cache, data/faiss_index)
    rel_str = str(rel).replace("\\", "/")
    for ex in _EXCLUDE_DIRS:
        if "/" in ex and rel_str.startswith(ex + "/"):
            return True
    return False


def _discover_py_files(root: Path) -> List[Path]:
    """Return all .py files under root that are not in excluded directories."""
    out: List[Path] = []
    for p in root.rglob("*.py"):
        if _is_excluded(p, root):
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_section(name: str, findings: List[Dict[str, object]]) -> str:
    n = len(findings)
    flag = "OK" if n == 0 else "WARN"
    return f"  {name:<28}  {n} issues  {flag}"


def _print_report(findings_by_section: Dict[str, List[Dict[str, object]]],
                  project_root: Path) -> None:
    print("=" * 60)
    print("VERIFIER AUDIT REPORT")
    print("=" * 60)
    print(f"PROJECT  : {project_root}")
    print(f"SCAN DATE: {_dt.date.today().isoformat()}")
    print()
    for section in ("SECRETS", "BUGS", "DATA LEAKS"):
        print(_format_section(section, findings_by_section.get(section, [])))
    print()

    # Detailed listings
    any_details = any(findings_by_section.get(s) for s in findings_by_section)
    if any_details:
        print("DETAILS:")
        for section, findings in findings_by_section.items():
            label = section.lower().replace(" ", "_")
            for f in findings:
                line = f.get("line", 0)
                # Print pattern/message but never echo a matched value.
                msg = f.get("message", f.get("pattern", ""))
                print(f"  {label}: {f.get('file', '?')}:{line} - {msg}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    root = _project_root()
    py_files = _discover_py_files(root)

    findings = {
        "SECRETS":    _secrets.scan(root, py_files),
        "BUGS":       _bugs.scan(root, py_files),
        "DATA LEAKS": _data_leaks.scan(root, py_files),
    }
    _print_report(findings, root)

    secret_count = len(findings["SECRETS"])
    exit_code = 1 if secret_count > 0 else 0
    print(f"EXIT CODE: {exit_code} "
          f"({'blocking - secrets found' if exit_code else 'no secrets; bugs/leaks are warnings'})")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
