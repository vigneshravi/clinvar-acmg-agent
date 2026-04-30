"""AST-based bug-pattern scanner for PathoMAN 2.0 source.

Flags four common antipatterns:
    1. print(os.environ[...])          -> potential key leak via stdout
    2. except: pass / except Exception: pass -> silent failures
    3. Mutable default arguments       -> shared-state bugs
    4. Comparison to None with == / != -> use 'is' / 'is not'
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List


def _is_print_environ(node: ast.AST) -> bool:
    """Detect `print(os.environ[...])` and `print(os.environ.get(...))`."""
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if not (isinstance(fn, ast.Name) and fn.id == "print"):
        return False
    for arg in node.args:
        # os.environ[...]
        if isinstance(arg, ast.Subscript):
            val = arg.value
            if (isinstance(val, ast.Attribute)
                    and val.attr == "environ"
                    and isinstance(val.value, ast.Name)
                    and val.value.id == "os"):
                return True
        # os.environ.get(...) or os.getenv(...)
        if isinstance(arg, ast.Call):
            f = arg.func
            if (isinstance(f, ast.Attribute)
                    and f.attr == "get"
                    and isinstance(f.value, ast.Attribute)
                    and f.value.attr == "environ"
                    and isinstance(f.value.value, ast.Name)
                    and f.value.value.id == "os"):
                return True
            if (isinstance(f, ast.Attribute)
                    and f.attr == "getenv"
                    and isinstance(f.value, ast.Name)
                    and f.value.id == "os"):
                return True
    return False


def _is_silent_except(node: ast.ExceptHandler) -> bool:
    """Detect `except [...]: pass` blocks."""
    body = node.body
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def _has_mutable_default(args: ast.arguments) -> List[int]:
    """Return offsets of args with mutable default literals."""
    bad = []
    defaults = list(args.defaults) + list(args.kw_defaults)
    for d in defaults:
        if isinstance(d, (ast.List, ast.Dict, ast.Set)):
            bad.append(getattr(d, "lineno", 0))
    return bad


def _is_none_eq_compare(node: ast.Compare) -> bool:
    """Detect `x == None` or `x != None`."""
    for op, comparator in zip(node.ops, node.comparators):
        if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comparator, ast.Constant) and comparator.value is None:
            return True
    # Also catch None on left side
    if isinstance(node.left, ast.Constant) and node.left.value is None:
        for op in node.ops:
            if isinstance(op, (ast.Eq, ast.NotEq)):
                return True
    return False


def _scan_one(path: Path, project_root: Path) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        return findings

    rel = str(path.relative_to(project_root))

    for node in ast.walk(tree):
        # 1. print(os.environ[...])
        if _is_print_environ(node):
            findings.append({
                "file": rel,
                "line": getattr(node, "lineno", 0),
                "pattern": "print_environ",
                "message": "print() of os.environ — potential key leak",
            })
        # 2. silent except
        if isinstance(node, ast.ExceptHandler) and _is_silent_except(node):
            findings.append({
                "file": rel,
                "line": getattr(node, "lineno", 0),
                "pattern": "silent_except",
                "message": "bare/silent except: pass — failures hidden",
            })
        # 3. mutable default args
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for ln in _has_mutable_default(node.args):
                findings.append({
                    "file": rel,
                    "line": ln or getattr(node, "lineno", 0),
                    "pattern": "mutable_default",
                    "message": f"mutable default argument in def {node.name}(...)",
                })
        # 4. == None / != None
        if isinstance(node, ast.Compare) and _is_none_eq_compare(node):
            findings.append({
                "file": rel,
                "line": getattr(node, "lineno", 0),
                "pattern": "none_eq_compare",
                "message": "comparison to None with == / != — use 'is' / 'is not'",
            })
    return findings


def scan(project_root: Path, py_files: List[Path]) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []
    for path in py_files:
        findings.extend(_scan_one(path, project_root))
    return findings
