# verifier — PathoMAN 2.0 audit module

Independent code-reviewer agent that walks the project tree and runs three
categories of static checks. Designed to be runnable from the project root
without any third-party dependencies.

```
python -m verifier.run_audit
```

## Layout

```
verifier/
├── __init__.py
├── run_audit.py             # main entry — discovers .py files, runs checks
├── checks/
│   ├── __init__.py
│   ├── secrets.py           # hardcoded API-key patterns + .env hygiene
│   ├── bugs.py              # AST-based common-bug detection
│   └── data_leaks.py        # PII / file-path / contact-info scanner
└── README.md
```

## What each check does

### `secrets.py`
Regex scan over every `.py` file. The audit log records the pattern label and
location only — it never echoes the matched value, so the audit output itself
is safe to commit. Patterns:

- `sk-ant-…`        Anthropic API key prefix
- `sk-proj-…`       OpenAI project key
- `sk-…` (32+ char) OpenAI legacy key
- `[a-f0-9]{32}`    32-character lowercase hex (NCBI E-utilities key shape)
- `pat_…`           HuggingFace personal access token
- `LANGSMITH_API_KEY` followed by an assigned value
- `os.environ["X"] = "literal"`  hardcoded environment assignment

It also flags:
- A `.env` file present in the project that is not covered by `.gitignore`.

False positives are suppressed when the line contains `EXAMPLE`,
`your-key-here`, `<replace-with-...>`, or the explicit marker `# noqa: secret`.

### `bugs.py`
Parses every `.py` file with `ast` and flags:

- `print(os.environ[...])`, `print(os.environ.get(...))`, `print(os.getenv(...))`
  — risk of leaking key values to stdout.
- `except [...]: pass` — silent failures hide bugs.
- Mutable default arguments (`def f(x=[]): ...`, `def f(x={}): ...`).
- Comparison to `None` with `==` / `!=` (should be `is` / `is not`).

### `data_leaks.py`
Regex scan for incidental PII:

- Absolute home-directory paths (`/Users/...`, `/home/...`)
- Email addresses (with allowlist for `noreply@`, `@example.com`,
  `@anthropic.com`, `@github.com`, etc.)
- US-style phone numbers

Suppress on a line with the marker `# noqa: data-leak`.

## Exit code

- `1` if any **SECRETS** findings are reported. This is the only blocking
  category; CI pipelines should fail here.
- `0` otherwise. **BUGS** and **DATA LEAKS** are warnings — they print but do
  not fail the build.

## Excluded directories

`run_audit.py` skips: `.venv`, `venv`, `env`, `__pycache__`, `cache`,
`data/cache`, `data/faiss_index`, `.git`, `.mypy_cache`, `.pytest_cache`,
`.ruff_cache`, `node_modules`.
