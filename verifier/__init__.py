"""Independent code-reviewer / audit module for PathoMAN 2.0.

Walks the project tree and runs three categories of checks:
    secrets   - hardcoded API keys, key-leak patterns, .env hygiene
    bugs      - AST-based common bug patterns
    data_leaks - PII, file paths, contact info in source

Run with: python -m verifier.run_audit
"""
