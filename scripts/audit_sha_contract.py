"""Reject SHA-256 admission comparisons while permitting provenance and RNG use."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "scripts")
FINGERPRINT_WORDS = ("fingerprint", "sha256")


def _mentions_fingerprint(node: ast.AST) -> bool:
    text = ast.unparse(node).casefold()
    if "fingerprint_schema" in text and "sha256" not in text:
        return False
    return any(word in text for word in FINGERPRINT_WORDS)


def _compares_two_fingerprints(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and _mentions_fingerprint(node.left)
        and any(_mentions_fingerprint(value) for value in node.comparators)
        and any(isinstance(operator, (ast.Eq, ast.NotEq)) for operator in node.ops)
    )


def audit() -> list[str]:
    findings: list[str] = []
    for root in SEARCH_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if path == Path(__file__).resolve():
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.If) or not any(
                    _compares_two_fingerprints(candidate)
                    for candidate in ast.walk(node.test)
                ):
                    continue
                if any(
                    isinstance(descendant, ast.Raise)
                    for descendant in ast.walk(ast.Module(body=node.body, type_ignores=[]))
                ):
                    findings.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: "
                        "fingerprint/checksum controls an admission raise"
                    )
    return findings


def main() -> None:
    findings = audit()
    if findings:
        print("\n".join(findings))
        raise SystemExit(1)
    print("SHA contract audit passed: no fingerprint equality admission gate found.")


if __name__ == "__main__":
    main()
