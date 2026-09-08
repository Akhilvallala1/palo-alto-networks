"""AC-16: no file outside `src/conduit/providers/` imports a vendor SDK.

This is the epic's load-bearing constraint. Conduit's claim is total model-vendor
independence: application code talks to Conduit, and only the adapter layer knows
a vendor exists. That claim is worth exactly as much as this test.

Parsed with `ast` rather than grepped, so `import anthropic`, `from anthropic import x`,
`import anthropic.foo as bar`, and a nested-scope import inside a function are all
caught, while the string "anthropic" in a comment, a model id, or a config key is not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "conduit"
PROVIDERS = SRC / "providers"

# Top-level module names that mean "a vendor SDK is being spoken to directly".
VENDOR_ROOTS = {
    "anthropic",
    "openai",
    "google",  # google.generativeai and friends
    "cohere",
    "mistralai",
    "boto3",  # bedrock
    "vertexai",
}


def _python_files() -> list[Path]:
    return sorted(
        p for p in SRC.rglob("*.py") if PROVIDERS not in p.parents and p.parent != PROVIDERS
    )


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        # `from . import x` has module None and level > 0; relative imports are ours.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_provider_package_is_the_only_place_vendors_are_named() -> None:
    """The guarded set is non-empty, so a rename of src/ can't silently pass this suite."""
    files = _python_files()
    assert files, f"found no python files to guard under {SRC}"
    assert PROVIDERS.is_dir(), "providers package is missing; the exemption points at nothing"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_vendor_sdk_import_outside_providers(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offending = _imported_roots(tree) & VENDOR_ROOTS
    assert not offending, (
        f"{path.relative_to(SRC.parent.parent)} imports vendor SDK(s) {sorted(offending)}.\n"
        "AC-16: only src/conduit/providers/ may name a vendor. Route model traffic through "
        "the Provider protocol instead."
    )


def test_vendor_sdks_are_not_core_dependencies() -> None:
    """A vendor SDK in [project.dependencies] would make the zero-key path (AC-17) a lie."""
    pyproject = (SRC.parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    core = pyproject.split("[project.optional-dependencies]")[0]
    deps = core.split("dependencies", 1)[1] if "dependencies" in core else ""
    for vendor in ("anthropic", "openai", "google-generativeai", "boto3"):
        assert f'"{vendor}' not in deps, (
            f"{vendor} is a core dependency; it must be an optional extra so a "
            "zero-key install still runs on mock + ollama."
        )
