"""Suite-wide collection rules.

CI reports lint, type-check, *unit* and *integration* as separate stages, so the
split has to exist somewhere. Putting it here rather than in a `pytest.mark`
sprinkled across 40 modules keeps the rule in one readable place and makes it
apply to files that do not exist yet: a module named `*_integration.py` or
`*_e2e.py` is integration, everything else is a unit test. A new test opts in by
what it is called, which is the same convention the existing suite already
follows by accident.
"""

from pathlib import Path

import pytest

#: Filename suffixes that mean "this module wires several packages together, or
#: talks to something outside the process".
INTEGRATION_SUFFIXES = ("_integration.py", "_e2e.py")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        name = Path(str(item.path)).name
        if name.endswith(INTEGRATION_SUFFIXES):
            item.add_marker(pytest.mark.integration)
