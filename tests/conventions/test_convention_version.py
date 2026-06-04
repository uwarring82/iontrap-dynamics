# SPDX-License-Identifier: MIT
"""Pin the ``CONVENTION_VERSION`` literal at the current Convention Freeze.

Guards against a silent bump-skip: the ``tests/unit`` references import
``CONVENTION_VERSION`` as a symbol to stamp or compare metadata — none asserts
the literal value — so a freeze that forgot to bump the constant would pass
unnoticed. This pins it. Added by the shared v0.3 seal (``WP/FREEZE-v0.3.md`` §3);
bumped to 0.4 at the WP-03 RLA seal (§25 reduced-model conventions + §5 scope).
"""

from __future__ import annotations

import pytest

from iontrap_dynamics.conventions import CONVENTION_VERSION

pytestmark = pytest.mark.convention


def test_convention_version_is_v0_4() -> None:
    assert CONVENTION_VERSION == "0.4"
