# SPDX-License-Identifier: MIT
"""Pin the ``CONVENTION_VERSION`` literal at the v0.3 Convention Freeze.

Guards against a silent bump-skip: the ``tests/unit`` references import
``CONVENTION_VERSION`` as a symbol to stamp or compare metadata — none asserts
the literal value — so a freeze that forgot to bump the constant would pass
unnoticed. This pins it. Added by the shared v0.3 seal (``WP/FREEZE-v0.3.md`` §3).
"""

from __future__ import annotations

import pytest

from iontrap_dynamics.conventions import CONVENTION_VERSION

pytestmark = pytest.mark.convention


def test_convention_version_is_v0_3() -> None:
    assert CONVENTION_VERSION == "0.3"
