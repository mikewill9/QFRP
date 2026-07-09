"""Style: rcParams applied, no display calls."""

from __future__ import annotations

import matplotlib.pyplot as plt

from quantlab.viz import PALETTE, apply_style


def test_apply_style_sets_rcparams():
    plt.rcdefaults()
    apply_style()
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["figure.facecolor"] == "white"


def test_apply_style_is_idempotent():
    apply_style()
    first = dict(plt.rcParams)
    apply_style()
    second = dict(plt.rcParams)
    assert first["axes.grid"] == second["axes.grid"]
    assert first["grid.color"] == second["grid.color"]


def test_palette_has_distinct_colors():
    assert len(PALETTE) == len(set(PALETTE))
    assert all(c.startswith("#") for c in PALETTE)
