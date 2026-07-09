"""Consistent plotting look and feel.

A single :func:`apply_style` call standardizes every figure the viz
package produces — color cycle, gridlines, font sizes — so figures
compare cleanly across a notebook session instead of each inheriting
whatever style happened to be active when it was drawn.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

__all__ = ["PALETTE", "apply_style"]

#: Shared color cycle: distinguishable in both screen and grayscale prints.
PALETTE = [
    "#2563eb",  # blue
    "#16a34a",  # green
    "#d97706",  # amber
    "#dc2626",  # red
    "#7c3aed",  # purple
    "#0891b2",  # teal
    "#db2777",  # pink
    "#65a30d",  # olive
]


def apply_style() -> None:
    """Set the matplotlib rcParams used by every function in this package.

    Idempotent and process-global, matching matplotlib's usual style
    model (like ``plt.style.use``); call once per script or notebook.
    """
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "savefig.dpi": 150,
        }
    )
