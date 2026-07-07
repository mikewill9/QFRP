"""Signal transforms.

Small composable objects mapping one wide score frame to another via
``transform(frame) -> frame``. Cross-sectional transforms (:class:`ZScore`,
:class:`Rank`, :class:`Winsorize`, :class:`Demean`, :class:`Neutralize`)
operate row by row and are safe by construction with respect to lookahead.
Time-series transforms (:class:`Decay`, :class:`Lag`) operate down each
column using only past values.

Compose with :class:`SignalPipeline`:

>>> pipe = SignalPipeline([Winsorize(), ZScore()])
>>> signal = pipe.transform(raw_scores)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from quantlab.exceptions import DataError
from quantlab.types import WideFrame

__all__ = [
    "Clip",
    "Decay",
    "Demean",
    "Lag",
    "Neutralize",
    "Rank",
    "SignalPipeline",
    "Transform",
    "Winsorize",
    "ZScore",
]


class Transform(ABC):
    """Base class for signal transforms."""

    @abstractmethod
    def transform(self, frame: WideFrame) -> WideFrame:
        """Return the transformed frame (input is never mutated)."""

    def __call__(self, frame: WideFrame) -> WideFrame:
        return self.transform(frame)

    def __repr__(self) -> str:
        args = ", ".join(
            f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_")
        )
        return f"{type(self).__name__}({args})"


class Winsorize(Transform):
    """Clip each row at cross-sectional quantiles (default 1%/99%)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        if not 0.0 <= lower < upper <= 1.0:
            raise DataError("require 0 <= lower < upper <= 1")
        self.lower = lower
        self.upper = upper

    def transform(self, frame: WideFrame) -> WideFrame:
        lo = frame.quantile(self.lower, axis=1)
        hi = frame.quantile(self.upper, axis=1)
        return frame.clip(lower=lo, upper=hi, axis=0)


class ZScore(Transform):
    """Standardize each row to zero mean, unit standard deviation."""

    def transform(self, frame: WideFrame) -> WideFrame:
        mu = frame.mean(axis=1)
        sd = frame.std(axis=1)
        return frame.sub(mu, axis=0).div(sd, axis=0)


class Rank(Transform):
    """Cross-sectional percentile rank.

    ``center=True`` (default) shifts ranks into ``(-0.5, 0.5)`` so the
    output is usable directly as a dollar-neutral weight seed.
    """

    def __init__(self, center: bool = True) -> None:
        self.center = center

    def transform(self, frame: WideFrame) -> WideFrame:
        if not self.center:
            return frame.rank(axis=1, pct=True)
        ranked = frame.rank(axis=1)
        n = frame.notna().sum(axis=1)
        return ranked.sub((n + 1) / 2.0, axis=0).div(n, axis=0)


class Demean(Transform):
    """Subtract the cross-sectional mean from each row."""

    def transform(self, frame: WideFrame) -> WideFrame:
        return frame.sub(frame.mean(axis=1), axis=0)


class Neutralize(Transform):
    """Demean within groups (e.g. sectors) per row.

    Parameters
    ----------
    groups:
        Mapping ``symbol -> group label``. Symbols absent from the mapping
        form their own ``"_other"`` group. After the transform every group's
        cross-sectional mean is zero, removing e.g. sector bets from a
        signal.
    """

    def __init__(self, groups: Mapping[str, str]) -> None:
        if not groups:
            raise DataError("groups mapping is empty")
        self.groups = dict(groups)

    def transform(self, frame: WideFrame) -> WideFrame:
        out = frame.copy()
        labels = [self.groups.get(str(c), "_other") for c in frame.columns]
        by_group: dict[str, list[str]] = {}
        for col, lab in zip(frame.columns, labels, strict=True):
            by_group.setdefault(lab, []).append(col)
        for cols in by_group.values():
            block = out[cols]
            out[cols] = block.sub(block.mean(axis=1), axis=0)
        return out


class Decay(Transform):
    """Exponential smoothing down each column (halflife in bars).

    Trades signal freshness for lower turnover; apply after cross-sectional
    normalization.
    """

    def __init__(self, halflife: float = 5.0) -> None:
        if halflife <= 0:
            raise DataError("halflife must be positive")
        self.halflife = halflife

    def transform(self, frame: WideFrame) -> WideFrame:
        return frame.ewm(halflife=self.halflife, ignore_na=True).mean()


class Lag(Transform):
    """Shift the signal forward in time by ``periods`` bars.

    Useful for studying signal decay or enforcing implementation delay
    beyond the backtester's own next-open fill timing.
    """

    def __init__(self, periods: int = 1) -> None:
        if periods < 0:
            raise DataError("periods must be >= 0")
        self.periods = periods

    def transform(self, frame: WideFrame) -> WideFrame:
        return frame.shift(self.periods)


class Clip(Transform):
    """Absolute clip at fixed bounds (post-zscore tail control)."""

    def __init__(self, lower: float = -3.0, upper: float = 3.0) -> None:
        if lower >= upper:
            raise DataError("lower must be < upper")
        self.lower = lower
        self.upper = upper

    def transform(self, frame: WideFrame) -> WideFrame:
        return frame.clip(lower=self.lower, upper=self.upper)


class SignalPipeline(Transform):
    """Apply a sequence of transforms left to right."""

    def __init__(self, steps: Sequence[Transform]) -> None:
        if not steps:
            raise DataError("pipeline needs at least one step")
        for s in steps:
            if not isinstance(s, Transform):
                raise DataError(f"{s!r} is not a Transform")
        self.steps = list(steps)

    def transform(self, frame: WideFrame) -> WideFrame:
        out = frame
        for step in self.steps:
            out = step.transform(out)
        return out

    def __repr__(self) -> str:
        inner = ", ".join(repr(s) for s in self.steps)
        return f"SignalPipeline([{inner}])"
