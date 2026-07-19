"""Signal quality score.

Condenses the four things that make an EPS + MA-crossover signal worth a look
into one 0-100 triage number, plus a per-factor breakdown for the UI's
"why this scores" panel. Pure math — no DB access — so it is trivially testable
and reusable from the scanner, the API, and the detail view.
"""
import math

# Relative importance of each factor. Must sum to 1.0.
WEIGHTS = {"eps": 0.35, "rvol": 0.30, "trend": 0.20, "timing": 0.15}


def _eps_factor(eps_change_pct: float | None) -> float:
    """Bigger earnings surprise = stronger catalyst. Log-scaled: EPS % ranges
    from single digits to 1000%+, so raw magnitude would swamp everything."""
    m = abs(eps_change_pct or 0.0)
    if m <= 0:
        return 0.0
    # log10(10)=1 -> ~0.36, log10(100)=2 -> ~0.68, log10(1000)=3 -> ~0.99
    return max(0.0, min(1.0, math.log10(max(m, 1.0)) / 3.2 + 0.05))


def _rvol_factor(rvol: float | None) -> float:
    """Relative volume — conviction behind the move. 1x normal -> 0, 3.5x -> 1."""
    if rvol is None:
        return 0.4  # unknown volume: neutral-ish, don't punish
    return max(0.0, min(1.0, (rvol - 1.0) / 2.5))


def _trend_factor(trend_aligned: bool | None) -> float:
    """A signal running WITH the long-term (SMA200) trend beats one fighting it:
    bullish above SMA200 / bearish below SMA200 is aligned."""
    if trend_aligned is None:
        return 0.5
    return 1.0 if trend_aligned else 0.35


def _timing_factor(days_between: int | None, trend_window: int) -> float:
    """A crossover that fires soon after the report is a sharper, fresher reaction
    to the earnings catalyst than one that drifts in weeks later."""
    if days_between is None or not trend_window:
        return 0.5
    return max(0.0, min(1.0, 1.0 - days_between / max(trend_window, 1)))


def score_signal(
    *,
    eps_change_pct: float | None,
    rvol: float | None = None,
    trend_aligned: bool | None = None,
    days_between: int | None = None,
    trend_window: int = 30,
) -> dict:
    """Return {"score": 0-100, "factors": {name: 0-100}}.

    Every input is optional; missing inputs fall back to a neutral contribution
    so a partially-enriched signal still gets a sensible score.
    """
    factors = {
        "eps": _eps_factor(eps_change_pct),
        "rvol": _rvol_factor(rvol),
        "trend": _trend_factor(trend_aligned),
        "timing": _timing_factor(days_between, trend_window),
    }
    score = sum(WEIGHTS[k] * factors[k] for k in WEIGHTS) * 100
    return {
        "score": round(score),
        "factors": {k: round(v * 100) for k, v in factors.items()},
    }
