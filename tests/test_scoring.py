from core.scoring import WEIGHTS, score_signal


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_score_in_range_and_shape():
    r = score_signal(eps_change_pct=50, rvol=2.0, trend_aligned=True, days_between=5, trend_window=30)
    assert 0 <= r["score"] <= 100
    assert set(r["factors"]) == set(WEIGHTS)
    assert all(0 <= v <= 100 for v in r["factors"].values())


def test_bigger_eps_scores_higher():
    lo = score_signal(eps_change_pct=12, rvol=1.5, trend_aligned=True, days_between=10)
    hi = score_signal(eps_change_pct=800, rvol=1.5, trend_aligned=True, days_between=10)
    assert hi["score"] > lo["score"]


def test_higher_rvol_scores_higher():
    lo = score_signal(eps_change_pct=50, rvol=1.0, trend_aligned=True, days_between=10)
    hi = score_signal(eps_change_pct=50, rvol=3.5, trend_aligned=True, days_between=10)
    assert hi["score"] > lo["score"]


def test_trend_alignment_matters():
    aligned = score_signal(eps_change_pct=50, rvol=2.0, trend_aligned=True, days_between=10)
    fighting = score_signal(eps_change_pct=50, rvol=2.0, trend_aligned=False, days_between=10)
    assert aligned["score"] > fighting["score"]


def test_sooner_cross_scores_higher():
    soon = score_signal(eps_change_pct=50, rvol=2.0, trend_aligned=True, days_between=2, trend_window=30)
    late = score_signal(eps_change_pct=50, rvol=2.0, trend_aligned=True, days_between=28, trend_window=30)
    assert soon["score"] > late["score"]


def test_missing_inputs_still_scores():
    r = score_signal(eps_change_pct=50)
    assert 0 <= r["score"] <= 100
