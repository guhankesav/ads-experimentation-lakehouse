import numpy as np

from adlake.anomaly import detect, evaluate


def grid(days=14, imps=5000, ctr=0.03, seed=3):
    rng = np.random.default_rng(seed)
    g = np.zeros((days, 24, 2))
    g[..., 0] = imps
    g[..., 1] = rng.binomial(imps, ctr, (days, 24))
    return g


def test_quiet_series_raises_no_alarms():
    assert detect(grid()) == []


def test_tracking_outage_hours_are_flagged():
    g = grid()
    g[10, 9:15, 1] *= 0.3
    found = detect(g)
    assert {(d, h) for d, h, _ in found} == {(10, h) for h in range(9, 15)}
    assert evaluate({"today_tab": found}, "today_tab", 10, (9, 15))["recall"] == 1.0


def test_low_volume_hours_are_skipped():
    g = grid()
    g[10, 3] = (50, 0)
    assert detect(g) == []
