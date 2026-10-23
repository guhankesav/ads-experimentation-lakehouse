import numpy as np

from adlake.experiment import Users, aa_false_positive_rate, ctr_test, cuped_test, ratio_delta, srm


def synthetic(n=60_000, lift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 19.4, n)
    treated = rng.random(n) < 0.5
    pre_imps = rng.poisson(10, n).astype(float)
    imps = rng.poisson(20, n).astype(float)
    pre_clicks = rng.binomial(pre_imps.astype(int), p).astype(float)
    clicks = rng.binomial(imps.astype(int), np.minimum(p * np.where(treated, 1 + lift, 1), 1)).astype(float)
    return Users(treated, clicks, imps, pre_clicks, pre_imps)


def test_delta_method_matches_bootstrap_variance():
    u = synthetic(n=5000)
    _, var = ratio_delta(u.clicks, u.imps)
    rng = np.random.default_rng(0)
    boot = [u.clicks[i].sum() / u.imps[i].sum() for i in (rng.integers(0, 5000, 5000) for _ in range(400))]
    assert abs(np.sqrt(var) / np.std(boot) - 1) < 0.15


def test_recovers_a_planted_lift():
    r = ctr_test(synthetic(lift=0.08))
    assert r["ci"][0] < 0.08 < r["ci"][1] and r["p_value"] < 0.01


def test_cuped_keeps_the_estimate_and_shrinks_the_interval():
    u = synthetic(lift=0.08)
    plain, cuped = ctr_test(u), cuped_test(u)
    assert cuped["variance_reduction"] > 0.1
    assert cuped["ci"][1] - cuped["ci"][0] < plain["ci"][1] - plain["ci"][0]
    assert cuped["ci"][0] < 0.08 < cuped["ci"][1]


def test_aa_tests_reject_about_five_percent():
    assert 0.025 < aa_false_positive_rate(synthetic(lift=0.0), runs=400)["false_positive_rate"] < 0.08


def test_srm_flags_a_skewed_split():
    assert srm(50_000, 50_300)["p_value"] > 0.01
    assert srm(52_000, 48_000)["p_value"] < 1e-6
