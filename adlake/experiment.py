"""A/B analysis for a ratio metric (CTR) randomized by user.

CTR = clicks / impressions is a ratio of per-user sums, so a plain t-test on impressions would
treat correlated impressions of one user as independent. The delta method gives the right
variance; CUPED then removes the part of each user's variance predicted by their pre-period CTR.
"""
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy import stats

from adlake.assignment import arm_col
from adlake.tables import SILVER

Z = stats.norm.ppf(0.975)


@dataclass
class Users:
    """Per-user sums: experiment-period clicks/impressions and pre-period clicks/impressions."""
    treated: np.ndarray
    clicks: np.ndarray
    imps: np.ndarray
    pre_clicks: np.ndarray
    pre_imps: np.ndarray


def per_user(spark: SparkSession, experiment: str, start: date) -> Users:
    s = spark.table(SILVER)
    split = F.lit(datetime(start.year, start.month, start.day))
    during = (s.where(F.col("event_ts") >= split).groupBy("user_key")
              .agg(F.sum("clicked").alias("clicks"), F.count("*").alias("imps")))
    before = (s.where(F.col("event_ts") < split).groupBy("user_key")
              .agg(F.sum("clicked").alias("pre_clicks"), F.count("*").alias("pre_imps")))
    rows = (during.join(before, "user_key", "left").fillna(0)
            .withColumn("treated", (arm_col(experiment, F.col("user_key")) == "treatment").cast("int"))
            .select("treated", "clicks", "imps", "pre_clicks", "pre_imps").collect())
    a = np.array(rows, dtype=np.float64)
    return Users(a[:, 0].astype(bool), a[:, 1], a[:, 2], a[:, 3], a[:, 4])


def srm(n_treatment: int, n_control: int, share: float = 0.5) -> dict:
    """Sample ratio mismatch: a broken split invalidates every downstream metric."""
    n = n_treatment + n_control
    chi2, p = stats.chisquare([n_treatment, n_control], [n * share, n * (1 - share)])
    return {"treatment_users": int(n_treatment), "control_users": int(n_control), "chi2": float(chi2), "p_value": float(p)}


def ratio_delta(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Ratio of sums sum(y)/sum(x) and its delta-method variance, with users as the unit."""
    n, my, mx = len(y), y.mean(), x.mean()
    cov = np.cov(y, x)
    var = (cov[0, 0] / mx**2 - 2 * my * cov[0, 1] / mx**3 + my**2 * cov[1, 1] / mx**4) / n
    return my / mx, var


def ctr_test(u: Users) -> dict:
    """Relative CTR lift, treatment vs control, with a delta-method confidence interval."""
    rt, vt = ratio_delta(u.clicks[u.treated], u.imps[u.treated])
    rc, vc = ratio_delta(u.clicks[~u.treated], u.imps[~u.treated])
    lift = rt / rc - 1
    se = (rt / rc) * np.sqrt(vt / rt**2 + vc / rc**2)
    return {"ctr_treatment": rt, "ctr_control": rc, "relative_lift": lift, "ci": [lift - Z * se, lift + Z * se],
            "p_value": float(2 * stats.norm.sf(abs(lift / se))), "se": se}


def linearized(y: np.ndarray, x: np.ndarray, ratio: float, mean_x: float) -> np.ndarray:
    """Per-user contribution to the ratio, so mean differences equal ratio differences to first order."""
    return (y - ratio * x) / mean_x


def cuped_test(u: Users) -> dict:
    """CTR lift with CUPED, using each user's linearized pre-period CTR as the covariate."""
    ratio, mean_x = u.clicks.sum() / u.imps.sum(), u.imps.mean()
    pre_ratio, pre_mean = u.pre_clicks.sum() / u.pre_imps.sum(), u.pre_imps.mean()
    y = linearized(u.clicks, u.imps, ratio, mean_x)
    c = linearized(u.pre_clicks, u.pre_imps, pre_ratio, pre_mean)
    theta = np.cov(y, c)[0, 1] / c.var(ddof=1)
    adj = y - theta * (c - c.mean())
    diff, se = _welch(adj[u.treated], adj[~u.treated])
    _, raw_se = _welch(y[u.treated], y[~u.treated])
    rc = u.clicks[~u.treated].sum() / u.imps[~u.treated].sum()
    return {"relative_lift": diff / rc, "ci": [(diff - Z * se) / rc, (diff + Z * se) / rc],
            "p_value": float(2 * stats.norm.sf(abs(diff / se))), "theta": float(theta),
            "variance_reduction": float(1 - adj.var(ddof=1) / y.var(ddof=1)),
            "mde_80pct_power": {"without_cuped": 2.8 * raw_se / rc, "with_cuped": 2.8 * se / rc}}


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return a.mean() - b.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def aa_false_positive_rate(u: Users, runs: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Re-split control users at random many times; a calibrated test rejects about alpha of them."""
    rng = np.random.default_rng(seed)
    y, x = u.clicks[~u.treated], u.imps[~u.treated]
    rejections = 0
    for _ in range(runs):
        fake = rng.random(len(y)) < 0.5
        rejections += ctr_test(Users(fake, y, x, y, x))["p_value"] < alpha
    return {"runs": runs, "alpha": alpha, "false_positive_rate": rejections / runs}


def analyze(u: Users) -> dict:
    return {"srm": srm(int(u.treated.sum()), int((~u.treated).sum())),
            "ctr_delta_method": ctr_test(u), "ctr_cuped": cuped_test(u), "aa": aa_false_positive_rate(u)}
