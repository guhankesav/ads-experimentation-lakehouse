"""Hourly CTR anomaly detection per placement against a same-hour, trailing-week baseline."""
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from adlake.tables import SILVER


def hourly_ctr(spark: SparkSession) -> dict[str, np.ndarray]:
    """placement -> array[day, hour, (impressions, clicks)]."""
    rows = (spark.table(SILVER)
            .groupBy("placement", F.date_trunc("hour", "event_ts").alias("hour"))
            .agg(F.count("*").alias("imps"), F.sum("clicked").alias("clicks"))
            .orderBy("placement", "hour").collect())
    first = min(r["hour"] for r in rows)
    days = (max(r["hour"] for r in rows) - first).days + 1
    out: dict[str, np.ndarray] = {}
    for r in rows:
        grid = out.setdefault(r["placement"], np.zeros((days, 24, 2)))
        offset = r["hour"] - first
        grid[offset.days, offset.seconds // 3600] = (r["imps"], r["clicks"])
    return out


def detect(grid: np.ndarray, lookback: int = 7, threshold: float = 4.0, min_imps: int = 200) -> list[tuple[int, int, float]]:
    """(day, hour, z) for hours whose CTR departs from the median of the same hour on prior days.

    Scale is the larger of the robust spread (MAD) and the binomial noise expected at this hour's
    volume, so quiet hours with few impressions do not raise alarms on sampling noise alone.
    """
    imps, clicks = grid[..., 0], grid[..., 1]
    ctr = np.divide(clicks, imps, out=np.zeros_like(clicks), where=imps > 0)
    overall = clicks.sum() / imps.sum()
    flagged = []
    for d in range(lookback, grid.shape[0]):
        for h in range(24):
            if imps[d, h] < min_imps:
                continue
            hist = ctr[d - lookback:d, h]
            med = np.median(hist)
            mad = 1.4826 * np.median(np.abs(hist - med))
            p = med if med > 0 else overall
            z = (ctr[d, h] - med) / max(mad, np.sqrt(p * (1 - p) / imps[d, h]))
            if abs(z) > threshold:
                flagged.append((d, h, round(float(z), 1)))
    return flagged


def evaluate(flagged: dict[str, list], placement: str, day: int, hours: tuple[int, int]) -> dict:
    truth = {(placement, day, h) for h in range(*hours)}
    found = {(p, d, h) for p, hits in flagged.items() for d, h, _ in hits}
    tp = len(truth & found)
    return {"injected_hours": len(truth), "flagged_hours": len(found), "true_positives": tp,
            "precision": tp / len(found) if found else 0.0, "recall": tp / len(truth)}
