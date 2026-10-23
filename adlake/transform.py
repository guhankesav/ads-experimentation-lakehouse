"""Bronze -> silver attribution, daily metrics, and the privacy-thresholded advertiser report."""
from datetime import date, datetime, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from adlake.assignment import arm_col
from adlake.tables import BRONZE, DAILY, SILVER

ATTRIBUTION_WINDOW = "INTERVAL 30 MINUTES"


def _ts(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def attributed(spark: SparkSession, first: date, last: date, cutoff: datetime | None) -> DataFrame:
    """One row per impression in [first, last], with the first click inside the attribution window.

    `cutoff` limits both sides to events received by then, which is what a scheduled run sees;
    None means everything received so far (a backfill).
    """
    lo, hi = _ts(first), _ts(last + timedelta(days=1))
    bronze = spark.table(BRONZE)
    if "device_class" not in bronze.columns:  # before the v2 schema arrived
        bronze = bronze.withColumn("device_class", F.lit(None).cast("string"))
    if cutoff is not None:
        bronze = bronze.where(F.col("sent_ts") <= F.lit(cutoff))
    imps = (bronze.where((F.col("event_type") == "impression") & (F.col("event_ts") >= F.lit(lo))
                         & (F.col("event_ts") < F.lit(hi)))
            .select("impression_id", "user_key", "campaign_id", "placement", "device_class", "event_ts"))
    clicks = (bronze.where((F.col("event_type") == "click") & (F.col("event_ts") >= F.lit(lo))
                           & (F.col("event_ts") < F.lit(hi + timedelta(minutes=30))))
              .select("impression_id", F.col("event_ts").alias("click_ts"), "cost_micros"))
    first_click = (imps.select("impression_id", F.col("event_ts").alias("imp_ts"))
                   .join(clicks, "impression_id")
                   .where((F.col("click_ts") >= F.col("imp_ts"))
                          & (F.col("click_ts") <= F.col("imp_ts") + F.expr(ATTRIBUTION_WINDOW)))
                   .groupBy("impression_id")
                   .agg(F.min("click_ts").alias("click_ts"), F.min_by("cost_micros", "click_ts").alias("cost_micros")))
    return (imps.join(first_click, "impression_id", "left")
            .select("impression_id", "user_key", "campaign_id", "placement", "device_class", "event_ts",
                    F.col("click_ts").isNotNull().cast("int").alias("clicked"), "click_ts",
                    F.coalesce("cost_micros", F.lit(0)).alias("cost_micros"),
                    F.coalesce(F.lit(cutoff), F.current_timestamp()).alias("data_cutoff")))


def write_silver(spark: SparkSession, first: date, last: date, cutoff: datetime | None) -> int:
    """Overwrites only the day partitions in range, so reruns and backfills are idempotent."""
    df = attributed(spark, first, last, cutoff)
    df.writeTo(SILVER).overwritePartitions()
    return spark.sql(f"SELECT snapshot_id FROM {SILVER}.snapshots ORDER BY committed_at DESC LIMIT 1").first()[0]


def write_daily_metrics(spark: SparkSession, experiment: str) -> None:
    (spark.table(SILVER)
     .withColumn("arm", arm_col(experiment, F.col("user_key")))
     .groupBy(F.to_date("event_ts").alias("day"), "placement", "arm")
     .agg(F.count("*").alias("impressions"), F.sum("clicked").alias("clicks"),
          F.countDistinct("user_key").alias("users"), (F.sum("cost_micros") / 1e6).alias("spend_usd"))
     .writeTo(DAILY).overwritePartitions())


def advertiser_report(spark: SparkSession, min_users: int = 50) -> tuple[DataFrame, dict]:
    """Campaign x day x placement report; rows backed by fewer than `min_users` people are suppressed."""
    full = (spark.table(SILVER)
            .groupBy("campaign_id", F.to_date("event_ts").alias("day"), "placement")
            .agg(F.count("*").alias("impressions"), F.sum("clicked").alias("clicks"),
                 F.countDistinct("user_key").alias("users"), (F.sum("cost_micros") / 1e6).alias("spend_usd")))
    released = full.where(F.col("users") >= min_users).drop("users")
    totals = full.agg(F.count("*").alias("rows"), F.sum("impressions").alias("impressions"),
                      F.sum(F.when(F.col("users") < min_users, 1).otherwise(0)).alias("suppressed_rows"),
                      F.sum(F.when(F.col("users") < min_users, F.col("impressions")).otherwise(0)).alias("suppressed_impressions")).first()
    return released, {"min_users": min_users, "rows": totals["rows"], "suppressed_rows": totals["suppressed_rows"],
                      "suppressed_impression_share": round(totals["suppressed_impressions"] / totals["impressions"], 4)}
