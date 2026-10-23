"""Iceberg table definitions. Partitioning is hidden: queries filter on event_ts, never on a date column."""
from pyspark.sql import SparkSession

BRONZE = "lake.ads.events_bronze"
SILVER = "lake.ads.impressions"
DAILY = "lake.ads.daily_metrics"


def create_tables(spark: SparkSession) -> None:
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.ads")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {BRONZE} (
            event_id STRING, event_type STRING, impression_id STRING, user_key STRING,
            campaign_id INT, placement STRING, event_ts TIMESTAMP, sent_ts TIMESTAMP,
            cost_micros BIGINT, schema_version INT,
            kafka_partition INT, kafka_offset BIGINT, ingested_at TIMESTAMP)
        USING iceberg PARTITIONED BY (days(event_ts))
        TBLPROPERTIES ('format-version' = '2', 'write.merge.mode' = 'merge-on-read')""")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {SILVER} (
            impression_id STRING, user_key STRING, campaign_id INT, placement STRING, device_class STRING,
            event_ts TIMESTAMP, clicked INT, click_ts TIMESTAMP, cost_micros BIGINT, data_cutoff TIMESTAMP)
        USING iceberg PARTITIONED BY (days(event_ts))
        TBLPROPERTIES ('format-version' = '2')""")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {DAILY} (
            day DATE, placement STRING, arm STRING, impressions BIGINT, clicks BIGINT,
            users BIGINT, spend_usd DOUBLE)
        USING iceberg PARTITIONED BY (day)
        TBLPROPERTIES ('format-version' = '2')""")
