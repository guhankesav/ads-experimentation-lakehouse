"""Kafka -> Iceberg bronze with Spark Structured Streaming.

Exactly-once into the table comes from two pieces: the checkpoint replays a micro-batch whose
commit was not recorded, and the MERGE on event_id makes a replayed batch (or a producer retry
that landed in a different batch) a no-op.
"""
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType, TimestampType

from adlake.tables import BRONZE

SCHEMA_V1 = StructType([
    StructField("event_id", StringType()),
    StructField("event_type", StringType()),
    StructField("impression_id", StringType()),
    StructField("user_key", StringType()),
    StructField("campaign_id", IntegerType()),
    StructField("placement", StringType()),
    StructField("event_ts", TimestampType()),
    StructField("sent_ts", TimestampType()),
    StructField("cost_micros", LongType()),
    StructField("schema_version", IntegerType()),
])
SCHEMA_V2 = StructType(SCHEMA_V1.fields + [StructField("device_class", StringType())])


def evolve_schema(spark: SparkSession, table: str, incoming: StructType) -> list[str]:
    """Additive evolution only: new nullable columns are metadata changes, no data files are rewritten."""
    existing = {f.name for f in spark.table(table).schema.fields}
    added = []
    for f in incoming.fields:
        if f.name not in existing:
            spark.sql(f"ALTER TABLE {table} ADD COLUMN {f.name} {f.dataType.simpleString()}")
            added.append(f.name)
    return added


def merge_batch(df: DataFrame, table: str = BRONZE) -> dict:
    spark = df.sparkSession
    df = df.dropDuplicates(["event_id"])
    rows = df.count()
    if rows == 0:
        return {"rows": 0, "inserted": 0}
    added = evolve_schema(spark, table, df.schema)
    target = spark.table(table).columns
    df.select([F.col(c) if c in df.columns else F.lit(None).alias(c) for c in target]).createOrReplaceTempView("incoming")
    fmt = "yyyy-MM-dd HH:mm:ss.SSSSSS"
    lo, hi = df.agg(F.date_format(F.min("event_ts"), fmt), F.date_format(F.max("event_ts"), fmt)).first()
    snapshots = spark.sql(f"SELECT count(*) FROM {table}.snapshots").first()[0]
    start = time.perf_counter()
    # The event_ts range lets Iceberg prune the target scan to the days this batch touches.
    spark.sql(f"""
        MERGE INTO {table} t USING incoming s
        ON t.event_id = s.event_id AND t.event_ts BETWEEN TIMESTAMP '{lo}' AND TIMESTAMP '{hi}'
        WHEN NOT MATCHED THEN INSERT *""")
    elapsed = time.perf_counter() - start
    inserted = 0
    if spark.sql(f"SELECT count(*) FROM {table}.snapshots").first()[0] > snapshots:  # no-op merges commit nothing
        summary = spark.sql(f"SELECT summary FROM {table}.snapshots ORDER BY committed_at DESC LIMIT 1").first()[0]
        inserted = int(summary.get("added-records", 0))
    return {"rows": rows, "inserted": inserted,
            "merge_seconds": round(elapsed, 2), "columns_added": added}


def ingest(spark: SparkSession, bootstrap: str, topic: str, checkpoint: str, schema: StructType,
           max_offsets: int = 1_000_000) -> list[dict]:
    """Drains everything currently in the topic (availableNow), resuming from the checkpoint."""
    stats: list[dict] = []
    raw = (spark.readStream.format("kafka")
           .option("kafka.bootstrap.servers", bootstrap)
           .option("subscribe", topic)
           .option("startingOffsets", "earliest")
           .option("maxOffsetsPerTrigger", max_offsets)
           .load())
    events = (raw.select(F.from_json(F.col("value").cast("string"), schema).alias("e"),
                         F.col("partition").alias("kafka_partition"), F.col("offset").alias("kafka_offset"))
              .select("e.*", "kafka_partition", "kafka_offset", F.current_timestamp().alias("ingested_at"))
              .where(F.col("event_id").isNotNull()))

    def write(batch: DataFrame, batch_id: int) -> None:
        stats.append({"batch": batch_id, **merge_batch(batch)})

    query = (events.writeStream.foreachBatch(write)
             .option("checkpointLocation", checkpoint)
             .trigger(availableNow=True)
             .start())
    query.awaitTermination()
    return stats
