from adlake.ingest import SCHEMA_V2, merge_batch
from adlake.tables import BRONZE
from conftest import event, events_df


def with_kafka_cols(df):
    from pyspark.sql import functions as F
    return df.withColumn("kafka_partition", F.lit(0)).withColumn("kafka_offset", F.lit(0).cast("long")) \
             .withColumn("ingested_at", F.current_timestamp())


def test_replayed_batch_inserts_nothing(tables):
    batch = with_kafka_cols(events_df(tables, [event("e1"), event("e2", imp="i2")]))
    assert merge_batch(batch)["inserted"] == 2
    assert merge_batch(batch)["inserted"] == 0
    assert tables.table(BRONZE).count() == 2


def test_duplicates_inside_a_batch_and_across_batches_are_dropped(tables):
    merge_batch(with_kafka_cols(events_df(tables, [event("e1"), event("e1", sent="2026-09-01 10:00:03")])))
    merge_batch(with_kafka_cols(events_df(tables, [event("e1", sent="2026-09-01 10:00:05"), event("e9", imp="i9")])))
    assert sorted(r[0] for r in tables.table(BRONZE).select("event_id").collect()) == ["e1", "e9"]


def test_new_field_is_added_without_rewriting_old_rows(tables):
    merge_batch(with_kafka_cols(events_df(tables, [event("e1")])))
    v1_snapshot = tables.sql(f"SELECT snapshot_id FROM {BRONZE}.snapshots").first()[0]
    files_before = {r[0] for r in tables.sql(f"SELECT file_path FROM {BRONZE}.files").collect()}

    v2 = with_kafka_cols(events_df(tables, [(*event("e2", imp="i2"), "ipad")], SCHEMA_V2))
    assert merge_batch(v2)["columns_added"] == ["device_class"]

    rows = {r["event_id"]: r["device_class"] for r in tables.table(BRONZE).collect()}
    assert rows == {"e1": None, "e2": "ipad"}
    assert files_before <= {r[0] for r in tables.sql(f"SELECT file_path FROM {BRONZE}.files").collect()}
    assert "device_class" not in tables.sql(f"SELECT * FROM {BRONZE} VERSION AS OF {v1_snapshot}").columns
