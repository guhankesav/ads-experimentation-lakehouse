"""Iceberg table maintenance: compaction of streaming small files and snapshot expiry."""
import statistics
import time

from pyspark.sql import SparkSession


def file_stats(spark: SparkSession, table: str) -> dict:
    r = spark.sql(f"SELECT count(*) n, avg(file_size_in_bytes) avg_b FROM {table}.files").first()
    return {"data_files": r["n"], "avg_file_mb": round(r["avg_b"] / 2**20, 2)}


def time_query(spark: SparkSession, sql: str, runs: int = 5) -> float:
    spark.sql(sql).collect()  # warm-up
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        spark.sql(sql).collect()
        times.append(time.perf_counter() - t)
    return round(statistics.median(times), 3)


def compact(spark: SparkSession, table: str, catalog: str = "lake") -> dict:
    name = table.split(".", 1)[1]
    r = spark.sql(f"CALL {catalog}.system.rewrite_data_files(table => '{name}', options => map('min-input-files', '2'))").first()
    return {"rewritten_files": r["rewritten_data_files_count"], "added_files": r["added_data_files_count"]}


def expire_snapshots(spark: SparkSession, table: str, keep: int = 3, catalog: str = "lake") -> dict:
    name = table.split(".", 1)[1]
    before = spark.sql(f"SELECT count(*) FROM {table}.snapshots").first()[0]
    spark.sql(f"CALL {catalog}.system.expire_snapshots(table => '{name}', older_than => current_timestamp(), retain_last => {keep})")
    return {"snapshots_before": before, "snapshots_after": spark.sql(f"SELECT count(*) FROM {table}.snapshots").first()[0]}
