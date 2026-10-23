import os
import sys
import time

from pyspark.sql import SparkSession

PACKAGES = ",".join([
    "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1",
    "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1",
])


def build_session(warehouse: str, master: str = "local[*]", shuffle_partitions: int = 16) -> SparkSession:
    """Local session with an Iceberg catalog named `lake` backed by a directory warehouse."""
    # PySpark converts timestamps to and from naive datetimes in the Python process's local zone;
    # pinning it to UTC keeps day boundaries and MERGE bounds identical on both sides.
    os.environ["TZ"] = "UTC"
    time.tzset()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.master(master)
        .appName("ads-experimentation-lakehouse")
        .config("spark.jars.packages", PACKAGES)
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.lake", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.lake.type", "hadoop")
        .config("spark.sql.catalog.lake.warehouse", warehouse)
        # foreachBatch runs in a cloned session; without this its schema changes stay invisible here
        .config("spark.sql.catalog.lake.cache-enabled", "false")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "6g")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
