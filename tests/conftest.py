from datetime import datetime

import pytest
from pyspark.sql.types import StructType

from adlake.ingest import SCHEMA_V1
from adlake.spark import build_session
from adlake.tables import BRONZE, DAILY, SILVER, create_tables


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    session = build_session(str(tmp_path_factory.mktemp("warehouse")), master="local[2]", shuffle_partitions=2)
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def tables(spark):
    for t in (BRONZE, SILVER, DAILY):
        spark.sql(f"DROP TABLE IF EXISTS {t} PURGE")
    create_tables(spark)
    return spark


def event(event_id, kind="impression", imp="i1", user="u1", ts="2026-09-01 10:00:00", sent=None, cost=0,
          placement="search_results", campaign=1001):
    t = datetime.fromisoformat(ts)
    return (event_id, kind, imp, user, campaign, placement, t, datetime.fromisoformat(sent) if sent else t, cost, 1)


def events_df(spark, rows, schema: StructType = SCHEMA_V1):
    return spark.createDataFrame(rows, schema)
