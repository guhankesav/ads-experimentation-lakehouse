from pyspark.sql import functions as F

from adlake.assignment import arm, arm_col


def test_python_and_spark_assign_every_user_identically(spark):
    keys = [f"user{i}" for i in range(2000)]
    rows = spark.createDataFrame([(k,) for k in keys], ["k"]).select("k", arm_col("exp", F.col("k")).alias("arm")).collect()
    assert {r["k"]: r["arm"] for r in rows} == {k: arm("exp", k) for k in keys}


def test_split_is_balanced_and_salted_by_experiment():
    keys = [f"user{i}" for i in range(20000)]
    share = sum(arm("exp", k) == "treatment" for k in keys) / len(keys)
    assert 0.48 < share < 0.52
    assert sum(arm("exp", k) != arm("other", k) for k in keys) > 8000  # independent across experiments
