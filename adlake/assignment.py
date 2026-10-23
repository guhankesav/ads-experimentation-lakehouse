"""Deterministic experiment assignment, identical in Python (simulator) and Spark (analysis)."""
import hashlib

from pyspark.sql import Column
from pyspark.sql import functions as F

BUCKETS = 1000


def bucket(experiment: str, user_key: str) -> int:
    return int(hashlib.sha256(f"{experiment}:{user_key}".encode()).hexdigest()[:8], 16) % BUCKETS


def arm(experiment: str, user_key: str, treatment_share: float = 0.5) -> str:
    return "treatment" if bucket(experiment, user_key) < treatment_share * BUCKETS else "control"


def arm_col(experiment: str, user_key: Column, treatment_share: float = 0.5) -> Column:
    digest = F.sha2(F.concat(F.lit(experiment + ":"), user_key), 256)
    b = F.conv(F.substring(digest, 1, 8), 16, 10).cast("long") % BUCKETS
    return F.when(b < int(treatment_share * BUCKETS), "treatment").otherwise("control")
