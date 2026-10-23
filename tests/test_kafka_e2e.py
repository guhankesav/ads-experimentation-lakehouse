"""Kafka -> Iceberg end to end. Needs a broker (docker compose up -d); skipped otherwise."""
import os
import socket
from pathlib import Path

import pytest

from adlake.ingest import SCHEMA_V2, ingest
from adlake.produce import publish, reset_topic
from adlake.run import replay_last_batch
from adlake.simulate import Config, simulate
from adlake.tables import BRONZE

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")


def broker_up() -> bool:
    host, port = BOOTSTRAP.split(":")
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, int(port))) == 0


if os.environ.get("REQUIRE_KAFKA") and not broker_up():
    raise RuntimeError(f"REQUIRE_KAFKA is set but no broker answers at {BOOTSTRAP}")
pytestmark = pytest.mark.skipif(not broker_up(), reason="no Kafka broker")


def test_stream_lands_each_event_once_even_after_a_replay(tables, tmp_path):
    ev = simulate(Config(users=400, pre_days=2, experiment_days=2, schema_v2_day=0, outage_day=3, seed=5))
    topic = "ad-events-test"
    reset_topic(BOOTSTRAP, topic, partitions=3)
    publish(ev, ev.send_order(), BOOTSTRAP, topic)
    checkpoint = tmp_path / "ckpt"

    stats = ingest(tables, BOOTSTRAP, topic, str(checkpoint), SCHEMA_V2, max_offsets=2_000)
    assert len(stats) > 1
    assert tables.table(BRONZE).count() == ev.truth["unique_events"]

    replay_last_batch(Path(checkpoint))
    replayed = ingest(tables, BOOTSTRAP, topic, str(checkpoint), SCHEMA_V2, max_offsets=2_000)
    assert [b["inserted"] for b in replayed] == [0]
    assert tables.table(BRONZE).count() == ev.truth["unique_events"]
