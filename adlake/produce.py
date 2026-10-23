"""Publishes simulated events to Kafka in the order devices would have sent them."""
import time

import numpy as np
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from adlake.simulate import Events


def reset_topic(bootstrap: str, topic: str, partitions: int = 6) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    if topic in admin.list_topics(timeout=10).topics:
        for f in admin.delete_topics([topic]).values():
            f.result()
        while topic in admin.list_topics(timeout=10).topics:
            time.sleep(0.5)
    for _ in range(20):  # deletion is asynchronous on the broker
        try:
            # Messages carry their simulated send time, so time-based retention would delete them on arrival.
            new = NewTopic(topic, partitions, 1, config={"retention.ms": "-1", "retention.bytes": "-1"})
            for f in admin.create_topics([new]).values():
                f.result()
            return
        except Exception as e:  # noqa: BLE001
            if "TOPIC_ALREADY_EXISTS" not in str(e) and "marked for deletion" not in str(e):
                raise
            time.sleep(1)


def publish(events: Events, idx: np.ndarray, bootstrap: str, topic: str) -> dict:
    """Keyed by user so one user's events stay ordered within a partition; idempotent producer."""
    producer = Producer({
        "bootstrap.servers": bootstrap, "enable.idempotence": True, "acks": "all",
        "linger.ms": 50, "batch.size": 1_000_000, "compression.type": "lz4",
    })
    start = time.perf_counter()
    for i in idx:
        while True:
            try:
                producer.produce(topic, key=events.user_keys[events.user[i]], value=events.to_json(i),
                                 timestamp=int(events.sent_ms[i]))
                break
            except BufferError:
                producer.poll(0.05)
        producer.poll(0)
    if producer.flush(60) != 0:
        raise RuntimeError("messages left undelivered")
    elapsed = time.perf_counter() - start
    return {"messages": int(len(idx)), "seconds": round(elapsed, 1), "per_second": int(len(idx) / elapsed)}
