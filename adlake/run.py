"""End-to-end run: simulate -> Kafka -> Iceberg bronze -> daily silver runs -> backfill -> analysis.

Every stage is checked against the simulator's ground truth and the results go to docs/run_summary.json.
"""
import argparse
import json
import shutil
import time
from datetime import timedelta
from pathlib import Path

from pyspark.sql import functions as F

from adlake import anomaly, experiment, maintenance
from adlake.ingest import SCHEMA_V1, SCHEMA_V2, ingest
from adlake.produce import publish, reset_topic
from adlake.simulate import Config, simulate
from adlake.spark import build_session
from adlake.tables import BRONZE, SILVER, create_tables
from adlake.transform import advertiser_report, write_daily_metrics, write_silver


def clicks_by_day(spark, table_ref: str) -> dict[str, list[int]]:
    rows = (spark.sql(f"SELECT to_date(event_ts) d, count(*) imps, sum(clicked) clicks FROM {table_ref} GROUP BY 1 ORDER BY 1")
            .collect())
    return {str(r["d"]): [r["imps"], r["clicks"]] for r in rows}


def replay_last_batch(checkpoint: Path) -> int:
    """Simulates a crash after the table commit but before the checkpoint commit."""
    commits = sorted((checkpoint / "commits").glob("[0-9]*"), key=lambda p: int(p.name))
    last = commits[-1]
    last.unlink()
    (last.parent / f".{last.name}.crc").unlink(missing_ok=True)  # Hadoop's checksum sidecar
    return int(last.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300_000)
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--topic", default="ad-events")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="docs/run_summary.json")
    args = ap.parse_args()

    work = Path(args.work).resolve()
    shutil.rmtree(work, ignore_errors=True)
    checkpoint = work / "checkpoints" / "bronze"
    summary: dict = {"timings_s": {}}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        summary["timings_s"][name] = round(time.perf_counter() - clock, 1)
        clock = time.perf_counter()

    cfg = Config(users=args.users)
    ev = simulate(cfg)
    summary["ground_truth"] = ev.truth
    lap("simulate")

    spark = build_session(str(work / "warehouse"))
    spark.sparkContext.setLogLevel("ERROR")
    create_tables(spark)
    reset_topic(args.bootstrap, args.topic)

    # Phase 1: schema v1 producers and pipeline. Phase 2: v2 adds device_class mid-stream.
    order = ev.send_order()
    v2_from = cfg.day_start_ms(cfg.schema_v2_day)
    phase1, phase2 = order[ev.sent_ms[order] < v2_from], order[ev.sent_ms[order] >= v2_from]
    summary["kafka"] = {"phase1": publish(ev, phase1, args.bootstrap, args.topic)}
    lap("publish_phase1")
    b1 = ingest(spark, args.bootstrap, args.topic, str(checkpoint), SCHEMA_V1)
    lap("ingest_phase1")

    rows_before = spark.table(BRONZE).count()
    replayed = replay_last_batch(checkpoint)
    b_replay = ingest(spark, args.bootstrap, args.topic, str(checkpoint), SCHEMA_V1)
    summary["exactly_once_replay"] = {"replayed_batch": replayed, "batch_stats": b_replay,
                                      "rows_before": rows_before, "rows_after": spark.table(BRONZE).count()}
    lap("replay_check")

    summary["kafka"]["phase2"] = publish(ev, phase2, args.bootstrap, args.topic)
    lap("publish_phase2")
    b2 = ingest(spark, args.bootstrap, args.topic, str(checkpoint), SCHEMA_V2)
    lap("ingest_phase2")

    bronze = spark.table(BRONZE)
    agg = bronze.agg(F.count("*").alias("rows"), F.countDistinct("event_id").alias("ids"),
                     F.sum(F.col("device_class").isNotNull().cast("int")).alias("with_device")).first()
    summary["bronze"] = {
        "messages_published": summary["kafka"]["phase1"]["messages"] + summary["kafka"]["phase2"]["messages"],
        "rows": agg["rows"], "distinct_event_ids": agg["ids"], "expected_unique_events": ev.truth["unique_events"],
        "rows_with_device_class": agg["with_device"],
        "micro_batches": len(b1) + len(b2), "columns_added": [c for b in b2 for c in b.get("columns_added", [])],
        "median_merge_s": sorted(b["merge_seconds"] for b in b1 + b2 if "merge_seconds" in b)[len(b1 + b2) // 2],
    }

    # Daily scheduled runs: day D runs at 02:00 on D+1 and sees only events received by then.
    first_day = cfg.start.date()
    for d in range(cfg.days):
        day = first_day + timedelta(days=d)
        cutoff = cfg.start.replace(tzinfo=None) + timedelta(days=d + 1, hours=2)
        write_silver(spark, day, day, cutoff)
    daily_snapshot = spark.sql(f"SELECT snapshot_id FROM {SILVER}.snapshots ORDER BY committed_at DESC LIMIT 1").first()[0]
    lap("silver_daily_runs")

    last_day = first_day + timedelta(days=cfg.days - 1)
    write_silver(spark, first_day, last_day, None)
    after_backfill = clicks_by_day(spark, SILVER)
    before_backfill = clicks_by_day(spark, f"{SILVER} VERSION AS OF {daily_snapshot}")
    write_silver(spark, first_day, last_day, None)  # rerun: overwrite of the same partitions
    lap("backfill")
    totals = lambda m, i: sum(v[i] for v in m.values())  # noqa: E731
    summary["backfill"] = {
        "daily_runs": {"impressions": totals(before_backfill, 0), "clicks": totals(before_backfill, 1)},
        "after_backfill": {"impressions": totals(after_backfill, 0), "clicks": totals(after_backfill, 1)},
        "expected": {"impressions": ev.truth["impressions"],
                     "attributed_clicks": ev.truth["clicks"] - ev.truth["clicks_beyond_window"]},
        "rerun_identical": clicks_by_day(spark, SILVER) == after_backfill,
        "per_day": {d: {"daily_run": before_backfill.get(d), "backfill": v} for d, v in after_backfill.items()},
    }

    write_daily_metrics(spark, cfg.experiment)
    _, summary["privacy_report"] = advertiser_report(spark, min_users=50)
    lap("gold")

    users = experiment.per_user(spark, cfg.experiment, first_day + timedelta(days=cfg.pre_days))
    summary["experiment"] = {"true_lift": cfg.true_lift, **experiment.analyze(users)}
    lap("experiment")

    grids = anomaly.hourly_ctr(spark)
    flagged = {p: anomaly.detect(g) for p, g in grids.items()}
    summary["anomaly"] = {"flagged": {p: f for p, f in flagged.items() if f},
                          **anomaly.evaluate(flagged, "today_tab", cfg.outage_day, cfg.outage_hours)}
    lap("anomaly")

    probe = (f"SELECT placement, count(*), sum(cost_micros) FROM {BRONZE} "
             f"WHERE event_ts >= TIMESTAMP '2026-09-15' AND event_ts < TIMESTAMP '2026-09-16' GROUP BY placement")
    before = {**maintenance.file_stats(spark, BRONZE), "query_s": maintenance.time_query(spark, probe)}
    compaction = maintenance.compact(spark, BRONZE)
    after = {**maintenance.file_stats(spark, BRONZE), "query_s": maintenance.time_query(spark, probe)}
    summary["maintenance"] = {"before": before, "compaction": compaction, "after": after,
                              "snapshots": maintenance.expire_snapshots(spark, BRONZE)}
    lap("maintenance")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "backfill"}, indent=2, default=str))


if __name__ == "__main__":
    main()
