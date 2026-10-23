from datetime import date, datetime

from adlake.ingest import merge_batch
from adlake.tables import SILVER
from adlake.transform import advertiser_report, attributed, write_silver
from conftest import event, events_df
from test_ingest import with_kafka_cols

DAY = date(2026, 9, 1)
CUTOFF = datetime(2026, 9, 2, 2, 0)


def load(spark, rows):
    merge_batch(with_kafka_cols(events_df(spark, rows)))


def silver(spark):
    return {r["impression_id"]: r["clicked"] for r in spark.table(SILVER).collect()}


def test_click_counts_only_inside_the_30_minute_window_and_only_once(tables):
    load(tables, [
        event("a", imp="in"), event("a1", "click", imp="in", ts="2026-09-01 10:29:00", cost=500_000),
        event("a2", "click", imp="in", ts="2026-09-01 10:29:30", cost=900_000),
        event("b", imp="out"), event("b1", "click", imp="out", ts="2026-09-01 10:31:00"),
    ])
    rows = {r["impression_id"]: r for r in attributed(tables, DAY, DAY, None).collect()}
    assert rows["in"]["clicked"] == 1 and rows["in"]["cost_micros"] == 500_000  # first click wins
    assert rows["out"]["clicked"] == 0


def test_click_before_midnight_impression_crosses_the_day_boundary(tables):
    load(tables, [event("a", imp="late", ts="2026-09-01 23:50:00"), event("a1", "click", imp="late", ts="2026-09-02 00:05:00")])
    assert {r["impression_id"]: r["clicked"] for r in attributed(tables, DAY, DAY, None).collect()} == {"late": 1}


def test_scheduled_run_misses_late_click_and_backfill_recovers_it(tables):
    load(tables, [event("a", imp="i1"),
                  event("a1", "click", imp="i1", ts="2026-09-01 10:05:00", sent="2026-09-02 07:00:00")])
    write_silver(tables, DAY, DAY, CUTOFF)
    assert silver(tables) == {"i1": 0}
    write_silver(tables, DAY, DAY, None)
    assert silver(tables) == {"i1": 1}


def test_rerunning_a_day_replaces_it_and_leaves_other_days_alone(tables):
    load(tables, [event("a", imp="d1"), event("b", imp="d2", ts="2026-09-02 09:00:00")])
    write_silver(tables, DAY, date(2026, 9, 2), None)
    write_silver(tables, DAY, DAY, None)
    write_silver(tables, DAY, DAY, None)
    assert sorted(silver(tables)) == ["d1", "d2"]
    assert tables.table(SILVER).count() == 2


def test_report_suppresses_rows_backed_by_too_few_people(tables):
    rows = [event(f"e{u}", imp=f"i{u}", user=f"u{u}", campaign=1001) for u in range(60)]
    rows += [event(f"f{u}", imp=f"j{u}", user=f"u{u}", campaign=2002) for u in range(5)]
    load(tables, rows)
    write_silver(tables, DAY, DAY, None)
    released, stats = advertiser_report(tables, min_users=50)
    assert [r["campaign_id"] for r in released.collect()] == [1001]
    assert "users" not in released.columns
    assert stats["suppressed_rows"] == 1
