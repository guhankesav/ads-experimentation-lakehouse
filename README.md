# Ads Experimentation Lakehouse

A streaming ads data pipeline and A/B analysis stack: simulated ad impressions and clicks flow
through **Kafka** into **Apache Iceberg** tables with **Spark Structured Streaming**, then through
attribution, daily metrics, a privacy-thresholded advertiser report, experiment analysis
(SRM, delta method, CUPED, A/A calibration), and hourly anomaly detection.

The events are **simulated**, and that is deliberate: the true treatment effect and every injected
defect (duplicates, late arrivals, a schema change, a tracking outage) are known, so each stage is
checked against ground truth instead of eyeballed.

```
simulator ──► Kafka (6 partitions, idempotent producer)
                │
                ▼  Structured Streaming, foreachBatch MERGE on event_id
        lake.ads.events_bronze      Iceberg, days(event_ts), schema v1 → v2 mid-stream
                │
                ▼  30-minute click attribution; daily runs at 02:00, then backfill
        lake.ads.impressions        partition overwrite = idempotent reruns
          │            │                 │
          ▼            ▼                 ▼
  daily_metrics   advertiser report   experiment analysis + hourly CTR anomaly detection
                  (k ≥ 50 users)
```

## Run

```bash
docker compose up -d                                            # Kafka 3.9 (KRaft)
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # needs Java 17+
.venv/bin/python -m adlake.run --users 300000                  # ~11 min on a laptop
.venv/bin/python -m pytest                                      # 24 tests, Kafka test needs the broker
```

## Results (300,000 users, 21 days, 9.77M Kafka messages)

All numbers are from [`docs/run_summary.json`](docs/run_summary.json).

### Exactly-once ingestion

| | |
|---|---|
| Messages published (including 48,840 producer retries) | 9,766,756 |
| Rows in bronze | **9,717,916**, equal to the unique events generated |
| Distinct `event_id`s in bronze | 9,717,916 (no duplicates) |
| Crash replay: last micro-batch reprocessed after deleting its commit marker | 641,386 rows read, **0 inserted** |

Two pieces make this work. The checkpoint replays a micro-batch whose commit was not recorded,
and the `MERGE ... WHEN NOT MATCHED THEN INSERT` on `event_id` turns a replayed batch, or a retry
that landed in a different batch, into a no-op. The merge condition also carries the batch's
`event_ts` range, so Iceberg scans only the day partitions the batch touches.

### Schema evolution

Producers added `device_class` (schema v2) from day 10. The pipeline switched parsers on the same
checkpoint, and `ALTER TABLE ADD COLUMN` changed only metadata: no data files were rewritten.
Old rows read the new column as null, and earlier snapshots still return the v1 schema
(the tests check both).

### Late data and backfill

Each day was first processed the way a 02:00 scheduled job would see it (events received by then),
then backfilled once all data had arrived:

| | Impressions | Attributed clicks |
|---|---|---|
| Daily runs | 9,399,391 | 252,523 |
| After backfill | **9,456,489** | **255,525** |
| Ground truth (clicks within 30 min) | 9,456,489 | 255,525 |

The backfill recovered 57,098 late impressions and 3,002 late clicks, matching ground truth exactly.
It overwrites only the affected day partitions, so running it twice gives identical results;
Iceberg time travel (`VERSION AS OF`) provides the before and after comparison.

### Experiment analysis

The simulator plants a **+4.0%** relative CTR lift for the treatment arm (users split 50/50 by a
salted SHA-256 hash, identical in Python and Spark).

| Check | Result |
|---|---|
| Sample ratio mismatch | 149,053 vs 148,721 users, p = 0.54 (no SRM) |
| CTR lift, delta method | +2.4%, 95% CI [0.9%, 3.9%], p = 0.001 |
| CTR lift, **CUPED** (pre-period CTR as covariate) | **+3.0%, 95% CI [1.7%, 4.3%]**, p < 0.0001 |
| CUPED variance reduction | **23%** (minimum detectable effect 2.1% → 1.8%) |
| A/A calibration, 1,000 random re-splits of control | 5.9% false positives at α = 0.05 |

CTR is a ratio of per-user sums, so impressions from the same user are correlated. The delta
method uses users as the unit of analysis. The raw estimate (+2.4%) sits below the planted 4%
because, in this sample, treatment users happened to have lower baseline CTRs: computing the lift
directly from the simulator's events gives +2.36%. CUPED adjusts for that pre-period imbalance and
its interval covers the true effect.

### Anomaly detection

A tracking outage drops 70% of `today_tab` clicks for six hours on day 15. Hourly CTR per placement
is compared with the median of the same hour over the previous 7 days, scaled by the larger of the
robust spread (MAD) and the expected binomial noise at that hour's volume:

| Injected hours | Detected | False alarms |
|---|---|---|
| 6 | **6 of 6** (z = −6.2 to −7.1) | 2 (low-traffic 02:00 hours, z ≈ 4.7) |

### Privacy

User keys are HMAC-SHA256 pseudonyms, so raw ids never enter the stream. The advertiser report
(campaign × day × placement) releases only rows backed by **at least 50 distinct users** and drops
the user count from the output. At this scale that suppresses 45% of rows but only **4.4% of
impressions**: the long tail of small campaigns.

### Table maintenance

Streaming micro-batches left 52 data files averaging 6.2 MB in bronze. `rewrite_data_files`
compacted them to 22 files averaging 14.6 MB, and a one-day aggregate query went from
0.158 s to 0.100 s (median of 5). `expire_snapshots` then trimmed 13 snapshots to 3.

## Design notes

- **Partitioning** is hidden (`days(event_ts)`): queries filter on the timestamp and Iceberg prunes.
- **Time zones**: PySpark converts timestamps through the Python process's local zone. The first
  version computed MERGE bounds 4 hours off (EDT) and let 4,456 duplicates through on a test run;
  the session now pins both Python and the JVM to UTC.
- **Catalog cache**: `foreachBatch` runs in a cloned Spark session, so the Iceberg catalog cache is
  disabled to make its schema changes visible to the main session.
- **Kafka retention**: messages carry their simulated send time, so the topic is created with
  `retention.ms = -1`; otherwise the broker deleted them as already expired.

## Tests

24 pytest tests, run in CI against a real Kafka broker: assignment parity between Python and Spark,
simulator ground truth, MERGE idempotency and cross-batch deduplication, schema evolution without
file rewrites, attribution window edges and day boundaries, late data and backfill, partition
overwrite isolation, privacy suppression, delta-method variance against a bootstrap, lift recovery,
CUPED, A/A calibration, SRM, anomaly detection, and a Kafka-to-Iceberg run with a crash replay.
