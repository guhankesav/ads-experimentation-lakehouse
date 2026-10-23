"""Simulated ad-serving event stream with a planted experiment effect and realistic defects.

Everything here is synthetic. Because the true effect and every injected defect are known,
the pipeline and the statistics can be checked against ground truth instead of eyeballed.
"""
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from adlake.assignment import BUCKETS, bucket

PLACEMENTS = np.array(["search_results", "today_tab", "search_tab", "product_page"])
PLACEMENT_SHARE = np.array([0.55, 0.20, 0.15, 0.10])
PLACEMENT_CTR = np.array([1.0, 0.6, 0.8, 1.2])
DEVICES = np.array(["iphone", "ipad"])
ATTRIBUTION_WINDOW_S = 30 * 60
MS_PER_DAY = 86_400_000


@dataclass
class Config:
    users: int = 300_000
    pre_days: int = 7
    experiment_days: int = 14
    start: datetime = datetime(2026, 9, 1, tzinfo=timezone.utc)
    experiment: str = "ranker_v2"
    treatment_share: float = 0.5
    true_lift: float = 0.04          # relative CTR lift for treatment during the experiment
    campaigns: int = 400
    late_share: float = 0.03         # events delayed by device offline time
    late_mean_hours: float = 6.0
    duplicate_share: float = 0.005   # producer retries that resend an event
    schema_v2_day: int = 10          # events from this day on carry device_class
    outage_day: int = 15             # today_tab click tracking outage
    outage_hours: tuple = (9, 15)
    outage_drop: float = 0.7
    seed: int = 7
    salt: bytes = b"rotating-salt-2026-09"

    @property
    def days(self) -> int:
        return self.pre_days + self.experiment_days

    def day_start_ms(self, day: int) -> int:
        return int((self.start + timedelta(days=day)).timestamp() * 1000)


@dataclass
class Events:
    """Columnar event arrays; one row per event, duplicates included, unordered."""
    kind: np.ndarray           # 0 impression, 1 click
    event_id: np.ndarray
    impression_id: np.ndarray
    user: np.ndarray           # index into user_keys
    campaign: np.ndarray
    placement: np.ndarray
    device: np.ndarray
    event_ms: np.ndarray
    sent_ms: np.ndarray
    cost_micros: np.ndarray
    user_keys: np.ndarray
    cfg: Config
    truth: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.kind)

    def send_order(self) -> np.ndarray:
        return np.argsort(self.sent_ms, kind="stable")

    def to_json(self, i: int) -> str:
        e = {
            "event_id": f"e{self.event_id[i]}",
            "event_type": "click" if self.kind[i] else "impression",
            "impression_id": f"i{self.impression_id[i]}",
            "user_key": self.user_keys[self.user[i]],
            "campaign_id": int(self.campaign[i]),
            "placement": PLACEMENTS[self.placement[i]],
            "event_ts": _iso(self.event_ms[i]),
            "sent_ts": _iso(self.sent_ms[i]),
            "cost_micros": int(self.cost_micros[i]),
            "schema_version": 1,
        }
        if self.event_ms[i] >= self.cfg.day_start_ms(self.cfg.schema_v2_day):
            e["schema_version"] = 2
            e["device_class"] = DEVICES[self.device[i]]
        return json.dumps(e, separators=(",", ":"))


def _iso(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def user_keys(cfg: Config) -> np.ndarray:
    """Pseudonymous keys: HMAC of an internal id, so raw ids never enter the stream."""
    return np.array([hmac.new(cfg.salt, str(u).encode(), "sha256").hexdigest()[:24] for u in range(cfg.users)])


def simulate(cfg: Config) -> Events:
    rng = np.random.default_rng(cfg.seed)
    keys = user_keys(cfg)
    treated = np.array([bucket(cfg.experiment, k) < cfg.treatment_share * BUCKETS for k in keys])
    imps_per_day = rng.gamma(2.0, 0.75, cfg.users)          # heavy and light users
    base_ctr = rng.beta(0.6, 19.4, cfg.users)               # mean 3%, wide spread across users
    device = (rng.random(cfg.users) < 0.25).astype(np.int8)  # 25% iPad
    campaign_p = 1 / np.arange(1, cfg.campaigns + 1) ** 1.1
    campaign_p /= campaign_p.sum()
    hour_p = 1 + 0.8 * np.sin((np.arange(24) - 9) / 24 * 2 * np.pi)  # diurnal traffic
    hour_p /= hour_p.sum()

    # Impressions
    counts = rng.poisson(np.tile(imps_per_day, cfg.days))
    user = np.repeat(np.tile(np.arange(cfg.users), cfg.days), counts)
    day = np.repeat(np.repeat(np.arange(cfg.days), cfg.users), counts)
    n = len(user)
    hour = rng.choice(24, n, p=hour_p)
    event_ms = (np.array([cfg.day_start_ms(d) for d in range(cfg.days)])[day]
                + hour * 3_600_000 + rng.integers(0, 3_600_000, n))
    placement = rng.choice(len(PLACEMENTS), n, p=PLACEMENT_SHARE)
    campaign = rng.choice(cfg.campaigns, n, p=campaign_p) + 1001

    # Clicks: user propensity x placement effect x planted treatment lift during the experiment
    in_experiment = day >= cfg.pre_days
    p_click = base_ctr[user] * PLACEMENT_CTR[placement] * np.where(treated[user] & in_experiment, 1 + cfg.true_lift, 1.0)
    clicked = rng.random(n) < np.minimum(p_click, 1.0)
    ci = np.flatnonzero(clicked)
    delay_s = np.where(rng.random(len(ci)) < 0.03, rng.uniform(0, 7200, len(ci)), rng.lognormal(np.log(40), 1.0, len(ci)))
    click_ms = event_ms[ci] + (delay_s * 1000).astype(np.int64)

    # Tracking outage: most today_tab clicks in a six-hour window are never logged
    o_start = cfg.day_start_ms(cfg.outage_day) + cfg.outage_hours[0] * 3_600_000
    o_end = cfg.day_start_ms(cfg.outage_day) + cfg.outage_hours[1] * 3_600_000
    in_outage = (placement[ci] == 1) & (click_ms >= o_start) & (click_ms < o_end)
    lost = in_outage & (rng.random(len(ci)) < cfg.outage_drop)
    ci, click_ms, delay_s = ci[~lost], click_ms[~lost], delay_s[~lost]
    cost = np.round(rng.lognormal(np.log(0.8), 0.6, len(ci)) * 1e6).astype(np.int64)

    kind = np.concatenate([np.zeros(n, np.int8), np.ones(len(ci), np.int8)])
    rows = np.concatenate([np.arange(n), ci])
    ev_ms = np.concatenate([event_ms, click_ms])
    total = len(kind)

    # Delivery: most events arrive in seconds, some after a device was offline for hours
    late = rng.random(total) < cfg.late_share
    lag_ms = np.where(late, rng.exponential(cfg.late_mean_hours * 3_600_000, total),
                      rng.exponential(2_000, total)).astype(np.int64)
    sent_ms = ev_ms + lag_ms

    # Producer retries: the same event (same id) sent twice
    dup = np.flatnonzero(rng.random(total) < cfg.duplicate_share)
    take = np.concatenate([np.arange(total), dup])
    sent_ms = np.concatenate([sent_ms, sent_ms[dup] + rng.integers(200, 5_000, len(dup))])

    ev = Events(
        kind=kind[take], event_id=take.astype(np.int64), impression_id=rows[take].astype(np.int64),
        user=user[rows][take], campaign=campaign[rows][take], placement=placement[rows][take],
        device=device[user[rows]][take], event_ms=ev_ms[take], sent_ms=sent_ms,
        cost_micros=np.concatenate([np.zeros(n, np.int64), cost])[take], user_keys=keys, cfg=cfg,
    )
    ev.truth = {
        "impressions": int(n), "clicks": int(len(ci)), "unique_events": int(total),
        "duplicate_sends": int(len(dup)), "late_events": int(late.sum()),
        "clicks_beyond_window": int((delay_s > ATTRIBUTION_WINDOW_S).sum()),
        "clicks_lost_in_outage": int(lost.sum()), "true_lift": cfg.true_lift,
        "outage": {"day": cfg.outage_day, "placement": "today_tab", "hours": list(cfg.outage_hours)},
        "treated_users": int(treated.sum()), "users": cfg.users,
    }
    return ev
