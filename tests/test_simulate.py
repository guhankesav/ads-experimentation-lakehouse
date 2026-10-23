import numpy as np

from adlake.simulate import Config, simulate

CFG = Config(users=3000, seed=11)


def test_same_seed_same_stream():
    a, b = simulate(CFG), simulate(CFG)
    assert np.array_equal(a.event_id, b.event_id) and np.array_equal(a.sent_ms, b.sent_ms)


def test_truth_counts_match_the_emitted_events():
    ev = simulate(CFG)
    assert len(ev) == ev.truth["unique_events"] + ev.truth["duplicate_sends"]
    assert len(np.unique(ev.event_id)) == ev.truth["unique_events"]
    first = np.unique(ev.event_id, return_index=True)[1]
    assert (ev.kind[first] == 0).sum() == ev.truth["impressions"]
    assert (ev.kind[first] == 1).sum() == ev.truth["clicks"]


def test_duplicates_repeat_the_same_event_and_arrive_later():
    ev = simulate(CFG)
    ids, counts = np.unique(ev.event_id, return_counts=True)
    dup_id = ids[counts == 2][0]
    rows = np.flatnonzero(ev.event_id == dup_id)
    assert ev.event_ms[rows[0]] == ev.event_ms[rows[1]] and ev.sent_ms[rows[0]] != ev.sent_ms[rows[1]]


def test_device_class_appears_only_from_the_v2_day():
    ev = simulate(CFG)
    order = ev.send_order()
    v2_from = CFG.day_start_ms(CFG.schema_v2_day)
    for i in order[::500]:
        assert ('"device_class"' in ev.to_json(i)) == (ev.event_ms[i] >= v2_from)


def test_events_never_arrive_before_they_happen():
    ev = simulate(CFG)
    assert (ev.sent_ms >= ev.event_ms).all()
