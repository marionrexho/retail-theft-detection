from theft.geometry import point_in_polygon
from theft.scoring import score_video, summarize


def alert(t, kind="bagged"):
    return {"t": t, "kind": kind, "person_id": 1, "alert": True}


def test_alert_close_to_label_is_caught():
    s = score_video("v", 60, [{"t": 10.0, "label": "bagged"}], [alert(11.5)])
    assert len(s.caught) == 1 and not s.missed and not s.false_alarms


def test_alert_too_far_counts_as_miss_and_false_alarm():
    s = score_video("v", 60, [{"t": 10.0, "label": "bagged"}], [alert(20.0)])
    assert len(s.missed) == 1 and len(s.false_alarms) == 1


def test_one_alert_cannot_catch_two_thefts():
    labels = [{"t": 10.0, "label": "bagged"}, {"t": 11.0, "label": "pocketed"}]
    s = score_video("v", 60, labels, [alert(10.5)])
    assert len(s.caught) == 1 and len(s.missed) == 1


def test_false_alarm_reports_the_normal_action_next_to_it():
    labels = [{"t": 30.0, "label": "carted"}]
    s = score_video("v", 60, labels, [alert(31.0, "pocketed")])
    (_, near), = s.false_alarms
    assert near["label"] == "carted"


def test_summary_rates():
    a = score_video("a", 1800, [{"t": 5, "label": "bagged"}], [alert(5), alert(100)])
    b = score_video("b", 1800, [{"t": 5, "label": "pocketed"}], [])
    sm = summarize([a, b])
    assert sm["catch_rate"] == 0.5
    assert sm["precision"] == 0.5
    assert sm["false_alarms_per_hour"] == 1.0


def test_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon(5, 5, sq)
    assert not point_in_polygon(15, 5, sq)
