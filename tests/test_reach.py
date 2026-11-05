"""Tests for the reach logic, using made-up people instead of real model output."""

from theft.reach import L_HIP, L_SHOULDER, L_WRIST, R_HIP, R_SHOULDER, R_WRIST, Params, Person, ReachTracker

SHELF = [[(0, 0), (100, 0), (100, 400), (0, 400)]]  # shelf on the left of the frame
BOX = (150, 100, 350, 500)  # person standing right of the shelf, 400 px tall
HIP_L, HIP_R = (220, 330), (280, 330)


def person(wrist, pid=1, conf=0.9):
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[L_SHOULDER] = (220, 180, 0.9)
    kps[R_SHOULDER] = (280, 180, 0.9)
    kps[L_HIP] = (*HIP_L, 0.9)
    kps[R_HIP] = (*HIP_R, 0.9)
    kps[L_WRIST] = (*wrist, conf)
    kps[R_WRIST] = (300, 250, 0.9)  # other hand idle, mid-body
    return Person(track_id=pid, box=BOX, keypoints=kps)


def run(path, bags=(), params=None, dt=0.1):
    """path: list of wrist positions, one per frame. Returns all events."""
    tr = ReachTracker(params or Params())
    events = []
    for i, w in enumerate(path):
        events += tr.update(i * dt, [person(w)], list(bags), SHELF)
    return events


IN_SHELF = (50, 200)
MID_AIR = (180, 150)  # outside the shelf, far from hips and bags
AT_WAIST = (225, 340)


def test_reach_to_waist_is_pocketed():
    ev = run([IN_SHELF] * 5 + [MID_AIR] * 3 + [AT_WAIST])
    assert [e.kind for e in ev] == ["pocketed"]
    assert ev[0].alert and ev[0].wrist == "left"


def test_reach_into_own_bag_is_bagged():
    bag = (160, 300, 230, 380)  # overlaps the person
    ev = run([IN_SHELF] * 5 + [(190, 340)], bags=[bag])
    assert [e.kind for e in ev] == ["bagged"]


def test_bag_that_is_not_near_the_person_does_not_count():
    far_bag = (600, 300, 700, 380)
    ev = run([IN_SHELF] * 5 + [(650, 340)] * 2, bags=[far_bag])
    assert ev == []


def test_quick_pass_through_shelf_is_ignored():
    # 0.1 s in the zone is shorter than the 0.2 s dwell
    ev = run([MID_AIR, IN_SHELF, MID_AIR, AT_WAIST])
    assert ev == []


def test_hand_returning_to_shelf_cancels_the_reach():
    ev = run([IN_SHELF] * 5 + [MID_AIR] * 3 + [IN_SHELF] * 2)
    assert ev == []


def test_hand_that_goes_nowhere_times_out_as_unknown():
    ev = run([IN_SHELF] * 5 + [MID_AIR] * 40)  # 4 s outside, follow window is 3 s
    assert [e.kind for e in ev] == ["unknown"]
    assert not ev[0].alert


def test_waist_without_a_shelf_reach_is_ignored():
    ev = run([MID_AIR] * 5 + [AT_WAIST] * 5)
    assert ev == []


def test_same_person_is_not_alerted_twice_in_a_row():
    reach = [IN_SHELF] * 5 + [AT_WAIST]
    ev = run(reach + reach)  # second reach 0.6 s later, inside the 2 s cooldown
    assert [e.kind for e in ev] == ["pocketed"]


def test_low_confidence_wrist_is_not_used():
    tr = ReachTracker()
    events = []
    for i in range(5):
        events += tr.update(i * 0.1, [person(IN_SHELF)], [], SHELF)
    events += tr.update(0.5, [person(AT_WAIST, conf=0.1)], [], SHELF)
    assert events == []
