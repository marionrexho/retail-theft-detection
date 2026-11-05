"""The core decision logic, kept free of any model code so it can be unit tested.

For every tracked person and each wrist:
  1. The wrist must stay inside a shelf zone for at least `dwell_s` (a real reach,
     not a hand passing by).
  2. Once it leaves the zone, it is followed for `follow_s` seconds.
  3. The first place it goes decides the event:
       - into a bag that belongs to the person  -> "bagged"   (alert)
       - to the person's own waist              -> "pocketed" (alert)
       - back into the shelf zone               -> reach cancelled (put back / re-reach)
       - nowhere in time                        -> "unknown"  (no alert)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .geometry import Box, box_contains, boxes_overlap, distance, expand_box, in_any_polygon

# COCO keypoint indexes used by YOLO pose models.
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
WRISTS = {"left": L_WRIST, "right": R_WRIST}


@dataclass
class Params:
    dwell_s: float = 0.2          # time a wrist must spend in the shelf zone
    follow_s: float = 3.0         # how long to follow the wrist after it leaves
    kp_conf: float = 0.3          # ignore keypoints the model is unsure about
    waist_frac: float = 0.12      # wrist-to-hip distance, as a fraction of person height
    bag_margin_frac: float = 0.03 # grow bag boxes a little, as a fraction of person height
    cooldown_s: float = 2.0       # min gap between alerts for the same person
    forget_s: float = 5.0         # drop state for people not seen this long


@dataclass
class Person:
    track_id: int
    box: Box
    keypoints: list  # 17 entries of (x, y, conf), in pixels


@dataclass
class Event:
    t: float
    kind: str
    person_id: int
    wrist: str
    box: Box
    reach_started: float
    alert: bool = field(default=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["t"] = round(self.t, 2)
        d["reach_started"] = round(self.reach_started, 2)
        d["box"] = [round(v, 1) for v in self.box]
        return d


@dataclass
class _WristState:
    in_zone_since: float | None = None
    reach_left_at: float | None = None
    reach_started: float | None = None


def _kp(p: Person, idx: int, min_conf: float):
    x, y, c = p.keypoints[idx]
    return (x, y) if c >= min_conf else None


class ReachTracker:
    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.wrists: dict[tuple[int, str], _WristState] = {}
        self.last_seen: dict[int, float] = {}
        self.last_alert: dict[int, float] = {}

    def update(self, t: float, people: list[Person], bags: list[Box],
               zones: list[list[tuple[float, float]]]) -> list[Event]:
        """Feed one frame. Returns the events that finished in this frame."""
        out: list[Event] = []
        for person in people:
            self.last_seen[person.track_id] = t
            for wrist_name, idx in WRISTS.items():
                ev = self._update_wrist(t, person, wrist_name, idx, bags, zones)
                if ev:
                    out.append(ev)
        self._forget(t)
        return out

    def _update_wrist(self, t, person, wrist_name, idx, bags, zones) -> Event | None:
        p = self.p
        st = self.wrists.setdefault((person.track_id, wrist_name), _WristState())
        wrist = _kp(person, idx, p.kp_conf)
        if wrist is None:
            # Keep state; a wrist hidden for a moment is common. The follow
            # window still runs out, so a lost wrist ends as "unknown".
            return self._maybe_timeout(t, person, wrist_name, st)

        if in_any_polygon(wrist[0], wrist[1], zones):
            if st.in_zone_since is None:
                st.in_zone_since = t
            st.reach_left_at = None  # back at the shelf: cancel any pending reach
            st.reach_started = None
            return None

        # Wrist is outside the shelf zone.
        if st.in_zone_since is not None:
            if t - st.in_zone_since >= p.dwell_s:
                st.reach_left_at = t
                st.reach_started = st.in_zone_since
            st.in_zone_since = None

        if st.reach_left_at is None:
            return None

        kind = self._where_is_wrist(wrist, person, bags)
        if kind is None:
            return self._maybe_timeout(t, person, wrist_name, st)
        return self._finish(t, kind, person, wrist_name, st)

    def _where_is_wrist(self, wrist, person: Person, bags: list[Box]) -> str | None:
        p = self.p
        height = max(1.0, person.box[3] - person.box[1])

        # Into a bag the person is carrying (the bag overlaps the person).
        person_area = expand_box(person.box, 0.05 * height)
        for bag in bags:
            if not boxes_overlap(bag, person_area):
                continue
            if box_contains(expand_box(bag, p.bag_margin_frac * height), *wrist):
                return "bagged"

        # To their own waist: close to a hip, and below the shoulders.
        hips = [h for h in (_kp(person, L_HIP, p.kp_conf), _kp(person, R_HIP, p.kp_conf)) if h]
        if hips:
            shoulders = [s for s in (_kp(person, L_SHOULDER, p.kp_conf),
                                     _kp(person, R_SHOULDER, p.kp_conf)) if s]
            below_shoulders = all(wrist[1] > s[1] for s in shoulders)
            near_hip = min(distance(wrist, h) for h in hips) <= p.waist_frac * height
            if near_hip and below_shoulders:
                return "pocketed"
        return None

    def _maybe_timeout(self, t, person, wrist_name, st) -> Event | None:
        if st.reach_left_at is not None and t - st.reach_left_at > self.p.follow_s:
            return self._finish(t, "unknown", person, wrist_name, st)
        return None

    def _finish(self, t, kind, person, wrist_name, st) -> Event | None:
        started = st.reach_started if st.reach_started is not None else t
        st.reach_left_at = None
        st.reach_started = None
        alert = kind in ("bagged", "pocketed")
        if alert:
            last = self.last_alert.get(person.track_id)
            if last is not None and t - last < self.p.cooldown_s:
                return None  # same person, same moment: don't double-alert
            self.last_alert[person.track_id] = t
        return Event(t=t, kind=kind, person_id=person.track_id, wrist=wrist_name,
                     box=person.box, reach_started=started, alert=alert)

    def _forget(self, t: float) -> None:
        gone = [pid for pid, seen in self.last_seen.items() if t - seen > self.p.forget_s]
        for pid in gone:
            self.last_seen.pop(pid, None)
            self.last_alert.pop(pid, None)
            for w in WRISTS:
                self.wrists.pop((pid, w), None)
