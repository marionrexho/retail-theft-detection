"""Compare the detector's alerts with a person's labels."""

from __future__ import annotations

from dataclasses import dataclass, field

from .labels import THEFT_LABELS


@dataclass
class VideoScore:
    name: str
    duration_s: float
    caught: list = field(default_factory=list)        # (label, alert) pairs
    missed: list = field(default_factory=list)        # labels with no alert
    false_alarms: list = field(default_factory=list)  # (alert, nearest normal label or None)

    @property
    def n_thefts(self) -> int:
        return len(self.caught) + len(self.missed)


def score_video(name: str, duration_s: float, labels: list[dict], alerts: list[dict],
                tolerance_s: float = 3.0) -> VideoScore:
    """Greedy matching: each theft label is matched to the nearest unused alert
    within `tolerance_s`. Alerts left over are false alarms."""
    thefts = sorted((l for l in labels if l["label"] in THEFT_LABELS), key=lambda l: l["t"])
    normals = [l for l in labels if l["label"] not in THEFT_LABELS]
    unused = sorted(alerts, key=lambda a: a["t"])
    s = VideoScore(name=name, duration_s=duration_s)

    for lab in thefts:
        best = None
        for a in unused:
            gap = abs(_alert_time(a) - lab["t"])
            if gap <= tolerance_s and (best is None or gap < abs(_alert_time(best) - lab["t"])):
                best = a
        if best is None:
            s.missed.append(lab)
        else:
            s.caught.append((lab, best))
            unused.remove(best)

    for a in unused:
        near = [n for n in normals if abs(n["t"] - _alert_time(a)) <= tolerance_s]
        near.sort(key=lambda n: abs(n["t"] - _alert_time(a)))
        s.false_alarms.append((a, near[0] if near else None))
    return s


def _alert_time(alert: dict) -> float:
    # A person labels the moment the item goes in, which falls between the end of
    # the reach and when the detector decided. Use the decision time.
    return alert["t"]


def summarize(scores: list[VideoScore]) -> dict:
    thefts = sum(s.n_thefts for s in scores)
    caught = sum(len(s.caught) for s in scores)
    fas = sum(len(s.false_alarms) for s in scores)
    hours = sum(s.duration_s for s in scores) / 3600
    return {
        "videos": len(scores),
        "thefts_labeled": thefts,
        "caught": caught,
        "missed": thefts - caught,
        "false_alarms": fas,
        "catch_rate": (caught / thefts) if thefts else None,
        "precision": (caught / (caught + fas)) if (caught + fas) else None,
        "false_alarms_per_hour": (fas / hours) if hours else None,
        "hours_of_video": hours,
    }
