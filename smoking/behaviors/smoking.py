from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from models.loader import Detection
from utils.config import SmokingBehaviorConfig


@dataclass(slots=True)
class SmokingBehaviorResult:
    detections: list[Detection]
    derived_events: list[Detection]
    status_lines: list[str]


@dataclass(slots=True)
class SmokingTrackState:
    smoking_score: int = 0
    last_seen_monotonic: float = 0.0
    last_event_monotonic: float | None = None


class SmokingBehaviorAnalyzer:
    def __init__(self, config: SmokingBehaviorConfig, logger: Any) -> None:
        self.config = config
        self.logger = logger
        self._tracks: dict[int, SmokingTrackState] = {}

    def process(self, detections: list[Detection]) -> SmokingBehaviorResult:
        if not self.config.enabled:
            return SmokingBehaviorResult(detections=detections, derived_events=[], status_lines=[])

        now = time.monotonic()
        relevant = [
            detection
            for detection in detections
            if detection.model_alias == self.config.model
        ]
        persons = [
            detection
            for detection in relevant
            if detection.label.lower() == self.config.person_label
        ]
        cigarettes = [
            detection
            for detection in relevant
            if detection.label.lower() == self.config.cigarette_label
        ]
        smokes = [
            detection
            for detection in relevant
            if self.config.smoke_label and detection.label.lower() == self.config.smoke_label
        ]

        derived_events: list[Detection] = []
        active_smokers = 0
        active_candidates = 0
        seen_track_ids: set[int] = set()

        for person in persons:
            if person.track_id is None and self.config.require_person_track:
                person.overlay_label = "PERSON untracked"
                person.overlay_color = (0, 200, 120)
                continue

            track_id = int(person.track_id if person.track_id is not None else hash(person.bbox))
            seen_track_ids.add(track_id)
            state = self._tracks.setdefault(track_id, SmokingTrackState())
            state.last_seen_monotonic = now

            matched_cigarette = _match_object_to_person(
                person=person,
                candidates=cigarettes,
                max_distance=self.config.max_distance_px,
            )
            matched_smoke = _match_object_to_person(
                person=person,
                candidates=smokes,
                max_distance=int(self.config.max_distance_px * self.config.smoke_distance_multiplier),
            )

            score_increment = 1 if matched_cigarette else 0
            if matched_smoke:
                score_increment += self.config.smoke_boost_frames

            if score_increment > 0:
                state.smoking_score += score_increment
            else:
                state.smoking_score = max(0, state.smoking_score - self.config.decay_frames)

            person.metadata.update(
                {
                    "behavior": "smoking_association",
                    "track_id": track_id,
                    "smoking_score": state.smoking_score,
                    "smoking_threshold": self.config.min_frames,
                    "matched_cigarette": matched_cigarette is not None,
                    "matched_smoke": matched_smoke is not None,
                }
            )

            if matched_cigarette or matched_smoke:
                active_candidates += 1

            if matched_cigarette is not None:
                matched_cigarette.overlay_label = f"CIGARETTE linked #{track_id}"
                matched_cigarette.overlay_color = (0, 80, 255)

            if matched_smoke is not None:
                matched_smoke.overlay_label = f"SMOKE linked #{track_id}"
                matched_smoke.overlay_color = (180, 180, 180)

            if state.smoking_score >= self.config.min_frames:
                active_smokers += 1
                person.overlay_label = f"SMOKING #{track_id} score={state.smoking_score}"
                person.overlay_color = (0, 0, 255)
                person.metadata["smoking_confirmed"] = True

                if self._should_emit_event(state, now):
                    derived_events.append(
                        Detection(
                            label=self.config.event_type,
                            confidence=_derived_confidence(person, matched_cigarette, matched_smoke),
                            bbox=person.bbox,
                            model_alias=self.config.model,
                            event_type=self.config.event_type,
                            trigger_in_zones_only=False,
                            track_id=track_id,
                            cooldown_seconds=self.config.event_cooldown_seconds,
                            emit_event=True,
                            display=False,
                            metadata={
                                "behavior": "smoking_association",
                                "track_id": track_id,
                                "smoking_score": state.smoking_score,
                                "evidence": {
                                    "person_confidence": person.confidence,
                                    "cigarette_confidence": matched_cigarette.confidence if matched_cigarette else None,
                                    "smoke_confidence": matched_smoke.confidence if matched_smoke else None,
                                },
                            },
                        )
                    )
                    state.last_event_monotonic = now
            elif matched_cigarette or matched_smoke:
                person.overlay_label = (
                    f"POSSIBLE SMOKING #{track_id} {state.smoking_score}/{self.config.min_frames}"
                )
                person.overlay_color = (0, 215, 255)
            else:
                person.overlay_label = f"PERSON #{track_id}"
                person.overlay_color = (0, 220, 120)

        self._cleanup_tracks(now, seen_track_ids)

        status_lines = [
            f"smokers={active_smokers} candidates={active_candidates} tracked={len(self._tracks)}"
        ]
        return SmokingBehaviorResult(
            detections=detections,
            derived_events=derived_events,
            status_lines=status_lines,
        )

    def _should_emit_event(self, state: SmokingTrackState, now: float) -> bool:
        if state.last_event_monotonic is None:
            return True
        return (now - state.last_event_monotonic) >= self.config.event_cooldown_seconds

    def _cleanup_tracks(self, now: float, seen_track_ids: set[int]) -> None:
        stale_tracks = [
            track_id
            for track_id, state in self._tracks.items()
            if track_id not in seen_track_ids
            and (now - state.last_seen_monotonic) > self.config.stale_track_seconds
        ]
        for track_id in stale_tracks:
            del self._tracks[track_id]


def _match_object_to_person(
    person: Detection,
    candidates: list[Detection],
    max_distance: int,
) -> Detection | None:
    best_candidate: Detection | None = None
    best_distance: float | None = None

    expanded_box = _expand_box(person.bbox, max_distance)
    for candidate in candidates:
        cx, cy = _box_center(candidate.bbox)

        if _point_inside_box(cx, cy, expanded_box):
            distance = _distance_point_to_box(cx, cy, person.bbox)
        else:
            distance = _distance_point_to_box(cx, cy, person.bbox)
            if distance > max_distance:
                continue

        if best_candidate is None or best_distance is None or distance < best_distance:
            best_candidate = candidate
            best_distance = distance

    return best_candidate


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance_point_to_box(px: float, py: float, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    closest_x = max(x1, min(px, x2))
    closest_y = max(y1, min(py, y2))
    return math.hypot(px - closest_x, py - closest_y)


def _point_inside_box(px: float, py: float, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def _expand_box(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def _derived_confidence(
    person: Detection,
    cigarette: Detection | None,
    smoke: Detection | None,
) -> float:
    confidences = [person.confidence]
    if cigarette is not None:
        confidences.append(cigarette.confidence)
    if smoke is not None:
        confidences.append(smoke.confidence)
    return max(confidences)
