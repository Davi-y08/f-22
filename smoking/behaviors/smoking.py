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
    smoking_score: float = 0.0
    evidence_frames: int = 0
    cigarette_evidence_frames: int = 0
    smoke_evidence_frames: int = 0
    last_seen_monotonic: float = 0.0
    last_evidence_monotonic: float = 0.0
    last_cigarette_monotonic: float | None = None
    last_event_monotonic: float | None = None


@dataclass(slots=True)
class EvidenceMatch:
    detection: Detection
    distance_px: float
    max_distance_px: float
    distance_score: float
    body_position_score: float
    association_score: float


class SmokingBehaviorAnalyzer:
    def __init__(self, config: SmokingBehaviorConfig, logger: Any) -> None:
        self.config = config
        self.logger = logger
        self._tracks: dict[int, SmokingTrackState] = {}

    def process(self, detections: list[Detection]) -> SmokingBehaviorResult:
        if not self.config.enabled:
            return SmokingBehaviorResult(detections=detections, derived_events=[], status_lines=[])

        now = time.monotonic()
        self._cleanup_tracks(now, set())
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
        used_cigarette_ids: set[int] = set()
        used_smoke_ids: set[int] = set()

        for person in sorted(persons, key=lambda item: item.confidence, reverse=True):
            if person.confidence < self.config.min_person_confidence:
                person.overlay_label = f"PERSON low-conf {person.confidence:.2f}"
                person.overlay_color = (0, 170, 120)
                continue

            if person.track_id is None and self.config.require_person_track:
                person.overlay_label = "PERSON untracked"
                person.overlay_color = (0, 200, 120)
                continue

            track_id = int(person.track_id if person.track_id is not None else hash(person.bbox))
            seen_track_ids.add(track_id)
            state = self._tracks.setdefault(track_id, SmokingTrackState())
            state.last_seen_monotonic = now
            if state.last_cigarette_monotonic is not None and now - state.last_cigarette_monotonic > self.config.stale_track_seconds:
                state.cigarette_evidence_frames = 0
                state.last_cigarette_monotonic = None

            matched_cigarette = _match_object_to_person(
                person=person,
                candidates=[
                    detection
                    for detection in cigarettes
                    if detection.confidence >= self.config.min_cigarette_confidence
                    and id(detection) not in used_cigarette_ids
                ],
                max_distance=self.config.max_distance_px,
                min_association_score=self.config.min_association_score,
                distance_person_scale=self.config.distance_person_scale,
            )
            matched_smoke = _match_object_to_person(
                person=person,
                candidates=[
                    detection
                    for detection in smokes
                    if detection.confidence >= self.config.min_smoke_confidence
                    and id(detection) not in used_smoke_ids
                ],
                max_distance=int(self.config.max_distance_px * self.config.smoke_distance_multiplier),
                min_association_score=self.config.min_association_score,
                distance_person_scale=self.config.distance_person_scale * self.config.smoke_distance_multiplier,
            )

            score_increment = 0.0
            if matched_cigarette:
                score_increment += _weighted_evidence_increment(
                    matched_cigarette,
                    base_weight=1.35,
                    minimum_increment=0.65,
                )
            if matched_smoke:
                score_increment += _weighted_evidence_increment(
                    matched_smoke,
                    base_weight=float(self.config.smoke_boost_frames),
                    minimum_increment=0.0,
                )

            if score_increment > 0:
                state.smoking_score = min(
                    float(self.config.min_frames) * 2.5,
                    state.smoking_score + score_increment,
                )
                state.evidence_frames += 1
                state.last_evidence_monotonic = now
                if matched_cigarette:
                    state.cigarette_evidence_frames += 1
                    state.last_cigarette_monotonic = now
                if matched_smoke:
                    state.smoke_evidence_frames += 1
            else:
                state.smoking_score = max(0, state.smoking_score - self.config.decay_frames)
                if state.smoking_score <= 0:
                    state.evidence_frames = 0
                    state.cigarette_evidence_frames = 0
                    state.smoke_evidence_frames = 0

            person.metadata.update(
                {
                    "behavior": "smoking_association",
                    "track_id": track_id,
                    "smoking_score": round(state.smoking_score, 2),
                    "smoking_threshold": self.config.min_frames,
                    "evidence_frames": state.evidence_frames,
                    "cigarette_evidence_frames": state.cigarette_evidence_frames,
                    "smoke_evidence_frames": state.smoke_evidence_frames,
                    "matched_cigarette": matched_cigarette is not None,
                    "matched_smoke": matched_smoke is not None,
                    "cigarette_association_score": _optional_round_score(matched_cigarette),
                    "smoke_association_score": _optional_round_score(matched_smoke),
                }
            )

            if matched_cigarette or matched_smoke:
                active_candidates += 1

            if matched_cigarette is not None:
                used_cigarette_ids.add(id(matched_cigarette.detection))
                matched_cigarette.detection.overlay_label = (
                    f"CIGARETTE linked #{track_id} {matched_cigarette.association_score:.2f}"
                )
                matched_cigarette.detection.overlay_color = (0, 80, 255)

            if matched_smoke is not None:
                used_smoke_ids.add(id(matched_smoke.detection))
                matched_smoke.detection.overlay_label = f"SMOKE linked #{track_id} {matched_smoke.association_score:.2f}"
                matched_smoke.detection.overlay_color = (180, 180, 180)

            if _is_confirmed_smoking(state, self.config):
                active_smokers += 1
                person.overlay_label = f"SMOKING #{track_id} score={state.smoking_score:.1f}"
                person.overlay_color = (0, 0, 255)
                person.metadata["smoking_confirmed"] = True

                if (matched_cigarette or matched_smoke) and self._should_emit_event(state, now):
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
                                "smoking_score": round(state.smoking_score, 2),
                                "evidence_frames": state.evidence_frames,
                                "cigarette_evidence_frames": state.cigarette_evidence_frames,
                                "smoke_evidence_frames": state.smoke_evidence_frames,
                                "evidence": {
                                    "person_confidence": person.confidence,
                                    "cigarette_confidence": (
                                        matched_cigarette.detection.confidence if matched_cigarette else None
                                    ),
                                    "smoke_confidence": matched_smoke.detection.confidence if matched_smoke else None,
                                    "cigarette_distance_px": (
                                        round(matched_cigarette.distance_px, 2) if matched_cigarette else None
                                    ),
                                    "smoke_distance_px": round(matched_smoke.distance_px, 2) if matched_smoke else None,
                                    "cigarette_association_score": _optional_round_score(matched_cigarette),
                                    "smoke_association_score": _optional_round_score(matched_smoke),
                                },
                            },
                        )
                    )
                    state.last_event_monotonic = now
            elif matched_cigarette or matched_smoke:
                person.overlay_label = (
                    f"POSSIBLE SMOKING #{track_id} {state.smoking_score:.1f}/{self.config.min_frames}"
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
    min_association_score: float,
    distance_person_scale: float,
) -> EvidenceMatch | None:
    best_match: EvidenceMatch | None = None
    effective_max_distance = _effective_max_distance(
        person.bbox,
        configured_max=max_distance,
        distance_person_scale=distance_person_scale,
    )
    expanded_box = _expand_box(person.bbox, int(effective_max_distance))
    for candidate in candidates:
        cx, cy = _box_center(candidate.bbox)

        if _point_inside_box(cx, cy, expanded_box):
            distance = _distance_point_to_box(cx, cy, person.bbox)
        else:
            distance = _distance_point_to_box(cx, cy, person.bbox)
            if distance > effective_max_distance:
                continue

        if distance > effective_max_distance:
            continue

        distance_score = 1.0 - min(1.0, distance / max(effective_max_distance, 1.0))
        body_position_score = _body_position_score(person.bbox, candidate.bbox)
        association_score = candidate.confidence * (
            (0.65 * distance_score) + (0.35 * body_position_score)
        )
        if association_score < min_association_score:
            continue

        match = EvidenceMatch(
            detection=candidate,
            distance_px=distance,
            max_distance_px=effective_max_distance,
            distance_score=distance_score,
            body_position_score=body_position_score,
            association_score=association_score,
        )
        if best_match is None:
            best_match = match
            continue
        if match.association_score > best_match.association_score:
            best_match = match
            continue
        if (
            math.isclose(match.association_score, best_match.association_score)
            and match.distance_px < best_match.distance_px
        ):
            best_match = match

    return best_match


def _effective_max_distance(
    person_box: tuple[int, int, int, int],
    configured_max: int,
    distance_person_scale: float,
) -> float:
    x1, y1, x2, y2 = person_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    diagonal = math.hypot(width, height)
    scaled_limit = max(32.0, diagonal * max(0.05, distance_person_scale))
    return max(1.0, min(float(configured_max), scaled_limit))


def _body_position_score(
    person_box: tuple[int, int, int, int],
    candidate_box: tuple[int, int, int, int],
) -> float:
    _cx, cy = _box_center(candidate_box)
    _x1, y1, _x2, y2 = person_box
    height = max(1.0, float(y2 - y1))
    relative_y = (cy - y1) / height
    if relative_y <= 0.68:
        return 1.0
    if relative_y <= 0.85:
        return 0.62
    return 0.35


def _weighted_evidence_increment(
    match: EvidenceMatch,
    base_weight: float,
    minimum_increment: float,
) -> float:
    increment = base_weight * match.association_score
    return max(minimum_increment, increment)


def _is_confirmed_smoking(state: SmokingTrackState, config: SmokingBehaviorConfig) -> bool:
    return (
        state.smoking_score >= config.min_frames
        and state.evidence_frames >= config.min_evidence_frames
        and state.cigarette_evidence_frames >= config.min_cigarette_evidence_frames
    )


def _optional_round_score(match: EvidenceMatch | None) -> float | None:
    if match is None:
        return None
    return round(match.association_score, 3)


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
    cigarette: EvidenceMatch | None,
    smoke: EvidenceMatch | None,
) -> float:
    if cigarette is not None and smoke is not None:
        confidence = (
            0.30 * person.confidence
            + 0.50 * cigarette.detection.confidence
            + 0.20 * smoke.detection.confidence
        )
        confidence *= 0.75 + (0.25 * max(cigarette.association_score, smoke.association_score))
        return max(0.0, min(1.0, confidence))

    if cigarette is not None:
        confidence = 0.35 * person.confidence + 0.65 * cigarette.detection.confidence
        confidence *= 0.75 + (0.25 * cigarette.association_score)
        return max(0.0, min(1.0, confidence))

    if smoke is not None:
        confidence = 0.40 * person.confidence + 0.60 * smoke.detection.confidence
        confidence *= 0.65 + (0.20 * smoke.association_score)
        return max(0.0, min(1.0, confidence))

    return max(0.0, min(1.0, person.confidence))
