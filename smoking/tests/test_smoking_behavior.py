from __future__ import annotations

import unittest
from unittest.mock import patch

from behaviors.smoking import SmokingBehaviorAnalyzer
from models.loader import Detection
from utils.config import SmokingBehaviorConfig


def _detection(
    label: str,
    confidence: float,
    bbox: tuple[int, int, int, int],
    track_id: int | None = None,
) -> Detection:
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        model_alias="smoking_monitor",
        event_type=label,
        trigger_in_zones_only=False,
        track_id=track_id,
        emit_event=False,
    )


class SmokingBehaviorAnalyzerTests(unittest.TestCase):
    def test_returning_person_does_not_inherit_expired_evidence(self) -> None:
        analyzer = SmokingBehaviorAnalyzer(SmokingBehaviorConfig(enabled=True, min_frames=3), None)
        def objects():
            return [
                _detection("person", .95, (50, 40, 160, 240), track_id=1),
                _detection("cigarette", .9, (78, 95, 88, 105)),
            ]
        with patch("behaviors.smoking.time.monotonic", return_value=100):
            analyzer.process(objects())
            analyzer.process(objects())
        with patch("behaviors.smoking.time.monotonic", return_value=120):
            result = analyzer.process(objects())
        self.assertEqual(result.derived_events, [])
        self.assertEqual(analyzer._tracks[1].evidence_frames, 1)

    def test_old_cigarette_does_not_validate_smoke_indefinitely(self) -> None:
        analyzer = SmokingBehaviorAnalyzer(SmokingBehaviorConfig(enabled=True, min_frames=2, min_evidence_frames=1), None)
        person = lambda: _detection("person", .95, (50, 40, 160, 240), track_id=1)
        with patch("behaviors.smoking.time.monotonic", return_value=100):
            analyzer.process([person(), _detection("cigarette", .9, (78, 95, 88, 105))])
        for now in range(101, 131):
            with patch("behaviors.smoking.time.monotonic", return_value=now):
                result = analyzer.process([person(), _detection("smoke", .9, (85, 70, 130, 115))])
            if now > 105:
                self.assertEqual(result.derived_events, [])

    def test_confirms_smoking_after_consistent_near_cigarette_evidence(self) -> None:
        analyzer = SmokingBehaviorAnalyzer(
            SmokingBehaviorConfig(
                enabled=True,
                min_frames=3,
                min_evidence_frames=2,
                min_cigarette_evidence_frames=1,
            ),
            logger=None,
        )

        emitted = []
        for _ in range(4):
            result = analyzer.process(
                [
                    _detection("person", 0.90, (50, 40, 160, 240), track_id=10),
                    _detection("cigarette", 0.82, (78, 95, 88, 105), track_id=30),
                ]
            )
            emitted.extend(result.derived_events)

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].label, "smoking")
        self.assertEqual(emitted[0].track_id, 10)
        self.assertGreaterEqual(emitted[0].metadata["cigarette_evidence_frames"], 1)

    def test_does_not_emit_for_smoke_only_by_default(self) -> None:
        analyzer = SmokingBehaviorAnalyzer(
            SmokingBehaviorConfig(
                enabled=True,
                min_frames=3,
                min_evidence_frames=2,
                min_cigarette_evidence_frames=1,
            ),
            logger=None,
        )

        emitted = []
        for _ in range(5):
            result = analyzer.process(
                [
                    _detection("person", 0.92, (50, 40, 160, 240), track_id=11),
                    _detection("smoke", 0.90, (85, 70, 130, 115), track_id=31),
                ]
            )
            emitted.extend(result.derived_events)

        self.assertEqual(emitted, [])

    def test_ignores_far_cigarette_candidates(self) -> None:
        analyzer = SmokingBehaviorAnalyzer(
            SmokingBehaviorConfig(
                enabled=True,
                min_frames=2,
                min_evidence_frames=1,
                min_cigarette_evidence_frames=1,
            ),
            logger=None,
        )

        result = analyzer.process(
            [
                _detection("person", 0.90, (50, 40, 160, 240), track_id=12),
                _detection("cigarette", 0.95, (400, 400, 430, 430), track_id=32),
            ]
        )

        self.assertEqual(result.derived_events, [])
        self.assertFalse(result.detections[0].metadata["matched_cigarette"])


if __name__ == "__main__":
    unittest.main()
