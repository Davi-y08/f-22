from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.config import load_config
from utils.redaction import redact_url_credentials


class RedactionTests(unittest.TestCase):
    def test_redacts_rtsp_credentials(self) -> None:
        source = "rtsp://usuario:senha@192.168.1.10:554/stream"

        self.assertEqual(
            redact_url_credentials(source),
            "rtsp://***:***@192.168.1.10:554/stream",
        )

    def test_keeps_public_urls_unchanged(self) -> None:
        source = "http://192.168.1.244:8080/video"

        self.assertEqual(redact_url_credentials(source), source)

    def test_handles_invalid_port_without_crashing(self) -> None:
        source = "rtsp://user:pass@example.local:invalid/stream"

        self.assertEqual(
            redact_url_credentials(source),
            "rtsp://***:***@example.local/stream",
        )


class ConfigLoadingTests(unittest.TestCase):
    def test_display_target_fps_is_clamped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "agent_id": "test-agent",
                        "model_catalog": {
                            "smoking_monitor": {
                                "path": "models/smoking_monitor.onnx",
                                "backend": "onnx",
                                "class_names": ["cigarette", "person", "smoke"],
                            }
                        },
                        "cameras": [
                            {
                                "id": "local-0",
                                "name": "Local",
                                "source": 0,
                                "models": ["smoking_monitor"],
                                "display": {"enabled": True, "target_fps": 120},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.cameras[0].display.target_fps, 60.0)


if __name__ == "__main__":
    unittest.main()
