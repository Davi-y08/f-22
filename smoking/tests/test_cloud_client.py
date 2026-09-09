from __future__ import annotations

import unittest
from unittest.mock import patch

from cloud.client import CloudClient
from discovery.service import DiscoveredCamera
from utils.config import CloudConfig


class CloudClientTests(unittest.TestCase):
    def test_sync_discovered_camera_uses_agent_key_header(self) -> None:
        client = CloudClient(
            CloudConfig(
                enabled=True,
                api_base_url="https://api.example.test/",
                agent_access_key="agent-secret",
            ),
            agent_id="agent-1",
        )

        with patch.object(client, "_post_json", return_value={"created": 1, "updated": 0}) as post_json:
            result = client.sync_discovered_cameras(
                [
                    DiscoveredCamera(
                        key="network:192.168.1.20",
                        kind="network",
                        name="Entrada",
                        host="192.168.1.20",
                        rtsp_ports=[554],
                        rtsp_stream_paths=["live"],
                        status="rtsp_stream_detected",
                    )
                ]
            )

        self.assertTrue(result.success)
        post_json.assert_called_once()
        path, payload = post_json.call_args.args
        self.assertEqual(path, "/agent/cameras/sync")
        self.assertEqual(payload["agent_id"], "agent-1")
        self.assertEqual(payload["cameras"][0]["url"], "rtsp://192.168.1.20:554/live")
        self.assertEqual(payload["cameras"][0]["status"], "online")

    def test_configured_local_camera_is_sent_as_local_url(self) -> None:
        client = CloudClient(
            CloudConfig(
                enabled=True,
                api_base_url="https://api.example.test",
                agent_access_key="agent-secret",
            ),
            agent_id="agent-1",
        )

        with patch.object(client, "_post_json", return_value={"created": 0, "updated": 1}) as post_json:
            result = client.sync_configured_camera(
                external_id="webcam-local-0",
                name="Webcam",
                source=0,
                status="online",
            )

        self.assertTrue(result.success)
        payload = post_json.call_args.args[1]
        self.assertEqual(payload["cameras"][0]["url"], "local://0")
        self.assertEqual(payload["cameras"][0]["source"], "local://0")

    def test_missing_key_disables_client_without_network_call(self) -> None:
        client = CloudClient(
            CloudConfig(enabled=True, api_base_url="https://api.example.test", agent_access_key=""),
            agent_id="agent-1",
        )

        with patch.object(client, "_post_json") as post_json:
            result = client.sync_discovered_cameras([])

        self.assertFalse(result.success)
        post_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
