"""Unit tests for the Docker-isolated OpenHands agent rollout.

Tests that _run_openhands_container correctly:
- Calls docker.containers.run with the right image, env vars, and volume mounts
- Cleans up the container on failure
- rollout() returns the expected trajectory dict shape
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure rllm repo root is on the path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.openhands.openhands_agent import (
    _run_openhands_container,
    rollout,
)


class TestRunOpenHandsContainer(unittest.TestCase):
    """Tests for _run_openhands_container with a mocked Docker client."""

    def _make_mock_client(self, logs: bytes = b"Agent finished. Task complete."):
        mock_client = MagicMock()
        mock_client.containers.run.return_value = logs
        return mock_client

    @patch("examples.openhands.openhands_agent._get_docker_client")
    def test_correct_image_used(self, mock_get_client):
        """Container must be started with the requested runtime image."""
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client

        _run_openhands_container(
            workspace="/tmp/ws",
            base_url="http://localhost:4000/v1",
            instruction="Fix the bug",
            runtime_image="test-oh-image:latest",
            model_name="test-model",
            max_iterations=5,
            timeout=60,
        )

        call_kwargs = mock_client.containers.run.call_args
        assert call_kwargs is not None, "containers.run was not called"
        assert call_kwargs.kwargs.get("image") == "test-oh-image:latest" or call_kwargs.args[0] == "test-oh-image:latest"

    @patch("examples.openhands.openhands_agent._get_docker_client")
    def test_env_vars_forwarded(self, mock_get_client):
        """LLM config env vars must be passed to the container."""
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client

        _run_openhands_container(
            workspace="/tmp/ws",
            base_url="http://proxy:4000/meta/slug/v1",
            instruction="test",
            runtime_image="oh:latest",
            model_name="my-model",
            max_iterations=10,
            timeout=60,
        )

        call_kwargs = mock_client.containers.run.call_args.kwargs
        env = call_kwargs.get("environment", {})
        self.assertEqual(env.get("LLM_BASE_URL"), "http://proxy:4000/meta/slug/v1")
        self.assertEqual(env.get("LLM_MODEL"), "my-model")
        self.assertEqual(env.get("MAX_ITERATIONS"), "10")

    @patch("examples.openhands.openhands_agent._get_docker_client")
    def test_workspace_volume_mounted(self, mock_get_client):
        """Workspace must be mounted at /workspace inside the container."""
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client

        import tempfile
        with tempfile.TemporaryDirectory() as ws:
            _run_openhands_container(
                workspace=ws,
                base_url="http://localhost:4000/v1",
                instruction="test",
                runtime_image="oh:latest",
                model_name="m",
                max_iterations=5,
                timeout=60,
            )

        volumes = mock_client.containers.run.call_args.kwargs.get("volumes", {})
        ws_real = os.path.realpath(ws)
        self.assertIn(ws_real, volumes)
        self.assertEqual(volumes[ws_real]["bind"], "/workspace")

    @patch("examples.openhands.openhands_agent._get_docker_client")
    def test_docker_socket_mounted(self, mock_get_client):
        """Docker socket must be mounted so OpenHands can spawn its own containers."""
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client

        _run_openhands_container(
            workspace="/tmp/ws",
            base_url="http://localhost:4000/v1",
            instruction="test",
            runtime_image="oh:latest",
            model_name="m",
            max_iterations=5,
            timeout=60,
        )

        volumes = mock_client.containers.run.call_args.kwargs.get("volumes", {})
        self.assertIn("/var/run/docker.sock", volumes)

    @patch("examples.openhands.openhands_agent._get_docker_client")
    def test_remove_true(self, mock_get_client):
        """Container must be auto-removed after exit."""
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client

        _run_openhands_container(
            workspace="/tmp/ws",
            base_url="http://localhost:4000/v1",
            instruction="test",
            runtime_image="oh:latest",
            model_name="m",
            max_iterations=5,
            timeout=60,
        )

        self.assertTrue(mock_client.containers.run.call_args.kwargs.get("remove"))

    @patch("examples.openhands.openhands_agent._get_docker_client")
    def test_force_remove_on_failure(self, mock_get_client):
        """On container failure, force-remove must be attempted."""
        mock_client = MagicMock()
        mock_client.containers.run.side_effect = RuntimeError("container died")
        # make containers.get return a mock container
        stale_container = MagicMock()
        mock_client.containers.get.return_value = stale_container
        mock_get_client.return_value = mock_client

        with self.assertRaises(RuntimeError):
            _run_openhands_container(
                workspace="/tmp/ws",
                base_url="http://localhost:4000/v1",
                instruction="test",
                runtime_image="oh:latest",
                model_name="m",
                max_iterations=5,
                timeout=60,
            )

        stale_container.remove.assert_called_once_with(force=True)


class TestRolloutShape(unittest.TestCase):
    """Tests for the rollout() function's return value shape."""

    @patch("examples.openhands.openhands_agent._run_openhands")
    @patch("examples.openhands.openhands_agent._setup_workspace")
    def test_rollout_returns_list_of_dicts(self, mock_setup, mock_run):
        """rollout() must return a list with one trajectory dict."""
        mock_setup.return_value = "/tmp/fake-ws"
        mock_run.return_value = "Task complete"

        result = rollout(
            {"instruction": "hello"},
            {"base_url": "http://localhost:4000/v1"},
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        traj = result[0]
        self.assertEqual(traj["name"], "openhands")
        self.assertIn("reward", traj)
        self.assertIn("steps", traj)

    @patch("examples.openhands.openhands_agent._run_openhands")
    @patch("examples.openhands.openhands_agent._setup_workspace")
    def test_rollout_reward_zero_on_exception(self, mock_setup, mock_run):
        """rollout() must return reward=0.0 when the container fails."""
        mock_setup.return_value = "/tmp/fake-ws"
        mock_run.side_effect = RuntimeError("kaboom")

        result = rollout(
            {"instruction": "hello"},
            {"base_url": "http://localhost:4000/v1"},
        )

        self.assertEqual(result[0]["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()