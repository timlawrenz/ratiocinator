"""Tests for HuggingFace data provisioning utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from ratiocinator.fleet.hf_data import (
    build_data_volume,
    build_output_volume,
    build_script_volume,
    upload_arm_script,
)


class TestBuildDataVolume:
    def test_dataset_source(self):
        vol = build_data_volume("hf-dataset", "user/my-dataset", "/data")
        assert vol == {
            "type": "dataset",
            "source": "user/my-dataset",
            "mount_path": "/data",
        }

    def test_bucket_source(self):
        vol = build_data_volume("hf-bucket", "user/my-bucket", "/data")
        assert vol == {
            "type": "bucket",
            "source": "user/my-bucket",
            "mount_path": "/data",
        }

    def test_unknown_source_defaults_to_dataset(self):
        vol = build_data_volume("something-else", "user/x", "/mnt")
        assert vol["type"] == "dataset"

    def test_custom_mount_path(self):
        vol = build_data_volume("hf-dataset", "user/data", "/workspace/data")
        assert vol["mount_path"] == "/workspace/data"


class TestBuildOutputVolume:
    def test_default_mount(self):
        vol = build_output_volume("user/artifacts")
        assert vol == {
            "type": "bucket",
            "source": "user/artifacts",
            "mount_path": "/output",
        }

    def test_custom_mount(self):
        vol = build_output_volume("user/out", mount_path="/results")
        assert vol["mount_path"] == "/results"


class TestBuildScriptVolume:
    def test_default_mount(self):
        vol = build_script_volume("user/scripts")
        assert vol == {
            "type": "bucket",
            "source": "user/scripts",
            "mount_path": "/input",
        }

    def test_custom_mount(self):
        vol = build_script_volume("user/scripts", mount_path="/scripts")
        assert vol["mount_path"] == "/scripts"


class TestUploadArmScript:
    async def test_upload_creates_unique_path(self):
        mock_client = AsyncMock()
        mock_client.upload_to_bucket = AsyncMock()

        remote = await upload_arm_script(
            client=mock_client,
            bucket_name="user/my-bucket",
            arm_name="baseline",
            script_content="#!/bin/bash\necho hello\n",
            experiment_name="my-experiment",
        )

        assert remote == "my-experiment/baseline/run.sh"
        mock_client.upload_to_bucket.assert_called_once()

        call_args = mock_client.upload_to_bucket.call_args
        assert call_args[0][0] == "user/my-bucket"
        # local_path is a temp file, just check it's a string
        assert isinstance(call_args[0][1], str)
        assert call_args[0][2] == "my-experiment/baseline/run.sh"

    async def test_parallel_arms_have_distinct_paths(self):
        mock_client = AsyncMock()

        path1 = await upload_arm_script(
            mock_client, "bucket", "arm-a", "script-a", "exp",
        )
        path2 = await upload_arm_script(
            mock_client, "bucket", "arm-b", "script-b", "exp",
        )

        assert path1 != path2
        assert "arm-a" in path1
        assert "arm-b" in path2

    async def test_temp_file_cleanup(self, tmp_path):
        """The temp file should be cleaned up after upload."""
        mock_client = AsyncMock()

        # After upload_arm_script returns, the temp file should be deleted
        await upload_arm_script(
            mock_client, "bucket", "arm-x", "#!/bin/bash\n", "exp",
        )

        # upload_to_bucket was called — check local_path was cleaned
        call_args = mock_client.upload_to_bucket.call_args
        local_path = call_args[0][1]
        import os
        assert not os.path.exists(local_path)


class TestVolumesFromDicts:
    def test_converts_to_volume_objects(self):
        """volumes_from_dicts requires huggingface_hub — test with mock."""
        volume_dicts = [
            {"type": "dataset", "source": "user/data", "mount_path": "/data"},
            {"type": "bucket", "source": "user/out", "mount_path": "/output"},
        ]

        mock_volume_cls = MagicMock()
        mock_volume_cls.side_effect = [
            MagicMock(type="dataset"),
            MagicMock(type="bucket"),
        ]

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(Volume=mock_volume_cls)}):
            # Re-import to pick up the mock
            import importlib

            import ratiocinator.fleet.hf_data as hf_data_mod
            importlib.reload(hf_data_mod)

            result = hf_data_mod.volumes_from_dicts(volume_dicts)
            assert len(result) == 2
            assert mock_volume_cls.call_count == 2
