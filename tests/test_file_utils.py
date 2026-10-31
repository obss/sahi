"""Tests for file utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


class TestFileUtils:
    """Test file utility functions."""

    def test_list_files(self) -> None:
        """Test listing files with filters."""
        from sahi.utils.file import list_files

        directory = "tests/data/coco_utils/"
        filepath_list = list_files(directory, contains=["json"], verbose=False)
        assert len(filepath_list) == 11

    def test_list_files_recursively(self, tmp_path: Path) -> None:
        """Test recursive file listing."""
        import shutil
        from pathlib import Path

        from sahi.utils.file import list_files_recursively

        # Copy only the target json files into an isolated temp directory to avoid race
        # with other tests that may create additional *coco.json files recursively.
        src_dir = Path("tests/data/coco_utils/")
        for f in src_dir.iterdir():
            if f.is_file() and "coco.json" in f.name:
                shutil.copy(f, tmp_path / f.name)

        relative_filepath_list, abs_filepath_list = list_files_recursively(
            str(tmp_path), contains=["coco.json"], verbose=False
        )
        assert len(relative_filepath_list) == 7
        assert len(abs_filepath_list) == 7

    def test_increment_path(self) -> None:
        """Test path incrementing for duplicate filenames."""
        from sahi.utils.file import increment_path

        with patch("sahi.utils.file.Path.exists", return_value=False):
            path = increment_path("test.txt")
            assert path == "test.txt"
        with patch("sahi.utils.file.Path.exists", return_value=True):
            path = increment_path("test.txt", exist_ok=False)
            assert path == "test.txt2"

    def test_download_from_url(self, tmp_path: Path) -> None:
        """A finished download lands at the target path with no leftover part file."""
        from sahi.utils.file import download_from_url

        source = tmp_path / "source.bin"
        source.write_bytes(b"weights")
        target = tmp_path / "models" / "model.pt"

        download_from_url(source.as_uri(), str(target))

        assert target.read_bytes() == b"weights"
        assert list(target.parent.iterdir()) == [target]

    def test_download_from_url_interrupted(self, tmp_path: Path) -> None:
        """An interrupted download leaves nothing at the target path, so the next call downloads again."""
        import urllib.request

        from sahi.utils.file import download_from_url

        source = tmp_path / "source.bin"
        source.write_bytes(b"weights")
        target = tmp_path / "models" / "model.pt"

        def fail_midway(url: str, filename: str) -> None:
            Path(filename).write_bytes(b"wei")
            raise ConnectionError("connection dropped")

        with patch.object(urllib.request, "urlretrieve", side_effect=fail_midway):
            try:
                download_from_url(source.as_uri(), str(target))
            except ConnectionError:
                pass

        assert list(target.parent.iterdir()) == []

        download_from_url(source.as_uri(), str(target))
        assert target.read_bytes() == b"weights"
