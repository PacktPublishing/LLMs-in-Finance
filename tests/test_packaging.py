from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.package_release import ARCHIVE_ROOT, release_files, sha256, write_archive
from scripts.verify_archive import validate_members


class PackagingTests(unittest.TestCase):
    def test_release_archive_is_deterministic_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            archive_one, checksum_one = write_archive(Path(first))
            archive_two, checksum_two = write_archive(Path(second))
            self.assertEqual(sha256(archive_one), sha256(archive_two))
            self.assertEqual(checksum_one.read_text(), checksum_two.read_text())
            with ZipFile(archive_one) as archive:
                self.assertEqual(validate_members(archive), ARCHIVE_ROOT)
                names = archive.namelist()
            self.assertTrue(names)
            self.assertTrue(all(name.startswith(f"{ARCHIVE_ROOT}/") for name in names))
            self.assertFalse(any(name.endswith(".zip") for name in names))
            release_names = [path.relative_to(Path.cwd()).as_posix() for path in release_files()]
            self.assertFalse(any("/build/" in f"/{name}/" for name in release_names))
            self.assertFalse(any(".egg-info/" in f"{name}/" for name in release_names))


if __name__ == "__main__":
    unittest.main()
