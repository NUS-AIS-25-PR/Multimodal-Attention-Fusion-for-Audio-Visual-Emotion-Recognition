"""Read-only archive contracts; corruption tests modify disposable copies only."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_archive import sha256, verify


class ArchiveIntegrityTests(unittest.TestCase):
    archive = Path(__file__).resolve().parent

    def test_published_snapshot_is_complete_without_local_checkpoints(self):
        summary = verify(self.archive)
        self.assertEqual(summary["completed_runs"], 30)
        self.assertEqual(summary["figure_exports"], 198)
        self.assertEqual(summary["held_out_samples_per_method"], 1440)
        self.assertEqual(summary["local_checkpoint_hashes_checked"], 0)

    def test_corrupted_history_rejects_file_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "archive"
            shutil.copytree(self.archive, copy, ignore=shutil.ignore_patterns("__pycache__"))
            path = copy / "outputs/speaker_independent_tracked_v1/fold_01/audio/history.csv"
            with path.open("a") as stream:
                stream.write("corrupted history\n")
            with self.assertRaisesRegex(AssertionError, "Size mismatch"):
                verify(copy)

    def test_changed_predictions_reject_even_if_file_hash_is_refreshed(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "archive"
            shutil.copytree(self.archive, copy, ignore=shutil.ignore_patterns("__pycache__"))
            relative = "outputs/speaker_independent_tracked_v1/fold_01/audio/test_predictions.csv"
            path = copy / relative
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            rows[0]["prediction"] = str((int(rows[0]["prediction"]) + 1) % 8)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            inventory_path = copy / "artifact_inventory.json"
            inventory = json.loads(inventory_path.read_text())
            record = next(r for r in inventory["files"] if r["archive"] == relative)
            record.update(bytes=path.stat().st_size, sha256=sha256(path))
            inventory_path.write_text(json.dumps(inventory))
            with self.assertRaisesRegex(AssertionError, "Confusion matrix disagrees"):
                verify(copy)


if __name__ == "__main__":
    unittest.main()
