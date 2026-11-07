"""Focused checks for split manifests; use python -m unittest discover here."""
import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("prepare_splits", Path(__file__).with_name("prepare_splits.py"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class SplitTests(unittest.TestCase):
    def test_headers_duplicates_partitions_overlap_blocks_and_reproducible_gzip(self):
        for header in ("", "Chromosome\tStart\tEnd\tvalue\n"):
            with self.subTest(header=bool(header)), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "fixture.bed"
                source.write_text(header + "chr1\t3000\t5000\t1\nchr1\t3000\t5000\t2\n"
                                  "chr1\t4000\t6000\t3\nchr14\t3000\t5000\t4\n"
                                  "chr16\t3000\t5000\t5\nchr8\t3000\t5000\t6\n"
                                  "chr9\t3000\t5000\t7\nchrX\t3000\t5000\t8\n")
                out1, out2 = root / "one.gz", root / "two.gz"
                args = ({"chr14"}, {"chr16", "chr8", "chr9"}, 2048)
                audit = module.prepare(source, out1, *args)
                module.prepare(source, out2, *args)
                self.assertEqual(out1.read_bytes(), out2.read_bytes())
                self.assertEqual(audit["split_unique_window_counts"], {"test": 3, "train": 2, "val": 1})
                self.assertEqual(audit["collapsed_repeated_input_windows"], 1)
                self.assertEqual(audit["excluded_non_autosomal_rows"], {"chrX": 1})
                self.assertEqual(audit["split_overlap_block_counts"]["train"], 1)
                self.assertEqual(audit["cross_split_input_window_overlap_count"], 0)
                with gzip.open(out1, "rt") as handle:
                    self.assertEqual(handle.readline().strip().split("\t"), ["chrom", "input_start", "input_end", "split", "overlap_block", "source_lines"])

    def test_missing_holdout_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.bed"
            output = Path(tmp) / "out.gz"
            path.write_text("chr1\t3000\t5000\n")
            with self.assertRaisesRegex(ValueError, "holdout chromosomes absent"):
                module.prepare(path, output, {"chr14"}, {"chr16"}, 2048)
            self.assertFalse(output.exists())

    def test_input_window_boundary_and_invalid_interval_rejected(self):
        for row in ("chr1\t0\t10\n", "chr1\t5000\t4000\n"):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "source.bed"
                path.write_text(row)
                with self.assertRaises(ValueError):
                    module.load_windows(path, 2048)

    def test_transitive_overlap_and_half_open_boundary(self):
        windows = [("chr1", 0, 10), ("chr1", 9, 19), ("chr1", 18, 28), ("chr1", 28, 38), ("chr2", 0, 10)]
        result = module.component_ids(windows)
        self.assertEqual(result[windows[0]], result[windows[2]])
        self.assertNotEqual(result[windows[2]], result[windows[3]])
        self.assertNotEqual(result[windows[0]], result[windows[4]])

    def test_existing_artifact_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp) / "source.bed", Path(tmp) / "out.gz"
            path.write_text("chr14\t3000\t5000\nchr16\t3000\t5000\n")
            output.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                module.prepare(path, output, {"chr14"}, {"chr16"}, 2048)
            self.assertEqual(output.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
