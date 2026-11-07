import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numcodecs
import numpy as np


SPEC = importlib.util.spec_from_file_location("mouse_cache", Path(__file__).with_name("prepare_mouse_cache.py"))
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def array_fixture(root, chrom, values, width=8):
    target = root / chrom
    target.mkdir(parents=True)
    columns = 1 if values.ndim == 1 else values.shape[1]
    chunks = [width] if columns == 1 else [width, columns]
    (target / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "dtype": values.dtype.str, "shape": list(values.shape), "chunks": chunks,
        "order": "C", "fill_value": 0, "filters": None, "compressor": {"id": "blosc"}}))
    for index in range((len(values) + width - 1) // width):
        piece = np.zeros(chunks, dtype=values.dtype)
        chunk = values[index * width:(index + 1) * width]
        piece[:len(chunk)] = chunk
        name = str(index) if columns == 1 else f"{index}.0"
        (target / name).write_bytes(numcodecs.Blosc().encode(piece))
    return target


class MouseCacheChecks(unittest.TestCase):
    def test_cross_chunk_signal_and_audit_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = np.arange(23, dtype=np.uint16)
            target = array_fixture(root, "chr1", values)
            reader = builder.DenseZarrReader(root, "<u2", max_cached_chunks=1)
            np.testing.assert_array_equal(reader.read("chr1", 6, 20), values[6:20])
            self.assertLessEqual(len(reader.cache), 1)
            self.assertEqual(reader.sources[str(target / "0")]["sha256"], builder.digest(target / "0"))
            self.assertIn(str(target / ".zarray"), reader.sources)
            with self.assertRaises(ValueError):
                reader.read("chr1", 0, 24)

    def test_declared_missing_chunk_fill_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = array_fixture(root, "chr1", np.arange(16, dtype=np.uint16))
            (target / "1").unlink()
            reader = builder.DenseZarrReader(root, "<u2")
            np.testing.assert_array_equal(reader.read("chr1", 8, 16), np.zeros(8, np.uint16))
            self.assertEqual(reader.sources[str(target / "1")]["kind"], "absent_chunk_declared_fill")

    def test_genome_onehot_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = np.eye(4, dtype=np.uint8)[np.arange(20) % 4]
            array_fixture(root, "chr16", values)
            reader = builder.DenseZarrReader(root, "u1", columns=4)
            np.testing.assert_array_equal(reader.read("chr16", 5, 13), values[5:13])

    def test_frozen_partitions_categories_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "peaks.tsv"
            header = "Chromosome\tStart\tEnd\tpeak_type\n"
            source.write_text(header + "chr1\t2000\t4000\tfor_profile\nchr14\t2000\t4000\tfor_count\n"
                              "chr16\t2000\t4000\tbackground\nchr8\t2000\t4000\tfor_profile\nchrX\t2000\t4000\tbackground\n")
            rows, excluded = builder.source_regions(source)
            self.assertEqual(excluded, {"chrX": 1})
            self.assertEqual({r["chrom"]: r["split"] for r in rows}, {"chr1": "train", "chr8": "test", "chr14": "valid", "chr16": "test"})
            self.assertEqual(next(r["is_peak"] for r in rows if r["chrom"] == "chr16"), 0)
            with source.open("a") as f:
                f.write("chr1\t2000\t4000\tbackground\n")
            with self.assertRaisesRegex(ValueError, "Repeated input center"):
                builder.source_regions(source)

    def test_frozen_N_mask_bounds_and_zero_target_retention(self):
        codes = np.zeros(3114, np.uint8)
        codes[:155] = 4
        self.assertIsNone(builder.eligibility("chr1", 3000, 8000, codes, .05))
        codes[155] = 4
        self.assertIsNotNone(builder.eligibility("chr1", 3000, 8000, codes, .05))
        self.assertIsNotNone(builder.eligibility("chr1", 1000, 8000, None, .05))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            array_fixture(root / "signal", "chr1", np.zeros(8000, np.uint16), width=1000)
            signal = builder.DenseZarrReader(root / "signal", "<u2")
            output = root / "out"
            output.mkdir()
            rejects = output / "excluded.jsonl"
            fasta = {"chr1": "A" * 6000 + "N" * 160 + "A" * 1840}
            rows = [{"region_id": "keep_zero", "chrom": "chr1", "center": 2500, "is_peak": 1},
                    {"region_id": "exclude_N", "chrom": "chr1", "center": 6000, "is_peak": 0}]
            report = builder.cache_one_split("test", rows, fasta, signal, output, .05, rejects)
            self.assertEqual(report["regions"], 1)
            self.assertEqual(report["excluded"], 1)
            y = np.load(output / "test.y.npy")
            self.assertEqual(y.shape, (1, 2000))
            self.assertEqual(float(y.sum()), 0)
            self.assertEqual(np.load(output / "test.x.npy").shape, (1, 3114))
            with (output / "test.regions.tsv").open() as f:
                self.assertEqual(next(csv.DictReader(f, delimiter="\t"))["region_id"], "keep_zero")


if __name__ == "__main__":
    unittest.main()
