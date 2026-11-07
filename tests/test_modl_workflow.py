"""Local workflow contracts: coordinates, molecule counts, grouping and split isolation."""

from contextlib import redirect_stdout
import gzip
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("modl_workflow_cli", ROOT / "scripts/modl.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class LocalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="modl-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fasta = self.root / "reference.fa"
        self.fasta.write_text("".join(f">chr{i}\n" + "ACGTN" * 40 + "\n" for i in range(1, 4)))
        self.peaks = self.root / "peaks.bed"
        self.peaks.write_text("".join(f"chr{i}\t90\t110\n" for i in range(1, 4)))
        self.fragments = self.root / "fragments.tsv.gz"
        with gzip.open(self.fragments, "wt") as handle:
            handle.write("# adjusted, deduplicated fragments\n")
            for i in range(1, 4):
                handle.write(f"chr{i}\t90\t110\tA-1\t9\n")
                handle.write(f"chr{i}\t100\t120\tB-1\t2\n")
                handle.write(f"chr{i}\t130\t140\tOTHER-1\t1\n")
        self.cells = self.root / "cells.tsv"
        self.cells.write_text("barcode\tgroup\nA-1\ttumor\nB-1\tnormal\n")
        self.out = self.root / "prepared"

    def invoke(self, *args):
        stream = StringIO()
        with patch("sys.argv", ["modl.py", *map(str, args)]), redirect_stdout(stream):
            status = cli.main()
        return status, json.loads(stream.getvalue())

    def prepare(self, *extra):
        return self.invoke("prepare", "--fasta", self.fasta, "--peaks", self.peaks,
                           "--fragments", self.fragments, "--barcode-groups", self.cells,
                           "--assembly", "mm10", "--context-bp", "64", "--chunk-bp", "32",
                           "--out", self.out, *extra)

    def test_fragments_keep_group_coordinates_and_ignore_pcr_support(self):
        status, result = self.prepare()
        self.assertEqual(status, 0, result)
        signal = zarr.open_group(str(self.out / "atac.zarr"), mode="r")
        self.assertEqual(signal["tumor"].attrs["library_size"], 6)
        self.assertEqual(signal["normal"].attrs["library_size"], 6)
        expected = np.zeros(200)
        expected[[90, 110]] = 1
        np.testing.assert_array_equal(signal["tumor/chrs/chr1"][:], expected)
        manifest = json.loads((self.out / "prepared.json").read_text())
        self.assertEqual(manifest["signal"]["unselected_barcodes"], 3)
        self.assertEqual(len(manifest["inputs"]["fragments"]["sha256"]), 64)
        genome = zarr.open_group(str(self.out / "genome.zarr"), mode="r")
        np.testing.assert_array_equal(genome["chrs/chr1"][:5], np.vstack([np.eye(4), np.zeros(4)]))

    def test_pooling_requires_explicit_selection(self):
        status, result = self.invoke("prepare", "--fasta", self.fasta, "--peaks", self.peaks,
                          "--fragments", self.fragments, "--assembly", "mm10", "--context-bp", "64", "--out", self.out)
        self.assertEqual(status, 2)
        self.assertIn("--pool-all", result["message"])
        self.assertFalse(self.out.exists())

    def test_refuse_existing_output(self):
        self.out.mkdir()
        marker = self.out / "keep.txt"
        marker.write_text("original")
        status, _ = self.prepare()
        self.assertEqual(status, 2)
        self.assertEqual(list(self.out.iterdir()), [marker])

    def test_coordinate_mismatch_fails_before_writing(self):
        self.peaks.write_text("chrMissing\t90\t110\n")
        status, result = self.prepare()
        self.assertEqual(status, 2)
        self.assertIn("FASTA", result["message"])
        self.assertFalse(self.out.exists())

    def test_peak_dedup_boundary_and_full_context_blacklist(self):
        self.peaks.write_text("chr1\t90\t110\nchr1\t90\t110\nchr1\t0\t2\nchr2\t90\t110\nchr3\t90\t110\n")
        blacklist = self.root / "blacklist.bed"
        blacklist.write_text("chr2\t70\t75\n")  # Context overlaps; original peak does not.
        status, result = self.prepare("--blacklist", blacklist)
        self.assertEqual(status, 0, result)
        self.assertEqual(result["peaks"], {"input": 5, "retained": 2, "duplicates_removed": 1,
                                           "boundary_excluded": 1, "blacklist_excluded": 1})

    def test_split_isolation_and_empty_split_failure(self):
        self.assertEqual(self.prepare()[0], 0)
        base = ("validate", "--data", self.out, "--track-id", "tumor", "--val-chr", "chr2")
        status, result = self.invoke(*base, "--test-chr", "chr3")
        self.assertEqual(status, 0, result)
        self.assertEqual(result["splits"]["counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(self.invoke(*base, "--test-chr", "chr2")[0], 2)
        self.assertEqual(self.invoke(*base, "--test-chr", "chrAbsent")[0], 2)

    def test_dry_run_is_local_explicit_and_creates_no_run_directory(self):
        self.assertEqual(self.prepare()[0], 0)
        run = self.root / "train"
        status, result = self.invoke("train", "--data", self.out, "--track-id", "tumor",
                                   "--val-chr", "chr2", "--test-chr", "chr3", "--out", run, "--dry-run")
        self.assertEqual(status, 0, result)
        self.assertFalse(run.exists())
        args = result["command"]
        self.assertEqual(args[args.index("--split-strategy") + 1], "chromosome")
        self.assertEqual(args[args.index("--wandb-mode") + 1], "disabled")
        self.assertEqual(args[args.index("--max-regions") + 1], "3")
        self.assertIn("--no-compile", args)

    def test_fragment_chromosome_end_is_reported_without_moving_the_cut(self):
        with gzip.open(self.fragments, "wt") as handle:
            handle.write("chr1\t190\t200\tA-1\t1\nchr1\t100\t120\tB-1\t1\n")
        status, result = self.prepare()
        self.assertEqual(status, 0, result)
        manifest = json.loads((self.out / "prepared.json").read_text())
        self.assertEqual(manifest["signal"]["out_of_bounds_endpoints"], 1)
        signal = zarr.open_group(str(self.out / "atac.zarr"), mode="r")
        self.assertEqual(signal["tumor/chrs/chr1"][190], 1)
        self.assertEqual(signal["tumor/chrs/chr1"][199], 0)

    def test_failed_fragment_conversion_records_failure_without_a_success_manifest(self):
        with gzip.open(self.fragments, "wt") as handle:
            handle.write("chrWrong\t90\t110\tA-1\t1\n")
        status, _ = self.prepare()
        self.assertEqual(status, 2)
        self.assertTrue((self.out / "prepare.failure.json").exists())
        self.assertFalse((self.out / "prepared.json").exists())

    def test_bigwig_conversion_preserves_signal_and_missing_chromosomes(self):
        import pyBigWig
        bw_path = self.root / "signal.bw"
        with pyBigWig.open(str(bw_path), "w") as handle:
            handle.addHeader([("chr1", 200)])
            handle.addEntries(["chr1", "chr1"], [90, 110], ends=[91, 111], values=[3.0, 2.0])
        status, result = self.invoke("prepare", "--fasta", self.fasta, "--peaks", self.peaks,
                          "--bigwig", bw_path, "--track-id", "bulk", "--assembly", "mm10",
                          "--context-bp", "64", "--chunk-bp", "32", "--out", self.out)
        self.assertEqual(status, 0, result)
        root = zarr.open_group(str(self.out / "atac.zarr"), mode="r")
        self.assertEqual(root["bulk"].attrs["library_size"], 5)
        self.assertEqual(root["bulk/chrs/chr1"][90], 3)
        self.assertEqual(root["bulk/chrs/chr2"][:].sum(), 0)

    def test_fragment_conversion_remains_correct_after_cache_eviction(self):
        # Three chromosomes x two tracks x seven chunks exceed the 64-chunk
        # buffer. Revisit early chunks to exercise read-modify-write after flush.
        with gzip.open(self.fragments, "wt") as handle:
            for repeat in range(2):
                for group in ["A-1", "B-1"]:
                    for i in range(1, 4):
                        for start in range(0, 180, 16):
                            handle.write(f"chr{i}\t{start}\t{start + 5}\t{group}\t1\n")
        status, result = self.prepare("--chunk-bp", "8")
        self.assertEqual(status, 0, result)
        root = zarr.open_group(str(self.out / "atac.zarr"), mode="r")
        self.assertEqual(root["tumor/chrs/chr1"][0], 2)
        self.assertEqual(root["normal"].attrs["library_size"], 144)

    def test_interpret_exports_the_requested_coordinates(self):
        from tests.test_model_studio_core import FakeProfileAdapter
        fake = FakeProfileAdapter()
        fake.model = object()
        self.assertEqual(self.prepare()[0], 0)
        checkpoint = self.root / "trusted_test.ckpt"
        checkpoint.write_bytes(b"test fixture; adapter is replaced by a deterministic model")
        output = self.root / "interpret"
        with patch("model_studio.adapters.cd200.CD200MotifCNNAdapter", return_value=fake):
            status, result = self.invoke("interpret", "--data", self.out, "--checkpoint", checkpoint,
                    "--region", "chr1:90-110", "--deletion-size", "3", "--out", output, "--device", "cpu")
        self.assertEqual(status, 0, result)
        payload = json.loads((output / "result.json").read_text())
        self.assertEqual(payload["nucleotide_positions"], list(range(90, 110)))
        self.assertEqual(payload["sequence"]["length"], 64)
        self.assertEqual(len((output / "deletions.csv").read_text().splitlines()), 21)


if __name__ == "__main__":
    unittest.main()
