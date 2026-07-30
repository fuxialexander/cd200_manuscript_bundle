from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from numcodecs import get_codec

from model_studio.adapters.base import SequenceModelAdapter
from model_studio.adapters.registry import ModelAdapterRegistry
from model_studio.annotations import MotifAnnotationCatalog
from model_studio.devices import DeviceManager
from model_studio.domain import (
    ArchitectureDescriptor,
    DeletionSettings,
    ModelMetadata,
    SequenceRecord,
)
from model_studio.perturbations import (
    DeletionAnalyzer,
    manuscript_plot_effects,
)
from model_studio.service import ModelStudioService
from model_studio.sequences import (
    _read_local_zarr_v2_1d_slice,
    _read_local_zarr_v2_slice,
    normalize_sequence,
    one_hot_encode,
)


class FakeProfileAdapter(SequenceModelAdapter):
    descriptor = ArchitectureDescriptor(
        id="fake",
        label="Fake profile",
        description="Test adapter",
    )

    def __init__(self) -> None:
        super().__init__("fake.ckpt", "cpu")
        self.batch_lengths: list[int] = []
        self._metadata = ModelMetadata(
            architecture_id="fake",
            architecture_label="Fake profile",
            checkpoint="fake.ckpt",
            device="cpu",
            output_heads=("regression",),
        )

    def load(self) -> ModelMetadata:
        return self.metadata

    def predict(self, sequences: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
        self.batch_lengths.append(len(sequences))
        return {"regression": sequences[:, :, 0]}


class ModelStudioCoreTests(unittest.TestCase):
    def test_sequence_normalization_and_one_hot(self) -> None:
        sequence = normalize_sequence(
            ">record\nacgtnacacgtnacacgtnacacgtnacacgtnac\n"
        )
        self.assertEqual(sequence, "ACGTNACACGTNACACGTNACACGTNACACGTNAC")
        encoded = one_hot_encode(sequence)
        self.assertEqual(encoded.shape, (35, 4))
        self.assertEqual(encoded[0].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(encoded[4].tolist(), [0.0, 0.0, 0.0, 0.0])

    def test_batched_real_deletion_effects_are_aligned(self) -> None:
        adapter = FakeProfileAdapter()
        record = SequenceRecord("A" + "C" * 63, "synthetic")
        result = DeletionAnalyzer(adapter).run(
            record,
            DeletionSettings(
                deletion_size=1,
                stride=1,
                batch_size=7,
                perturbation="real",
                output_head="regression",
            ),
        )
        self.assertEqual(result.baseline["regression"][0], 1.0)
        self.assertEqual(len(result.window_positions), 64)
        self.assertEqual(result.window_effects[0], 1.0)
        self.assertTrue(all(effect == 0.0 for effect in result.window_effects[1:]))
        self.assertEqual(result.nucleotide_effects[0], 1.0)
        self.assertEqual(result.plot_effects[0], 100.0)
        self.assertTrue(result.plot_effect["applied"])
        self.assertLessEqual(max(adapter.batch_lengths[1:]), 7)

    def test_display_only_scan_matches_full_scan_inside_display(self) -> None:
        record = SequenceRecord(
            "ACGT" * 16,
            "chr1:100-164",
            chromosome="chr1",
            start=100,
            end=164,
        )
        common = dict(
            deletion_size=5,
            stride=1,
            batch_size=8,
            perturbation="real",
            output_head="regression",
        )
        full = DeletionAnalyzer(FakeProfileAdapter()).run(
            record,
            DeletionSettings(**common, scan_scope="context"),
            display_region=(120, 140),
        )
        displayed = DeletionAnalyzer(FakeProfileAdapter()).run(
            record,
            DeletionSettings(**common, scan_scope="display"),
            display_region=(120, 140),
        )
        expected = np.asarray(full.plot_effects)[20:40]
        np.testing.assert_allclose(displayed.plot_effects, expected)
        self.assertEqual(
            displayed.nucleotide_positions,
            list(range(120, 140)),
        )
        self.assertLess(
            displayed.settings["scanned_window_count"],
            displayed.settings["full_context_window_count"],
        )

    def test_manuscript_plot_effect_transformation(self) -> None:
        transformed, metadata = manuscript_plot_effects(
            np.asarray([-2.0, 1.0, 0.0]),
            np.asarray([-4.0, 2.0]),
        )
        np.testing.assert_allclose(transformed, [50.0, 25.0, 0.0])
        self.assertEqual(metadata["denominator"], 4.0)
        self.assertEqual(metadata["unit"], "%")
        self.assertEqual(
            metadata["source"],
            "analysis/plot_scan_results.py::prepare_plot_effects",
        )

    def test_differential_sign_and_regression_core_scoring(self) -> None:
        differential = DeletionAnalyzer._score(
            "differential",
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([[3.5], [-2.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(differential, [2.5, -3.0])
        regression = DeletionAnalyzer._score(
            "regression",
            np.asarray([[1.0, 10.0, 3.0, 2.0]], dtype=np.float32),
            np.asarray(
                [
                    [1.0, 8.0, 5.0, 20.0],
                    [1.0, 12.0, 4.0, 20.0],
                ],
                dtype=np.float32,
            ),
            core_slice=(1, 3),
        )
        np.testing.assert_allclose(regression, [2.0, -2.0])

    def test_manuscript_context_and_display_geometry(self) -> None:
        context = ModelStudioService._centered_interval(
            45408000,
            45409426,
            4096,
        )
        display = ModelStudioService._centered_interval(
            *context,
            2000,
            inclusive_center=True,
        )
        self.assertEqual(context, (45406665, 45410761))
        self.assertEqual(display, (45407712, 45409712))

    def test_preset_sequence_keeps_browser_selected_region(self) -> None:
        payload = {
            "source": {
                "kind": "region",
                "region": "chr16:45408400-45409000",
                "genome_zarr": "/untrusted/genome.zarr",
            },
            "display_bp": 600,
        }
        preset = {
            "source": {
                "kind": "region",
                "region": "chr16:45408000-45409426",
                "genome_zarr": "/trusted/mm10.zarr",
            },
            "context_bp": 4096,
            "display_bp": 2000,
        }
        merged = ModelStudioService._apply_preset_sequence(payload, preset)
        self.assertEqual(
            merged["source"]["region"],
            "chr16:45408400-45409000",
        )
        self.assertEqual(
            merged["source"]["genome_zarr"],
            "/trusted/mm10.zarr",
        )
        self.assertEqual(merged["context_bp"], 4096)
        self.assertEqual(merged["display_bp"], 600)

    def test_nearest_smoothing_matches_even_window_reference(self) -> None:
        observed = ModelStudioService._smooth_nearest(
            np.arange(8, dtype=np.float32),
            4,
        )
        np.testing.assert_allclose(
            observed,
            [0.75, 1.5, 2.5, 3.5, 4.5, 5.5, 6.25, 6.75],
        )

    def test_motif_catalog_rna_and_direct_tf_filters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motif_path = root / "motifs.tsv"
            motif_path.write_text(
                "chromosome\tstart\tend\tgene_name\tmotif_name\tp_value\tscore\n"
                "chr1\t110\t120\tTF1\tTF1.a\t0.01\t4\n"
                "chr1\t118\t130\tTF1\tTF1.b\t0.001\t5\n"
                "chr1\t140\t150\tTF2\tTF2.a\t0.02\t3\n"
                "chr2\t110\t120\tTF3\tTF3.a\t0.03\t2\n"
            )
            rna_path = root / "rna.csv"
            rna_path.write_text(
                "gene,baseMean,log2FoldChange\n"
                "TF1,100,3\n"
                "TF2,5,-4\n"
            )
            catalog = MotifAnnotationCatalog(
                motif_path,
                chromosome="chr1",
                start=100,
                end=200,
                rna_path=rna_path,
            )
            summary = catalog.summary()
            self.assertEqual(summary["raw_motif_count"], 3)
            self.assertEqual(summary["merged_motif_count"], 2)
            self.assertEqual(summary["rna_columns"]["expression"], "baseMean")
            plot_effects = [0.0] * 100
            plot_effects[15] = 20.0
            plot_effects[45] = 30.0
            result = {
                "nucleotide_positions": list(range(100, 200)),
                "plot_effects": plot_effects,
            }
            rna_filtered = catalog.filter(
                result,
                mode="rna",
                min_abs_logfc=2.0,
                min_expression=10.0,
                min_effect_pct=10.0,
            )
            self.assertEqual(
                [motif["tf"] for motif in rna_filtered["motifs"]],
                ["TF1"],
            )
            self.assertEqual(rna_filtered["motifs"][0]["hit_count"], 2)
            direct = catalog.filter(
                result,
                mode="tf",
                selected_tf="TF2",
                min_effect_pct=10.0,
            )
            self.assertEqual(
                [motif["tf"] for motif in direct["motifs"]],
                ["TF2"],
            )

    def test_gene_filter_modes_remain_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motif_path = root / "motifs.tsv"
            motif_path.write_text(
                "chromosome\tstart\tend\tgene_name\tmotif_name\n"
                "chr1\t110\t120\tTF1\tTF1.a\n"
                "chr1\t130\t140\tTF2\tTF2.a\n"
            )
            first = root / "first.csv"
            first.write_text(
                "gene,baseMean,log2FoldChange,padj\n"
                "TF1,100,1.2,0.01\n"
                "TF2,100,-1.2,0.01\n"
            )
            second = root / "second.csv"
            second.write_text(
                "gene,baseMean,log2FoldChange,padj\n"
                "TF1,100,0.8,0.20\n"
                "TF2,100,1.5,0.01\n"
            )
            genes = root / "genes.csv"
            genes.write_text("gene\nTF2\n")
            catalog = MotifAnnotationCatalog(
                motif_path,
                chromosome="chr1",
                start=100,
                end=160,
                rna_path=first,
                rna_path_2=second,
                gene_list_path=genes,
            )
            result = {
                "nucleotide_positions": list(range(100, 160)),
                "plot_effects": [20.0] * 60,
            }
            expression = catalog.filter(
                result,
                mode="expression",
                min_expression=10,
                min_effect_pct=10,
            )
            single = catalog.filter(
                result,
                mode="differential",
                min_expression=10,
                min_abs_logfc=1,
                max_padj=0.05,
                min_effect_pct=10,
            )
            consensus = catalog.filter(
                result,
                mode="consensus_de",
                min_expression=10,
                min_abs_logfc=0.5,
                max_padj=0.05,
                min_effect_pct=10,
            )
            gene_list = catalog.filter(
                result,
                mode="gene_list",
                min_effect_pct=10,
            )
            self.assertEqual(expression["tf_count"], 2)
            self.assertEqual(single["tf_count"], 2)
            self.assertEqual(
                [item["tf"] for item in consensus["motifs"]],
                ["TF1"],
            )
            self.assertEqual(
                [item["tf"] for item in gene_list["motifs"]],
                ["TF2"],
            )

    def test_figure_group_deduplicates_numbered_tf_family(self) -> None:
        groups = MotifAnnotationCatalog._figure_groups(
            [
                {"tf": "SNAI1", "start": 10, "end": 20, "overlap_effect_pct": 20},
                {"tf": "SNAI3", "start": 11, "end": 19, "overlap_effect_pct": 22},
                {"tf": "SNAI2", "start": 12, "end": 18, "overlap_effect_pct": 21},
                {"tf": "CEBPB", "start": 40, "end": 50, "overlap_effect_pct": 15},
            ]
        )
        self.assertEqual([group["label"] for group in groups], ["SNAI", "CEBPB"])
        self.assertEqual(groups[0]["full_label"], "SNAI3/SNAI2/SNAI1")

    def test_registry_exposes_bundled_architecture(self) -> None:
        ids = {
            descriptor["id"]
            for descriptor in ModelAdapterRegistry().descriptors()
        }
        self.assertEqual(ids, {"cd200_motif_cnn"})

    def test_local_zarr_v2_reader_only_decodes_needed_chunks(self) -> None:
        compressor = {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 3,
            "shuffle": 2,
            "blocksize": 0,
        }
        source = np.arange(48, dtype=np.uint8).reshape(12, 4)
        with tempfile.TemporaryDirectory() as directory:
            array_path = Path(directory)
            (array_path / ".zarray").write_text(
                json.dumps(
                    {
                        "zarr_format": 2,
                        "shape": [12, 4],
                        "chunks": [8, 4],
                        "dtype": "|u1",
                        "compressor": compressor,
                        "fill_value": 0,
                        "filters": None,
                        "order": "C",
                    }
                )
            )
            codec = get_codec(compressor)
            (array_path / "0.0").write_bytes(codec.encode(source[:8].tobytes()))
            (array_path / "1.0").write_bytes(codec.encode(source[8:].tobytes()))
            observed, chromosome_length = _read_local_zarr_v2_slice(
                array_path,
                6,
                11,
            )
        self.assertEqual(chromosome_length, 12)
        np.testing.assert_array_equal(observed, source[6:11])

    def test_local_zarr_v2_1d_reader_decodes_signal_chunks(self) -> None:
        compressor = {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 3,
            "shuffle": 2,
            "blocksize": 0,
        }
        source = np.arange(13, dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            array_path = Path(directory)
            (array_path / ".zarray").write_text(
                json.dumps(
                    {
                        "zarr_format": 2,
                        "shape": [13],
                        "chunks": [8],
                        "dtype": "<u2",
                        "compressor": compressor,
                        "fill_value": 0,
                        "filters": None,
                        "order": "C",
                    }
                )
            )
            codec = get_codec(compressor)
            (array_path / "0").write_bytes(codec.encode(source[:8].tobytes()))
            (array_path / "1").write_bytes(codec.encode(source[8:].tobytes()))
            observed, chromosome_length = _read_local_zarr_v2_1d_slice(
                array_path,
                6,
                12,
            )
        self.assertEqual(chromosome_length, 13)
        np.testing.assert_array_equal(observed, source[6:12])

    def test_auto_device_prefers_mps_when_cuda_is_absent(self) -> None:
        with (
            mock.patch("model_studio.devices.torch.cuda.is_available", return_value=False),
            mock.patch(
                "model_studio.devices.torch.backends.mps.is_available",
                return_value=True,
            ),
        ):
            self.assertEqual(DeviceManager.resolve("auto"), "mps")


if __name__ == "__main__":
    unittest.main()
