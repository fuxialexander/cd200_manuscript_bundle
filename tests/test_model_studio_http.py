from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from model_studio.server import ModelStudioHTTPServer


class ModelStudioHTTPTests(unittest.TestCase):
    def test_capabilities_endpoint(self) -> None:
        server = ModelStudioHTTPServer(("127.0.0.1", 0))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/capabilities",
                timeout=10,
            ) as response:
                payload = json.load(response)
            architecture_ids = {
                architecture["id"] for architecture in payload["architectures"]
            }
            self.assertEqual(
                architecture_ids,
                {"cd200_motif_cnn"},
            )
            preset_ids = {
                preset["id"] for preset in payload["analysis_presets"]
            }
            self.assertIn("manuscript_differential_promoter", preset_ids)
            manuscript = next(
                preset
                for preset in payload["analysis_presets"]
                if preset["id"] == "manuscript_differential_promoter"
            )
            self.assertEqual(manuscript["context_bp"], 4096)
            self.assertEqual(
                manuscript["analysis_model"]["output_head"],
                "differential",
            )
            self.assertIn("profile_model", manuscript)
            self.assertTrue(payload["devices"]["cpu_available"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_local_genome_and_observed_track_endpoints(self) -> None:
        server = ModelStudioHTTPServer(("127.0.0.1", 0))
        server.service.chromosome_sizes = mock.Mock(
            return_value=[("chr1", 1000), ("chrX", 500)]
        )
        server.service.observed_atac_bedgraph = mock.Mock(
            return_value=(
                "chr1\t100\t110\t2.5\n",
                {
                    "chromosome": "chr1",
                    "start": 100,
                    "end": 110,
                    "bin_bp": 10,
                    "source": "local_zarr",
                },
            )
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/genome/mm10.chrom.sizes",
                timeout=10,
            ) as response:
                chromosome_sizes = response.read().decode()
            self.assertEqual(chromosome_sizes, "chr1\t1000\nchrX\t500\n")

            with urllib.request.urlopen(
                "http://127.0.0.1:"
                f"{port}/api/tracks/observed-atac.bedgraph"
                "?preset_id=manuscript_differential_promoter"
                "&chr=chr1&start=100&end=110&max_points=1000",
                timeout=10,
            ) as response:
                bedgraph = response.read().decode()
                bin_bp = response.headers["X-Track-Bin-Bp"]
            self.assertEqual(bedgraph, "chr1\t100\t110\t2.5\n")
            self.assertEqual(bin_bp, "10")
            server.service.observed_atac_bedgraph.assert_called_once_with(
                "manuscript_differential_promoter",
                "chr1",
                100,
                110,
                max_points=1000,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_annotation_load_and_filter_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motif_path = root / "motifs.tsv"
            motif_path.write_text(
                "chromosome\tstart\tend\tgene_name\tmotif_name\n"
                "chr1\t110\t120\tTF1\tTF1.a\n"
            )
            rna_path = root / "rna.csv"
            rna_path.write_text(
                "gene,baseMean,log2FoldChange\n"
                "TF1,100,3\n"
            )
            result = {
                "sequence": {
                    "chromosome": "chr1",
                    "start": 100,
                    "end": 200,
                },
                "nucleotide_positions": list(range(100, 200)),
                "plot_effects": [20.0] * 100,
            }
            server = ModelStudioHTTPServer(("127.0.0.1", 0))
            server.service.result = mock.Mock(return_value=result)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                load_request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/annotations/load",
                    data=json.dumps(
                        {
                            "job_id": "test-job",
                            "motif_path": str(motif_path),
                            "rna_path": str(rna_path),
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(load_request, timeout=10) as response:
                    loaded = json.load(response)["annotations"]
                self.assertEqual(loaded["available_tfs"], ["TF1"])

                filter_request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/annotations/filter",
                    data=json.dumps(
                        {
                            "job_id": "test-job",
                            "mode": "rna",
                            "min_abs_logfc": 2,
                            "min_expression": 10,
                            "min_effect_pct": 10,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(filter_request, timeout=10) as response:
                    filtered = json.load(response)["annotations"]
                self.assertEqual(filtered["motif_count"], 1)
                self.assertEqual(filtered["motifs"][0]["tf"], "TF1")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
