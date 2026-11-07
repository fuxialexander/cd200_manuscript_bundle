"""Focused background-gate and all-component chromosome-leakage checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("mouse_native", Path(__file__).with_name("train_mouse_chrombpnet.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class MouseBiasChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.manifest = self.cache / "cache_manifest.json"
        self.manifest.write_text(json.dumps({"assembly":"mm10"}))
        for split, chrom in [("train","chr2"),("valid","chr14")]:
            np.save(self.cache / f"{split}.x.npy",np.zeros((2,3114),dtype=np.uint8))
            np.save(self.cache / f"{split}.y.npy",np.ones((2,2000),dtype=np.float32))
            (self.cache / f"{split}.regions.tsv").write_text(
                f"region_id\tchrom\tcenter\tis_peak\n{split}1\t{chrom}\t10000\t1\n{split}2\t{chrom}\t20000\t0\n")
        self.bias = self.root / "bias_model_scaled.h5"
        self.bias.write_bytes(b"test-placeholder-no-tensorflow-required")
        self.args = SimpleNamespace(cache_dir=self.cache,bias_model=self.bias,heldout_chroms=["chr16","chr8","chr9"],max_train_steps=0,max_valid_batches=0)

    def tearDown(self):
        self.tmp.cleanup()

    def provenance(self, **changes):
        p = {"stage":"completed_mouse_bias_fit_and_train_only_rescaling","assembly":"mm10",
             "initialization":"random; no pretrained checkpoint","scaled_model_sha256":M.shared.sha256(self.bias),
             "cache_manifest_sha256":M.shared.sha256(self.manifest),"training_chromosomes":["chr2"],
             "validation_chromosomes":["chr14"],"pilot":False}
        p.update(changes)
        (self.root / "bias_provenance.json").write_text(json.dumps(p))

    def test_background_threshold_fits_only_training_and_is_frozen_for_validation(self):
        c=np.array([100,120,140,0,1,2,3,4,5,10,20,30,40,50,75],dtype=float)
        p=np.zeros(len(c),dtype=bool);p[:3]=True
        selected,g=M.background_gates(c,p)
        self.assertAlmostEqual(g["background_count_upper_cutoff"],50.2)
        self.assertTrue(np.all(c[selected]<g["background_count_upper_cutoff"]))
        before=dict(g)
        v=np.array([0,1,25,49,55],dtype=float)
        np.testing.assert_array_equal(M.apply_background_gates(v,np.zeros(5,dtype=bool),g),[1,2,3])
        self.assertEqual(g,before)

    def test_zero_threshold_is_not_silently_repaired(self):
        with self.assertRaisesRegex(ValueError,"first percentile is zero"):
            M.background_gates(np.array([0,0,100,1,2,3]),np.array([True,True,True,False,False,False]))

    def test_mouse_partitions_allow_train_and_validation_but_never_chr16(self):
        M.mouse_inputs(self.args)
        p=self.cache / "train.regions.tsv"
        p.write_text(p.read_text().replace("chr2","chr16"))
        with self.assertRaisesRegex(ValueError,"Held-out chromosomes"):
            M.mouse_inputs(self.args)

    def test_unknown_or_human_assembly_rejected(self):
        self.manifest.write_text(json.dumps({"assembly":"GRCh38"}))
        with self.assertRaisesRegex(ValueError,"mm10"):
            M.mouse_inputs(self.args)

    def test_verified_mouse_component_accepted(self):
        self.provenance()
        M.require_mouse_bias(self.args)

    def test_human_or_unknown_bias_rejected(self):
        self.provenance(assembly="GRCh38")
        with self.assertRaisesRegex(ValueError,"human/unknown"):
            M.require_mouse_bias(self.args)

    def test_bias_that_saw_chr16_rejected(self):
        self.provenance(training_chromosomes=["chr2","chr16"])
        with self.assertRaisesRegex(ValueError,"held-out chromosome"):
            M.require_mouse_bias(self.args)

    def test_bias_weight_substitution_rejected(self):
        self.provenance()
        self.bias.write_bytes(b"substituted-weights")
        with self.assertRaisesRegex(ValueError,"weights differ"):
            M.require_mouse_bias(self.args)

    def test_pilot_component_rejected_for_final_fit(self):
        self.provenance(pilot=True)
        with self.assertRaisesRegex(ValueError,"bias pilot"):
            M.require_mouse_bias(self.args)

    def test_target_contract_overrides_inherited_human_units(self):
        original=M.shared.save_json
        try:
            M.install_mouse_metadata_contract()
            p=self.root / "prediction_metadata.json"
            M.shared.save_json(p,{"count_transform":"ln1p_raw_insertion_count","prediction_counts_units":"raw insertions"})
            obj=json.loads(p.read_text())
            self.assertEqual(obj["count_transform"],"ln1p_stored_atac_count")
            self.assertEqual(obj["target_units"],"unsmoothed_per_base_ATAC_counts_from_bundled_Zarr")
            self.assertEqual(obj["genome_build"],"mm10")
            self.assertEqual(obj["target_kind"],"stored_atac_signal_counts")
            self.assertEqual((obj["input_length"],obj["output_length"]),(2114,1000))
            self.assertIn("have not been established",obj["target_provenance_limitation"])
            self.assertNotEqual(obj["prediction_counts_units"],"raw insertions")
        finally:
            M.shared.save_json=original


if __name__ == "__main__":
    unittest.main()
