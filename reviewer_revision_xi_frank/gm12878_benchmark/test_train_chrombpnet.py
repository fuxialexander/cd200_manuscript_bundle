"""Geometry and leakage checks runnable without TensorFlow/CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("train_cbp", Path(__file__).with_name("train_chrombpnet.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class CacheChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        x = np.tile(np.arange(3114, dtype=np.int64) % 5, (2, 1)).astype(np.uint8)
        y = np.tile(np.arange(2000, dtype=np.float32), (2, 1))
        for split, chrom in [("train", "chr2"), ("valid", "chr8")]:
            np.save(self.root / f"{split}.x.npy", x)
            np.save(self.root / f"{split}.y.npy", y)
            (self.root / f"{split}.regions.tsv").write_text(
                f"region_id\tchrom\tcenter\tis_peak\n{split}1\t{chrom}\t5000\t1\n{split}2\t{chrom}\t6000\t0\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_center_and_count_target(self):
        d = M.Cache(self.root, "train")
        x, (y, c) = d.batch([0, 1])
        self.assertEqual(x.shape, (2, 2114, 4))
        np.testing.assert_array_equal(y[0], np.arange(500, 1500))
        np.testing.assert_allclose(c[:,0], np.log1p(y.sum(-1)))
        self.assertEqual(x[0,4].sum(), 0)  # N remains all-zero.

    def test_training_negative_is_not_jittered(self):
        d = M.Cache(self.root, "train")
        _, (y, _) = d.batch([0, 1], np.random.default_rng(1234))
        self.assertEqual(set(y[1]), set(np.arange(500,1500)))
        self.assertNotEqual(set(y[0]), set(np.arange(500,1500)))

    def test_disjoint_partition_gate(self):
        a, b = M.Cache(self.root,"train"), M.Cache(self.root,"valid")
        M.assert_partitions(a,b)
        b.regions[0]["chrom"] = "chr2"
        with self.assertRaises(ValueError):
            M.assert_partitions(a,b)

    def test_duplicate_identity_rejected(self):
        p = self.root / "train.regions.tsv"
        p.write_text(p.read_text().replace("train2", "train1"))
        with self.assertRaises(ValueError):
            M.Cache(self.root,"train")


@unittest.skipUnless(os.environ.get("CHROMBPNET_PARITY_MODELS"), "Set CHROMBPNET_PARITY_MODELS to released fold_0 directory for independent H5 graph parity")
class ReleasedGraphParity(unittest.TestCase):
    def test_combined_graph_matches_components_without_marshaled_lambda(self):
        import h5py
        import tensorflow as tf
        root = Path(os.environ["CHROMBPNET_PARITY_MODELS"])
        experiment = os.environ.get("CHROMBPNET_PARITY_EXPERIMENT", "ENCSR637XSC")
        full = root / f"model.chrombpnet.fold_0.{experiment}.h5"
        with h5py.File(full) as handle:
            config = json.loads(handle.attrs["model_config"])
        lambdas = [layer for layer in config["config"]["layers"] if layer["class_name"] == "Lambda"]
        self.assertEqual(len(lambdas), 1)
        layer = lambdas[0]["config"]
        self.assertEqual(layer["name"], "logcount_predictions")
        self.assertEqual(layer["function_type"], "lambda")
        self.assertEqual(layer["arguments"], {})
        layer.update(function="verified_reduce_logsumexp", function_type="function", module="__main__")
        def verified_reduce_logsumexp(t):
            return tf.reduce_logsumexp(t, axis=-1, keepdims=True)
        direct = tf.keras.models.model_from_json(json.dumps(config), custom_objects={"verified_reduce_logsumexp": verified_reduce_logsumexp})
        direct.load_weights(full)
        components = M.released_model(tf, root / f"model.bias_scaled.fold_0.{experiment}.h5",
                                     root / f"model.chrombpnet_nobias.fold_0.{experiment}.h5")
        x = np.eye(4,dtype=np.float32)[np.random.default_rng(1234).integers(4,size=(8,2114))]
        original, composed = direct(x,training=False), components(x,training=False)
        differences = []
        for a,b in zip(original,composed):
            a,b = a.numpy(), b.numpy()
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
            differences.append(float(np.max(np.abs(a-b))))
        M.save_json(Path(__file__).parent / "logs/chrombpnet_component_parity.json", {
            "status":"PASS", "validation":"Original combined graph with only inspected count Lambda replaced by named reduce_logsumexp; original combined H5 weights compared against independent component reconstruction",
            "unsafe_marshaled_code_executed":False, "n_random_sequences":8, "sequence_length":2114,
            "max_abs_profile_logit_difference":differences[0], "max_abs_logcount_difference":differences[1],
            "full_combined_model_sha256":M.sha256(full), "tensorflow":tf.__version__,
            "parity_device":str(original[0].device)})


if __name__ == "__main__":
    unittest.main()
