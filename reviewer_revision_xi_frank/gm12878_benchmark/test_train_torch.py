import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch


SPEC = importlib.util.spec_from_file_location("gm_train", Path(__file__).with_name("train_torch.py"))
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


class TorchTrainingChecks(unittest.TestCase):
    def test_jitter_alignment_and_reverse_complement(self):
        x = (np.arange(3114) % 5).astype(np.uint8)
        y = np.arange(2000, dtype=np.float32)
        for j in (0, 500, 1000):
            seq, target = trainer.aligned_window(x, y, j)
            expected = torch.nn.functional.one_hot(torch.from_numpy(x[j:j + 2114].astype(np.int64)), 5)[..., :4].T.float()
            torch.testing.assert_close(seq, expected)
            np.testing.assert_array_equal(target.numpy(), y[j:j + 1000])
            seq_rc, target_rc = trainer.aligned_window(x, y, j, True)
            torch.testing.assert_close(seq_rc, seq.flip((0, 1)))
            torch.testing.assert_close(target_rc, target.flip(0))

    def test_invalid_values_fail(self):
        x, y = np.zeros(3114, np.uint8), np.zeros(2000, np.float32)
        with self.assertRaises(ValueError):
            trainer.aligned_window(x, y, 1001)
        x[0] = 5
        with self.assertRaises(ValueError):
            trainer.aligned_window(x, y, 0)
        x[0] = 0
        y[0] = np.nan
        with self.assertRaises(ValueError):
            trainer.aligned_window(x, y, 0)

    def test_training_sampler_has_fixed_negative_ratio(self):
        mask = np.array([True] * 100 + [False] * 1000)
        a = trainer.sample_training_indices(mask, .1, 1234)
        b = trainer.sample_training_indices(mask, .1, 1234)
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(set(a)))
        self.assertEqual(sum(mask[a]), 100)
        self.assertEqual(sum(~mask[a]), 10)
        self.assertNotEqual(a, trainer.sample_training_indices(mask, .1, 1235))

    def test_count_target_and_profile_loss(self):
        y = torch.ones(2, 1000)
        target = torch.log1p(y.sum(-1))
        q = torch.ones(2, 1000) / 1000
        total, shape, count = trainer.losses(target, q, y, 5)
        self.assertAlmostEqual(count.item(), 0)
        self.assertAlmostEqual(shape.item(), 1000 * np.log(1000), places=2)
        zero, _, _ = trainer.losses(torch.zeros(2), None, torch.zeros(2, 1000), 5)
        self.assertEqual(zero.item(), 0)

    def test_cache_rows_and_chromosome_partition_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, chrom in (("train", "chr1"), ("valid", "chr2")):
                np.save(root / f"{split}.x.npy", np.zeros((2, 3114), dtype=np.uint8))
                np.save(root / f"{split}.y.npy", np.zeros((2, 2000), dtype=np.float32))
                (root / f"{split}.regions.tsv").write_text("region_id\tchrom\tcenter\tis_peak\n" +
                    f"{split}_a\t{chrom}\t10000\t1\n{split}_b\t{chrom}\t20000\t0\n")
            train = trainer.CacheDataset(root, "train", True)
            valid = trainer.CacheDataset(root, "valid", False)
            trainer.check_partition_pair(train, valid)
            x, y, index = valid[0]
            self.assertEqual(tuple(x.shape), (4, 2114))
            self.assertEqual(tuple(y.shape), (1000,))
            self.assertEqual(index, 0)
            with mock.patch.object(trainer.np.random, "randint", side_effect=[17, 0]) as rng:
                train[0]
                self.assertEqual(rng.call_args_list, [mock.call(0, 1001), mock.call(2)])
            with mock.patch.object(trainer.np.random, "randint", side_effect=[0]) as rng:
                train[1]
                self.assertEqual(rng.call_args_list, [mock.call(2)])
            valid.regions[0]["chrom"] = "chr1"
            with self.assertRaisesRegex(ValueError, "chromosomes overlap"):
                trainer.check_partition_pair(train, valid)

    def test_no_test_cache_needed_to_construct_training_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "train.x.npy", np.zeros((1, 3114), dtype=np.uint8))
            np.save(root / "train.y.npy", np.zeros((1, 2000), dtype=np.float32))
            (root / "train.regions.tsv").write_text("region_id\tchrom\tcenter\tis_peak\na\tchr1\t1\t1\n")
            data = trainer.CacheDataset(root, "train", True)
            self.assertEqual(len(data), 1)
            self.assertFalse((root / "test.x.npy").exists())


if __name__ == "__main__":
    unittest.main()
