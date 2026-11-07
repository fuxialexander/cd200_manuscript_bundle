"""Designed statistical/contract tests; run with python -m unittest discover."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

spec=importlib.util.spec_from_file_location('benchmark_evaluate',Path(__file__).with_name('evaluate.py'))
e=importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


class EvaluatorTests(unittest.TestCase):
    def test_count_stats_match_explicit_resampling(self):
        target=np.array([0.,1.,2.,4.,3.,6.]);pred=np.array([1.,1.2,2.5,3.,4.,5.])
        blocks=np.array([0,0,1,1,2,2]);mask=np.ones(6,dtype=bool)
        stats=e.count_block_stats(target,pred,mask,blocks,3)
        w=np.array([[2,0,1],[0,1,2],[1,1,1]])
        result=e.weighted_count_metrics(w@stats)
        for i,row in enumerate(w):
            indices=np.concatenate([np.repeat(np.where(blocks==j)[0],k) for j,k in enumerate(row)])
            self.assertAlmostEqual(result['count_pearson_ln1p'][i],pearsonr(target[indices],pred[indices]).statistic)
            self.assertAlmostEqual(result['count_rmse_ln1p'][i],np.sqrt(np.mean((target[indices]-pred[indices])**2)))

    def test_profile_distance_matches_native_and_bins(self):
        y=np.zeros((3,1000));y[0,4]=1;y[1,4]=1
        p=np.zeros((3,1000));p[0,4]=1;p[1,24]=1;p[2,24]=1
        out=e.profile_metrics(y,p)
        np.testing.assert_allclose(out['profile_js_distance'][:2],[0,np.sqrt(np.log(2))])
        self.assertAlmostEqual(out['profile_js_distance'][1],jensenshannon(y[1],p[1]))
        self.assertAlmostEqual(out['profile_js_divergence_nats'][1],np.log(2))
        self.assertAlmostEqual(out['profile_pearson_1bp'][0],1)
        self.assertAlmostEqual(out['profile_pearson_20bp'][0],1)
        self.assertTrue(all(np.isnan(x[2]) for x in out.values()))

    def test_constant_profiles_have_undefined_correlation(self):
        out=e.profile_metrics(np.ones((1,1000)),np.full((1,1000),.001))
        self.assertAlmostEqual(out['profile_js_distance'][0],0,places=14)
        self.assertTrue(np.isnan(out['profile_pearson_1bp'][0]))
        self.assertTrue(np.isnan(out['profile_pearson_20bp'][0]))

    def test_profiles_fail_bad_probabilities(self):
        for p in [np.zeros((1,1000)),np.full((1,1000),-0.001),np.full((1,1000),np.nan)]:
            with self.assertRaises(ValueError):e.profile_metrics(np.ones((1,1000)),p)

    def test_blocks_merge_overlapping_input_across_bin_boundary(self):
        rows=[{'chrom':'chr1','center':998950},{'chrom':'chr1','center':1001000},
              {'chrom':'chr1','center':2500000},{'chrom':'chr3','center':999500}]
        blocks,chroms=e.genomic_blocks(rows)
        self.assertEqual(blocks[0],blocks[1])
        self.assertNotEqual(blocks[0],blocks[2])
        self.assertNotEqual(blocks[0],blocks[3])
        self.assertEqual(len(chroms),3)
        w=e.bootstrap_weights(chroms,50,10)
        np.testing.assert_array_equal(w,e.bootstrap_weights(chroms,50,10))
        for c in set(chroms):
            idx=np.where(np.array(chroms)==c)[0]
            np.testing.assert_array_equal(w[:,idx].sum(axis=1),len(idx))

    def make_fixture(self,root,constant_profile=False):
        data=root/'data';data.mkdir();preds={}
        ids=[f'r{i}' for i in range(8)]
        with (data/'test.regions.tsv').open('w') as f:
            f.write('region_id\tchrom\tcenter\tis_peak\n')
            for i,x in enumerate(ids):f.write(f'{x}\tchr{1 if i<4 else 3}\t{2000+(i%4)*1500000}\t{i%2}\n')
        y=np.zeros((8,2000),dtype=np.float32)
        for i in range(1,8):y[i,500+i]=i
        np.save(data/'test.y.npy',y)
        for name,shift in [('MoDL',0),('ChromBPNet',.1),('ExplaiNN',.2)]:
            d=root/name;d.mkdir();preds[name]=d
            np.save(d/'prediction_counts.npy',np.log1p(np.arange(8))+shift)
            (d/'prediction_metadata.json').write_text(json.dumps({'region_ids_sha256':e.region_order_hash(ids),
                'target_start':500,'target_end':1500,'count_transform':e.COUNT_TRANSFORM}))
            if name!='ExplaiNN':
                profile=np.zeros((8,1000));profile[np.arange(8),np.arange(8)]=1
                if constant_profile and name=='ChromBPNet':profile[1]=.001
                np.save(d/'prediction_profiles.npy',profile)
        return data,preds

    def test_end_to_end_masks_zero_targets_pairing_and_na(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data,preds=self.make_fixture(root,constant_profile=True)
            result=e.evaluate(data,preds,root/'out',n_bootstrap=30)
            self.assertEqual(result['scopes']['all']['MoDL']['zero_count_targets'],1)
            self.assertEqual(result['scopes']['all']['MoDL']['metrics']['profile_js_distance']['n_defined'],7)
            self.assertEqual(result['scopes']['all']['ExplaiNN']['profile_status'],'N/A: checkpoint provides scalar counts only')
            self.assertEqual(result['scopes']['nonpeaks']['MoDL']['n_regions'],4)
            self.assertAlmostEqual(result['scopes']['all']['MoDL']['metrics']['count_rmse_ln1p']['value'],0)
            differences=[x for x in result['paired_deltas'] if x['scope']=='all' and x['model_a']=='MoDL' and x['model_b']=='ChromBPNet']
            d=next(x for x in differences if x['metric']=='profile_pearson_1bp')
            self.assertEqual(d['n_common_defined_regions'],6)
            self.assertAlmostEqual(d['delta_a_minus_b'],0)
            d=next(x for x in differences if x['metric']=='count_rmse_ln1p')
            self.assertAlmostEqual(d['delta_a_minus_b'],-.1)
            self.assertIsNone(result['scopes']['all']['MoDL']['metrics']['count_spearman_ln1p']['ci95'])
            saved=json.loads((root/'out/evaluation.json').read_text())
            self.assertEqual(saved['contract']['region_ids_sha256'],result['contract']['region_ids_sha256'])
            self.assertTrue((root/'out/metrics.tsv').exists())

    def test_bad_order_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data,preds=self.make_fixture(root)
            p=preds['MoDL']/'prediction_metadata.json';m=json.loads(p.read_text());m['region_ids_sha256']='bad';p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'order/hash mismatch'):e.evaluate(data,preds,root/'out',20)

    def test_bad_count_shape_and_nonfinite_fail(self):
        for bad in [np.ones(7),np.full(8,np.inf)]:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);data,preds=self.make_fixture(root)
                np.save(preds['MoDL']/'prediction_counts.npy',bad)
                with self.assertRaisesRegex(ValueError,'finite vector'):e.evaluate(data,preds,root/'out',20)

    def test_bad_raw_target_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data,preds=self.make_fixture(root)
            y=np.load(data/'test.y.npy');y[0,700]=-1;np.save(data/'test.y.npy',y)
            with self.assertRaisesRegex(ValueError,'nonnegative'):e.evaluate(data,preds,root/'out',20)

    def test_constant_count_target_undefined(self):
        stats=e.count_block_stats(np.ones(4),np.arange(4.),np.ones(4,dtype=bool),np.zeros(4,dtype=int),1)
        self.assertTrue(np.isnan(e.weighted_count_metrics(stats.sum(axis=0))['count_pearson_ln1p']))
        self.assertEqual(e.interval(np.array([np.nan,np.nan]))['valid_replicates'],0)


if __name__=='__main__':unittest.main()
