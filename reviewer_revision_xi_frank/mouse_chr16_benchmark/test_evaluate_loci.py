import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import multinomial

spec=importlib.util.spec_from_file_location('evaluate_loci',Path(__file__).with_name('evaluate_loci.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


class LocusEvaluationTests(unittest.TestCase):
    def test_frozen_loci_and_full_coverage(self):
        rows=e.planned_windows()
        self.assertEqual(len(rows),7)
        self.assertEqual(e.LOCI['P1'],('chr16',45383863,45384762))
        self.assertEqual(e.LOCI['P2'],('chr16',45356164,45357191))
        self.assertEqual(e.LOCI['promoter'],('chr16',45408000,45409426))
        for name in e.LOCI:
            profile,coverage=e.stitch_interval(rows,np.ones((7,1000)),name)
            self.assertTrue((coverage>0).all())
            np.testing.assert_array_equal(profile,np.ones(e.LOCI[name][2]-e.LOCI[name][1]))

    def test_nll_matches_discrete_multinomial(self):
        y=np.array([2.,1.,0.,3.]);p=np.array([.3,.2,.1,.4])
        out=e.shape_metrics(y,p,'raw_insertion_counts')
        self.assertTrue(out['multinomial_likelihood_valid'])
        self.assertAlmostEqual(out['profile_nll_with_constant'],-multinomial.logpmf(y,6,p))
        self.assertAlmostEqual(out['profile_nll_without_constant'],-np.dot(y,np.log(p)))
        self.assertAlmostEqual(out['profile_cross_entropy_per_target_unit'],-np.dot(y,np.log(p))/6)

    def test_fractional_targets_explicitly_qualified(self):
        out=e.shape_metrics(np.array([.2,.8]),np.array([.4,.6]),'processed_signal')
        self.assertFalse(out['target_values_integer'])
        self.assertFalse(out['multinomial_likelihood_valid'])
        self.assertIn('not a discrete count likelihood',out['loss_interpretation'])
        out=e.shape_metrics(np.array([2.,8.]),np.array([.4,.6]),'processed_signal')
        self.assertFalse(out['multinomial_likelihood_valid'])

    def test_zero_target_and_zero_probability(self):
        out=e.shape_metrics(np.zeros(4),np.full(4,.25),'raw_insertion_counts')
        self.assertEqual(out['profile_nll_with_constant'],0)
        self.assertIsNone(out['profile_cross_entropy_per_target_unit'])
        self.assertIsNone(out['profile_js_distance'])
        out=e.shape_metrics(np.array([1.,0]),np.array([0.,1]),'raw_insertion_counts')
        self.assertIsNone(out['profile_nll_without_constant'])
        self.assertIn('infinite',out['nll_status'])
        self.assertAlmostEqual(out['profile_js_divergence_nats'],np.log(2))

    def test_invalid_profile_fails(self):
        for p in [np.array([0.,0]),np.array([-1.,2]),np.array([np.nan,1])]:
            with self.assertRaises(ValueError):e.shape_metrics(np.array([1.,1]),p,'raw_insertion_counts')

    def test_constant_correlation_and_partial_bin(self):
        self.assertIsNone(e.pearson(np.ones(5),np.arange(5)))
        np.testing.assert_array_equal(e.bin_sum(np.arange(43)),[sum(range(20)),sum(range(20,40)),sum(range(40,43))])
        out=e.shape_metrics(np.ones(43),np.full(43,1/43),'raw_insertion_counts')
        self.assertIsNone(out['profile_pearson_1bp'])
        self.assertEqual(out['profile_bin20_final_width'],3)

    def test_observation_overlap_disagreement_fails(self):
        rows=e.planned_windows();values=np.ones((len(rows),1000))
        values[2]+=1
        with self.assertRaisesRegex(ValueError,'observed windows disagree'):
            e.stitch_interval(rows,values,'P2',True)

    def fixture(self,root,central_only=False):
        data=root/'data';data.mkdir();manifest=data/'locus_windows.tsv'
        e.write_window_manifest(manifest)
        if central_only:
            text=manifest.read_text().splitlines();manifest.write_text('\n'.join([text[0]]+[x for x in text[1:] if x.endswith('\tcentral')])+'\n')
        rows=e.read_windows(manifest)
        raw=np.stack([1+4*(np.arange(r['output_start'],r['output_end'])%23==0) for r in rows]).astype(float)
        np.save(data/'observed_profiles.npy',raw)
        meta={'genome_build':'mm10','target_units':'unsmoothed_per_base_ATAC_counts_from_bundled_Zarr','target_kind':'stored_atac_signal_counts',
              'input_length':2114,'output_length':1000}
        (data/'target_metadata.json').write_text(json.dumps(meta))
        predictions={}
        for model in ['MoDL','ExplaiNN']:
            d=root/model;d.mkdir();predictions[model]=d
            np.save(d/'prediction_counts.npy',np.log1p(raw.sum(axis=1)))
            if model=='MoDL':np.save(d/'prediction_profiles.npy',raw/raw.sum(axis=1,keepdims=True))
            m={**meta,'window_ids_sha256':e.window_order_hash([r['window_id'] for r in rows]),'count_transform':e.COUNT_TRANSFORM}
            (d/'prediction_metadata.json').write_text(json.dumps(m))
        return data,predictions,rows,raw

    def test_end_to_end_no_double_counting_and_scalar_na(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data,pred,rows,raw=self.fixture(root)
            out=e.evaluate(data,pred,root/'out')
            central=out['windows']['promoter_central']['MoDL']
            self.assertAlmostEqual(central['count_log_mse'],0)
            self.assertAlmostEqual(central['profile_js_distance'],0,places=12)
            self.assertAlmostEqual(central['profile_pearson_1bp'],1)
            full=out['full_intervals']['promoter']['MoDL']
            _,a,b=e.LOCI['promoter'];truth=float((1+4*(np.arange(a,b)%23==0)).sum())
            self.assertAlmostEqual(full['observed_total_signal'],truth)
            self.assertAlmostEqual(full['reconstructed_predicted_total_signal'],truth)
            self.assertAlmostEqual(full['reconstructed_count_log_mse'],0)
            self.assertGreater(sum(raw[i].sum() for i,r in enumerate(rows) if r['locus']=='promoter'),truth)
            self.assertIn('scalar-only',out['full_intervals']['promoter']['ExplaiNN']['profile_status'])
            self.assertEqual(len(out['windows']),7)
            self.assertTrue((root/'out/locus_metrics.tsv').is_file())
            json.loads((root/'out/locus_evaluation.json').read_text())
            self.assertNotIn('count_pearson',out)

    def test_central_only_reports_incomplete_full_interval(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data,pred,rows,raw=self.fixture(root,central_only=True)
            out=e.evaluate(data,pred,root/'out')
            self.assertTrue(out['full_intervals']['P1']['MoDL']['coverage_complete'])
            self.assertFalse(out['full_intervals']['promoter']['MoDL']['coverage_complete'])
            self.assertNotIn('reconstructed_count_log_mse',out['full_intervals']['promoter']['MoDL'])

    def test_mismatched_order_or_units_fails(self):
        for key,value in [('window_ids_sha256','wrong'),('target_units','CPM')]:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);data,pred,rows,raw=self.fixture(root)
                p=pred['MoDL']/'prediction_metadata.json';meta=json.loads(p.read_text());meta[key]=value;p.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):e.evaluate(data,pred,root/'out')

    def test_stored_counts_and_standard_region_order_alias(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data,pred,rows,raw=self.fixture(root)
            p=pred['MoDL']/'prediction_metadata.json';meta=json.loads(p.read_text())
            meta['region_ids_sha256']=meta.pop('window_ids_sha256');p.write_text(json.dumps(meta))
            out=e.evaluate(data,pred,root/'out')
            record=out['windows']['P1_central']['MoDL']
            self.assertTrue(record['multinomial_likelihood_valid'])
            self.assertIn('upstream Tn5 shift is not established',record['loss_interpretation'])
            self.assertEqual(out['contract']['count_transform'],'ln1p_stored_atac_count')
            self.assertEqual(out['contract']['target_units'],'unsmoothed_per_base_ATAC_counts_from_bundled_Zarr')

    def test_negative_native_count_not_clipped_for_stitching(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data,pred,rows,raw=self.fixture(root)
            counts=np.load(pred['MoDL']/'prediction_counts.npy');counts[0]=-.5;np.save(pred['MoDL']/'prediction_counts.npy',counts)
            out=e.evaluate(data,pred,root/'out')
            self.assertIn('negative inverse',out['full_intervals']['P1']['MoDL']['profile_status'])
            self.assertGreater(out['windows']['P1_central']['MoDL']['count_log_mse'],0)


if __name__=='__main__':unittest.main()
