import csv,hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np

class CommonLossTests(unittest.TestCase):
 def test_common_loss_has_native_count_geometry_and_multinomial_constant(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cache=root/'cache';pred=root/'pred';cache.mkdir();pred.mkdir()
   rows=[{'region_id':'a','chrom':'chr16','is_peak':'1'},{'region_id':'b','chrom':'chr8','is_peak':'0'}]
   with (cache/'test.regions.tsv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
   y=np.zeros((2,2000),np.float32);y[0,500:502]=[2,1];np.save(cache/'test.y.npy',y)
   np.save(pred/'prediction_counts.npy',np.array([np.log(4),0]))
   q=np.zeros((2,1000));q[0,:2]=[2/3,1/3];q[1]=1/1000;np.save(pred/'prediction_profiles.npy',q)
   (pred/'prediction_metadata.json').write_text(json.dumps({'region_ids_sha256':hashlib.sha256(b'a\nb\n').hexdigest(),'count_transform':'ln1p_stored_atac_count'}))
   subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_losses.py')),'--cache-dir',str(cache),'--prediction',f'MoDL={pred}','--output',str(root/'out'),'--count-weight','35.3'],check=True,capture_output=True)
   r=json.loads((root/'out/losses.json').read_text());v=next(x for x in r['rows'] if x['scope']=='peaks')
   self.assertAlmostEqual(v['count_log_mse'],0,places=12)
   self.assertAlmostEqual(v['profile_nll_without_constant']-v['profile_multinomial_nll'],np.log(3),places=10)
   self.assertEqual(next(x for x in r['rows'] if x['scope']=='nonpeaks')['profile_nll_without_constant'],0)

if __name__=='__main__':unittest.main()
