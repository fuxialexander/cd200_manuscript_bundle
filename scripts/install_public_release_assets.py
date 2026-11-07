"""Verify v0.2.0 archives and place historical weights at the source's expected paths.

Run from an extracted modl_revision directory. Uses only the Python standard library.
This does not install a scientific runtime, download genomic stores, or run inference.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,tarfile

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True,help='Directory containing original archives and RELEASE_MANIFEST.json')
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--install-historical-weights',action='store_true')
    a=p.parse_args();manifest=json.loads((a.assets/'RELEASE_MANIFEST.json').read_text())
    verified=[]
    for item in manifest['assets']:
        f=a.assets/item['file']
        if f.stat().st_size!=item['bytes'] or sha(f)!=item['sha256']:raise ValueError('Release asset checksum mismatch: '+f.name)
        verified.append(f.name)
    placed=[]
    if a.install_historical_weights:
        with tarfile.open(a.assets/'modl_checkpoints_20261107.tar.gz') as tf:
            rows=json.load(tf.extractfile('weights/CHECKPOINT_MANIFEST.json'))
            for row in rows:
                if row['dataset']!='historical_cSCC':continue
                target=a.bundle/'analysis/checkpoints'/Path(row['path']).name
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if sha(target)==row['sha256']:placed.append(str(target.relative_to(a.bundle)));continue
                    if not target.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1\n'):raise FileExistsError('Refusing to replace different checkpoint: '+str(target))
                temp=target.with_name(target.name+'.partial')
                if temp.exists():raise FileExistsError(str(temp))
                with tf.extractfile('weights/'+row['path']) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
                if sha(temp)!=row['sha256']:raise ValueError('Checkpoint checksum mismatch')
                temp.replace(target);placed.append(str(target.relative_to(a.bundle)))
    print(json.dumps({'verified_assets':verified,'historical_weights':placed,'inference_or_training_run':False},indent=2))

if __name__=='__main__':main()
