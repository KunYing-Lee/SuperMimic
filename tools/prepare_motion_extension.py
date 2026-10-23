"""Import a manifest-described G1 qpos36 bundle without modifying its motion poses.

Inputs remain private; output is an append-only library with explicit per-clip rates.
Only quaternion column ordering changes. MotionLib performs the direct resampling.
"""
import argparse,csv,hashlib,io,json,shutil,zipfile
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 with zipfile.ZipFile(a.archive) as z:
  manifests=[n for n in z.namelist() if n.endswith('/manifest.csv') and not n.startswith('__MACOSX/')]
  if len(manifests)!=1:raise ValueError('Expected one manifest.csv')
  root=manifests[0].rsplit('/',1)[0]+'/'
  checks={line.split(None,1)[1].strip():line.split()[0] for line in z.read(root+'CHECKSUMS_SHA256.txt').decode().splitlines() if line.strip()}
  def read(name):
   content=z.read(root+name)
   if hashlib.sha256(content).hexdigest()!=checks[name]:raise ValueError(f'Checksum mismatch: {name}')
   return content
  rows=list(csv.DictReader(io.StringIO(read('manifest.csv').decode('utf-8-sig'))))
  if len(rows)!=20:raise ValueError('Expected exactly 20 requested motions')
  a.output.mkdir(parents=True);rates={};record=[]
  base_rates=json.loads((a.base/'motion_rates.json').read_text()) if (a.base/'motion_rates.json').exists() else {}
  for f in sorted(a.base.glob('*.csv')):shutil.copy2(f,a.output/f.name);rates[f.name]=base_rates.get(f.name,30.0)
  originals=sorted(rates)
  for row in rows:
   if row['delivery_quaternion_order']!='wxyz':raise ValueError('Expected delivered wxyz convention')
   raw=read(row['omg_csv_path']);x=np.loadtxt(io.BytesIO(raw),delimiter=',');fps=float(row['delivery_fps'])
   if x.shape!=(int(row['delivery_g1_frames']),36) or not np.isfinite(x).all():raise ValueError('Invalid qpos36 data')
   err=float(np.max(np.abs(np.linalg.norm(x[:,3:7],axis=1)-1)))
   if err>1e-4:raise ValueError(f'Non-unit quaternion: {err}')
   y=x.copy();y[:,3:7]=x[:,[4,5,6,3]]
   dest='zz_motioncode_'+row['stem']+'.csv'
   if '/' in dest or dest in rates:raise ValueError('Unsafe or duplicate sample name')
   np.savetxt(a.output/dest,y,delimiter=',',fmt='%.17g');rates[dest]=fps
   record.append(dict(file=dest,source=row['omg_csv_path'],source_sha256=hashlib.sha256(raw).hexdigest(),source_fps=fps,source_frames=len(x),frames_50hz=int(np.floor(((len(x)-1)/fps)/.02))+1,quaternion_norm_max_error=err))
  if sorted(rates)[:len(originals)]!=originals:raise ValueError('Extension must preserve original clip order')
  (a.output/'motion_rates.json').write_text(json.dumps(rates,indent=2)+'\n')
  (a.output/'import_provenance.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),base_clips=len(originals),added_clips=len(rows),transform='wxyz to xyzw column permutation only; original positions/joints/rates retained',motions=record),indent=2)+'\n')
  print(json.dumps(dict(base=len(originals),added=len(rows),new_duration_seconds=sum((r['source_frames']-1)/r['source_fps'] for r in record),new_frames_50hz=sum(r['frames_50hz'] for r in record)),indent=2))

if __name__=='__main__':main()
