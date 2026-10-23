"""Explicit append-only dataset migration of a full training checkpoint.

Preserve networks, optimizers, RNG, simulator, replay, observation history, global
iteration and learning-rate schedule. Preserve old bin returns/counts, initialize
new bins normally, and restart the historical sampling average for the new domain.
This operation is intentionally separate from strict same-dataset resume.
"""
import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.reproducibility import manifest,sha256,validate_resume,write_json
from supermimic.robot import MJCF_PATH


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--motion-dir',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists() or a.run_dir.exists():raise FileExistsError('Use fresh output and run paths')
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);old=ck['manifest']
 if ck.get('schema_version')!=2:raise ValueError('Full schema-v2 checkpoint required')
 args=argparse.Namespace(**ck['args']);args.motion_dir=str(a.motion_dir.resolve());args.log_dir=str(a.run_dir.resolve());args.resume=str(a.output.resolve())
 motion=MotionLib.from_dir(args.motion_dir,G1Kinematics('cpu'),'cpu');nold=len(old['dataset']['clips'])
 assert len(motion.names)>nold and motion.names[:nold]==old['dataset']['clips'],'Must append clips without reordering old clips'
 assert int(motion.clip_start[nold])==old['dataset']['frames_50hz'],'Old frame indices changed'
 rates=json.loads((a.motion_dir/'motion_rates.json').read_text())
 assert all(rates[Path(f).name]==30.0 for f in motion.files[:nold]),'This migrator requires the original 30 Hz library'
 current=manifest(str(ROOT),args,motion,MJCF_PATH)
 changed=[k for k in old['source_sha256'].keys()|current['source_sha256'].keys() if old['source_sha256'].get(k)!=current['source_sha256'].get(k)]
 assert set(changed)<= {'supermimic/motion.py','supermimic/reproducibility.py'},f'Unrelated source changes: {changed}'
 assert old['packages']==current['packages'] and old['cuda']==current['cuda'],'Runtime differs'
 for path,digest in old['asset_sha256'].items():
  f=Path(path)
  if f.is_relative_to(Path(old['args']['motion_dir'])):new=a.motion_dir/f.relative_to(Path(old['args']['motion_dir']))
  else:new=ROOT/f.relative_to(Path(old['repository_root']))
  assert sha256(new)==digest,f'Original asset changed: {new}'
 ep=ck['env']['episodes'];nb=len(ep['bin_G']);nn=motion.num_bins
 assert nn>nb and int(motion.bin_clip[nb])==nold,'Old bin indexing changed'
 assert int(ck['replay']['data']['f'].max())<int(motion.clip_start[nold])
 assert int(ep['f'].max())<int(motion.clip_start[nold]) and int(ep['start_bin'].max())<nb
 for key in ['bin_G','bin_count']:
  previous=ep[key];ep[key]=torch.zeros(nn,dtype=previous.dtype);ep[key][:nb].copy_(previous)
 ep['bin_w_avg']=torch.full((nn,),1.0/nn,dtype=ep['bin_w_avg'].dtype)
 ep['sampling_updates'].zero_()
 source_hash=sha256(a.checkpoint)
 lineage=dict(parent_checkpoint=str(a.checkpoint.resolve()),parent_sha256=source_hash,parent_iteration=ck['iteration'],old_clips=nold,new_clips=len(motion.names),old_bins=nb,new_bins=nn,old_frames=int(motion.clip_start[nold]),new_frames=motion.num_frames,changed_source_files=changed,preserved='models, normalizers, optimizers, schedule, replay, simulator integration, active episodes, observations, RNG, old-bin return estimates/counts',reinitialized='historical sampling average and its update count on the expanded domain; new-bin return/count start at zero',dataset_provenance=json.loads((a.motion_dir/'import_provenance.json').read_text()))
 ck['manifest']=current;ck['args']=vars(args);ck['dataset_extension']=lineage;validate_resume(ck,current)
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(a.output)
 lineage['migrated_checkpoint_sha256']=sha256(a.output);write_json(str(a.output.with_suffix('.json')),lineage)
 print(json.dumps({k:v for k,v in lineage.items() if k!='dataset_provenance'},indent=2),flush=True)

if __name__=='__main__':main()
