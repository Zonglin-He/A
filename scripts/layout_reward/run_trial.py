"""Finite HC128/Vid128 layout trial, isolated from paused formal prediction."""
import os,sys,json,time,hashlib,gc,subprocess,contextlib,tempfile
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['STVG_PROJECT_ROOT']).resolve()
TRIAL=HERE.parent
from stvg.runtime import setup,lease,load_source
from stvg.inputs import validate_input
from stvg.decode import decode
from stvg.query import QuerySubjectParser
from stvg.source import frozen_forward
from stvg.tensors import detached
from stvg.readout import prediction
from stvg.state import Replay
from stvg.initialization import initialize
from stvg.predictor import output_writeback
from stvg.layout_reward import LayoutReward
from stvg.layout_fit import fit as layout_fit
from stvg.policy import fit as geometry_fit
from stvg.teacher_bridge import ResidentTeacher

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(z,indent=2)+'\n');os.replace(t,p)
def save(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');torch.save(z,t);os.replace(t,p)

class FeatureTeacher(ResidentTeacher):
 def _start(self):
  if self.process is None:
   self.log=(Path(self.directory.name)/'teacher.log').open('wb')
   self.process=subprocess.Popen([sys.executable,'-B','-m','stvg.layout_teacher_worker'],cwd=HERE,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))

def capture(model,record,parser,teacher,cfg,gold):
 meta=record['metadata'];frames,ids=decode(meta,record['dataset']);validate_input(frames,ids,meta)
 torch.manual_seed(20260920);np.random.seed(20260920);model.cuda()
 row=dict(input=meta,frame_ids=ids,parses=dict(subject=parser(meta['caption'])['subject']))
 start=time.perf_counter();batch,recs,base=frozen_forward(model,frames,row)
 native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids);source=native['boxes'].cpu();spatial=detached(base.spatial_inputs,'cpu')
 assert np.array_equal(source.numpy(),gold['Source']), 'Source changed'
 from stvg.positions import observation_positions
 positions=observation_positions(ids,native['indices']);assert positions==gold['positions8'].tolist()
 del batch,base;model.cpu();gc.collect();torch.cuda.empty_cache();source_seconds=time.perf_counter()-start
 start=time.perf_counter();t=teacher(frames[positions],meta['caption'],positions=positions,frame_ids=ids,dataset=record['dataset'])
 raw,init=initialize(source.double().numpy(),positions,np.asarray(t['boxes'],dtype=np.float64),cfg['E_calibration'])
 E=output_writeback(source.numpy(),raw,cfg['initialization_gamma']);assert np.array_equal(E,gold['E_Ref']), 'Teacher/initialization changed'
 return dict(Source=source,E=E,spatial_inputs=spatial,planned_positions=positions,teacher=t,frame_ids=ids,interval=gold['Strong_interval'],
             cost=dict(source_capture_seconds=source_seconds,teacher_wall_seconds=time.perf_counter()-start,teacher=t['cost']),GT_read=False)

def main(limit=None):
 setup();lock=lease();cfg=read(TRIAL/'config.json');manifest=ROOT/'data/evaluation/development.jsonl';assert sha(manifest)=='25bb82e71489ec35aca2e821dde6330d07b29107d551300922f51abe38a10f4b'
 ref=ROOT/'outputs/reference';seal=read(ref/'sealed.json');rows=[json.loads(s)for s in manifest.read_text().splitlines()if s.strip()]
 assert len(rows)==256
 # Verify the complete matched reference before any work; no labels are opened.
 for row in rows:
  p=ref/seal['files'][row['id']]['path'];assert sha(p)==seal['files'][row['id']]['sha256']
 fingerprint=hashlib.sha256((TRIAL/'config.json').read_bytes()+b''.join(p.read_bytes()for p in sorted((HERE/'stvg').glob('*.py')))+(HERE/'run_trial.py').read_bytes()).hexdigest()
 atomic(TRIAL/'run.json',dict(status='running',fingerprint=fingerprint,manifest_sha256=sha(manifest),GT_read=False,formal_status='paused_by_user',queries=256))
 parser=QuerySubjectParser(HERE/'checkpoints/stanza');teacher=FeatureTeacher(TRIAL/'teacher_cache',fingerprint=fingerprint)
 completed=0
 try:
  for ds in ['hc2','vidstg']:
   model=None
   for record in [r for r in rows if r['dataset']==ds]:
    key=record['id'];dest=TRIAL/'predictions'/(key+'.npz');receipt=dest.with_suffix('.json')
    if receipt.exists():
     rr=read(receipt);assert rr['fingerprint']==fingerprint and sha(dest)==rr['sha256'];completed+=1;continue
    if limit is not None and completed>=limit:break
    start=time.perf_counter();p=ref/seal['files'][key]['path']
    with np.load(p,allow_pickle=False)as z:gold={k:z[k].copy()for k in z.files}
    cachepath=TRIAL/'student_cache'/(key+'.pt');cc=TRIAL/'student_cache'/(key+'.json')
    if cachepath.exists():
     assert cc.exists() and read(cc)['fingerprint']==fingerprint and sha(cachepath)==read(cc)['sha256'];cache=torch.load(cachepath,map_location='cpu',weights_only=False)
    else:
     if model is None:model=load_source(cfg['datasets'][ds]);model.cpu()
     cache=capture(model,record,parser,teacher,cfg['datasets'][ds],gold);save(cachepath,cache);atomic(cc,dict(fingerprint=fingerprint,sha256=sha(cachepath),GT_read=False))
    if model is None:model=load_source(cfg['datasets'][ds]);model.cpu()
    model.ground_decoder.decoder.cuda();model.ground_decoder.decoder.eval().requires_grad_(False)
    feature=cache['teacher'];reward=LayoutReward(feature['layout_features'],feature['layout_mask'],feature['boxes'])
    # Old geometry result is reused only after exact Source and E/position checks.
    result=dict(Source=gold['Source'],E_Ref=gold['E_Ref'],GeometryOPD=gold['Live'],frame_ids=gold['frame_ids'],positions8=gold['positions8'],Strong_interval=gold['Strong_interval'])
    costs={};traces={}
    for arm in ['LayoutOPD','LayoutDirect']:
     out=layout_fit(model,cache,record['key'],arm,cfg['datasets'][ds],reward)
     assert torch.equal(out['path'][0]['boxes'],torch.as_tensor(cache['E']))
     result[arm]=out['final'].numpy();costs[arm]=dict(seconds=out['seconds'],updates=out['actual_updates']);traces[arm]=out
    tracepath=TRIAL/'traces'/(key+'.pt');save(tracepath,traces)
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.npz.tmp')
    with tmp.open('wb')as h:np.savez(h,**result)
    os.replace(tmp,dest);atomic(receipt,dict(fingerprint=fingerprint,sha256=sha(dest),reference_sha256=sha(p),trace_sha256=sha(tracepath),costs=costs,capture_cost=cache['cost'],total_seconds=time.perf_counter()-start,GT_read=False))
    completed+=1;atomic(TRIAL/'progress.json',dict(completed=completed,total=256,id=key,dataset=ds));print(json.dumps(dict(completed=completed,total=256,id=key,dataset=ds,seconds=time.perf_counter()-start)),flush=True)
    del reward,cache,result,traces,out,feature;model.cpu();gc.collect();torch.cuda.empty_cache()
   if model is not None:del model;gc.collect();torch.cuda.empty_cache()
   if limit is not None and completed>=limit:break
  if completed==256:
   files={r['id']:dict(path='predictions/'+r['id']+'.npz',sha256=sha(TRIAL/'predictions'/(r['id']+'.npz')))for r in rows}
   atomic(TRIAL/'sealed.json',dict(status='sealed_before_offline_scoring',GT_read=False,fingerprint=fingerprint,queries=256,manifest_sha256=sha(manifest),files=files))
   atomic(TRIAL/'run.json',dict(status='completed_sealed',fingerprint=fingerprint,manifest_sha256=sha(manifest),GT_read=False,queries=256))
 finally:teacher.close();lock.close()

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();main(a.limit)
