"""Independent fixed-parent corruption study for the selected E-ROI method."""
import os,sys,json,time,hashlib,shutil,gc,copy,importlib.util
from pathlib import Path
from functools import lru_cache
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R/'studies/shared_corrected_roi_v1'),str(R)]
import shared_common as selected
from shared_common import read,sha,atomic,identity,digest
parent=selected.parent
D=R/'outputs/corruption_subset_shared_roi_v1';P=R/'results/corruption_subset_shared_roi_v1'
COUNTS={'hc2':128,'vidstg':128};fixed=lru_cache(None)(read)
spec=importlib.util.spec_from_file_location('_subset_original_corruption_operators',R/'studies/corruption_full_v1/operators.py');operators=importlib.util.module_from_spec(spec);spec.loader.exec_module(operators)
conditions=operators.conditions
def rows(ds):return fixed(parent.OLD/'selection.json')[ds]['confirm']
def unit(ds,name):
 assert ds in COUNTS and name in {c['id']for c in conditions()}
 return D/'units'/name/ds
def condition(name):return next(c for c in conditions()if c['id']==name)
def validate():
 p=fixed(D/'plan.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 selected.validate();return p
def resources(start):
 assert time.perf_counter()-start<259200
 assert shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<160*2**30
def save(path,value):
 import torch
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);os.replace(tmp,path)
def old_entry(ds,key,view):
 import torch
 e=fixed(parent.PREV/'corruptions/sealed.json')['files'][ds][key][view]
 for f,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(parent.PREV/e[f])==e[h]
 c=torch.load(parent.PREV/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);assert not c['GT_read'];return c,e
def clean(ds,row):
 import numpy as np
 e=fixed(selected.D/f'components_{ds}_sealed.json')['files'][row['id']]
 for f,h in [('path','sha256'),('receipt_path','receipt_sha256')]:assert sha(selected.D/e[f])==e[h]
 with np.load(selected.D/e['path'])as z:a={k:z[k].copy()for k in ['Source','Source_interval','FullMAP','FullMAP_interval','frame_ids','positions8']}
 return a,dict(prediction_path=str(selected.D/e['path']),prediction_sha256=e['sha256'],receipt_path=str(selected.D/e['receipt_path']),receipt_sha256=e['receipt_sha256'],inherited_original_acquisition_nonzero=True)
def replay(c):
 import numpy as np
 if c['kind']=='clean_identity':
  a,proof=clean(c['dataset'],next(r for r in rows(c['dataset'])if r['id']==c['query']));assert proof==c['clean_proof'];return a
 from stvg.prototype import one_step
 from stvg.current_shared_roi import map_interval
 from stvg.spatial import SpatialGeometry,evidence_dp
 q,_=one_step(c['E_ROI_features'],c['q0'],kappa=0,beta=1);iv=map_interval(c['frame_ids'],q);g=SpatialGeometry.from_cached_geometry(c['geometry'])
 return dict(Source=c['Source'].copy(),Source_interval=np.asarray(c['Source_interval']),FullMAP=evidence_dp(g.for_interval(iv))['final'].numpy(),FullMAP_interval=np.asarray(iv),frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']))
def from_capture(cap,native):
 import numpy as np
 from stvg.representation import extract
 f,_=extract(cap,np.asarray(cap['E']))
 return dict(kind='acquired',Source=np.asarray(cap['Source']).copy(),Source_interval=native,frame_ids=cap['frame_ids'],positions8=cap['planned_positions'],geometry=cap['geometry'].state_dict(),q0=cap['posterior']['Native'],E_ROI_features=f,GT_read=False,first_aux_DP_in_selected_path=False)
def from_legacy(ds,key,view,raw):
 import numpy as np
 from stvg.spatial import SpatialGeometry,evidence_dp
 from stvg.representation import extract
 from stvg.prototype import one_step
 from stvg.current_shared_roi import map_interval
 old,e=old_entry(ds,key,view);ids=old['frame_ids'];assert raw['frame_ids']==ids
 g=SpatialGeometry.from_cached_geometry(old['geometry']);aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();f,_=extract(raw,aux);assert np.array_equal(f,old['content_features']),'Original corrupted auxiliary ROI pool must be exact'
 q,_=one_step(f,old['native_prior'],kappa=64,beta=1);iv=map_interval(ids,q)
 with np.load(parent.PREV/e['path'])as z:
  assert np.array_equal(z['Source'],old['Source'])and np.array_equal(z['FullMAP_interval'],iv)and np.array_equal(z['FullMAP'],evidence_dp(g.for_interval(iv))['final'].numpy())
 f,_=extract(raw,g.E.numpy())
 c=dict(kind='legacy_prefix',Source=np.asarray(old['Source']).copy(),Source_interval=old['Source_interval'],frame_ids=ids,positions8=old['positions8'],geometry=old['geometry'],q0=old['native_prior'],E_ROI_features=f,GT_read=False,first_aux_DP_in_selected_path=False)
 return c,dict(old_prediction_sha256=e['sha256'],old_compact_sha256=e['compact_sha256'],old_receipt_sha256=e['receipt_sha256'],old_aux_pool_all_arrays_exact=True,new_actual_E_ROI_pool=True,compatibility_only_first_aux_DP=True)
class Teacher:
 def __init__(self,root):
  from stvg.teacher_bridge import ResidentTeacher
  self.root=root;self.fingerprint=fixed(parent.OLD/'shifts_plan.json')['runtime_fingerprint'];self.live=ResidentTeacher(cache_dir=root/'teacher_cache',fingerprint=self.fingerprint);self.last=None
 def __call__(self,frames,caption,**kw):
  import torch
  from stvg.teacher_bridge import evidence_key
  key,context=evidence_key(frames,caption,fingerprint=self.fingerprint,**kw)
  for cache in [parent.OLD/'shift_teacher_cache',parent.PREV/'corruptions/teacher_cache']:
   rp=cache/(key+'.json');pp=cache/(key+'.pt')
   if rp.exists():
    rc=read(rp);assert rc['key']==key and rc['context']==context and sha(pp)==rc['sha256']
    z=copy.deepcopy(torch.load(pp,map_location='cpu',weights_only=False));cost=copy.deepcopy(z['cost']);z['evidence_original_cost']=cost;z['cost']={k:0. for k in cost};z.update(cache_hit=True,cache_key=key,model_load_seconds=0.,process_wall_seconds=0.,query_forwards=0,peak_GPU_bytes=0)
    self.last=dict(path=str(pp),sha256=sha(pp),receipt=str(rp),receipt_sha256=sha(rp),owned=False,original_cost=cost);return z
  z=self.live(frames,caption,**kw);pp=self.root/'teacher_cache'/(key+'.pt');rp=pp.with_suffix('.json')
  assert sha(pp)==read(rp)['sha256'];self.last=dict(path=str(pp),sha256=sha(pp),receipt=str(rp),receipt_sha256=sha(rp),owned=True,original_cost=z.get('evidence_original_cost',z['cost']));return z
 def close(self):
  # Preserve own resident request/reply/log before its original temporary cleanup.
  dest=self.root/'teacher_runtime_preserved';dest.mkdir(parents=True,exist_ok=True)
  if self.live.log:self.live.log.flush()
  files={}
  for f in Path(self.live.directory.name).glob('*'):
   if f.is_file():shutil.copy2(f,dest/f.name);files[f.name]=sha(dest/f.name)
  atomic(dest/'preservation.json',dict(files=files,actual_teacher_pid=self.live.process.pid if self.live.process else None,before_original_temp_cleanup=True))
  self.live.close()
def checkpoint_class():
 import inspect,textwrap
 import stvg.predictor as old
 from stvg.current_shared_roi import map_interval
 code=textwrap.dedent(inspect.getsource(old.BoxFixedPredictor.capture));needle="native_interval=final_mbr(U,q)['interval']";assert code.count(needle)==1;code=code.replace(needle,'native_interval=map_interval(U,q)')
 needle="    native=prediction(base.zero['logits'],base.zero['boxes'],recs,frame_ids)";assert code.count(needle)==1
 code=code.replace(needle,"    self.full_checkpoint(dict(zero=detached(base.zero,'cpu'),spatial_inputs=detached(base.spatial_inputs,'cpu'),records=recs,native_local_grid=native_grid,frame_ids=list(frame_ids),GT_read=False))\n"+needle)
 needle='    if teacher_override is not None:';assert code.count(needle)==1
 code=code.replace(needle,"    self.processed_checkpoint(dict(Source=src.numpy(),frame_ids=U,planned_positions=positions,spatial_inputs=spatial,native_logits=logits,native_local_grid=native_grid,records=recs,posterior={'Native':q},GT_read=False))\n"+needle)
 scope={**vars(old),'map_interval':map_interval};exec(compile(code,'<subset_selected_capture_checkpoint_before_validation_and_teacher>','exec'),scope)
 return type('SubsetCheckpointedSelected',(old.BoxFixedPredictor,),{'capture':scope['capture']})
