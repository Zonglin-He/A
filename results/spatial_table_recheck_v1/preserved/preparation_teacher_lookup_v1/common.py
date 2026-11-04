"""Independent whole-list table recheck, preserving all earlier artifacts."""
from pathlib import Path
import os,sys,json,time,hashlib,functools,shutil
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/spatial_table_recheck_v1';P=R/'results/spatial_table_recheck_v1'
sys.path[:0]=[str(R/'studies/reviewer_key_controls_v1'),str(R),str(R/'tools')]
import common_v1 as old
import numpy as np,torch
read=old.read;sha=old.sha;atomic=old.atomic;digest=old.digest;identity=old.identity;fixed=functools.lru_cache(None)(read)
RUN=old.selected.RUN;PURE=R/'outputs/spatial_temporal_six_v1';TEMP=R/'outputs/reviewer_key_controls_v1';COMBO=R/'outputs/reviewer_combo_random_roi_v1'
ARMS=['Source','SourceTemporal','ComboLog','Full','PureLogFixed','ComboLinear','PureLinearFixed'];PAIRS=[('Full','ComboLog'),('Full','PureLogFixed'),('Full','ComboLinear'),('Full','PureLinearFixed')]
COUNTS={'hc2':1902,'vidstg':10303}
@functools.lru_cache(None)
def rows(ds):return old.selected.records(ds)
def resources(start):
 assert time.perf_counter()-start<72*3600 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30

def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for grp in ['code_sha256','input_sha256']:
  for f,h in p[grp].items():assert sha(R/f)==h,f
 return p

def corner(boxes,W=1.,H=1.):
 b=np.asarray(boxes,dtype=np.float64);o=np.empty_like(b);o[:,0]=(b[:,0]-b[:,2]*.5)*W;o[:,1]=(b[:,1]-b[:,3]*.5)*H;o[:,2]=(b[:,0]+b[:,2]*.5)*W;o[:,3]=(b[:,1]+b[:,3]*.5)*H;return o

def rebuild(ids,pos,raw,rule):
 ids=np.asarray(ids,dtype=np.float64);pos=np.asarray(pos,int);raw=np.asarray(raw,dtype=np.float64);at=ids[pos];assert raw.shape==(8,4)and len(set(pos))==8 and np.all(np.diff(at)>0)and np.isfinite(raw).all()and(raw[:,2:]>0).all()
 if rule=='center_logsize':
  a=np.column_stack([np.interp(ids,at,raw[:,k])for k in range(2)]);s=np.exp(np.column_stack([np.interp(ids,at,np.log(raw[:,k]))for k in [2,3]]));out=np.column_stack([a,s])
 else:
  xy=corner(raw);v=np.column_stack([np.interp(ids,at,xy[:,k])for k in range(4)]);out=np.column_stack([(v[:,0]+v[:,2])*.5,(v[:,1]+v[:,3])*.5,v[:,2]-v[:,0],v[:,3]-v[:,1]])
 out[pos]=raw;out[:pos[0]]=raw[0];out[pos[-1]+1:]=raw[-1];assert np.array_equal(out[pos],raw)and np.array_equal(out[:pos[0]],np.repeat(raw[:1],pos[0],axis=0))and np.array_equal(out[pos[-1]+1:],np.repeat(raw[-1:],len(ids)-pos[-1]-1,axis=0));return out

def context_key(ds,caption,ids,pos):return digest(dict(dataset=ds,caption=caption,frame_ids=list(map(int,ids)),positions=list(map(int,pos))))

def inputs(ds,row):
 key=row['id'];a=fixed(RUN/ds/'sealed.json')['files'][key];b=fixed(TEMP/'temporal'/ds/'sealed.json')['files'][key];c=fixed(PURE/'pure_full'/ds/'sealed.json')['files'][key];z=fixed(COMBO/'combination'/ds/'sealed.json')['files'][key];inventory=fixed(D/'teacher_inventory.json')[ds][key]
 for root,e in [(RUN/ds,a),(TEMP,b),(PURE,c),(COMBO,z)]:assert sha(root/e['path'])==e['sha256']and sha(root/e['receipt_path'])==e['receipt_sha256']
 tp=Path(inventory['payload']);rp=Path(inventory['receipt']);assert sha(tp)==inventory['payload_sha256']and sha(rp)==inventory['receipt_sha256'];tm=read(rp);assert tm['sha256']==inventory['payload_sha256'];teacher=torch.load(tp,map_location='cpu',weights_only=False,mmap=True);raw=np.asarray(teacher['boxes'],dtype=np.float64)
 with np.load(RUN/ds/a['path'])as original,np.load(TEMP/b['path'])as temporal,np.load(PURE/c['path'])as pure,np.load(COMBO/z['path'])as combo:
  ids=original['frame_ids'].copy();pos=original['positions8'].copy();assert context_key(ds,row['metadata']['caption'],ids,pos)==inventory['context_without_RGB_key'];assert np.array_equal(raw,original['native_boxes'])
  for q in [temporal,pure,combo]:assert np.array_equal(ids,q['frame_ids'])and np.array_equal(pos,q['positions8'])
  assert np.array_equal(temporal['Ours'],pure['Current'])and np.array_equal(temporal['Ours_interval'],pure['Current_interval'])and np.array_equal(temporal['SourceTR_interval'],combo['SimpleCombination_interval'])and np.array_equal(pure['PureMatched'],combo['SimpleCombination'])and np.array_equal(temporal['Source'],pure['Source'])
  log=rebuild(ids,pos,raw,'center_logsize');linear=rebuild(ids,pos,raw,'corner_linear');delta=float(np.max(np.abs(log-pure['PureMatched'])));assert delta<2e-14,delta
  arrays=dict(frame_ids=ids,positions8=pos,raw_expert_boxes=raw,Source=temporal['Source'].copy(),SourceTemporal=temporal['SourceTR'].copy(),ComboLog=log,Full=temporal['Ours'].copy(),PureLogFixed=log.copy(),ComboLinear=linear,PureLinearFixed=linear.copy(),E_observed=original['SharedInit'][pos].copy())
  assert np.array_equal(arrays['Source'],arrays['SourceTemporal'])
  for arm in ARMS:arrays[arm+'_interval']=(temporal['Source_interval']if arm=='Source'else temporal['SourceTR_interval']if arm in ['SourceTemporal','ComboLog','ComboLinear']else temporal['Ours_interval']).copy()
  assert all(np.array_equal(arrays[a+'_interval'],arrays['Full_interval'])for a in ['PureLogFixed','PureLinearFixed'])and all(np.array_equal(arrays[a+'_interval'],arrays['SourceTemporal_interval'])for a in ['ComboLog','ComboLinear'])
  full=arrays['Full'][pos];e=arrays['E_observed'];flags=np.all(full==raw,axis=1);eqE=np.all(full==e,axis=1);assert(flags|eqE).all();inside=(ids[pos]>=arrays['Full_interval'][0])&(ids[pos]<arrays['Full_interval'][1]);proof=dict(old_production_sha256=a['sha256'],old_temporal_sha256=b['sha256'],old_full_pure_sha256=c['sha256'],old_simple_combo_sha256=z['sha256'],raw_teacher_payload_sha256=inventory['payload_sha256'],raw_teacher_receipt_sha256=inventory['receipt_sha256'],raw_RGB_context_sha256=tm['context']['RGB_sha256'],all8_raw_payload_boxes_exact_production=True,independent_log_interp_max_coordinate_difference=delta,all_query_grid_positions_and_reference_arrays_exact=True,common_Full_interval_exact=True,own_SourceTR_combo_interval_exact=True,observed_full_rawexpert_count=int(flags.sum()),observed_full_E_not_expert_count=int((~flags&eqE).sum()),inside_event_E_not_expert_count=int((inside&~flags&eqE).sum()),GT_read=False)
 return arrays,proof
