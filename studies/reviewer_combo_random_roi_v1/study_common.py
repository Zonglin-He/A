"""Two CPU-only reviewer additions; all previous science and results immutable."""
from pathlib import Path
import os,sys,json,time,hashlib,shutil,functools,ast,math
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/reviewer_combo_random_roi_v1';P=R/'results/reviewer_combo_random_roi_v1'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path[:0]=[str(R/'studies/reviewer_key_controls_v1'),str(R),str(R/'tools')]
import common_v1 as legacy
import numpy as np,torch
from stvg.representation import extract,pool,weights
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval
read=legacy.read;sha=legacy.sha;atomic=legacy.atomic;digest=legacy.digest;identity=legacy.identity;fixed=functools.lru_cache(None)(read);selected=legacy.selected
OLD=R/'outputs/reviewer_key_controls_v1';SUB=OLD/'subset128_v1';PURE=R/'outputs/spatial_temporal_six_v1/pure_full'
COUNTS={'hc2':1902,'vidstg':10303}
MAIN=['Source','SourceTemporal','SimpleCombination','InitializedTemporal','Full']
MPAIRS=[('Full','SimpleCombination'),('SimpleCombination','SourceTemporal'),('Full','InitializedTemporal'),('InitializedTemporal','SourceTemporal'),('SourceTemporal','Source')]
REGIONS=['Source','SourceROI','CorrectedROI','GlobalAppearance','BackgroundAppearance','ShuffledBoxes','RandomRegion']
RPAIRS=[('CorrectedROI',x)for x in REGIONS if x not in ['Source','CorrectedROI']]+[('RandomRegion','SourceROI')]
_p=R/'studies/pure_expert_interpolation_v1/run.py';_t=_p.read_text();_n=next(x for x in ast.parse(_t).body if isinstance(x,ast.FunctionDef)and x.name=='pure_interpolation');_ns={'np':np};exec(compile(ast.get_source_segment(_t,_n),str(_p),'exec'),_ns);pure=_ns['pure_interpolation']
def rows(task,ds):return selected.records(ds)if task=='combination'else fixed(SUB/'selection.json')[ds]
def resources(start):
 assert time.perf_counter()-start<72*3600 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
def guard():legacy.infer_guard()
def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 return p
def old_file(root,entry,key='path',hashkey='sha256'):
 p=root/entry[key];assert sha(p)==entry[hashkey],p;return p
def combination(ds,row):
 key=row['id'];t=fixed(OLD/'temporal'/ds/'sealed.json')['files'][key];p=fixed(PURE/ds/'sealed.json')['files'][key]
 tp=old_file(OLD,t);pp=old_file(PURE.parent,p);assert sha(OLD/t['receipt_path'])==t['receipt_sha256']and sha(PURE.parent/p['receipt_path'])==p['receipt_sha256']
 with np.load(tp)as z,np.load(pp)as q:
  assert np.array_equal(z['frame_ids'],q['frame_ids'])and np.array_equal(z['positions8'],q['positions8'])and np.array_equal(z['Ours'],q['Current'])and np.array_equal(z['Ours_interval'],q['Current_interval'])
  assert np.array_equal(z['ExpertInitTR_interval'],z['Ours_interval'])
  a=dict(frame_ids=z['frame_ids'].copy(),positions8=z['positions8'].copy(),SimpleCombination=q['PureMatched'].copy(),SimpleCombination_interval=z['SourceTR_interval'].copy())
 return a,dict(temporal_prediction_sha256=t['sha256'],temporal_receipt_sha256=t['receipt_sha256'],pure_prediction_sha256=p['sha256'],pure_receipt_sha256=p['receipt_sha256'],strict_raw8_nearest_tail_pure=True,Source_only_temporal_interval=True,Full_initialized_intervals_exact=True,GT_read=False)
def region_inputs(ds,row):
 e=fixed(SUB/ds/'sealed.json')['files'][row['id']];cp=old_file(SUB,e,'compact_path','compact_sha256');assert sha(SUB/e['receipt_path'])==e['receipt_sha256'];c=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
 rawinfo=fixed(SUB/'raw_inventory.json')[ds][row['id']];assert sha(rawinfo['path'])==rawinfo['sha256'];raw=torch.load(rawinfo['path'],map_location='cpu',weights_only=False,mmap=True);assert list(map(int,raw['frame_ids']))==list(map(int,c['ids']))and not raw['GT_read']
 return c,raw,dict(old_compact_sha256=e['compact_sha256'],old_receipt_sha256=e['receipt_sha256'],raw_grid_sha256=rawinfo['sha256'])
def translated_support(target,valid,token):
 yy,xx=np.nonzero(target>0);assert len(xx)>0;top,bottom,left,right=int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1;template=target[top:bottom,left:right];h,w=template.shape
 windows=np.lib.stride_tricks.sliding_window_view(valid,(h,w));feasible=np.all(windows|~(template>0),axis=(-2,-1));ay,ax=np.nonzero(feasible);assert len(ax)>0
 index=int.from_bytes(hashlib.sha256(token.encode()).digest(),'big')%len(ax);y,x=int(ay[index]),int(ax[index]);out=np.zeros_like(target);out[y:y+h,x:x+w]=template
 assert np.array_equal(np.sort(target.ravel()),np.sort(out.ravel()))and math.fsum(target.ravel())==math.fsum(out.ravel())and not np.any((out>0)&~valid)
 return out,dict(top=y,left=x,height=h,width=w,feasible_placements=len(ax),identity=bool(np.array_equal(out,target)),row_displacement=y-top,column_displacement=x-left,area=math.fsum(target.ravel()),overlap_fraction=float(np.minimum(target,out).sum()/target.sum()))
def random_features(ds,row,c,raw,manifest=None):
 result=np.empty_like(c['region_features']['CorrectedROI']);details=[]
 for off,(grid,valid)in enumerate(raw['native_local_grid']):
  x=grid.permute(0,2,3,1).numpy()
  for j in range(len(x)):
   t=off+2*j;v=valid[j].numpy();support,fallback=weights(c['E'][t],*v.shape,v);token=f'20261104|{ds}|{row["id"]}|{int(c["ids"][t])}';s,d=translated_support(support,v,token);d.update(frame_index=t,physical_frame_id=int(c['ids'][t]),target_global_fallback=bool(fallback));details.append(d);result[t]=pool(x[j],s)
 details.sort(key=lambda x:x['frame_index']);assert len(details)==len(c['ids'])
 if manifest is not None:assert details==manifest
 return result,details
def region(ds,row):
 c,raw,proof=region_inputs(ds,row);f,manifest=random_features(ds,row,c,raw,fixed(D/'random_manifest.json')[ds][row['id']]);original,_=extract(raw,c['E']);assert np.array_equal(original,c['region_features']['CorrectedROI'])
 q,detail=one_step(f,c['q0'],kappa=0,beta=1);iv=np.asarray(map_interval(c['ids'],q));a=dict(frame_ids=np.asarray(c['ids']),positions8=np.asarray(c['pos']),RandomRegion=c['E'].copy(),RandomRegion_interval=iv,RandomRegion_evidence=detail['L'].copy())
 compact=dict(features=f,q0=c['q0'].copy(),E=c['E'].copy(),ids=list(c['ids']),pos=list(c['pos']),GT_read=False)
 proof.update(random_manifest_sha256=sha(D/'random_manifest.json'),exact_weight_multiset_area_shape_each_frame=True,spatial_output_fixed_same_E=True,features_changed=not np.array_equal(f,original),identity_frames=sum(d['identity']for d in manifest),frames=len(manifest),mean_overlap_fraction=float(np.mean([d['overlap_fraction']for d in manifest])),target_fallback_frames=sum(d['target_global_fallback']for d in manifest),GT_read=False)
 return a,compact,proof
def replay_random(c):
 q,d=one_step(c['features'],c['q0'],kappa=0,beta=1);return dict(frame_ids=np.asarray(c['ids']),positions8=np.asarray(c['pos']),RandomRegion=c['E'].copy(),RandomRegion_interval=np.asarray(map_interval(c['ids'],q)),RandomRegion_evidence=d['L'].copy())
