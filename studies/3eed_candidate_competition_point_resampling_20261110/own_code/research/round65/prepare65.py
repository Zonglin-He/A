"""Bind raw XYZ and freeze all policies/indices before model or task scoring."""
from common65 import *
import sampling65 as s,time
p=read(ROOT/'research/round65/experiment/EXPERIMENT_SPEC_LOCK.json');pin(p);out=ROOT/p['prepared_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();rows=read(ROOT/p['manifest']);roles={r['id_PRIVATE']:np.array(r['mask'],bool) for r in read(ROOT/'research/round62/role_diagnostics_attempt001/MASKS_PRIVATE.json') if r['role']=='reference'};stats=[];start=time.perf_counter()
for i,r in enumerate(rows):
 path=Path(r['pointpath']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,np.float32).reshape(-1,4);x=np.array(raw[:,:3],np.float32,copy=True);assert np.isfinite(x).all() and 0<len(x)<16384
 if r['platform']=='drone':x[:,2]+=1.8
 z=load(ROOT/p['capture_root']/'samples_PRIVATE'/(r['id']+'.npz'),['A_boxes','A_scores','A_query','A_tokens','A_point_indices']);d=s.design(x,z,roles[r['id']]);payload={'focus_indices':d['focus_union'],'context_indices':d['context_union'],'target_hypothesis_indices':np.array(d['targets'],np.int64),'reference_hypothesis_indices':np.array(d['refs'],np.int64)};rec={'raw_sha256':sha(path),'raw_n':len(x),'reference_available':bool(roles[r['id']].any()),'target_hypotheses':len(d['targets']),'reference_hypotheses':len(d['refs']),'raw_A_unique_n':len(np.unique(z['A_point_indices'])),'views':{}}
 for mode in p['sampling_modes']:
  for v in p['new_views']:
   ix,fallback=s.draw(len(x),d,r['id'],v,mode);payload[mode+'_'+v+'_point_indices']=ix;rec['views'][mode+'_'+v]={'global_fallback_draws':fallback,'seed':s.seed(r['id'],v),'unique_n':len(np.unique(ix))}
 np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),**payload);dump(out/'cases_PRIVATE'/(r['id']+'.json'),rec);stats.append({'platform':r['platform'],'raw_n':len(x),'A_unique_n':rec['raw_A_unique_n'],'reference_available':rec['reference_available']})
 if (i+1)%64==0:print('PREPARE65',i+1,flush=True)
dump(out/'RAW_COUNTS_PUBLIC_SAFE.json',stats);dump(out/'COSTS.json',{'n':len(rows),'seconds':time.perf_counter()-start,'task_labels_read':False});seal(out,'R65_ALL_RAW_BINDINGS_AND_PAIRED_SAMPLE_INDICES_SEALED_NO_GT',n=336)
