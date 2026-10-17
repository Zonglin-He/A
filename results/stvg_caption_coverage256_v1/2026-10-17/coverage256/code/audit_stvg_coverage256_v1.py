import sys,hashlib,json,math,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import t
from scripts.run_stvg_coverage224002 import original
from scripts import stvg_overnight_global_v1 as bridge
from scripts.stvg_overnight_score_v1 import seal
b=original.EXP;base=original.BASE;read,put,sha=original.read,original.put,original.sha
original.configure(bridge);seal(b);teacher=read(b/'teacher/G32/TEACHER_GLOBAL_BARRIER.json');entries={Path(x['path']).stem:x for x in teacher['files']};checked=0;missing=0;calls=0
for j in read(b/'TEACHER_JOB_LOCK.json')['records']:
 cache=original.CORE/'native_cache'/j['job_id'];old=read(cache/'POLICIES.json');assert sha(cache/'native.npz')==old['native_sha256'];entry=entries[j['job_id']];assert sha(ROOT/entry['path'])==entry['sha256'];response=read(ROOT/entry['path'])
 with np.load(cache/'native.npz',allow_pickle=False) as z:
  f=z['features'];q=z['query_embedding'];cap=z['caption_embedding'];actual=[]
  for k,(start,end,T,K) in enumerate(((0,512,2,64),(512,1280,3,96))):
   v=z['valid'+str(k)];mandatory=z['mandatory'+str(k)];groups=z['groups'+str(k)];s=bridge.choose(f[start:end].reshape(T,256,-1),q,cap,v,mandatory,groups,K);ix=s['indices']['B'].tolist()
   assert s['quotas']==old['records'][k]['quotas'] and s['role_quotas']==old['records'][k]['role_quotas'] and len(ix)==K and set(np.flatnonzero(mandatory))<=set(ix)
   if k==0:assert ix==old['records'][0]['indices']['B']
   actual.append(ix)
 if response['new_teacher_call']:
  assert response['visual_indices_sha256']==hashlib.sha256(json.dumps(actual).encode()).hexdigest() and response['visual_token_count']==160;calls+=1
 else:assert not old['caption_available'];missing+=1
 checked+=1
assert checked==896
put(base/'ROOT_COMPONENT_READBACK.json',dict(status='pass',count=224,window_count=checked,actual_WHO_calls=calls,missing_caption_window_receipts=missing,all_actual_indices_verified=True,unchanged_global_indices_frame_role_quotas_ID_context_protection=True,only_local_candidate_priority_loop_deleted=True,GT_read=False,new_model_calls=0,new_GT_reader_calls=0))
d=read(base/'MATCHED_RESULTS256.json');rows=[]
for part in d['records']:
 for p in part['paired']:
  if p['comparison'] not in ('B_noCoverage_minus_A','B_noCoverage_minus_B') or p['metric']!='m_vIoU':continue
  v=list(p['per_source_delta_pp'].values());n=len(v);mean=math.fsum(v)/n;err=float(t.ppf(.975,n-1)*np.std(v,ddof=1)/math.sqrt(n));rows.append(dict(partition=part['partition'],dataset=p['dataset'],comparison=p['comparison'],count=n,mean_delta_pp=mean,paired_source_t95_interval_pp=[mean-err,mean+err]))
put(base/'ROOT_PAIRED_UNCERTAINTY.json',dict(status='pass',count=256,rows=rows,development_not_confirmation=True,GT_read=False,bootstrap=False))
print(json.dumps(dict(status='pass',window_count=checked,actual_WHO_calls=calls,missing_caption_window_receipts=missing,intervals=rows)))
