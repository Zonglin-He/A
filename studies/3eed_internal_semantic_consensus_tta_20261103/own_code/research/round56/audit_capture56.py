"""Independent official-output audit of all semantic stages; four new source forwards."""
import sys,time,torch,hashlib
from common56 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint,native_matching,SENTINEL
rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');selected=[]
for pl in ['drone','quad']:selected+=sorted([r for r in rows if r['platform']==pl],key=lambda x:hashlib.sha256(('R56_OFFICIAL_STAGE_AUDIT:'+x['id']).encode()).digest())[:2]
start=time.perf_counter();model=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);count=0;error=0.
for row in selected:
 sid=row['id'];l=load(ROOT/'crossplatform/round53/latent336_attempt001/samples_PRIVATE'/(sid+'.npz'),['input_sample_xyz','point_indices']);cap=load(ROOT/'crossplatform/round56/semantic_capture336_attempt001/samples_PRIVATE'/(sid+'.npz'));masks=[torch.as_tensor(m,device='cuda') for m in cap['role_masks']]
 with torch.no_grad():
  z=model({'point_clouds':torch.as_tensor(l['input_sample_xyz'],device='cuda')[None],'text':[row['caption']+SENTINEL]});keys=['proposal_proj_queries']+[str(i)+'head_proj_queries' for i in range(5)]+['last_proj_queries']
  for stage,key in enumerate(keys):
   q=z[key];tokens=z['proj_tokens'];assert torch.equal(q[0],torch.as_tensor(cap['stage_proj_queries'][stage],device='cuda'))
   score=torch.stack([native_matching(q,tokens,m)[1] for m in masks]).cpu().numpy();e=float(abs(score-cap['stage_role_z'][stage]).max());error=max(error,e);assert e==0.;count+=1
assert fingerprint(model)==fp
out=ROOT/'research/round56/capture_audit_attempt001';assert not out.exists();out.mkdir(parents=True);result={'state':'PASS_R56_FOUR_INDEPENDENT_FULL_SOURCE_FORWARDS_ALL_SEVEN_STAGE_SEMANTICS_BITEXACT','new_full_description_forwards':4,'stages_checked':count,'max_role_logodds_error':error,'source_unchanged':True,'seconds':time.perf_counter()-start,'GT_reads':0,'program_sha256':sha(__file__)};dump(out/'AUDIT_RESULT.json',result);print(result,flush=True)
