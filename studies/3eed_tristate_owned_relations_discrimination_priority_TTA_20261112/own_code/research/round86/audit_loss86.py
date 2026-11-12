"""Independent actual full+masked candidate/recovery endpoint checks, pre-GT."""
from core86 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source
start=time.perf_counter();P=B86/'prediction366_attempt001';assert read(H86/'independent_native_audit_attempt001/AUDIT_RESULT.json')['prediction_seal_sha256']==sha(P/'SEALED.json');base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));torch.set_num_threads(4);m=Fusion85(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=load(ROOT/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz');worst=0.;checks=0;priority_up=0;sum_up=0
with torch.no_grad():
 for nr,r in enumerate(read(H86/'experiment/ALL366_PRIVATE.json')):
  sid=r['id'];a=load(P/'samples_PRIVATE'/(sid+'.npz'));rec=read(P/'records_PRIVATE'/(sid+'.json'));_,z,d=item80(r);ev=load(B86/'evidence366_attempt002/samples_PRIVATE'/(sid+'.npz'));pm=[torch.tensor(v,device='cuda') for v in a['mask_point']];im=[torch.tensor(v,device='cuda') for v in a['mask_image']];ws=local_weights85(d,pm,im,{'point':ev['region_point'],'image':ev['region_image']});restore(m,S)
  for k,(pp,ii) in enumerate([mask86(r,d,{'point':ev['region_point'],'image':ev['region_image']},v) for v in [86001,86002]]):assert torch.equal(pp,pm[k]) and torch.equal(ii,im[k])
  baseline=[m.main(base,z,d,'bidirectional')[0]]+[m.main(base,z,d,'bidirectional',point_mask=p,image_mask=i)[0] for p,i in zip(pm,im)]
  for b in ['S','F']:
   if not rec['bases'][b]['active']:continue
   s0=torch.tensor(a[b+'_s0'],device='cuda')
   for variant in ['candidate','joint','mismatch','sum']:
    pi=torch.tensor(a[b+('_mismatch_pi' if variant=='mismatch' else '_pi')],device='cuda');st=rec['bases'][b]['variants'][variant][0];restore(m,S);v0=objective86(m,base,z,d,s0,pi,baseline,pm,im,ws);put_patch(m,load(P/'states_PRIVATE'/(sid+'_'+b+'_'+variant+'1.npz')));v1=objective86(m,base,z,d,s0,pi,baseline,pm,im,ws)
    for actual,key in [(v0[0],'total_before'),(v0[2],'classification_before'),(v0[3],'recovery_before'),(v1[0],'total_after'),(v1[2],'classification_after'),(v1[3],'recovery_after')]:worst=max(worst,abs(float(actual)-st[key]));checks+=1
    assert worst<1e-10
    if st['accepted'] and variant=='joint':priority_up+=int(float(v1[2])>float(v0[2])+1e-10);assert float(v1[0])<=float(v0[0])+1e-10
    if st['accepted'] and variant=='sum':sum_up+=int(float(v1[2])>float(v0[2])+1e-10)
    restore(m,S)
  if (nr+1)%64==0:print('LOSS_AUDIT86',nr+1,flush=True)
assert priority_up==0;O=H86/'independent_loss_replay_audit_attempt001';O.mkdir();result={'state':'PASS_R86_ACTUAL_NATIVE_FIRST_STEP_FULL_MASKED_CLASS_AND_RECOVERY_ENDPOINTS','n':366,'scalar_loss_checks':checks,'max_loss_error':worst,'accepted_primary_candidate_increases':priority_up,'sum_control_first_step_candidate_increases':sum_up,'masking_regenerated_from_fixed_seed':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start,'task_GT_read':False};dump(O/'AUDIT_RESULT.json',result);print(result)
