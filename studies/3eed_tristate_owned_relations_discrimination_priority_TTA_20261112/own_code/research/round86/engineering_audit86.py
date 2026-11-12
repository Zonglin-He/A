"""Independent real-input acceptance, zero-base and masked-path probes."""
from core86 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source
P=B86/'engineering_attempt001';assert read(P/'SEALED.json')['n']==2;rows=read(H86/'experiment/ALL366_PRIVATE.json');chosen=[next(r for r in rows if r['platform']==p and read(B86/'evidence366_attempt002/cases_PRIVATE'/(r['id']+'.json'))['factor_kinds']) for p in ['drone','quad']];base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));torch.set_num_threads(4);m=Fusion85(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);restore(m,load(ROOT/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz'));up=0;priority_steps=0
with torch.no_grad():
 for r in chosen:
  sid=r['id'];a=load(P/'samples_PRIVATE'/(sid+'.npz'));rec=read(P/'records_PRIVATE'/(sid+'.json'));z0,z,d=item80(r);frozen=m.main(base,z,d,'bidirectional')[0]
  for b in ['S','F']:
   s0=torch.tensor(a[b+'_s0'],device='cuda');assert float(abs(anchored85(frozen,frozen,s0)-s0.softmax(-1)).max())==0
   for v in ['candidate','joint','mismatch']:
    traj=rec['bases'][b]['variants'][v];assert any(s['accepted'] and s['parameter_L2']>0 for s in traj)
    for s in traj:
     if s['accepted']:assert s['classification_after']<=s['classification_before']+1e-10;priority_steps+=1
   up+=sum(s['accepted'] and s['classification_after']>s['classification_before']+1e-10 for s in rec['bases'][b]['variants']['sum'])
   assert rec['bases'][b]['unknown_probability_error']==0
  pm=torch.tensor(a['mask_point'][0],device='cuda');im=torch.tensor(a['mask_image'][0],device='cuda');P0=z['vis0'][0];v=m.features(P0,d,'bidirectional',point_mask=pm,image_mask=im);wrongP=P0.clone();wrongP[pm]+=100.;w=m.features(wrongP,d,'bidirectional',point_mask=pm,image_mask=im);assert all(torch.equal(v[k],w[k]) for k in [0,1]);image=d['image'].clone();pixelmask=im.reshape(1,1,32,48).repeat_interleave(8,2).repeat_interleave(8,3).expand_as(image);image[pixelmask]=0.;w=m.features(P0,d,'bidirectional',image=image,point_mask=pm,image_mask=im);assert all(torch.equal(v[k],w[k]) for k in [0,1])
 gc=torch.tensor([1.,0.],device='cuda');gr=torch.tensor([-3.,4.],device='cuda');direction,projected=priority86(gc,gr);assert torch.equal(projected,torch.tensor([0.,4.],device='cuda')) and float(gc@direction)>0
result={'state':'PASS_R86_REAL_NONZERO_CANDIDATE_AND_PRIORITY_UPDATES_BOTH_BASES_AND_MASKED_INPUT_BLOCKING','n':2,'candidate_decline_checks':priority_steps,'sum_control_candidate_increases':up,'zero_base_probability_error':0,'unknown_probability_error':0,'masked_input_leak_error':0,'unilateral_projection_synthetic_verified':True,'task_GT_read':False};O=H86/'engineering_audit_attempt001';O.mkdir();dump(O/'AUDIT_RESULT.json',result);print(result)
