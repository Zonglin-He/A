"""Real-input checks: gating parity, native uniform loss, cap/permutation and SGD."""
from core79 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H79/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H79/'experiment/CODE_LOCK.json'));base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);native=Fusion(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);restore(native,S);records=[]
for j,r in enumerate(source_rows('VAL128')[:2]+[next(r for r in target_rows() if r['platform']==pl) for pl in ['drone','quad']]):
 z0=prefix(r);z=tensor_prefix(z0);d=tensor_data(load(casepath(r)),r);pm,im=maskpair79(r,d)
 with torch.no_grad():pn,_,_=native.main(base,z,d,'bidirectional')
 refp,refq,rp,ri,pr=sensitivity(m,base,z,d,z0['boxes']);assert float(abs(refp-pn).max())<1e-10 and np.isfinite(rp).all() and np.isfinite(ri).all();ws,fb=weight_sets(r,d,pm,im,rp,ri);err=0.
 with torch.no_grad():
  for a,b,w,v in zip(pm,im,ws['uniform'],ws['query']):
   orig=native.auxiliary(z,d,'bidirectional',a,b)[0];lp,li,ld=m.local_losses(z,d,a,b);agg=.1*(w['P']*lp).sum()+.1*(w['I']*li).sum()+.01*(w['D']*ld).sum();err=max(err,float(abs(orig-agg)))
  assert err<1e-7
 for wa,wb in zip(ws['query'],ws['shuffle']):
  for key in wa:assert torch.equal(wa[key].sort().values,wb[key].sort().values)
 g,delta,rec=gradient79(m,base,z,d,refp,refq,pm,im,ws['query']);assert g.isfinite().all();set_step(m,S,delta)
 with torch.no_grad():post,_,_=m.main(base,z,d,'bidirectional');assert post.isfinite().all()
 restore(m,S);assert sum(v.numel() for _,v in m.tta_parameters())==102498;assert m.analysis_point_gate is None and m.analysis_image_gate is None;records.append({'row':j+1,'platform':r['platform'],'uniform_native_loss_max_error':err,'gating_allones_native_max_error':float(abs(refp-pn).max()),'sensitivity_pair_count':pr['pair_count'],'point_saliency_sum':float(rp.sum()),'image_saliency_sum':float(ri.sum()),'weight_fallback_sets':fb,'query_gradient_norm':rec['gradient_norm'],'real_parameter_step_L2':rec['raw_proposal_L2']})
assert fingerprint(base)==fp;out=H79/'engineering_attempt001';out.mkdir();dump(out/'RESULT.json',{'state':'PASS_R79_REAL_SOURCE_TARGET_GATES_NATIVE_LOSS_WEIGHTS_AND_NONZERO_UPDATES','n':len(records),'rows':records,'no_GT_read':True,'source_training':False});print(read(out/'RESULT.json'))
