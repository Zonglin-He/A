"""Distinguish proposed FP32 update norms from actual quantized parameter writeback."""
from core97 import *
def main():
 checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   rows=rows97(tgt,'target');root=B97/(src+'_to_'+tgt)
   for idx in [0,len(rows)//2,len(rows)-1]:
    r=rows[idx];saved=jread(root/'records_PRIVATE'/(r['id']+'.json'))
    with torch.no_grad():z0,g,_,_=sample97(base,r)
    z,d=tensor_prefix(z0),tensor_data(g,r);pm,im=maskpair79(r,d);ws=weight_sets({'id':'UNUSED_UNIFORM'},d,pm,im,np.zeros(1024),np.zeros(1536))[0]['uniform']
    for wrong in [False,True]:
     grad,raw,rec=physical_direction91(m,base,z,d,S,pm,im,ws,wrong);direction=-(1/(grad.norm()+1e-6)).clamp_max(1)*grad;kind='wrong_physical' if wrong else 'physical'
     for mode in ['raw','bt']:
      arm=kind+'_'+mode;alpha=saved['updates'][arm]['alpha'];delta=alpha*direction;theta_step91(m,S,delta);actual=float(torch.sqrt(sum((p.detach()-torch.as_tensor(S[n],device='cuda')).double().square().sum() for n,p in m.tta_parameters())));proposal=float(delta.norm());assert abs(proposal-saved['updates'][arm]['actual_parameter_L2'])<1e-8
      if proposal>0:assert actual>0,'accepted nonzero proposal had no actual parameter writeback'
      checks.append({'source':src,'target':tgt,'row':'ROW'+str(idx+1).zfill(6),'arm':arm,'proposal_L2':proposal,'actual_quantized_L2':actual,'actual_nonzero':actual>0})
  del m,base;torch.cuda.empty_cache()
 assert len(checks)==72;path=H97/'NATIVE_AUDIT.json';audit=jread(path);audit.update(update_norm_field_semantics='Producer legacy actual_parameter_L2 field stores proposed update vector L2; it is not the exact FP32 quantized writeback norm. Counts use nonzero proposals. All sampled nonzero proposals also changed actual parameters.',quantized_writeback_checks=checks);jdump(path,audit);jdump(H97/'WRITEBACK_AUDIT.json',{'state':'PASS_R97_PROPOSED_VS_ACTUAL_QUANTIZED_WRITEBACK_DISCLOSED','checks':72,'all_sampled_nonzero_proposals_have_actual_writeback':True});print({'writeback_checks':len(checks)})
if __name__=='__main__':main()
