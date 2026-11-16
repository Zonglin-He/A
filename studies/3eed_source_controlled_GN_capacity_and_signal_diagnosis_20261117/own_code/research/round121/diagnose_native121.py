from core121 import *
setsource97('waymo');base=base97('waymo');m,S=model97(base,'waymo');rr=jread(H121/'SOURCE_CACHE_REGISTRY_PRIVATE.json')[:2];o=[]
for e in rr:
 z0,z,d,rt,*_=data121(e)
 with torch.no_grad():
  with functional112(m,'neither'):f=feature91(m,base,z,d,rt,S)
  rep=replay(base,z0);proj=base.contrastive_align_projection_image(torch.as_tensor(z0['h'],device='cuda')[None])[0];p=native_scores(F.normalize(proj,dim=-1),torch.as_tensor(z0['tokens'],device='cuda'),z['token_mask'].bool()).double().softmax(0);q=F.normalize(m.match(f['h'][None])[0],dim=-1);pp=native_scores(q,f['u'],z['token_mask'].bool()).double().softmax(0)
  o.append({'native_vs_cached_h':float(abs(f['h'].cpu().numpy()-z0['h']).max()),'native_vs_cached_u':float(abs(f['u'].cpu().numpy()-z0['tokens']).max()),'replay_vs_cached_h':float(abs(rep['h']-z0['h']).max()),'cached_native_match_vs_source_scores':float(abs(p.cpu().numpy()-z0['scores']).max()),'new_native_vs_replay_h':float(abs(f['h'].cpu().numpy()-rep['h']).max()),'native_vs_cached_prob':float(abs(pp.cpu().numpy()-z0['scores']).max()),'head_equal':all(torch.equal(m.match.state_dict()[n],v) for n,v in base.contrastive_align_projection_image.state_dict().items())})
jdump(H121/'NATIVE_QUALIFICATION_DIAGNOSIS.json',o);print(o)
