"""Native path identity, actual sensor provenance and independent match-head replay."""
from core111 import *
def main():
 checks=[];anchors={x['A']['metapath']:x for x in jread(R111/'research/round105/FULL_ANCHORS_PRIVATE.json')};pairs={p['pair_PRIVATE']:p for p in jread(R111/'research/round105/FULL_PAIRS_PRIVATE.json')}
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);s0=state(base.contrastive_align_projection_image);s1=state(m.match);assert set(s0)==set(s1)=={f'{j}.{k}' for j in [0,2,4] for k in ['weight','bias']}
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=[r for r in jread(R111/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json') if (R111/'crossplatform/round110/original_PRIVATE'/name/(r['id']+'.npz')).exists()]
   for r in rows[:2]:
    old=load(R111/'crossplatform/round110/original_PRIVATE'/name/(r['id']+'.npz'));oldrec=jread(B106/name/'records_PRIVATE'/(r['id']+'.json'));z0,receipt=sourceprefix111(base,r,src);z=tensor_prefix(z0);A=original111(base,z);assert torch.equal(A['h'],torch.as_tensor(old['h'],device='cuda')) and torch.equal(A['query'],torch.as_tensor(old['query'],device='cuda')) and torch.equal(A['tokens'],torch.as_tensor(old['tokens'],device='cuda'));assert float(abs(A['p']-torch.as_tensor(old['p'],device='cuda')).max())<1e-7
    ident=prefix106(z,[{'vis':z['vis0'][0],'valid':torch.ones(1024,device='cuda',dtype=torch.bool)}],True);assert torch.equal(ident['vis0'],z['vis0']);I=original111(base,ident);assert torch.equal(I['p'],A['p'])
    ar=anchors[str(Path(r['metapath']).resolve())];ob,proof=observations111(base,r,z0,ar,pairs,oldrec);vv=[ob['real'][v] for v in ['B','C'] if ob['real'][v] is not None];Z=prefix106(z,vv,True);f=original111(base,Z);errors=[]
    for F0 in [A,f]:
     for head,st in [(base.contrastive_align_projection_image,s0),(m.match,s1)]:
      with torch.no_grad():p,q=direct111(head,F0['h'],F0['tokens'],F0['mask'])
      cp,cq=cpu111(F0['h'].cpu().numpy(),F0['tokens'].cpu().numpy(),F0['mask'].cpu().numpy(),st);e=float(abs(p.cpu().numpy()-cp).max());assert e<5e-6 and int(p.argmax())==int(cp.argmax());errors.append(e)
    assert all(torch.equal(z['position'+str(j)],Z['position'+str(j)]) for j in range(6));assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());checks.append({'source':src,'target':tgt,'source_original_h_query_tokens_bitexact':True,'source_original_probability_error':float(abs(A['p'].cpu().numpy()-old['p']).max()),'identity_fusion_bitexact':True,'source_positions_fixed':True,'source_model_fingerprint_preserved':True,'sensor_provenance_same_as_R106':True,'real_B_available':ob['real']['B'] is not None,'real_C_available':ob['real']['C'] is not None,'independent_CPU_probability_error':max(errors),'GT_used':False});print('ENGINEERING111',src,tgt,len(checks),flush=True)
  del base,m;torch.cuda.empty_cache()
 jdump(H111/'ENGINEERING.json',{'state':'PASS_NATIVE_ORIGINAL_IDENTITY_SENSOR_PROVENANCE_FIXED_POSITIONS_INDEPENDENT_CPU_HEADS_SOURCE_RESTORE','n':len(checks),'checks':checks,'FINAL_access':False,'GT_used':False})
if __name__=='__main__':main()
