from core108 import *
sys.path.insert(0,str(H108));from replay108 import ce_cpu108
def main():
 checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=loadce108(src);assert all(torch.equal(v,head.state_dict()[k]) for k,v in base.prediction_heads[-1].sem_cls_scores_head.state_dict().items())
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   rows=jread(R108/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');root=B106/(src+'_to_'+tgt);eligible=[r for r in rows if not jread(root/'records_PRIVATE'/(r['id']+'.json'))['protected_alias_fallback']]
   for r in eligible[:2]:
    qs=[];ce=[];hk=base.decoder[-1].register_forward_hook(lambda _m,_a,o:qs.append(o.detach()));hc=base.prediction_heads[-1].sem_cls_scores_head.register_forward_hook(lambda _m,_a,o:ce.append(o.detach()))
    try:
     with torch.no_grad():z0,g,_=capture98(base,r,src)
    finally:hk.remove();hc.remove()
    assert len(qs)==len(ce)==1;assert torch.equal(head(qs[0].transpose(1,2)),ce[0]);z=tensor_prefix(z0);d=tensor_data(g,r);rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);live=[];hk=base.decoder[-1].register_forward_hook(lambda _m,_a,o:live.append(o.detach()))
    try:
     with torch.no_grad():f=native105(m,base,z,d,route,S,[])
    finally:hk.remove()
    cache=load(root/'features_PRIVATE'/(r['id']+'.npz'));assert torch.equal(live[0][0],f['h']) and np.array_equal(f['h'].cpu().numpy(),cache['real_A_h']);p,logits=ce108(head,f['h'],f['mask']);pc,lc=ce_cpu108(f['h'].cpu().numpy(),f['mask'].cpu().numpy(),statece108(head));err=float(abs(pc-p.cpu().numpy()).max());le=float(abs(lc-logits.cpu().numpy()).max());assert err<3e-6 and le<1e-4 and pc.argmax()==int(p.argmax());assert np.array_equal(z0['boxes'],load(root/'samples_PRIVATE'/(r['id']+'.npz'))['source_boxes']);checks.append({'source':src,'target':tgt,'source_native_CE_module_exact':True,'prepared_native_hidden_exact':True,'CPU_probability_error':err,'CPU_logit_error':le,'FINAL_access':False});restore(m,S);print('ENGINEERING108',src,tgt,len(checks),flush=True)
  assert fingerprint(base)==fp;del base,m,head;torch.cuda.empty_cache()
 jdump(H108/'ENGINEERING.json',{'state':'PASS_SOURCE_CHECKPOINT_NATIVE_SOFTTOKEN_MODULE_AND_PREPARED_ACTUAL_DECODER_HIDDEN_CPU_PARITY','n':len(checks),'checks':checks,'GT_used':False,'FINAL_access':False})
if __name__=='__main__':main()
