"""Real native states: exact identity, running-buffer-only write and functional/native replay."""
from core110 import *
sys.path.insert(0,str(R110/'research/round108'));from replay108 import ce_cpu108

def main():
 assert jread(B110/'original_PRIVATE/SEALED.json')['n']==724;checks=[]
 for src in ['waymo','drone','quad']:
  base=base97(src);head=base.prediction_heads[-1].sem_cls_scores_head;S=statece108(head);fp=fingerprint(base)
  for tgt in ['waymo','drone','quad']:
   if tgt==src:continue
   name=src+'_to_'+tgt;rows=jread(R110/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');rows=[r for r in rows if (B110/'original_PRIVATE'/name/(r['id']+'.npz')).exists()]
   for r in rows[:2]:
    sid=r['id'];old=load(B110/'original_PRIVATE'/name/(sid+'.npz'));O=torch.as_tensor(old['h'],device='cuda');mask=torch.as_tensor(old['mask'],device='cuda');putce110(head,S);before=ce108(head,O,mask)[0]
    ident,_,mi=transport110(head,O,O,S);assert all(np.array_equal(ident[k],v) for k,v in S.items());assert torch.equal(ce108(head,O,mask)[0],before)
    with np.load(B106/name/'features_PRIVATE'/(sid+'.npz')) as cache:
     P=torch.as_tensor(cache['real_A_h'],device='cuda');st,rec,info=transport110(head,O,P,S);assert info['changed_values']>0;events=['calibration_finished']
     # C/fused observations are only accessed AFTER state calibration.
     errs=[];funerrs=[]
     for view in ['A','C','fusedABC']:
      h=torch.as_tensor(cache['real_'+view+'_h'],device='cuda');m=torch.as_tensor(cache['real_'+view+'_mask'],device='cuda');putce110(head,st);p,logits=ce108(head,h,m);cpu,lc=ce_cpu108(h.cpu().numpy(),m.cpu().numpy(),st);e=float(abs(cpu-p.cpu().numpy()).max());assert e<5e-6 and int(cpu.argmax())==int(p.argmax());errs.append(e)
      fun=functional110(head,h,S,rec);ll=float(abs(fun-logits).max());fm=torch.zeros(256,device='cuda',dtype=torch.bool);fm[:len(m)]=m;mass=fun.double().log_softmax(-1).masked_fill(~fm[None],-torch.inf).logsumexp(-1);pp=mass.softmax(0);pe=float(abs(pp-p).max());assert pe<5e-6 and int(pp.argmax())==int(p.argmax());funerrs.append({'logit_error':ll,'probability_error':pe})
    putce110(head,S);assert fingerprint(base)==fp;checks.append({'source':src,'target':tgt,'identity_buffers_and_probability_bitexact':True,'only_BN_running_stats_changed':True,'calibration':info,'independent_CPU_max_error':max(errs),'functional_vs_actual_write':funerrs,'full_source_fingerprint_restored':True,'C_after_calibration':True});print('ENGINEERING110',src,tgt,len(checks),flush=True)
  del base,head;torch.cuda.empty_cache()
 jdump(H110/'ENGINEERING.json',{'state':'PASS_REAL_NATIVE_ORIGINAL_PREPARED_PAIR_IDENTITY_EPS_TRANSPORT_ACTUAL_STATS_WRITE_CPU_AND_FUNCTIONAL_PARITY_RESTORE','n':len(checks),'checks':checks,'GT_used':False,'FINAL_access':False})
if __name__=='__main__':main()
