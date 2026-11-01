"""Finite authorized R44 phase2: two cached query heads, no task labels."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,time,json,hashlib,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round24'),str(Path(__file__).resolve().parent)]
import fit24 as old
import phase2_draft44 as algebra
import torch
F=torch.nn.functional
EXP=ROOT/'research/round44/phase2';OUT=ROOT/'crossplatform/round44/fit272_attempt001'
DIRECT=ROOT/'crossplatform/round44/direct208_attempt001'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def bound(folder,rel,seal):
 p=folder/rel;assert sha(p)==seal['files'][rel],str(p);return p
def check():
 auth=read(EXP/'ROOT_EXECUTION_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_FINITE_PHASE2_MATCHING_TTA'
 for p,h in auth['references_sha256'].items():assert sha(ROOT/p)==h,p
 lock=read(EXP/'PLAN_LOCK.json');assert lock['root_authorization_sha256']==sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json')
 for p,h in dict(lock['program_sha256'],**lock['input_sha256']).items():assert sha(ROOT/p)==h,p
 assert old.feature.__module__=='common24' and 'fit29' not in sys.modules
 for folder,split in [(ROOT/'crossplatform/round24/capture336_attempt001','samples_PRIVATE/'),(ROOT/'crossplatform/round23/bank272_attempt001','samples_PRIVATE/')]:
  seal=read(folder/'SEALED.json')
  for r in old.rows('D208')+old.rows('V64'):bound(folder,split+r['id']+'.npz',seal)
 cs=read(ROOT/'crossplatform/round24/capture336_attempt001/SEALED.json');bound(ROOT/'crossplatform/round24/capture336_attempt001','projection_source_PRIVATE.npz',cs)
 return lock
def native_and_z(m,b):
 q=F.normalize(m(b['h']),dim=-1);v=q@b['tokens'].transpose(-1,-2)/.07
 s=(v.masked_fill(b['padding'][:,None],-torch.inf).softmax(-1)*b['mask'][:,None]).sum(-1)
 return s,algebra.exact_z_from_logits(v,b['mask'],b['padding'])
def keys(rows):
 k=np.stack([old.interact(old.feature(r)) for r in rows]);k/=np.maximum(np.linalg.norm(k,axis=-1,keepdims=True),1e-12);assert k.dtype==np.float32;return k
def main():
 check();OUT.mkdir(exist_ok=False)
 for p in ['states_PRIVATE','predictions_PRIVATE','input_contract_PRIVATE','cases_PRIVATE','memory_PRIVATE']:(OUT/p).mkdir()
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 start=time.perf_counter();receipts=[];ns=0
 directseal=read(DIRECT/'SEALED.json');geomfolder=ROOT/'crossplatform/round37/prepared208_attempt001';geomseal=read(geomfolder/'SEALED.json')
 try:
  for pl,expected in [('drone',14),('quad',18)]:
   A=[r for r in old.rows('D208') if r['platform']==pl];B=[r for r in old.rows('V64') if r['platform']==pl];assert len(A)==104 and len(B)==32
   prefs=[];groups=[];direct=[]
   for j,r in enumerate(A):
    c=read(bound(DIRECT,'cases_PRIVATE/'+r['id']+'.json',directseal));g=read(bound(geomfolder,'cases_PRIVATE/'+r['id']+'.json',geomseal))['full_groups'];groups.append(g);direct.append(c)
    f=c['decision']['accepted_preference']
    if f is not None:
     assert c['decision']['eligible'] and not set(f['positive_queries'])&set(f['negative_queries'])
     assert g[f['positive_mask']]==f['positive_queries'] and g[f['negative_mask']]==f['negative_queries']
     prefs.append(dict(f,row_index=j,alias=r['alias'],id_PRIVATE=r['id']))
   E=[f['row_index'] for f in prefs];U=[j for j in range(104) if j not in set(E)];assert len(E)==len(set(E))==expected and len(U)==104-expected and sorted(E+U)==list(range(104))
   a=old.batch(A);b=old.batch(B);m=old.module('query_mlp');assert sum(p.numel() for p in m.parameters())==184960
   source={n:p.detach().clone() for n,p in m.named_parameters()};q0=a['scores'].argmax(1)
   with torch.no_grad():p0,z0=native_and_z(m,a);pv0,zv0=native_and_z(m,b)
   assert not z0.requires_grad and not zv0.requires_grad
   errs={'A':float(abs(p0-a['scores']).max()),'V':float(abs(pv0-b['scores']).max())};assert max(errs.values())<2e-5
   assert torch.equal(p0.argmax(1),q0) and torch.equal(pv0.argmax(1),b['scores'].argmax(1))
   for j in range(104):assert algebra.fixed_source_group_readout(a['scores'][j].cpu().numpy(),a['scores'][j].cpu().numpy(),groups[j])==int(q0[j])
   np.savez_compressed(OUT/'input_contract_PRIVATE'/(pl+'.npz'),A_ids=np.array([r['id'] for r in A]),V_ids=np.array([r['id'] for r in B]),A_mask=a['mask'].cpu().numpy(),A_padding=a['padding'].cpu().numpy(),V_mask=b['mask'].cpu().numpy(),V_padding=b['padding'].cpu().numpy(),source_A_scores=a['scores'].cpu().numpy(),source_V_scores=b['scores'].cpu().numpy(),source_A_replayed=p0.cpu().numpy(),source_V_replayed=pv0.cpu().numpy(),source_A_z=z0.cpu().numpy(),source_V_z=zv0.cpu().numpy(),source_q0=q0.cpu().numpy(),E_indices=np.array(E),U_indices=np.array(U))
   dump(OUT/'input_contract_PRIVATE'/(pl+'_preferences.json'),{'preferences':prefs,'U_indices':U,'fixed_denominator':expected,'all_phase1_feedback_retained':True})
   opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=0);trace=[]
   with torch.no_grad():initialloss,initialterms=algebra.proposed_loss(z0,z0,q0,prefs,U,expected)
   for step in range(1,17):
    opt.zero_grad();_,z=native_and_z(m,a);loss,terms=algebra.proposed_loss(z,z0,q0,prefs,U,expected);assert torch.isfinite(loss);loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())));opt.step()
    trace.append({'step':step,'total':float(loss.detach()),'preference':float(terms['preference'].detach()),'relative_z_keep':float(terms['relative_z_keep'].detach()),'gradient_L2':grad})
   with torch.no_grad():pa,za=native_and_z(m,a);pb,zb=native_and_z(m,b);finalloss,ft=algebra.proposed_loss(za,z0,q0,prefs,U,expected)
   sa=pa.cpu().numpy();sb=pb.cpu().numpy();az=za.cpu().numpy();bz=zb.cpu().numpy();state={n:p.detach().cpu().numpy() for n,p in m.state_dict().items()};np.savez_compressed(OUT/'states_PRIVATE'/(pl+'_query_mlp.npz'),**state);ns+=1
   gaps0=initialterms['gaps'].cpu().numpy();gaps1=ft['gaps'].cpu().numpy();prefreceipts=[]
   for f,g0,g1 in zip(prefs,gaps0,gaps1):
    j=f['row_index'];raw=int(sa[j].argmax());rep=algebra.fixed_source_group_readout(sa[j],a['scores'][j].cpu().numpy(),groups[j]);P=f['positive_queries'];N=f['negative_queries']
    prefreceipts.append({'alias':f['alias'],'row_index':j,'initial_gap_z':float(g0),'final_gap_z':float(g1),'pair_satisfied_z':bool(g1>0),'raw_query':raw,'fixed_source_group_query':rep,'winner_in_positive':rep in P,'winner_in_negative':rep in N,'third_unjudged_winner':rep not in set(P+N),'third_not_automatically_negative':True})
   ak=keys(A);bk=keys(B);values=algebra.original_probability_memory_values(a['scores'].cpu().numpy(),prefs);assert np.count_nonzero(values[U])==0
   np.savez_compressed(OUT/'memory_PRIVATE'/(pl+'_library.npz'),A_keys=ak,V_keys=bk,values_probability_units=values,E_indices=np.array(E),U_indices=np.array(U))
   memory={}
   for name,lib in [('memory_E_only',E),('memory_allA_zeroU',list(range(104)))]:
    corrs=[]
    for j,r in enumerate(B):
     corr,nn=algebra.retrieve_probability_memory(bk[j],ak,values,np.array(lib));corrs.append(corr)
     np.savez_compressed(OUT/'memory_PRIVATE'/(r['id']+'_'+name+'.npz'),correction_probability_units=corr,**{k:v for k,v in nn.items() if k!='topk_tie_contract'})
    memory[name]=np.stack(corrs)
   for split,rr,ss,zz in [('D208',A,sa,az),('V64',B,sb,bz)]:
    for j,r in enumerate(rr):
     f=old.feature(r);FG=old.teacher(r,f);delta=FG-f['scores'].astype(float)
     if split=='D208':
      names=['source','eligible_G_direct','oldG_unrestricted','TTA_group_fixed_source_rep','TTA_raw_query'];scores=np.stack([f['scores'],FG,FG,ss[j],ss[j]]);queries=np.array([int(f['scores'].argmax()),direct[j]['decision']['query'],int(FG.argmax()),algebra.fixed_source_group_readout(ss[j],f['scores'],groups[j]),int(ss[j].argmax())]);case={'alias':r['alias'],'platform':pl,'split':split,'eligible':j in set(E),'source_q0':int(q0[j]),'full_groups':groups[j],'phase1_decision':direct[j]['decision'],'A_readout':'adapted_groupmax_to_frozen_source_rep_unmapped_retained'}
     else:
      names=['source','TTA','memory_E_only','memory_allA_zeroU','oldG_direct_current','TTA_plus_current_G'];scores=np.stack([f['scores'],ss[j],f['scores']+memory['memory_E_only'][j],f['scores']+memory['memory_allA_zeroU'][j],FG,ss[j].astype(float)+delta]);queries=scores.argmax(-1);case={'alias':r['alias'],'platform':pl,'split':split,'no_V_used_for_fitting':True,'memory_addition_coordinate':'native_probability_only','current_G_arms_are_extra_information':True}
     np.savez_compressed(OUT/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(names),scores=scores,queries=queries,boxes=f['boxes'][queries],adapted_native_probability=ss[j],adapted_token_logodds=zz[j],source_native_probability=f['scores'],source_replayed_native_probability=(p0 if split=='D208' else pv0)[j].cpu().numpy(),source_token_logodds=(z0 if split=='D208' else zv0)[j].cpu().numpy(),oldG_F64=FG,deltaG_F64=delta)
     dump(OUT/'cases_PRIVATE'/(r['id']+'.json'),case)
   receipt={'platform':pl,'trainable_params':184960,'steps':16,'lr':.0005,'beta':1,'pair_temperature':.1,'E':expected,'U':len(U),'source_probability_replay_maxerrors':errs,'source_query_replay_exact':True,'z0_no_grad':True,'trace':trace,'initial_terms':{'total':float(initialloss),'preference':float(initialterms['preference']),'relative_z_keep':float(initialterms['relative_z_keep'])},'final_terms_after_step16':{'total':float(finalloss),'preference':float(ft['preference']),'relative_z_keep':float(ft['relative_z_keep'])},'parameter_delta_L2':float(torch.sqrt(sum((p-source[n]).square().sum() for n,p in m.named_parameters()))),'preferences':prefreceipts};receipts.append(receipt);dump(OUT/'FIT_RECEIPTS.json',receipts);print('FIT_R44',pl,'finalpref',receipt['final_terms_after_step16']['preference'],'pairs',sum(x['pair_satisfied_z'] for x in prefreceipts),'/',expected,flush=True)
   del a,b,m,opt,p0,pv0,z0,zv0,pa,pb,za,zb
   torch.cuda.empty_cache()
  files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()};dump(OUT/'SEALED.json',{'state':'R44_PHASE2_ALL_A208_V64_HEAD_MEMORY_PREDICTIONS_BEFORE_TASKSCORE','files':files,'n':272,'A':208,'V':64,'states':2,'optimizer_steps':32,'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'root_authorization_sha256':sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json'),'seconds':time.perf_counter()-start,'peak_bytes':torch.cuda.max_memory_allocated(),'all_boxes_original':True,'fullgrounder_closure':False,'no_task_GT_newG_FINAL_U_T':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print('R44_PHASE2_SEALED',time.perf_counter()-start,flush=True)
 except BaseException as e:
  dump(OUT/('FAILED_'+str(time.time_ns())+'.json'),{'state':'TECHNICAL_FAILURE_PRESERVED_NOT_METHOD_RESULT','type':type(e).__name__,'message':str(e),'completed_states':ns,'automatic_retry':False});raise
if __name__=='__main__':main()
