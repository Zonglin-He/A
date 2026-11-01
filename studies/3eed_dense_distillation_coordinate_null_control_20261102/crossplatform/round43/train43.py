"""Authorized eight cached query heads, literal R24 control, no task GT."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,json,hashlib,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round24'))
import fit24 as old
import torch
F=torch.nn.functional
EXP=ROOT/'research/round43/experiment';OUT=ROOT/'crossplatform/round43/fit272_attempt001'
CONFIGS=['full_probability','full_token_logodds','null_probability','null_token_logodds']
ARMS=['source']+CONFIGS+['direct_G','full_probability_plus_current_G','full_token_logodds_plus_current_G','zero_delta_memory']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def check():
 a=read(EXP/'ROOT_EXECUTION_AUTHORIZATION.json');assert a['state']=='ROOT_AUTHORIZED_PAIRED_DENSE_COORDINATE_DIAGNOSTIC'
 for p,h in a['references_sha256'].items():assert sha(ROOT/p)==h,p
 lock=read(EXP/'PLAN_LOCK.json');assert lock['root_authorization_sha256']==sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json')
 for p,h in dict(lock['program_sha256'],**lock['input_sha256']).items():assert sha(ROOT/p)==h,p
 assert old.feature.__module__=='common24' and 'fit29' not in sys.modules,'legacy feature hook overridden'
 cs=read(ROOT/'crossplatform/round24/capture336_attempt001/SEALED.json');bs=read(ROOT/'crossplatform/round23/bank272_attempt001/SEALED.json')
 for r in old.rows('D208')+old.rows('V64'):
  rel='samples_PRIVATE/'+r['id']+'.npz';assert sha(ROOT/'crossplatform/round24/capture336_attempt001'/rel)==cs['files'][rel] and sha(ROOT/'crossplatform/round23/bank272_attempt001'/rel)==bs['files'][rel]
 assert sha(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz')==cs['files']['projection_source_PRIVATE.npz']
 return lock

def native_and_z(m,b):
 q=F.normalize(m(b['h']),dim=-1);v=q@b['tokens'].transpose(-1,-2)/.07;valid=~b['padding'];pos=b['mask']&valid;neg=~b['mask']&valid;assert pos.any(-1).all() and neg.any(-1).all()
 logits=v.masked_fill(b['padding'][:,None],-torch.inf);s=(logits.softmax(-1)*b['mask'][:,None]).sum(-1);z=torch.logsumexp(v.masked_fill(~pos[:,None],-torch.inf),-1)-torch.logsumexp(v.masked_fill(~neg[:,None],-torch.inf),-1);return s,z

def fit_one(pl,name,A,B,a,b):
 m=old.module('query_mlp');source={n:p.detach().clone() for n,p in m.named_parameters()};assert sum(p.numel() for p in m.parameters())==184960
 opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=0.);null=name.startswith('null_');usez=name.endswith('token_logodds');pT=a['p0'] if null else a['pT'];target=a['scores'].argmax(1) if null else a['target'];trace=[]
 with torch.no_grad():initial=old.scores(m,'query_mlp',a);initialerr=float(abs(initial-a['scores']).max());assert initialerr<2e-5
 for step in range(1,17):
  opt.zero_grad()
  if usez:s,z=native_and_z(m,a)
  else:s=old.scores(m,'query_mlp',a)
  lp=(s/.25).log_softmax(-1)
  # Full-probability operation order is literally the original fit24 loop.
  kl=F.kl_div((z/.25).log_softmax(-1) if usez else lp,pT,reduction='batchmean');anchor=F.kl_div(lp,a['p0'],reduction='batchmean');chosen=s.gather(1,target[:,None]).squeeze(1);other=s.clone().scatter(1,target[:,None],-torch.inf).max(-1).values;pair=F.relu(.05+other-chosen).mean();reg=sum((p-source[n]).square().mean() for n,p in m.named_parameters());loss=kl+.2*anchor+.5*pair+reg;assert torch.isfinite(loss);loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())));opt.step()
  if step in [1,4,8,16]:trace.append({'step':step,'total':float(loss.detach()),'teacher_KL':float(kl.detach()),'source_KL_native':float(anchor.detach()),'hinge_native':float(pair.detach()),'parameter_anchor':float(reg.detach()),'gradient_L2':grad})
 with torch.no_grad():sa=old.scores(m,'query_mlp',a).cpu().numpy();sb=old.scores(m,'query_mlp',b).cpu().numpy();_,za=native_and_z(m,a);_,zb=native_and_z(m,b);zg=za.cpu().numpy();zv=zb.cpu().numpy();sf=torch.tensor(sa,device='cuda');lf=(sf/.25).log_softmax(-1);kf=F.kl_div((za/.25).log_softmax(-1) if usez else lf,pT,reduction='batchmean');af=F.kl_div(lf,a['p0'],reduction='batchmean');cf=sf.gather(1,target[:,None]).squeeze(1);of=sf.clone().scatter(1,target[:,None],-torch.inf).max(-1).values;hf=F.relu(.05+of-cf).mean();rf=sum((p-source[n]).square().mean() for n,p in m.named_parameters());final_terms={'teacher_KL':float(kf),'source_KL_native':float(af),'hinge_native':float(hf),'parameter_anchor':float(rf),'total':float(kf+.2*af+.5*hf+rf)}
 state={n:p.detach().cpu().numpy() for n,p in m.state_dict().items()};np.savez_compressed(OUT/'states_PRIVATE'/(pl+'_'+name+'.npz'),**state)
 receipt={'platform':pl,'configuration':name,'params':184960,'steps':16,'lr':.0005,'teacher_mode':'null_deltaG' if null else 'full_original_G','teacher_KL_student_coordinate':'token_logodds' if usez else 'native_probability','sourceKL_hinge_readout_coordinate':'native_probability','initial_probability_replay_maxerror':initialerr,'trace':trace,'final_terms_after_step16':final_terms,'parameter_delta_L2':float(torch.sqrt(sum((p-source[n]).square().sum() for n,p in m.named_parameters()))),'final_A_teacher_top1_agreement_percent':float((sa.argmax(1)==target.cpu().numpy()).mean()*100)}
 if name=='full_probability':
  prior=npz(ROOT/'crossplatform/round24/screen_attempt001/states_PRIVATE'/(pl+'_query_mlp_lr1_s16.npz'));err=max(float(abs(state[k]-prior[k]).max()) for k in state);exact=all(np.array_equal(state[k],prior[k]) for k in state);nq=0
  for split,rr,ss in [('D208',A,sa),('V64',B,sb)]:
   for r,v in zip(rr,ss):
    p=npz(ROOT/'crossplatform/round24/screen_attempt001'/(split+'_predictions/predictions_PRIVATE')/(r['id']+'.npz'));ix=p['arm_names'].tolist().index('query_mlp_lr1_s16');assert int(v.argmax())==int(p['query_indices'][ix]),('R24_control_query_mismatch',r['alias']);nq+=1
  assert err<=1e-6,('R24_control_state_mismatch',pl,err);receipt['matched_original_control']={'state_maxerror':err,'state_bitexact':exact,'A_V_query_matches':nq,'state_tolerance_prespecified':1e-6,'query_contract':'exact, no label-based tolerance'}
 print('FIT_R43',pl,name,'teacherKL',trace[-1]['teacher_KL'],'delta',receipt['parameter_delta_L2'],flush=True)
 return {'A':sa,'V':sb,'A_z':zg,'V_z':zv,'state':state,'receipt':receipt}

def main():
 check();OUT.mkdir(exist_ok=False);(OUT/'states_PRIVATE').mkdir();(OUT/'predictions_PRIVATE').mkdir();(OUT/'input_contract_PRIVATE').mkdir();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.perf_counter();contexts={};results={};receipts=[]
 try:
  for pl in ['drone','quad']:
   A=[r for r in old.rows('D208') if r['platform']==pl];B=[r for r in old.rows('V64') if r['platform']==pl];assert len(A)==104 and len(B)==32;contexts[pl]=(A,B,old.batch(A,True),old.batch(B));aa=contexts[pl][2];bb=contexts[pl][3];np.savez_compressed(OUT/'input_contract_PRIVATE'/(pl+'.npz'),A_ids=np.array([r['id'] for r in A]),V_ids=np.array([r['id'] for r in B]),teacher_F32=aa['teacher'].cpu().numpy(),teacher_pT=aa['pT'].cpu().numpy(),source_p0=aa['p0'].cpu().numpy(),A_mask=aa['mask'].cpu().numpy(),A_padding=aa['padding'].cpu().numpy(),V_mask=bb['mask'].cpu().numpy(),V_padding=bb['padding'].cpu().numpy(),source_scores=aa['scores'].cpu().numpy())
  # Both historical controls must pass before any new-coordinate/null fitting.
  for pl in ['drone','quad']:
   A,B,a,b=contexts[pl];x=fit_one(pl,'full_probability',A,B,a,b);results[pl,'full_probability']=x;receipts.append(x['receipt'])
  dump(OUT/'MATCHED_CONTROL_REPLAY.json',{'state':'PASSED_BOTH_PLATFORMS_BEFORE_OTHER_STATES','receipts':receipts.copy(),'no_GT_or_taskscore':True})
  for name in CONFIGS[1:]:
   for pl in ['drone','quad']:
    A,B,a,b=contexts[pl];x=fit_one(pl,name,A,B,a,b);results[pl,name]=x;receipts.append(x['receipt'])
  for pl in ['drone','quad']:
   A,B,a,b=contexts[pl]
   for split,rr,key in [('D208',A,'A'),('V64',B,'V')]:
    for j,r in enumerate(rr):
     z=old.feature(r);fullG=old.teacher(r,z);delta=fullG-z['scores'].astype(float);score={'source':z['scores'],**{n:results[pl,n][key][j] for n in CONFIGS},'direct_G':fullG,'full_probability_plus_current_G':results[pl,'full_probability'][key][j].astype(float)+delta,'full_token_logodds_plus_current_G':results[pl,'full_token_logodds'][key][j].astype(float)+delta,'zero_delta_memory':z['scores']};ss=np.stack([score[n] for n in ARMS]);q=ss.argmax(-1);payload={'arm_names':np.array(ARMS),'scores':ss,'queries':q,'boxes':z['boxes'][q],'teacher_F32':fullG.astype(np.float32),'teacher_F64':fullG,'deltaG':delta,'source_native_probability':z['scores'],'trained_token_logodds':np.stack([results[pl,n]['A_z' if key=='A' else 'V_z'][j] for n in CONFIGS])}
     if split=='V64':
      mp=ROOT/'crossplatform/round24/memory_attempt001/V64_predictions/predictions_PRIVATE'/(r['id']+'.npz');mm=npz(mp);mi=mm['arm_names'].tolist().index('memory_a1.0');mq=int(mm['query_indices'][mi]);payload.update(legacy_fullG_memory_query=np.array(mq),legacy_fullG_memory_box=z['boxes'][mq],legacy_fullG_memory_original_file=np.array(str(mp.relative_to(ROOT))),legacy_fullG_memory_original_sha256=np.array(sha(mp)))
     np.savez_compressed(OUT/'predictions_PRIVATE'/(r['id']+'.npz'),**payload)
  dump(OUT/'FIT_RECEIPTS.json',receipts);files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()};dump(OUT/'SEALED.json',{'state':'R43_ALL8_HEADS_A208_V64_PREDICTIONS_BEFORE_TASKSCORE','files':files,'n':272,'A':208,'V':64,'states':8,'optimizer_steps':128,'same_forward_fullmodel_closure':False,'all3Dboxes_original':True,'no_GT_or_newG_or_FINAL_U_T':True,'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'root_authorization_sha256':sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json'),'seconds':time.perf_counter()-start,'peak_bytes':torch.cuda.max_memory_allocated(),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print('R43_ALL_PREDICTIONS_SEALED',time.perf_counter()-start,flush=True)
 except BaseException as e:dump(OUT/('FAILED_'+str(time.time_ns())+'.json'),{'state':'MEASUREMENT_FAILURE_PRESERVED_STOP_NOT_METHOD_RESULT','type':type(e).__name__,'message':str(e),'completed_states':len(receipts),'automatic_semantic_retry':False});raise
if __name__=='__main__':main()
