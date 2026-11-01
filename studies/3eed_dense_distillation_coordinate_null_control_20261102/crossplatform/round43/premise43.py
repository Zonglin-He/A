"""Approved CPU-only teacher representability premise; no model, GT, or fit."""
import json,hashlib,datetime,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'crossplatform/round43';EXP=ROOT/'research/round43/premise'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def check():
 p=read(EXP/'PLAN_LOCK.json');i=read(EXP/'IMPLEMENTATION_LOCK.json');assert i['plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 for f,h in dict(p['input_sha256'],**i['program_sha256']).items():assert sha(ROOT/f)==h,f
 return p

def lse(v):
 a=np.asarray(v,np.float64);m=a.max(-1,keepdims=True);return (m+np.log(np.exp(a-m).sum(-1,keepdims=True))).squeeze(-1)
def distribution(v):
 logs=np.asarray(v,np.float64)-lse(v);return np.exp(logs),logs

def entropy(p,logs):return float(-(p*logs).sum())
def KL(p,logs,otherlogs):return float((p*(logs-otherlogs)).sum())
def source_z(z):
 q=z['query'].astype(np.float64);t=z['tokens'].astype(np.float64);mask=np.asarray(z['token_mask'],bool);assert q.shape==(256,64) and t.shape[1]==64 and mask.shape==(len(t),) and mask.any() and (~mask).any();v=q@t.T/.07;return lse(v[:,mask])-lse(v[:,~mask])
def peak_bound_KL(t,u):
 if t<=u:return 0.
 return float(t*np.log(t/u)+(1-t)*np.log((1-t)/(1-u))) if t<1 else float(np.log(1/u))

def main():
 check();import torch
 torch.set_num_threads(4);assert not torch.cuda.is_initialized();rr=read(ROOT/'research/round24/input/D208_PRIVATE.json');assert len(rr)==208 and len({r['id'] for r in rr})==208;capdir=ROOT/'crossplatform/round24/capture336_attempt001';bankdir=ROOT/'crossplatform/round23/bank272_attempt001';cs=read(capdir/'SEALED.json');bs=read(bankdir/'SEALED.json');out=BASE/'teacher_premise208_attempt001';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();(out/'rows_PRIVATE').mkdir();cap=float(np.exp(4)/(255+np.exp(4)));toy=np.zeros(256);toy[0]=1;assert abs(distribution(toy/.25)[0].max()-cap)<1e-15;toy[0]=2;unbounded_peak=float(distribution(toy/.25)[0].max());assert unbounded_peak>cap;records=[];start=time.perf_counter()
 for r in rr:
  fn='samples_PRIVATE/'+r['id']+'.npz';fp=capdir/fn;bp=bankdir/fn;assert sha(fp)==cs['files'][fn] and sha(bp)==bs['files'][fn];z=npz(fp);b=npz(bp);s=z['scores'];assert s.shape==(256,) and s.dtype==np.float32 and np.isfinite(s).all();assert all(b[k].shape==(256,) and np.isfinite(b[k]).all() for k in ['vision','position','relation'])
  # Exact arithmetic/order of the original R24 teacher construction and cast.
  F64=s.astype(float)+2*b['vision']+2*b['position']+2*b['relation'];F32=F64.astype(np.float32);delta=F64-s.astype(float);pg,lg=distribution(F32.astype(np.float64)/.25);p0,l0=distribution(s.astype(np.float64)/.25);z0=source_z(z);pz,lz=distribution(z0/.25)
  with torch.inference_mode():pg32=torch.tensor(F32,device='cpu').div(.25).softmax(-1).numpy();p032=torch.tensor(s,device='cpu').div(.25).softmax(-1).numpy()
  qG=int(F32.argmax());q0=int(s.argmax());fullhinge=max(.05+float(np.delete(s,qG).max())-float(s[qG]),0.);nullhinge=max(.05+float(np.delete(s,q0).max())-float(s[q0]),0.);rangeF=float(F32.max()-F32.min());constant=bool(np.all(delta==delta[0]));zero=bool(np.all(delta==0));allcomponentzero=all(bool(np.all(b[k]==0)) for k in ['vision','position','relation']);signs={k:{'positive':int((b[k]>0).sum()),'negative':int((b[k]<0).sum()),'zero':int((b[k]==0).sum())} for k in ['vision','position','relation']}
  rec={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'F32_range':rangeF,'F32_range_gt1':rangeF>1,'teacher_peak_CPU64':float(pg.max()),'teacher_peak_CPU32':float(pg32.max()),'teacher_peak_gt_ideal_native_cap':bool(pg.max()>cap),'teacher_entropy_nats':entropy(pg,lg),'source_probability_posterior_entropy_nats':entropy(p0,l0),'source_z_posterior_entropy_nats':entropy(pz,lz),'KL_teacher_to_source_probability':KL(pg,lg,l0),'KL_teacher_to_source_z':KL(pg,lg,lz),'null_KL_source_probability_to_source_probability':KL(p0,l0,l0),'null_KL_source_probability_to_source_z':KL(p0,l0,lz),'deltaG_allzero':zero,'deltaG_constant':constant,'all_component_evidence_zero':allcomponentzero,'deltaG_range':float(delta.max()-delta.min()),'teacher_argmax_differs_source':qG!=q0,'initial_native_hinge_fullG':fullhinge,'initial_native_hinge_nullG':nullhinge,'teacher_peak_Bernoulli_KL_lowerbound':peak_bound_KL(float(pg.max()),cap),'evidence_sign_counts':signs,'CPU32_CPU64_teacher_posterior_maxerror':float(abs(pg32.astype(float)-pg).max()),'CPU32_CPU64_source_posterior_maxerror':float(abs(p032.astype(float)-p0).max()),'source_file_sha256':sha(fp),'bank_file_sha256':sha(bp),'no_GT_or_filtering':True,'zero_numeric_value_not_semantic_UNKNOWN_label':True}
  records.append(rec);np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),source_native_probability=s,vision=b['vision'],position=b['position'],relation=b['relation'],teacher_F64=F64,teacher_F32=F32,deltaG=delta,teacher_posterior_CPU64=pg,teacher_posterior_CPU32=pg32,source_probability_posterior_CPU64=p0,source_probability_posterior_CPU32=p032,source_token_logodds_CPU64=z0,source_z_posterior_CPU64=pz);dump(out/'rows_PRIVATE'/(r['id']+'.json'),rec)
 assert not torch.cuda.is_initialized();dump(out/'ALL_ROWS_PRIVATE.json',records);files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};dump(out/'SEALED.json',{'state':'R43_ALL208_LABEL_FREE_PREMISE_VECTORS_SEALED','files':files,'n':208,'no_GPU_model_fit_GT_taskscore':True,'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'implementation_sha256':sha(EXP/'IMPLEMENTATION_LOCK.json'),'seconds':time.perf_counter()-start,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 summary={}
 for pl in ['drone','quad','all']:
  rs=[r for r in records if pl=='all' or r['platform']==pl];N=len(rs);v={'n':N,'F_range_gt1':sum(r['F32_range_gt1'] for r in rs),'teacher_peak_gt_cap':sum(r['teacher_peak_gt_ideal_native_cap'] for r in rs),'deltaG_zero_rows':sum(r['deltaG_allzero'] for r in rs),'deltaG_constant_rows':sum(r['deltaG_constant'] for r in rs),'all_component_zero_rows':sum(r['all_component_evidence_zero'] for r in rs),'teacher_argmax_changed_from_source':sum(r['teacher_argmax_differs_source'] for r in rs),'evidence_sign_counts':{k:{a:sum(r['evidence_sign_counts'][k][a] for r in rs) for a in ['positive','negative','zero']} for k in ['vision','position','relation']}}
  for k in ['F32_range','teacher_peak_CPU64','teacher_entropy_nats','source_probability_posterior_entropy_nats','source_z_posterior_entropy_nats','KL_teacher_to_source_probability','KL_teacher_to_source_z','null_KL_source_probability_to_source_z','teacher_peak_Bernoulli_KL_lowerbound','initial_native_hinge_fullG','initial_native_hinge_nullG']:
   a=np.array([r[k] for r in rs]);v[k]={'min':float(a.min()),'median':float(np.median(a)),'mean':float(a.mean()),'max':float(a.max())}
  for group,flag in [('zero_delta','deltaG_allzero'),('constant_delta','deltaG_constant')]:
   ss=[r for r in rs if r[flag]];v[group]={'n':len(ss),'mean_teacher_entropy':float(np.mean([r['teacher_entropy_nats'] for r in ss])) if ss else None,'mean_source_probability_entropy':float(np.mean([r['source_probability_posterior_entropy_nats'] for r in ss])) if ss else None,'mean_source_z_entropy':float(np.mean([r['source_z_posterior_entropy_nats'] for r in ss])) if ss else None,'mean_KL_teacher_to_source_probability':float(np.mean([r['KL_teacher_to_source_probability'] for r in ss])) if ss else None,'mean_KL_teacher_to_source_z':float(np.mean([r['KL_teacher_to_source_z'] for r in ss])) if ss else None}
  summary[pl]=v
 result={'state':'COMPLETED_LABEL_FREE_TEACHER_EXPRESSION_PREMISE_NOT_TASK_RESULT','ideal_native_probability_posterior_cap':cap,'summary':summary,'rank_adapter_counterexample':{'bounded_probability_claim_applies':False,'old_linear_residual_has_no_range_norm_constraint':True,'toy_unbounded_score_span2_peak':unbounded_peak,'not_new_model_or_actual_adapter_trial':True},'unknown_preservation':'All original positive/negative/zero component arrays copied exactly with file hashes; no invented UNKNOWN labels fromzero; no row/candidate excluded','numeric_contract':'primaryCPU64softmax ofexactR24F32; CPUtorch32 recorded notGPUbitparityclaim; sourcezfromcachedsourcequery/tokens only','training_and_task_scoring_authorized':False,'prediction_seal_sha256':sha(out/'SEALED.json')};dump(EXP/'SUMMARY.json',result);print('R43_PREMISE_COMPLETED',json.dumps({p:{k:summary[p][k] for k in ['n','F_range_gt1','teacher_peak_gt_cap','deltaG_zero_rows','deltaG_constant_rows']} for p in summary}));print('R43_NULL_COORDINATE',json.dumps({p:summary[p]['null_KL_source_probability_to_source_z'] for p in summary}))
if __name__=='__main__':main()
