"""Independent CPU-only numeric premise replay. No task GT, training or model."""
import json,hashlib,time,collections
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
EXP=ROOT/'research/round43/premise';SEALED=ROOT/'crossplatform/round43/teacher_premise208_attempt001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+'\n')
def npz(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def posterior(x):
 logs=np.asarray(x,np.float64)-logsumexp(np.asarray(x,np.float64));return np.exp(logs),logs
def ent(p,l):return -float(np.dot(p,l))
def kl(p,l,q):return float(np.dot(p,l-q))
def near(a,b,tol=1e-10):
 if isinstance(a,(bool,np.bool_)) or isinstance(b,(bool,np.bool_)):assert bool(a)==bool(b),(a,b)
 else:assert abs(float(a)-float(b))<=tol,(a,b,tol)
def main():
 start=time.perf_counter();import torch
 torch.set_num_threads(4);assert not torch.cuda.is_initialized()
 plan=read(EXP/'PLAN_LOCK.json');impl=read(EXP/'IMPLEMENTATION_LOCK.json');assert impl['plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 for n,h in {**plan['input_sha256'],**impl['program_sha256']}.items():assert sha(ROOT/n)==h
 sseal=read(SEALED/'SEALED.json');assert sseal['plan_sha256']==sha(EXP/'PLAN_LOCK.json') and sseal['implementation_sha256']==sha(EXP/'IMPLEMENTATION_LOCK.json')
 for n,h in sseal['files'].items():assert sha(SEALED/n)==h
 summary=read(EXP/'SUMMARY.json');assert summary['prediction_seal_sha256']==sha(SEALED/'SEALED.json')
 cohort=read(ROOT/'research/round24/input/D208_PRIVATE.json');assert len(cohort)==208 and collections.Counter(r['platform'] for r in cohort)=={'drone':104,'quad':104}
 oldsrc=ROOT/'crossplatform/round24/capture336_attempt001';oldbank=ROOT/'crossplatform/round23/bank272_attempt001';ss=read(oldsrc/'SEALED.json');bs=read(oldbank/'SEALED.json');stored={r['id_PRIVATE']:r for r in read(SEALED/'ALL_ROWS_PRIVATE.json')};cap=1/(1+255*np.exp(-4));near(cap,summary['ideal_native_probability_posterior_cap'],1e-15);records=[];errors=collections.defaultdict(float);outside=0;source_low=1.;source_high=0.;tokens=[]
 # A sufficient attainable-coordinate upper bound; not a model-parameter oracle.
 one=np.zeros(256,np.float64);one[0]=1;near(posterior(one/.25)[0].max(),cap,1e-15);one[0]=2;unbounded=float(posterior(one/.25)[0].max());near(unbounded,summary['rank_adapter_counterexample']['toy_unbounded_score_span2_peak']);assert unbounded>cap and summary['rank_adapter_counterexample']['bounded_probability_claim_applies'] is False
 for r in cohort:
  n='samples_PRIVATE/'+r['id']+'.npz';sp=oldsrc/n;bp=oldbank/n;assert sha(sp)==ss['files'][n] and sha(bp)==bs['files'][n];z=npz(sp);b=npz(bp);v=npz(SEALED/n);s=z['scores'];assert s.dtype==np.float32 and s.shape==(256,) and np.isfinite(s).all();source_low=min(source_low,float(s.min()));source_high=max(source_high,float(s.max()));outside+=int(((s<0)|(s>1)).sum())
  F64=np.add(np.add(np.add(s.astype(np.float64),np.multiply(b['vision'],2)),np.multiply(b['position'],2)),np.multiply(b['relation'],2));F32=F64.astype(np.float32);delta=np.subtract(F64,s.astype(np.float64));assert np.array_equal(F64,v['teacher_F64']) and np.array_equal(F32,v['teacher_F32']) and np.array_equal(delta,v['deltaG']) and np.array_equal(s,v['source_native_probability'])
  for key in ['vision','position','relation']:assert np.array_equal(b[key],v[key])
  pG,lG=posterior(F32.astype(np.float64)*4);p0,l0=posterior(s.astype(np.float64)*4);mask=z['token_mask'].astype(bool);assert mask.any() and (~mask).any();matrix=np.matmul(z['query'].astype(np.float64),z['tokens'].astype(np.float64).T)/.07;z0=logsumexp(matrix[:,mask],axis=1)-logsumexp(matrix[:,~mask],axis=1);pz,lz=posterior(z0*4);tokens.append({'positive':int(mask.sum()),'all':len(mask)})
  for key,new in [('teacher_posterior_CPU64',pG),('source_probability_posterior_CPU64',p0),('source_token_logodds_CPU64',z0),('source_z_posterior_CPU64',pz)]:
   e=float(abs(new-v[key]).max());errors[key]=max(errors[key],e);assert e<=2e-12,(r['alias'],key,e)
  with torch.inference_mode():p32=torch.tensor(F32,device='cpu').mul(4).softmax(0).numpy();s32=torch.tensor(s,device='cpu').mul(4).softmax(0).numpy()
  assert np.array_equal(p32,v['teacher_posterior_CPU32']) and np.array_equal(s32,v['source_probability_posterior_CPU32'])
  qg=int(np.argmax(F32));q0=int(np.argmax(s));otherG=max(float(s[i]) for i in range(256) if i!=qg);other0=max(float(s[i]) for i in range(256) if i!=q0);rangeF=float(np.float32(F32.max()-F32.min()));peak=float(pG.max());lower=0. if peak<=cap else peak*np.log(peak/cap)+(1-peak)*np.log((1-peak)/(1-cap))
  rec={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'F32_range':rangeF,'F32_range_gt1':rangeF>1,'teacher_peak_CPU64':peak,'teacher_peak_CPU32':float(p32.max()),'teacher_peak_gt_ideal_native_cap':peak>cap,'teacher_entropy_nats':ent(pG,lG),'source_probability_posterior_entropy_nats':ent(p0,l0),'source_z_posterior_entropy_nats':ent(pz,lz),'KL_teacher_to_source_probability':kl(pG,lG,l0),'KL_teacher_to_source_z':kl(pG,lG,lz),'null_KL_source_probability_to_source_probability':kl(p0,l0,l0),'null_KL_source_probability_to_source_z':kl(p0,l0,lz),'deltaG_allzero':bool((delta==0).all()),'deltaG_constant':bool((delta==delta[0]).all()),'all_component_evidence_zero':all((b[k]==0).all() for k in ['vision','position','relation']),'deltaG_range':float(delta.max()-delta.min()),'teacher_argmax_differs_source':qg!=q0,'initial_native_hinge_fullG':max(.05+otherG-float(s[qg]),0),'initial_native_hinge_nullG':max(.05+other0-float(s[q0]),0),'teacher_peak_Bernoulli_KL_lowerbound':float(lower),'CPU32_CPU64_teacher_posterior_maxerror':float(abs(p32.astype(float)-pG).max()),'CPU32_CPU64_source_posterior_maxerror':float(abs(s32.astype(float)-p0).max()),'source_file_sha256':sha(sp),'bank_file_sha256':sha(bp),'evidence_sign_counts':{k:{'positive':int((b[k]>0).sum()),'negative':int((b[k]<0).sum()),'zero':int((b[k]==0).sum())} for k in ['vision','position','relation']}}
  ref=stored[r['id']]
  for k,new in rec.items():
   if k in ['alias','id_PRIVATE','platform','source_file_sha256','bank_file_sha256','evidence_sign_counts']:assert new==ref[k],(r['alias'],k)
   else:near(new,ref[k]);errors['scalar_'+k]=max(errors['scalar_'+k],abs(float(new)-float(ref[k])))
  rec['F32_interpreted_real_exact_span']=float(F32.astype(np.float64).max()-F32.astype(np.float64).min());records.append(rec)
 # All count, quantile and group reductions recomputed independently.
 reductions={}
 for pl in ['drone','quad','all']:
  rr=[r for r in records if pl=='all' or r['platform']==pl];v={'n':len(rr),'F_range_gt1':sum(r['F32_range_gt1'] for r in rr),'teacher_peak_gt_cap':sum(r['teacher_peak_gt_ideal_native_cap'] for r in rr),'deltaG_zero_rows':sum(r['deltaG_allzero'] for r in rr),'deltaG_constant_rows':sum(r['deltaG_constant'] for r in rr),'all_component_zero_rows':sum(r['all_component_evidence_zero'] for r in rr),'teacher_argmax_changed_from_source':sum(r['teacher_argmax_differs_source'] for r in rr),'evidence_sign_counts':{k:{sgn:sum(r['evidence_sign_counts'][k][sgn] for r in rr) for sgn in ['positive','negative','zero']} for k in ['vision','position','relation']}}
  reference=summary['summary'][pl]
  for k,x in v.items():assert x==reference[k],(pl,k,x,reference[k])
  for k in ['F32_range','teacher_peak_CPU64','teacher_entropy_nats','source_probability_posterior_entropy_nats','source_z_posterior_entropy_nats','KL_teacher_to_source_probability','KL_teacher_to_source_z','null_KL_source_probability_to_source_z','teacher_peak_Bernoulli_KL_lowerbound','initial_native_hinge_fullG','initial_native_hinge_nullG']:
   a=np.array([r[k] for r in rr]);v[k]={'min':float(a.min()),'median':float(np.quantile(a,.5)),'mean':float(a.mean()),'max':float(a.max())}
   for stat,val in v[k].items():near(val,reference[k][stat])
  for group,flag in [('zero_delta','deltaG_allzero'),('constant_delta','deltaG_constant')]:
   sub=[r for r in rr if r[flag]];keys={'mean_teacher_entropy':'teacher_entropy_nats','mean_source_probability_entropy':'source_probability_posterior_entropy_nats','mean_source_z_entropy':'source_z_posterior_entropy_nats','mean_KL_teacher_to_source_probability':'KL_teacher_to_source_probability','mean_KL_teacher_to_source_z':'KL_teacher_to_source_z'};v[group]={'n':len(sub),**{n:float(np.mean([r[k] for r in sub])) for n,k in keys.items()}}
   for k,val in v[group].items():near(val,reference[group][k])
  reductions[pl]=v
 assert not torch.cuda.is_initialized()
 result={'state':'PASS_INDEPENDENT_LABEL_FREE_PREMISE_REPLAY','n':208,'candidates_per_row':256,'teacher_F64_F32_cast_and_all_evidence_arrays_exact':True,'all_CPUtorch32_posteriors_bit_exact':True,'sourcez_recomputed_with_actual_tokenmask_and_all_tokens':True,'ideal_native_cap':float(cap),'max_numeric_errors':dict(errors),'all_scalar_and_platform_reductions_reproduced':True,'counts':{pl:{k:reductions[pl][k] for k in ['n','F_range_gt1','teacher_peak_gt_cap','deltaG_zero_rows','deltaG_constant_rows']} for pl in reductions},'null_initialKL_mean':reductions['all']['null_KL_source_probability_to_source_z']['mean'],'zeroDeltaG_mean_teacher_to_z_KL':reductions['all']['zero_delta']['mean_KL_teacher_to_source_z'],'exact_real_span_gt1':sum(r['F32_interpreted_real_exact_span']>1 for r in records),'rounded_span_count_differs_exact':any((r['F32_interpreted_real_exact_span']>1)!=r['F32_range_gt1'] for r in records),'source_probability_min':source_low,'source_probability_max':source_high,'source_probability_components_outside_ideal01':outside,'positive_token_count_range':[min(t['positive'] for t in tokens),max(t['positive'] for t in tokens)],'all_token_count_range':[min(t['all'] for t in tokens),max(t['all'] for t in tokens)],'rank_adapter_unbounded_counterexample_reproduced':True,'sealed_input_sha256':sha(SEALED/'SEALED.json'),'no_GT_source_training_GPU_model_optimizer_or_FINAL':True,'semantic_reliability_and_task_gain_not_established':True,'no_author_outputs_modified':True,'seconds':time.perf_counter()-start};dump(OUT/'AUDIT_RESULT.json',result);dump(OUT/'INDEPENDENT_ROWS_PRIVATE.json',records);dump(OUT/'INDEPENDENT_REDUCTIONS.json',reductions);print(json.dumps(read(OUT/'AUDIT_RESULT.json'),indent=2))
if __name__=='__main__':main()
