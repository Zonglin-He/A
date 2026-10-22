"""Isolate new temporal-context information from already available endpoint logits."""
import os,sys,time,hashlib,json,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_innovation_hc64_v1';P=R/'results/native_event_innovation_hc64_v1';C=R/'outputs/native_event_context_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';OLD=R/'outputs/event_query_hc64_v1'
ARMS=['NativeG6','ContextInnovation','SameInformationContext','CensoredContext','Sharpen','ReverseInnovation','NoInnovation']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert sha(C/'sealed.json')==p['context_seal_sha256'];assert sha(TEXT/'sealed.json')==p['text_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p
def prepare():
 assert not(D/'plan.json').exists();parent=read(C/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_text_v1/core.py','research/native_event_context_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_innovation_hc64_v1',primary='ContextInnovation',arms=ARMS,count=64,authorization='Same user overnight research authorization, no new expert, space fixed',code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],captures=parent['captures'],prior_files=parent['prior_files'],context_files=read(C/'sealed.json')['files'],text_files=parent['text_files'],context_seal_sha256=sha(C/'sealed.json'),text_seal_sha256=sha(TEXT/'sealed.json'),labels_sha256=parent['labels_sha256'],
 rationale='Context primary loses6.8349pp tIoU and shortens8.648s to7.293s while candidate ranking53.96%. Full-information q0²/uFull loses10.4859pp: repeated original confidence/structural-hull removal can dominate new context evidence. RC059 end collapses320frames; RC056 shrinks both boundaries. Test context-induced changes versus exactly same cropped support/hull using unchanged full-context logits, rather than dividing by an uninformative hull.',
 fixed=dict(windows={'Prefix':[0,48],'Suffix':[16,64]},same_information='For each window slice all6 original Full logits at identical physical IDs and parity, then construct original two-offset hull/mean. No fusion/decoder rerun. Call resulting posterior bW.',primary='q0*exp(mean of censored log(qContextW/bW)); joint or available-boundary marginal as parent; unavailable endpoints unknown. Cancels confidence and support geometry already present in original logits. Unit exponent, no scan.',controls='SameInformationContext q0*exp(mean censored log(bW/uW)); prior CensoredContext; Sharpen; reverse innovation; qContext=bW exact no-op',space='same original per-interval complete-box DP and same Source/NativeG6 outputs'),
 GT_in_inference=False,all64_seal_before_new_GT=True,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_decoder=0,new_optimizer=0,maximum_CPU_seconds=600,stop_epoch=1792708800,exposure='Repeatedly exposed HC64 development, not independent',qualification=parent['qualification'],references=parent['references'],scope='Mechanism correction using cached context outputs; no claim conditional posterior is calibrated under event truncation.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','captures','prior_files','context_files','text_files']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
 import torch,numpy as np
 sys.path.insert(0,str(R/'external/TA-STVG'));import utils.misc,utils.video_list
 from stvg.global6 import posterior
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_text_v1.core import normalize
 from research.native_event_context_v1.core import censored_evidence
 p=verify();torch.set_num_threads(1);start=time.perf_counter();files={};put(D/'state.json',dict(status='predicting_CPU_cache',fingerprint=p['fingerprint'],**who()))
 for row in records():
  verify();assert time.perf_counter()-start<p['maximum_CPU_seconds'];k=row['id'];e=p['context_files'][k];assert sha(C/e['path'])==e['sha256']and sha(C/e['trace_path'])==e['trace_sha256'];tr=torch.load(C/e['trace_path'],weights_only=False,map_location='cpu');te=p['text_files'][k];assert sha(TEXT/te['trace_path'])==te['trace_sha256'];old=torch.load(TEXT/te['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];tick=time.perf_counter();innov={};same={};bases={};nullcheck=[]
  for kind,(lo,hi)in [('Prefix',(0,48)),('Suffix',(16,64))]:
   recs=[];logs=[];wids=ids[lo:hi+1]
   for offset,z in enumerate(old['views']['Full']['logits']):
    positions=[i for i in range(offset,65,2)if lo<=i<=hi];sel=[(i-offset)//2 for i in positions];recs.append(dict(offset=offset,flip=False,frame_ids=[ids[i]for i in positions]));logs.append(z[:,:,sel,:].contiguous())
   bases[kind]=posterior(recs,logs,wids);q=tr['intervened_q'][kind];u=tr['null_q'][kind];innov[kind],_=censored_evidence(ids,wids,q,bases[kind]);same[kind],_=censored_evidence(ids,wids,bases[kind],u);nullcheck.append(censored_evidence(ids,wids,bases[kind],bases[kind])[0])
  effect=(innov['Prefix']+innov['Suffix'])/2;info=(same['Prefix']+same['Suffix'])/2;q0=tr['q']['NativeG6'];l=np.zeros_like(q0);take=q0>0;l[take]=np.log(q0[take]);assert all(np.array_equal(x,np.zeros_like(q0))for x in nullcheck)
  qs=dict(NativeG6=q0,ContextInnovation=normalize(q0,effect),SameInformationContext=normalize(q0,info),CensoredContext=tr['q']['CensoredContext'],Sharpen=normalize(q0,l),ReverseInnovation=normalize(q0,effect[::-1,::-1].T.copy()),NoInnovation=q0.copy());iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
  with np.load(C/e['path'],allow_pickle=False)as z:
   for a in ['NativeG6','CensoredContext','Sharpen']:assert np.array_equal(arrays[a],z[a])and iv[a]==z[a+'_interval'].tolist()
   arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
  f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);out=D/'traces'/f'{k}.pt';out.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,bases=bases,context_innovation=innov,score=effect,same_information_score=info,GT_read=False,no_innovation_exact_noop=True),out);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(out.relative_to(D)),trace_sha256=sha(out),seconds=time.perf_counter()-tick);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,**who()));print(k,iv['NativeG6'],iv['ContextInnovation'],flush=True)
 verify();assert not torch.cuda.is_initialized();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=dict(new_expert=0,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_decoder=0,new_optimizer=0),seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
