"""Two-process scoring/cold audit; original numeric blocks and merge order exact."""
from pathlib import Path
import os,sys,time,json,hashlib,inspect,importlib.util,subprocess,fcntl,signal,traceback,gc
R=Path(__file__).resolve().parents[2];OLD=R/'studies/temporal_feedback_risk_v1'
sys.path.insert(0,str(OLD))
import common as c
spec=importlib.util.spec_from_file_location('_unchanged_temporal_feedback_risk',OLD/'run_v1.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
D=R/'outputs/temporal_feedback_risk_acceleration_v2'
P=R/'results/temporal_feedback_risk_acceleration_v2'
CAP=160*2**30;RESERVE=4*2**30;MAX_JSON=512*2**20
def put(p,x):
 data=json.dumps(x,indent=2)+'\n';assert len(data.encode())<MAX_JSON
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(data);os.replace(q,p)
def live(w):
 try:s=(Path('/proc')/str(w['pid'])/'stat').read_text().split(') ',1)[1].split();return int(s[19])==w['startticks']and s[0]!='Z'
 except FileNotFoundError:return False
def private_size():return sum(p.stat().st_size for root in [c.D,D]for p in root.rglob('*')if p.is_file())
class Guard:
 def __init__(self):
  self.start=time.perf_counter();self.base=private_size();assert self.base+RESERVE<CAP
 def __call__(self,started):
  assert time.perf_counter()-started<259200 and c.shutil.disk_usage(R).free>=80*2**30
  assert next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:'))>=8*2**30
  assert self.base+RESERVE<CAP
def blocks(stage):
 src=inspect.getsource(getattr(original,stage))
 split='\n pub=P/ds;'if stage=='score'else"\n rr=read(folder/'metrics.json')"
 i=src.index(split);prefix=src[:i];suffix=src[i:]
 if stage=='audit':
  changed=prefix.replace('del a;gc.collect()','del a;\n  if i%32==0:gc.collect()')
  assert changed.replace('del a;\n  if i%32==0:gc.collect()','del a;gc.collect()')==prefix
  prefix=changed
 code=prefix.replace('def '+stage+'(ds):','def _prefix(ds):',1)
 code+='\n return dict(rows=out,checks=checks,maxerr=maxerr,elapsed_wall_seconds=time.perf_counter()-start)\n'if stage=='score'else'\n return dict(sha_checks=sha_checks,elapsed_wall_seconds=time.perf_counter()-start)\n'
 return src,code,suffix
def validate():
 c.validate();p=c.read(D/'execution_plan_v2.json');assert p['code_sha256']==c.sha(Path(__file__))
 assert p['original_science_plan_sha256']==c.sha(c.D/'plan_v1.json')
 for f,h in p['original_code_sha256'].items():assert c.sha(R/f)==h,f
 return p
def namespace(stage,rows,progress):
 ns=original.__dict__.copy();guard=Guard();ns['resources']=guard;ns['records']=lambda ds:rows
 def atomic(path,x):
  assert Path(path)==c.D/'vidstg'/(stage+'_progress.json')or Path(path)==c.D/'hc2'/(stage+'_progress.json')
  put(progress,x)
 ns['atomic']=atomic;_,code,_=blocks(stage);exec(compile(code,str(Path(__file__))+'::unchanged_'+stage+'_prefix','exec'),ns);return ns
def preflight():
 validate();rows=c.records('hc2')[:4];ns=namespace('score',rows,D/'preflight_progress.json');t=time.perf_counter();a=ns['_prefix']('hc2');old={r['id']:r for r in c.read(c.D/'hc2/metrics.json')['private_rows']}
 assert len(a['rows'])==4 and all(row==old[row['id']]for row in a['rows'])
 proof={}
 for stage in ['score','audit']:
  src,prefix,suffix=blocks(stage);proof[stage]=dict(original_function_source_sha256=hashlib.sha256(src.encode()).hexdigest(),generated_prefix_source_sha256=hashlib.sha256(prefix.encode()).hexdigest(),unchanged_original_suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest(),numeric_metric_derive_equality_provenance_and_tolerance_statements_unchanged=True)
 rec=dict(status='four_actual_published_HC_eight_arm_official_metrics_entire_rows_exact_and_original_numeric_blocks_preserved',cases=4,extra_official_checks=32,elapsed_wall_seconds=time.perf_counter()-t,block_proofs=proof,worker=c.identity(),new_model=0,new_Source=0,new_expert=0,new_optimizer=0,no_HC_whole_domain_rerun=True)
 put(D/'preflight_v2.json',rec);print(json.dumps(rec))
def shard(stage,part):
 validate();assert c.read(D/'preflight_v2.json')['cases']==4;rows=c.records('vidstg');cut=len(rows)//2;rr=rows[:cut]if part==0 else rows[cut:];folder=D/stage/str(part);assert not(folder/'result.json').exists();put(folder/'worker.json',c.identity())
 if stage=='audit':
  from stvg.runtime import guard
  sys.addaudithook(guard)
 ns=namespace(stage,rr,folder/'progress.json');start=time.perf_counter();x=ns['_prefix']('vidstg');x.update(query_ids=[r['id']for r in rr],queries=len(rr),worker=c.identity(),original_stage=stage,new_model=0,new_Source=0,new_expert=0,new_optimizer=0);put(folder/'result.json',x);put(folder/'receipt.json',dict(status='whole_disjoint_shard_complete_no_prefix_accuracy_published',queries=len(rr),result_sha256=c.sha(folder/'result.json'),worker=c.identity(),elapsed_wall_seconds=time.perf_counter()-start));assert private_size()<CAP
def merge(stage):
 validate();guard=Guard();guard(time.perf_counter());xs=[]
 for i in [0,1]:
  folder=D/stage/str(i);e=c.read(folder/'receipt.json');assert c.sha(folder/'result.json')==e['result_sha256'];xs.append(c.read(folder/'result.json'))
 ids=[r['id']for r in c.records('vidstg')];assert xs[0]['query_ids']+xs[1]['query_ids']==ids and len(set(ids))==10303
 ns=original.__dict__.copy();ns['resources']=guard
 allowed={c.D/'vidstg/metrics.json',c.D/'vidstg/score_receipt.json',c.D/'vidstg/reporting_receipt.json'}
 def atomic(path,x):
  p=Path(path);assert p in allowed or p.is_relative_to(c.P);put(p,x)
 ns['atomic']=atomic;_,_,suffix=blocks(stage)
 args='ds,out,checks,maxerr,start'if stage=='score'else'ds,sha_checks,start'
 code='def _merge('+args+'):\n import torch,numpy as np\n from full_score import metric,cluster\n torch.set_num_threads(1)'+suffix+'\n';exec(compile(code,str(Path(__file__))+'::original_'+stage+'_suffix','exec'),ns)
 start=c.read(D/(stage+'_start.json'))['monotonic_start']
 if stage=='score':
  out=xs[0]['rows']+xs[1]['rows'];assert [r['id']for r in out]==ids
  ns['_merge']('vidstg',out,sum(x['checks']for x in xs),max(x['maxerr']for x in xs),start)
 else:ns['_merge']('vidstg',sum(x['sha_checks']for x in xs),start)
 assert private_size()<CAP;put(D/(stage+'_merge_receipt.json'),dict(status='original_suffix_complete_in_original_full_manifest_order',queries=10303,worker=c.identity(),merged_shards=[c.sha(D/stage/str(i)/'receipt.json')for i in [0,1]],canonical_receipt_sha256=c.sha(c.D/'vidstg'/('score_receipt.json'if stage=='score'else'reporting_receipt.json')),original_numeric_suffix_unchanged=True))
def queue():
 p=validate();lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(D/'controller_v2.json').exists();who=c.identity();put(D/'controller_v2.json',who);prior=c.read(D/'preserved_original_serial_score/preservation.json');owner=prior['owner'];complete=False;children=[]
 try:
  assert live(owner)and(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()[0]=='T'
  for stage in ['score','audit']:
   put(D/'state_v2.json',dict(status=stage+'_two_disjoint_CPU_processes',controller=who));put(D/(stage+'_start.json'),dict(monotonic_start=time.perf_counter(),wall_time=time.time()));children=[]
   for part,cpu in [(0,14),(1,15)]:
    log=(D/(stage+'_'+str(part)+'.log')).open('xb');args=['taskset','-c',str(cpu),'nice','-n','19','ionice','-c','3',sys.executable,'-B',str(Path(__file__)),'shard',stage,str(part)];ch=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});put(D/(stage+'_'+str(part)+'_child.json'),c.identity(ch.pid));children.append((part,ch,log))
   for part,ch,log in children:
    code=ch.wait();log.close();put(D/(stage+'_'+str(part)+'_exit.json'),dict(code=code));assert code==0,stage+' shard failed; preserve and resume original'
   with(D/(stage+'_merge.log')).open('xb')as log:
    ch=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(Path(__file__)),'merge',stage],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});put(D/(stage+'_merge_child.json'),c.identity(ch.pid));code=ch.wait()
   put(D/(stage+'_merge_exit.json'),dict(code=code));assert code==0,stage+' merge failed'
  assert c.read(c.D/'vidstg/reporting_receipt.json')['all_arrays_cold_rederived_exact'];complete=True
  originals=[prior['owner'],prior['child']];assert all(live(x)for x in originals)
  os.killpg(owner['pid'],signal.SIGTERM);os.killpg(owner['pid'],signal.SIGCONT)
  for _ in range(200):
   if not any(live(x)for x in originals):break
   time.sleep(.05)
  assert not any(live(x)for x in originals)
  put(D/'original_owned_exit.json',dict(status='intentional_exit_after_accelerated_full_score_and_independent_full_cold_audit_verified',original_processes=originals,actual_owned_exit_verified=True,OS_wait_return_code_unavailable=True,SIGTERM_then_SIGCONT_deliver_pending_termination=True,old_partial_uncheckpointed_metric_work_not_reused=True,old_completed_query_lower_bound=prior['completed_lower_bound'],original_CPU_cost_retained_in_preservation=True))
  put(D/'state_v2.json',dict(status='Vid_whole_official_scored_and_allarray_audited_pending_root_semantic_cost_image_publication',controller=who,whole_authorized_scope_complete=False))
 except BaseException:
  put(D/'failure_v2.json',dict(error=traceback.format_exc(),worker=who,old_prediction_and_science_unchanged=True));raise
 finally:
  for _,ch,log in children:
   if ch.poll()is None:
    ch.terminate()
    try:ch.wait(timeout=10)
    except subprocess.TimeoutExpired:ch.kill();ch.wait()
   if not log.closed:log.close()
  if not complete and live(owner):
   os.killpg(owner['pid'],signal.SIGCONT);put(D/'original_serial_CPU_resume_after_failure.json',dict(**owner,SIGCONT_sent=True,original_RAM_preserved=True))
if __name__=='__main__':
 cmd=sys.argv[1]
 if cmd=='preflight':preflight()
 elif cmd=='shard':shard(sys.argv[2],int(sys.argv[3]))
 elif cmd=='merge':merge(sys.argv[2])
 elif cmd=='queue':queue()
