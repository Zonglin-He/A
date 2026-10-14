"""Finite official Table 1 prediction chain, never a recurring monitor.

Wait for actual complete Source and preflight roots, then process fixed whole
video/clock batches. One GPU stage at a time; no target labels or method search.
"""
from pathlib import Path
import argparse,collections,hashlib,json,os,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
B=ROOT/'artifacts/stvg_final_table1_v1';FULL=B/'FULL'
BASE=FULL/'batch_execution007';UNIT='stvg-table1-full-batches-007.service';GiB=1024**3
PY=ROOT/'.venv/stvg_native_distill_v1/bin/python'
from scripts import launch_stvg_instance_identity_v1 as resource
resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='waiting'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=canonical(v)
 if p.exists():assert p.read_bytes()==data,p;return
 tmp=p.with_name(p.name+'.tmp')
 with tmp.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.link(tmp,p);tmp.unlink()
def rel(p):return str(Path(p).relative_to(ROOT))
def status(stage,**kw):resource.write(BASE/'STATUS.json',dict(status=stage,GT_read=False,time=time.time(),**kw),mutable=True)
def wait_source():
 status('waiting_actual_full_source_exit_and_preflight_root')
 while True:
  # The estimate uses only the completed independent preflight. It may be
  # emitted while full Source capture is running, before any full teacher fit.
  if (B/'ROOT_PREFLIGHT64_READBACK.json').exists() and not (FULL/'PREFLIGHT_COST_FORECAST001.json').exists():
   run_stage('preflight_cost','forecast',['scripts/report_stvg_table1_preflight_cost_v1.py'],gpu=False)
   status('waiting_actual_full_source_exit_and_preflight_root',preflight_cost_estimate_complete=True)
  values=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show','stvg-table1-full-source-006.service','-p','ActiveState','-p','Result'],text=True).splitlines()if '='in x)
  if values['ActiveState']not in ('active','activating','deactivating'):break
  time.sleep(30)
 assert values['Result']=='success',values
 assert read(FULL/'source_execution006/STATUS.json')['status']=='source_complete_pending_expert_student'
 p=read(B/'ROOT_PREFLIGHT64_READBACK.json');assert p['status']=='pass'and p['count']==64 and not p['GT_read']
 assert read(FULL/'source_execution006/ROOT_REPEAT_HANDOFF.json')['status']=='pass'
 s=read(FULL/'SOURCE_CAPTURE_BARRIER.json');assert s['status']=='sealed'and s['count']==13785 and not s['GT_read']

def lock_runtimes(lock):
 # Instantiate missing dependency digests only after their actual immutable
 # barriers exist. The worker source/protocol/weights are already root-pinned.
 common=lock['worker_pins'];protected=lock['protected']
 roster=sha(FULL/'FULL_SOURCE_ROSTER_LOCK.json');manifest=sha(B/'INPUT_MANIFEST.json')
 source=sha(FULL/'SOURCE_RUNTIME.json');barrier=sha(FULL/'SOURCE_CAPTURE_BARRIER.json')
 tracking={'schema':'stvg-final-table1-full-tracking-runtime-v1','status':'locked','GT_read':False,
  'pins':common,'protected':protected,'bindings':{
   'full_source_runtime_sha256':source,'source_capture_barrier_sha256':barrier,
   'full_source_roster_sha256':roster,'input_manifest_sha256':manifest,
   'preflight_source_runtime_sha256':sha(B/'SOURCE_RUNTIME.json'),
   'preflight_tracking_runtime_sha256':sha(B/'TRACKING_RUNTIME.json')}}
 put(FULL/'TRACKING_RUNTIME.json',tracking)
 teacher={'schema':'stvg-final-table1-full-teacher-runtime-v1','status':'locked','GT_read':False,
  'pins':common,'protected':protected,'bindings':{
   'full_source_roster_sha256':roster,'input_manifest_sha256':manifest,'source_runtime_sha256':source,
   'source_capture_barrier_sha256':barrier,'tracking_runtime_sha256':sha(FULL/'TRACKING_RUNTIME.json'),
   'r1_execution_lock_sha256':sha(B/'R1_EXECUTION_LOCK.json')},
  'execution':lock['teacher_execution']}
 put(FULL/'TEACHER_RUNTIME.json',teacher)
 student={'schema':'stvg-final-table1-full-student-runtime-v1','status':'locked','GT_read':False,
  'pins':common,'protected':protected,'full_source_roster_sha256':roster,
  'source_barrier_sha256':barrier,'source_runtime_sha256':source,'teacher_runtime_sha256':sha(FULL/'TEACHER_RUNTIME.json')}
 put(FULL/'STUDENT_RUNTIME.json',student)

def make_batches(groups_per_batch):
 from scripts import run_stvg_table1_full_tracking_v1 as tracking
 from scripts import run_stvg_table1_full_teacher_v1 as teacher
 from scripts import run_stvg_table1_full_student_v1 as student
 rows=read(FULL/'FULL_SOURCE_ROSTER_LOCK.json')['records'];order={r['probe_alias']:i for i,r in enumerate(rows)}
 groups=collections.OrderedDict();student_partition={}
 for row in rows:
  identity=tracking._video_clock_identity(row)
  assert identity==teacher.video_clock_identity(row)
  group=hashlib.sha256(canonical(identity)).hexdigest();groups.setdefault(group,[]).append(row['probe_alias'])
  k=student._group_key(row)
  assert student_partition.setdefault(k,group)==group
 assert len(student_partition)==len(groups)
 batches=[];items=list(groups.items())
 for n in range(0,len(items),groups_per_batch):
  aliases=sorted([a for _,g in items[n:n+groups_per_batch]for a in g],key=order.__getitem__)
  selection=hashlib.sha256(canonical(aliases)).hexdigest()
  path=FULL/'batch_aliases'/(selection+'.json')
  put(path,{'schema':'stvg-final-table1-alias-batch-v1','aliases':aliases})
  batches.append({'index':len(batches),'aliases':aliases,'selection_sha256':selection,'alias_file_path':rel(path),'alias_file_sha256':sha(path)})
 assert len({a for x in batches for a in x['aliases']})==13785 and sum(len(x['aliases'])for x in batches)==13785
 plan={'schema':'stvg-final-table1-fixed-batch-plan-v1','status':'locked','count':13785,'video_clock_group_count':len(groups),
       'groups_per_batch':groups_per_batch,'batch_count':len(batches),'batches':batches,'GT_read':False,
       'full_source_roster_sha256':sha(FULL/'FULL_SOURCE_ROSTER_LOCK.json')}
 put(FULL/'BATCH_PLAN007.json',plan);return batches

def run_stage(batch_label,name,args,gpu=True):
 resource.STAGE=batch_label+'_'+name
 if gpu:
  v=resource.sample();assert v['available']>=8*GiB and v['disk_free']>=30*GiB and v['gpu_free']>=(26 if name=='teacher' else 22)*GiB,v
 folder=BASE/batch_label;folder.mkdir(parents=True,exist_ok=True)
 out=folder/(name+'_EXIT.json')
 if out.exists():
  assert read(out)['exit_code']==0,'failed stage needs separately reviewed recovery, not silent retry'
  return
 status('running',batch=batch_label,stage=name)
 with (folder/(name+'.log')).open('xb')as log:
  child=subprocess.Popen([str(PY),'-B',*args],cwd=ROOT,env=dict(os.environ,
   PYTHONPATH=':'.join(map(str,[ROOT,ROOT/'external/yolov12',ROOT/'external/Grounded-SAM-2-main'])),
   PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
  resource.write(folder/(name+'_LAUNCH.json'),dict(status='running',pid=child.pid,GT_read=False,time=time.time()))
  code=child.wait()
 resource.write(out,dict(status='actual_worker_exited',exit_code=code,GT_read=False,time=time.time()))
 if code:raise RuntimeError(f'{batch_label}/{name} actual worker exit {code}; preserved for root review')

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-batches',action='store_true');a=p.parse_args()
 lock=read(FULL/'BATCH_EXECUTION_LOCK007.json')
 for relative,digest in {**lock['pins'],**lock['protected']}.items():assert sha(ROOT/relative)==digest,relative
 if a.prepare_batches:
  batches=make_batches(lock['groups_per_batch']);print(json.dumps(dict(status='pass',batch_count=len(batches),count=13785,GT_read=False,CUDA_used=False)));return
 wait_source()
 run_stage('source_readback','baseline_custody',['scripts/review_stvg_table1_full_source_baseline_v1.py'],gpu=False)
 lock_runtimes(lock);batches=make_batches(lock['groups_per_batch'])
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if '='in x)
 assert [int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*GiB,20*GiB,GiB]
 stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 from scripts import run_stvg_table1_full_student_v1 as student
 try:
  for batch in batches:
   label=f'batch_{batch["index"]:04d}';selection=batch['selection_sha256'];aliaspath=batch['alias_file_path']
   tracking=FULL/f'TRACKING_BATCH_{selection}_BARRIER.json'
   teacher=FULL/'TEACHER_BATCH_BARRIERS'/f'TEACHER_BATCH_{selection}_BARRIER.json'
   run_stage(label,'tracking',['scripts/run_stvg_table1_full_tracking_v1.py','--aliases-file',aliaspath])
   run_stage(label,'teacher',['scripts/run_stvg_table1_full_teacher_v1.py','--aliases-file',aliaspath,'--tracking-barrier',rel(tracking)])
   batch_id=student._teacher_batch_id(batch['aliases'],batch['alias_file_sha256'],sha(teacher))
   studentbarrier=FULL/f'STUDENT_BATCH_BARRIER_{batch_id}.json'
   run_stage(label,'student',['scripts/run_stvg_table1_full_student_v1.py','--aliases-file',aliaspath,'--teacher-barrier',rel(teacher)])
   root=FULL/'ROOT_BATCH_READBACKS'/(selection+'.json');root.parent.mkdir(parents=True,exist_ok=True)
   run_stage(label,'root',['scripts/review_stvg_table1_full_batch_v1.py','--student-barrier',rel(studentbarrier),'--output',rel(root)],gpu=False)
   run_stage(label,'auxiliary_retire',['scripts/stvg_table1_full_auxiliary_cache_v1.py','--retire','--tracking',rel(tracking),'--teacher',rel(teacher),'--student',rel(studentbarrier),'--root',rel(root)],gpu=False)
   run_stage(label,'retire',['scripts/stvg_table1_full_cache_retirement_v1.py','--tracking',rel(tracking),'--teacher',rel(teacher),'--student',rel(studentbarrier),'--root',rel(root)],gpu=False)
  for phase in ['tracking','teacher','student']:
   run_stage('finalize',phase,[f'scripts/run_stvg_table1_full_{phase}_v1.py','--finalize'],gpu=False)
  run_stage('finalize','auxiliary_custody',['scripts/stvg_table1_full_auxiliary_cache_v1.py','--verify-all','--output',rel(FULL/'ROOT_AUXILIARY_CUSTODY_READBACK.json')],gpu=False)
  run_stage('finalize','root',['scripts/review_stvg_table1_full_batch_v1.py','--global','--output',rel(FULL/'ROOT_ALL_PREDICTIONS_READBACK.json')],gpu=False)
  status('all_13785_predictions_sealed_and_root_read_pending_evaluation',count=13785,not_FINAL=True)
 except Exception as e:
  status('failed_pending_root_engineering_review',error_type=type(e).__name__,error=str(e));raise
 finally:stop.set()
if __name__=='__main__':
 try:main()
 except Exception as e:
  status('failed_pending_root_engineering_review',error_type=type(e).__name__,error=str(e))
  raise
