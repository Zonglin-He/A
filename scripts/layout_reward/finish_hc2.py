"""External scope limiter: unchanged inference; stop only its verified group at HC128."""
import os,json,time,signal,hashlib,datetime,subprocess,traceback
from pathlib import Path
T=Path(os.environ['STVG_PROJECT_ROOT']).resolve()/'outputs/layout_reward_dev';D=T/'scope_hc2'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def ident(pid):
 p=Path('/proc')/str(pid)
 try:
  stat=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,status=stat[0],ppid=int(stat[1]),pgid=int(stat[2]),startticks=stat[19],command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
 except FileNotFoundError:return None
try:
 scope=read(D/'scope.json');ctl=read(T/'controller.json');pid=ctl['pid'];ticks=ctl['startticks'];start=time.monotonic()
 while True:
  z=ident(pid);assert z and z['startticks']==ticks and z['status']!='Z','Original inference exited unexpectedly'
  progress=read(T/'progress.json')
  if progress['completed']>=128:break
  assert time.monotonic()-start<3600,'HC completion deadline exceeded'
  time.sleep(.02)
 assert progress['completed']==128 and progress['dataset']=='hc2','Unexpected scope boundary'
 assert z['pgid']==pid and '/workspace/run_trial.py' in z['command']
 os.kill(pid,signal.SIGSTOP)
 members=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   a=ident(int(p.name))
   if a and a['pgid']==pid and a['status']!='Z':members.append(a)
 assert members and all(a['pid']==pid or 'stvg.layout_teacher_worker' in a['command'] for a in members),'Unexpected process group member'
 for a in members:
  if a['pid']!=pid:os.kill(a['pid'],signal.SIGSTOP)
 assert not list((T/'predictions').glob('BV*.npz')),'Vid prediction appeared'
 files={}
 for key in scope['ids']:
  p=T/'predictions'/(key+'.npz');r=read(p.with_suffix('.json'))
  assert r['fingerprint']==scope['inference_fingerprint'] and r['GT_read'] is False and sha(p)==r['sha256']
  files[key]=dict(path='predictions/'+key+'.npz',sha256=sha(p),receipt_sha256=sha(p.with_suffix('.json')))
 write(D/'boundary.json',dict(progress=progress,processes=members,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_128_predictions_verified=True,no_Vid_predictions=True))
 for a in members:os.kill(a['pid'],signal.SIGTERM);os.kill(a['pid'],signal.SIGCONT)
 for _ in range(200):
  active=[a for a in members if (ident(a['pid']) or {}).get('status') not in (None,'Z')]
  if not active:break
  time.sleep(.05)
 assert not active,'Owned processes did not exit'
 write(D/'sealed.json',dict(status='sealed_before_offline_scoring',GT_read=False,fingerprint=scope['inference_fingerprint'],queries=128,dataset='hc2',manifest_sha256=scope['manifest_sha256'],scope_sha256=sha(D/'scope.json'),files=files))
 run=read(T/'run.json');run.update(status='completed_hc2_sealed_by_user_scope',original_planned_queries=256,queries=128,excluded_dataset='vidstg',scope_seal='scope_hc2/sealed.json');write(T/'run.json',run)
 queue=read(T/'queue.json');queue.update(status='hc2_sealed_awaiting_report',controller_exited=True);write(T/'queue.json',queue)
 write(D/'progress.json',dict(completed=128,total=128,dataset='hc2',status='sealed'))
 score=subprocess.run([str(T/'workspace/.conda/tubedetr/bin/python'),'-B',str(T/'workspace/score_hc2.py')],cwd=T/'workspace',env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=(D/'score.log').open('w'),stderr=subprocess.STDOUT)
 assert score.returncode==0,'HC-only CPU scoring failed; see score.log'
 write(D/'completion.json',dict(status='scored_awaiting_publication',queries=128,metrics_sha256=sha(D/'metrics.json'),formal_remains_paused=True))
except BaseException:
 (D/'limiter_error.log').write_text(traceback.format_exc());raise
