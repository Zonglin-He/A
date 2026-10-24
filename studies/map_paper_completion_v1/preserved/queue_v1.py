"""One owned GPU queue and two CPU slots for the newly authorized remaining suite."""
from common import *
import argparse,subprocess,traceback,signal,shutil
JOBS=['published_TTA','corruptions','spatial_hc2','spatial_vidstg','qualitative']
def prepare():
 assert not(D/'execution_plan.json').exists();files=[*S.glob('*.py')];inputs=[D/'plan.json',D/'temporal_v2_plan.json',D/'spatial_plan.json',D/'deployment_plan.json',D/'corruptions_plan.json',D/'published_TTA_plan.json',D/'qualitative_plan.json',OLD/'supports_sealed.json',OLD/'shifts_plan.json',OLD/'shifts_confirm_sealed.json',R/'outputs/paper_followup_v1/tta_selected.json']
 p=dict(schema='MAP_paper_completion_queue_v1',suite_fingerprint=read(D/'plan.json')['fingerprint'],GPU_order=['finish_already_running_deployment','published_TTA_predict','corruptions_predict','spatial_hc2_predict','spatial_vidstg_predict'],CPU='Continue2existingtemporaldomains afterdeployment; launchpostseal officialscore→exactaudit orqualitativerender as2slotsfree. GPUproducerindependent; no prefixGT. QualitativePNG rootmustview.',limits=dict(CPU_slots=2,CPU_affinity=[14,15],nice=19,ionice='idleclass3',GPU_workers=1,min_available_RAM=8*2**30,min_free_disk=80*2**30),failure='Preserveactualfailure/noautoretry/skip/Sourcefallback; resumeonlyownCPUgroups pausedforcost onexit. Neverrestartoldqueues.',code_sha256={str(f.relative_to(R)):sha(f)for f in files},input_sha256={str(f):sha(f)for f in inputs});p['fingerprint']=digest(p);atomic(D/'execution_plan.json',p);atomic(P/'execution_protocol.json',p);print(p['fingerprint'])
def validate_queue():
 p=read(D/'execution_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for f,h in {**p['code_sha256'],**p['input_sha256']}.items():assert sha(R/f if not Path(f).is_absolute()else f)==h,f
 return p
def preflight_inputs():
 import torch
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);validate_queue();support=read(OLD/'supports_sealed.json');proof=[]
 for ds,groups in read(D/'selection.json').items():
  seal=fixed(RUN/ds/'sealed.json')
  for row in groups['confirm']:
   e=support['files'][ds]['confirm'][row['id']];f=OLD/e['shared_path'];assert sha(f)==e['shared_sha256'];raw=torch.load(f,map_location='cpu',weights_only=False,mmap=True);parent=seal['files'][row['id']];assert raw['parent_capture_sha256']==parent['capture_sha256'];assert sha(raw['parent_capture'])==raw['parent_capture_sha256'];cap=torch.load(raw['parent_capture'],map_location='cpu',weights_only=False,mmap=True);assert raw['frame_ids']==cap['frame_ids'];assert read(RUN/ds/parent['receipt_path'])['input_binding']==binding(row,read(RUN/'plan.json')['provider_runtime_fingerprint']);assert not raw['GT_read'];proof.append(dict(dataset=ds,shared_sha256=e['shared_sha256'],parent_capture_sha256=parent['capture_sha256']));del raw,cap
 atomic(D/'GPU_context_preflight_receipt.json',dict(status='passed',queries=256,all_shared_contexts_exact=True,code_sha256=sha(Path(__file__)),proof=proof,GT_read=False,new_Source=0,new_expert=0,new_GPU=0));atomic(P/'GPU_context_preflight_receipt.json',dict(status='passed',queries=256,all_shared_contexts_exact=True,GT_read=False,new_Source=0,new_expert=0,new_GPU=0))
def available_RAM():
 return int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
def cpu_job(name):
 validate_queue();who=identity();atomic(D/f'{name}_CPU_worker.json',who)
 commands=([('tta.py','score'),('tta.py','audit')]if name=='published_TTA'else[('corruptions.py','score'),('corruptions.py','audit')]if name=='corruptions'else[('qualitative.py','run')]if name=='qualitative'else[('spatial.py','score'),('spatial.py','audit')]);ds=name.split('_',1)[1]if name.startswith('spatial_')else None
 try:
  for script,command in commands:
   validate_queue();args=[sys.executable,'-B',str(S/script),command]+(['--dataset',ds]if ds else[]);log=(D/f'{name}_{command}.log').open('ab');c=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});atomic(D/f'{name}_{command}_child.json',identity(c.pid));rc=c.wait();log.close();atomic(D/f'{name}_{command}_exit.json',dict(code=rc));assert rc==0,'Preserved CPU stage failure'
  atomic(D/f'{name}_CPU_state.json',dict(status='complete',worker=who))
 except BaseException:atomic(D/f'{name}_CPU_state.json',dict(status='failed_preserved',worker=who,error=traceback.format_exc()));raise

def queue():
 validate_queue();assert not(D/'controller.json').exists(),'No duplicate suite queue';who=identity();atomic(D/'controller.json',who);atomic(D/'launch.json',dict(controller=who,suite_fingerprint=read(D/'plan.json')['fingerprint'],execution_fingerprint=read(D/'execution_plan.json')['fingerprint'],time=time.time()));cpu={};resumed=False
 def resume_own():
  nonlocal resumed
  if resumed:return
  pa=read(D/'deployment_CPU_pause.json')
  for w in pa['groups']:
   if not owned_exit(w):assert os.getpgid(w['pid'])==w['pid'];os.killpg(w['pid'],signal.SIGCONT)
  atomic(D/'deployment_CPU_resume.json',dict(status='only_own_temporal_groups_resumed',groups=pa['groups'],time=time.time()));resumed=True
 def existing_CPU():
  return [read(D/f'temporal_v2_{ds}_domain_child.json')for ds in COUNTS if not owned_exit(read(D/f'temporal_v2_{ds}_domain_child.json'))]
 def ready(name):
  return (G/'sealed.json').exists()if name=='published_TTA'else(D/'corruptions/sealed.json').exists()if name=='corruptions'else resumed if name=='qualitative'else(D/'spatial'/f'{name.split("_",1)[1]}_sealed.json').exists()
 def finished(name):
  return (D/'published_TTA/reporting_receipt.json').exists()if name=='published_TTA'else(D/'corruptions/reporting_receipt.json').exists()if name=='corruptions'else(D/'qualitative/render_receipt.json').exists()if name=='qualitative'else(D/'spatial'/f'{name.split("_",1)[1]}_reporting_receipt.json').exists()
 def maintain_CPU():
  for name,(c,w)in list(cpu.items()):
   if c.poll()is not None:assert c.returncode==0 and finished(name),'CPU stage failed, preserve';cpu.pop(name)
  for ds in COUNTS:
   f=D/f'temporal_v2_{ds}_state.json'
   if f.exists():assert not read(f)['status'].startswith('failed'),'Temporal failure preserved'
  for name in JOBS:
   if name in cpu or finished(name)or not ready(name):continue
   if len(cpu)+len(existing_CPU())>=2 or available_RAM()<8*2**30:break
   assert not(D/f'{name}_CPU_worker.json').exists(),'CPU receipt missing after oldworker, no retry';log=(D/f'{name}_CPU.log').open('ab');args=['taskset','-c','14,15','nice','-n','19','ionice','-c','3',sys.executable,'-B',__file__,'cpu-job','--name',name];c=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},start_new_session=True);log.close();w=identity(c.pid);cpu[name]=(c,w);atomic(D/f'{name}_CPU_child.json',w)
 def gpu_command(name,script,command,ds=None):
  validate_queue()
  while available_RAM()<8*2**30:maintain_CPU();time.sleep(5)
  assert shutil.disk_usage(R).free>80*2**30;assert not(D/f'{name}_GPU_child.json').exists(),'No GPU stage repeat';log=(D/f'{name}_GPU.log').open('ab');args=[sys.executable,'-B',str(S/script),command]+(['--dataset',ds]if ds else[]);c=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);w=identity(c.pid);atomic(D/f'{name}_GPU_child.json',w);atomic(D/'state.json',dict(status=name+'_running',controller=who,active_GPU=w,CPU_jobs={k:v[1]for k,v in cpu.items()},suite_fingerprint=read(D/'plan.json')['fingerprint']))
  while c.poll()is None:maintain_CPU();time.sleep(5)
  log.close();atomic(D/f'{name}_GPU_exit.json',dict(code=c.returncode));assert c.returncode==0,'Preserve GPU paidattempts and evidence; no retry'
 try:
  w=read(D/'deployment_child.json');atomic(D/'state.json',dict(status='waiting_actual_deployment_worker',controller=who,worker=w))
  while not owned_exit(w):time.sleep(5)
  resume_own();assert read(D/'deployment/state.json')['status']=='complete_measured_verified','Timing failed, futureGPUstages blocked';assert read(D/'deployment/sealed.json')['all_current_MAP_outputs_bitwise'];validate_queue()
  log=(D/'GPU_context_preflight.log').open('ab');c=subprocess.Popen([sys.executable,'-B',__file__,'preflight-inputs'],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});atomic(D/'GPU_context_preflight_child.json',identity(c.pid));assert c.wait()==0,'Context bridge failed before nextGPU';log.close()
  gpu_command('published_TTA','tta.py','predict');maintain_CPU();gpu_command('corruptions','corruptions.py','predict');maintain_CPU()
  for ds in COUNTS:gpu_command('spatial_'+ds,'spatial.py','predict',ds);maintain_CPU()
  while cpu or any(not finished(name)for name in JOBS)or existing_CPU():maintain_CPU();time.sleep(5)
  assert all((D/f'temporal_v2_{ds}_reporting_receipt.json').exists()for ds in COUNTS);atomic(D/'state.json',dict(status='all_measurements_scored_audited_pending_root_visual_review_analysis_export_publication',controller=who,suite_fingerprint=read(D/'plan.json')['fingerprint'],remaining=['rootviewqualitativePNG','timingpairedCIanalysis','allstageanalysis','publicationALLremoteverify'],time=time.time()))
 except BaseException:resume_own();atomic(D/'state.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc(),time=time.time()));raise
G=D/'published_TTA'
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','queue','cpu-job','preflight-inputs']);a.add_argument('--name',choices=JOBS);z=a.parse_args();dict(prepare=prepare,queue=queue,**{'cpu-job':lambda:cpu_job(z.name),'preflight-inputs':preflight_inputs})[z.command]()
