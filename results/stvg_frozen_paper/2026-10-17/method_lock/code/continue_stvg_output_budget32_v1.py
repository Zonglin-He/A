"""One serial post-integration QCQP experiment; no old queue recovery."""
from pathlib import Path
import os,select,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_output_budget_qp_v1 import BASE,read,put,verify
PAPER=ROOT/'artifacts/stvg_paper_t2_b_v1';UNIT='stvg-output-budget32-001.service'
def main(pid):
 verify();args=Path('/proc')/str(pid)/'cmdline';assert 'run_stvg_paper_t2_b_v1.py' in args.read_bytes().replace(b'\0',b' ').decode();put(BASE/'LAUNCH.json',dict(status='waiting',stage='paper_integrated_measurement_dependency',unit=UNIT,controller_pid=os.getpid(),dependency_pid=pid,GT_read=False));put(BASE/'STATUS.json',dict(status='waiting',stage='paper_integrated_measurement_dependency',count=32,GT_read=False),replace=True);fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN)
 try:
  while not poll.poll(1000):verify()
 finally:os.close(fd)
 assert subprocess.check_output(['systemctl','--user','show','stvg-paper-t2-b-001.service','--property=Result','--value'],text=True).strip()=='success'
 for p in ('arms/B/trials/G32/ROOT_READBACK.json','arms/B/trials/G32/GLOBAL_PREDICTION_BARRIER.json','CPU_COMPLETION.json'):
  d=read(PAPER/p);assert d['count']==256 and d['status'] in ('complete','pass')
 from scripts import launch_stvg_instance_identity_v1 as resource
 resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='output_budget32';g=2**30;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start();put(BASE/'EXECUTION_START.json',dict(status='running',epoch=time.time(),compute_cap_seconds=14400,GT_read=False))
 def timeout():
  put(BASE/'TIMEOUT_RECEIPT.json',dict(status='finite_compute_cap_reached',all_partial_outputs_preserved=True,GT_read=False));subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=SIGTERM',UNIT],check=True)
 timer=threading.Timer(14400,timeout);timer.daemon=True;timer.start()
 try:
  for stage in ('student','root','score'):
   verify();out=BASE/'execution'/stage;out.mkdir(parents=True,exist_ok=False);put(BASE/'STATUS.json',dict(status='running',stage=stage,count=32,GT_read=stage=='score'),replace=True);script='run_stvg_output_budget32_v1.py' if stage=='student' else 'score_stvg_output_budget32_v1.py';extra=[] if stage=='student' else [stage]
   with (out/'worker.log').open('xb') as log:
    tick=time.perf_counter();p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts'/script),*extra],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait()
   put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,(stage,rc)
  put(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,root_science_actual_pixels_anonymous_remote_archive_required=True,GT_read=False));put(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',count=32,GT_read=False),replace=True)
 except BaseException as e:put(BASE/'FAILURE.json',dict(status='technical_failure',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False));raise
 finally:timer.cancel();stop.set()
if __name__=='__main__':main(int(sys.argv[1]))
