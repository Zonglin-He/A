"""Finite serial 64 complete qualifications, 128 timing paths, original CPU root."""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
from scripts.run_stvg_opd_deployment_priority002 import BASE,LAT,verify,stopped
CLAIM=LAT.parent
def run():
    verify();stopped()
    from scripts.stvg_opd_expanded_state_common_v1 import ALLOCATOR
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def child(script,args,CPU=False):
            verify();stopped();cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B',script]+args
            log=BASE/'logs'/((('_'.join(args)) or Path(script).stem)+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':ALLOCATOR,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            if CPU:env['CUDA_VISIBLE_DEVICES']=''
            else:env.pop('CUDA_VISIBLE_DEVICES',None)
            with log.open('a') as f:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
                v=dict(status='running',stage=args[0] if args else Path(script).stem,dataset=args[1] if len(args)>1 else None,
                    controller_pid=os.getpid(),worker_pid=p.pid,CPU_only=CPU,GT_read=False,
                    original_science_unchanged=True,experiment_complete=False,command=cmd,time=time.time())
                status(BASE/'STATUS.json',v);status(CLAIM/'STATUS.json',dict(v,latest_serial_revision='deployment_latency/after_expanded_priority002'))
                assert p.wait()==0,'Actual child failure preserved: '+str(cmd)
        if not (LAT/'ROOT_QUALIFICATION_READBACK.json').exists():
            for ds in ['hc2','vidstg']:
                if not (LAT/'qualification'/ds/'COMPLETION.json').exists():child('scripts/run_stvg_opd_deployment_priority002.py',['qualification',ds])
            child('scripts/run_stvg_opd_deployment_latency_v1.py',['root_qualification'],True)
        q=read(LAT/'ROOT_QUALIFICATION_READBACK.json');assert q['status']=='pass' and q['actual_GPU_fits']==64
        write(BASE/'ROOT_PRIORITY_QUALIFICATION_READBACK.json',dict(status='pass',scope='complete original 32 query/64 actual complete GPU fit/trace-off/math/native/expert root qualification under measured storage policy; not future OOM guarantee or timed-path completion',
            qualification_unique_queries=32,actual_GPU_fits=64,actual_new_DINO_calls=q['actual_new_DINO_calls'],
            root_sha256=sha(LAT/'ROOT_QUALIFICATION_READBACK.json'),CPU_only=True,GT_read=False,time=time.time()))
        for ds in ['hc2','vidstg']:
            for method,repeat in [('Frozen',0),('Full',0),('Full',1),('Frozen',1)]:
                if not (LAT/'timing'/ds/method/str(repeat)/'COMPLETION.json').exists():
                    child('scripts/run_stvg_opd_deployment_priority002.py',['timing',ds,method,str(repeat)])
        if not (LAT/'CPU_COMPLETION.json').exists():
            subprocess.run(['/home/wwww/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node',
                'container_tools/mark_artifact_operation_started.mjs','--operation-kind','create',
                '--expected-output-count','1','--output-format','pdf'],
                cwd='/home/wwww/.codex/plugins/cache/openai-primary-runtime/pdf/26.1007.11041/skills/pdf',check=True)
            child('scripts/finalize_stvg_opd_deployment_latency_v1.py',[],True)
        assert read(LAT/'ROOT_COMPLETE_READBACK.json')['status']=='pass'
        v=dict(status='pending_actual_deployment_timing_root_PDF_pixel_view_publication_archive_then_LN_cost',
            scope='original 128 complete timed paths/32 matched queries/two repeats; not whole C/LN/cost completion',
            actual_timed_query_paths=128,root_sha256=sha(LAT/'ROOT_COMPLETE_READBACK.json'),
            experiment_complete=False,CPU_only=True,GT_read=False,controller_pid=os.getpid(),time=time.time())
        write(BASE/'CPU_COMPLETION.json',v);status(BASE/'STATUS.json',v);status(CLAIM/'STATUS.json',dict(v,latest_serial_revision='deployment_latency/after_expanded_priority002'))
if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),CPU_only=True,GT_read=False,time=time.time()));raise
