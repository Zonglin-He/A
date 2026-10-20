"""Finite formal evaluation controller. Planning alone never starts model inference.

Two TA-STVG jobs run sequentially; CGSTVG is deferred until this stage is complete. Jobs run in child processes. No retries, scheduling,
parameter selection, query filtering or expert changes are performed here.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

JOBS = [('tastvg', 'hc2'), ('tastvg', 'vidstg')]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(temporary,path)


def make_plan(root):
    root=Path(root).resolve();files={}
    for directory in ['stvg','tools','configs','external/TA-STVG','external/ReferDINO','external/CGSTVG']:
        for p in (root/directory).rglob('*'):
            if p.is_file() and not {'.git','__pycache__','build'}.intersection(p.parts) and p.suffix in {'.py','.json','.yaml','.yml','.cpp','.cu','.h','.cuh'}:
                files[str(p.relative_to(root))]=digest(p)
    for relative in ['docs/assets.json','requirements.txt','requirements-teacher.txt','requirements-cgstvg.txt']:
        p=root/relative
        if p.is_file():files[relative]=digest(p)
    jobs=[]
    for backbone,dataset in JOBS:
        manifest=root/'data/evaluation/formal'/f'{dataset}.jsonl'
        labels=root/'data/evaluation'/f'{dataset}_labels.jsonl'
        if not labels.is_file():raise FileNotFoundError(labels)
        records=[json.loads(s)for s in manifest.read_text().splitlines()if s.strip()]
        if not records or len({r['id']for r in records})!=len(records):raise ValueError('Empty cohort or repeated query IDs')
        if any(r['dataset']!=dataset for r in records):raise ValueError('Dataset differs from locked job')
        jobs.append(dict(id=backbone+'-'+dataset,backbone=backbone,dataset=dataset,
            manifest=str(manifest.relative_to(root)),manifest_sha256=digest(manifest),queries=len(records),
            labels=str(labels.relative_to(root)),visual_cache_mib=0 if dataset=='hc2'else 4096,output=f'outputs/formal/{backbone}/{dataset}'))
    return dict(schema=1,root=str(root),python=sys.executable,code_and_configuration=files,jobs=jobs,
                teacher_cache='outputs/formal/teacher_cache',visual_cache_mib=4096,
                policy='finite sequential jobs; frozen parameters; fail immediately; offline scoring only after complete seal')


def validate_worker(root,job,require_seal):
    """Check binding before resuming; sealed skips also hash every prediction."""
    output=root/job['output'];run=json.loads((output/'run.json').read_text())
    if run.get('manifest_sha256')!=job['manifest_sha256'] or run.get('backbone')!=job['backbone'] or run.get('queries')!=job['queries'] or run.get('labels_read')is not False:
        raise ValueError('Worker run metadata differs from frozen formal job')
    expected=hashlib.sha256(canonical(dict(manifest_sha256=run['manifest_sha256'],runtime_fingerprint=run['runtime_fingerprint'],backbone=job['backbone']))).hexdigest()
    if run.get('resume_hash')!=expected:raise ValueError('Worker resume binding is invalid')
    proof=json.loads((output/'runtime_fingerprint.json').read_text())
    if hashlib.sha256(canonical(proof)).hexdigest()!=run['runtime_fingerprint']:raise ValueError('Worker runtime proof is not bound to run')
    # Read source/configuration, never labels. The worker checks full assets and runtime on partial resume.
    for relative,expected_code in proof.get('code',{}).items():
        logical=Path(relative)
        if logical.is_absolute()or '..'in logical.parts:raise ValueError('Unsafe saved runtime code path')
        p=root/logical
        if digest(p)!=expected_code:raise ValueError('Saved worker code differs from current runtime')
    sealpath=output/'sealed.json'
    if require_seal or sealpath.exists():
        seal=json.loads(sealpath.read_text())
        for key in ['manifest_sha256','resume_hash','runtime_fingerprint','queries']:
            if seal.get(key)!=run.get(key):raise ValueError('Prediction seal differs from worker run')
        if seal.get('labels_read')is not False or seal.get('status')!='sealed_before_offline_scoring':raise ValueError('Invalid prediction seal status')
        records=[json.loads(s)for s in (root/job['manifest']).read_text().splitlines()if s.strip()]
        if {r['id']for r in records}!=set(seal['files']):raise ValueError('Seal query set differs from complete cohort')
        for record in records:
            entry=seal['files'][record['id']];p=(output/entry['path']).resolve()
            if not p.is_relative_to(output.resolve()) or digest(p)!=entry['sha256']:raise ValueError('Sealed prediction changed')
            receipt=json.loads(p.with_suffix('.json').read_text())
            if receipt.get('resume_hash')!=expected or receipt.get('request_sha256')!=hashlib.sha256(canonical(record)).hexdigest() or receipt.get('prediction_sha256')!=entry['sha256'] or receipt.get('labels_read')is not False:
                raise ValueError('Sealed query receipt differs from exact input/run')
    return expected


def process_identity(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,PermissionError,ValueError,IndexError):return None


def launch(command,root,log,state,path,jobstate,phase,popen):
    env=dict(os.environ)
    if phase=='score':env['CUDA_VISIBLE_DEVICES']=''
    with log.open('ab') as handle:
        process=popen(command,cwd=str(root),stdout=handle,stderr=subprocess.STDOUT,env=env)
        event=dict(phase=phase,pid=process.pid,process_start_ticks=process_identity(process.pid),started_utc=now(),command=command,log=str(log.relative_to(root)))
        jobstate.setdefault('processes',[]).append(event)
        jobstate.update(status='scoring'if phase=='score'else'running',phase=phase,pid=process.pid,process_start_ticks=event['process_start_ticks'],started_utc=event['started_utc'],command=command,log=str(log.relative_to(root)))
        write(path,state)
        try:code=process.wait()
        except BaseException:
            # Only the child created by this controller is eligible for termination.
            if process.poll()is None:process.terminate();process.wait()
            raise
    event.update(finished_utc=now(),exit_code=code)
    jobstate.update(pid=None,finished_utc=event['finished_utc'],exit_code=code)
    write(path,state)
    if code:raise RuntimeError(f'{phase} exited with code {code}')


def controller(root,execute=False,resume=False,popen=None):
    root=Path(root).resolve();folder=root/'outputs/formal';folder.mkdir(parents=True,exist_ok=True)
    popen=popen or subprocess.Popen
    with (folder/'controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=make_plan(root);planpath=folder/'plan.json';statepath=folder/'state.json'
        if planpath.exists():
            if json.loads(planpath.read_text())!=plan:raise ValueError('Frozen plan changed; refusing resume or overwrite')
        else:write(planpath,plan)
        plan_hash=hashlib.sha256(canonical(plan)).hexdigest()
        state=json.loads(statepath.read_text())if statepath.exists()else dict(schema=1,plan_sha256=plan_hash,status='planned',created_utc=now(),jobs={j['id']:dict(status='pending')for j in plan['jobs']})
        if state.get('plan_sha256')!=plan_hash:raise ValueError('State belongs to a different plan')
        if not execute:
            write(statepath,state);print(json.dumps(dict(status='planned_only',plan=str(planpath),plan_sha256=plan_hash)));return state
        if not resume and any(j['status']!='pending'for j in state['jobs'].values()):raise ValueError('Existing started jobs require explicit --resume')
        for previous in state['jobs'].values():
            pid=previous.get('pid')
            if pid and previous.get('status')in {'running','scoring'}:
                identity=process_identity(pid)
                if identity is not None and identity==previous.get('process_start_ticks'):
                    raise RuntimeError('A previously launched child is still running; refusing duplicate execution')
        state.update(status='running',controller_pid=os.getpid(),updated_utc=now());write(statepath,state)
        active=None
        try:
            for job in plan['jobs']:
                active=state['jobs'][job['id']];output=root/job['output'];output.mkdir(parents=True,exist_ok=True)
                runpath=output/'run.json';sealpath=output/'sealed.json';resume_hash=None
                if runpath.exists():
                    if not resume:raise ValueError('Existing worker output requires explicit --resume')
                    resume_hash=validate_worker(root,job,require_seal=sealpath.exists())
                elif any(output.iterdir()):raise ValueError('Unbound existing output; refusing overwrite')
                if active['status']=='scored':
                    validate_worker(root,job,require_seal=True)
                    if digest(output/'metrics.json')!=active.get('metrics_sha256'):raise ValueError('Scored metrics changed')
                    continue
                if not sealpath.exists():
                    command=[sys.executable,'-B','-m','stvg.worker','--backbone',job['backbone'],'--manifest',str(root/job['manifest']),'--output',str(output),'--teacher-cache',str(root/plan['teacher_cache']),'--visual-cache-mib',str(job['visual_cache_mib'])]
                    if resume_hash:command+=['--resume-hash',resume_hash]
                    launch(command,root,folder/(job['id']+'-worker.log'),state,statepath,active,'worker',popen)
                validate_worker(root,job,require_seal=True)
                active.update(status='completed',completed_utc=now(),seal_sha256=digest(sealpath));write(statepath,state)
                metrics=output/'metrics.json'
                if metrics.exists():raise ValueError('Unbound existing metrics; refusing overwrite')
                command=[sys.executable,'-B','-m','stvg','score','--manifest',str(root/job['manifest']),'--labels',str(root/job['labels']),'--predictions',str(output),'--output',str(metrics)]
                launch(command,root,folder/(job['id']+'-score.log'),state,statepath,active,'score',popen)
                parsed=json.loads(metrics.read_text())
                if parsed.get('status')!='scored_sealed_predictions':raise ValueError('Score did not produce a completed report')
                active.update(status='scored',scored_utc=now(),metrics_sha256=digest(metrics));write(statepath,state)
            state.update(status='completed',finished_utc=now(),controller_pid=None);write(statepath,state);return state
        except BaseException as error:
            if active is not None:active.update(status='failed',error_type=type(error).__name__,error=str(error),failed_utc=now())
            state.update(status='failed',error_type=type(error).__name__,error=str(error),failed_utc=now(),controller_pid=None);write(statepath,state)
            raise



def finish_ta_stage(root):
    """Finite post-score tasks. Publication is a separate verified GitHub step."""
    root=Path(root).resolve();folder=root/'outputs/formal';statepath=folder/'state.json'
    with (folder/'finish.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(statepath.read_text());plan=json.loads((folder/'plan.json').read_text())
        if plan!=make_plan(root)or state['plan_sha256']!=hashlib.sha256(canonical(plan)).hexdigest():
            raise ValueError('Postprocessing does not match the frozen plan')
        if any(state['jobs'][j['id']]['status']!='scored'for j in plan['jobs']):
            raise ValueError('Both complete TA-STVG cohorts must be scored first')
        tasks=[]
        for dataset in ['hc2','vidstg']:
            tasks.append(('benchmark-'+dataset,[sys.executable,'-B','tools/benchmark_formal.py','--backbone','tastvg','--dataset',dataset],['outputs/formal/tastvg/'+dataset+'/benchmark.json']))
        tasks.append(('qualitative',[sys.executable,'-B','tools/qualitative.py','--backbone','tastvg'],['outputs/formal/qualitative/tastvg/hc2/selection.json','outputs/formal/qualitative/tastvg/vidstg/selection.json']))
        tasks.append(('report',[sys.executable,'-B','tools/report_formal.py'],['results/formal/report.md','results/formal/results.json']))
        for mode in ['check','snapshot','check']:
            name='archive-'+mode+('-final'if mode=='check'and any(t[0]=='archive-check'for t in tasks)else'')
            tasks.append((name,[sys.executable,'-B','scripts/research_archive.py',mode],['docs/sources.json']if mode=='snapshot'else[]))
        post=state.setdefault('postprocess',{});state.update(status='postprocessing',controller_pid=os.getpid());write(statepath,state)
        try:
            for name,command,outputs in tasks:
                if name=='archive-check':
                    history=root/'docs/history.md'
                    text=history.read_text()
                    current='当前任务：TA-STVG两个完整方向已封存评分，效率和定性检查、五行统一报告已完成，等待GitHub结果同步核验；CGSTVG继续暂停。方法和参数未改。'
                    lines=text.splitlines();lines=[current if line.startswith('当前任务：')else line for line in lines]
                    marker='TA-STVG完整阶段本地完成：'
                    if marker not in text:lines+=['',marker+'HC2 3482条、VidSTG 10303条全部保留；预测先封存后独立评分，效率与定性产物绑定封存哈希。当前结果见 results/formal/report.md。公开同步尚待核验，不将本地完成写成GitHub已完成。']
                    history.write_text('\n'.join(lines)+'\n')
                old=post.get(name)
                if old and old.get('status')=='completed':
                    if any(digest(root/p)!=h for p,h in old['outputs'].items()):raise ValueError('Completed postprocessing output changed: '+name)
                    continue
                task=post.setdefault(name,{})
                launch(command,root,folder/(name+'.log'),state,statepath,task,'postprocess',subprocess.Popen)
                task.update(status='completed',outputs={p:digest(root/p)for p in outputs});write(statepath,state)
            state.update(status='awaiting_github_sync',controller_pid=None,finished_utc=now(),
                         next_stage='CGSTVG remains paused until TA-STVG publication is verified')
            write(statepath,state)
        except BaseException as error:
            state.update(status='postprocess_failed',controller_pid=None,error_type=type(error).__name__,error=str(error));write(statepath,state)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Execute only the two TA-STVG jobs; CGSTVG is not queued in this stage')
    parser.add_argument('--resume',action='store_true',help='Resume only outputs bound to the exact frozen plan/run')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    controller(root,execute=args.run,resume=args.resume)
    if args.run:finish_ta_stage(root)

if __name__=='__main__':main()
