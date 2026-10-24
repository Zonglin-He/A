"""Two-slot CPU scheduling for unchanged MAP paper computations."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import sys, json, time, signal, subprocess, traceback
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from studies.map_components_v1 import run as science
D, P = science.D, science.P
PLAN = D / 'parallel_v2_plan.json'
JOBS = [('components', 'vidstg'), ('budget', 'hc2'), ('budget', 'vidstg')]

def prepare():
    p = science.validate()
    assert not PLAN.exists()
    assert science.read(D/'components_hc2_reporting_receipt.json')['all_arrays_cold_replayed_exact']
    assert science.read(D/'preflight_receipt.json')['repeated_exact']
    existing = science.read(D/'components_vidstg_domain_child.json')
    assert not science.exited(existing)
    assert not list(D.glob('budget_*_domain_child.json'))
    v = dict(schema='MAP_components_CPU_parallel_schedule_v2',
             authorization='Direct user asks CPU parallelism; use the free slot for budget without interrupting original CG-Vid.',
             scientific_fingerprint=p['fingerprint'], scientific_plan_sha256=science.sha(D/'plan.json'),
             scheduler_sha256=science.sha(__file__), existing_component_domain=existing,
             previous_controller=science.read(D/'controller.json'),
             jobs=[list(x) for x in JOBS],
             limits=p['limits'],
             scope='Scheduling only. Original run.py/plan/derive/scoring/reporting unchanged. Existing Vid component domain continues; budget jobs may start before it ends. Each task/domain still seals before its own labels. No source/expert/GPU/strict launch.',
             resource_guard='CPU14-15/nice19/idleIO/one thread, max2 primary computation domains. Guard original Vid baseline, pause only owned CPU domain groups and existing CG-MAP consumer. Never signal original CG/teacher/services.')
    v['fingerprint'] = science.digest(v)
    science.atomic(PLAN, v)
    science.atomic(P/'parallel_v2_protocol.json', v)
    print(v['fingerprint'], flush=True)

def validate():
    p = science.validate()
    v = science.read(PLAN)
    assert v['fingerprint'] == science.digest({k:x for k,x in v.items() if k != 'fingerprint'})
    assert v['scientific_fingerprint'] == p['fingerprint']
    assert v['scientific_plan_sha256'] == science.sha(D/'plan.json')
    assert v['scheduler_sha256'] == science.sha(__file__)
    return p, v

def completed(task, ds):
    rp, sp = D/f'{task}_{ds}_reporting_receipt.json', D/f'{task}_{ds}_score_receipt.json'
    if not rp.exists():
        return False
    a = science.read(rp)
    assert a['fingerprint'] == science.read(D/'plan.json')['fingerprint']
    assert a['all_arrays_cold_replayed_exact'] and a['all_means_recomputed']
    assert a['score_sha256'] == science.sha(sp)
    assert science.read(sp)['seal_sha256'] == science.sha(D/f'{task}_{ds}_sealed.json')
    return True

def queue():
    p, v = validate()
    assert science.exited(v['previous_controller']), 'Previous queue owner must actually exit first'
    assert not (D/'parallel_v2_controller.json').exists()
    science.atomic(D/'parallel_v2_controller.json', science.identity())
    baseline = science.read(D/'Vid_baseline.json')
    active = {('components','vidstg'):v['existing_component_domain']}
    pending = [('budget','hc2'), ('budget','vidstg')]
    children, logs, done, failures = {}, {}, [], []
    paused, pause_at, last = False, 0., baseline['completed']
    cgmap = science.read(science.MAP/'CG_vidstg_domain_child.json')
    try:
        while active or pending:
            for key, w in list(active.items()):
                child = children.get(key)
                exited = child.poll() is not None if child else science.exited(w)
                if not exited:
                    continue
                if child:
                    logs[key].close()
                task, ds = key
                if completed(task, ds) and (not child or child.returncode == 0):
                    done.append(list(key))
                else:
                    failures.append(dict(job=list(key), worker=w, state=science.read(D/f'{task}_{ds}_state.json')))
                del active[key]
            current = science.rate()
            low = science.mem() < p['limits']['minimum_RAM_bytes']
            slow = current['completed']-last >= 25 and current['mean'] > 1.35*baseline['mean']
            groups = [*active.values(), cgmap]
            if not paused and (low or slow):
                for w in groups:
                    science.group_signal(w, signal.SIGSTOP)
                paused, pause_at, last = True, time.time(), current['completed']
                science.atomic(D/'resource_guard.json', dict(status='own_CPU_groups_paused', reason='RAM' if low else 'Vid_rate_precaution', current=current, time=pause_at, scheduler=v['fingerprint']))
            elif paused and science.mem() > p['limits']['resume_RAM_bytes'] and time.time()-pause_at >= 60:
                for w in groups:
                    science.group_signal(w, signal.SIGCONT)
                paused, last = False, current['completed']
                science.atomic(D/'resource_guard.json', dict(status='own_CPU_groups_resumed', current=current, time=time.time(), scheduler=v['fingerprint']))
            if not slow and current['completed']-last >= 25:
                last = current['completed']
            if failures:
                pending.clear()
            while pending and len(active) < p['limits']['max_compute_processes'] and not paused and not low:
                key = pending.pop(0)
                task, ds = key
                assert not (D/f'{task}_{ds}_domain_child.json').exists(), 'No duplicate domain launch'
                assert not (D/task/ds).exists(), 'No overwrite of partial paid or cached work'
                log = (D/f'{task}_{ds}_domain.log').open('a')
                c = subprocess.Popen([sys.executable,'-B',str(R/'studies/map_components_v1/run.py'),'domain','--task',task,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                w = science.identity(c.pid)
                science.atomic(D/f'{task}_{ds}_domain_child.json',w)
                children[key], logs[key], active[key] = c, log, w
            status = 'parallel_components_budget_CPU_running' if active else 'complete_scored_reported_pending_publication'
            if failures:
                status = 'failure_preserved_independent_running' if active else 'failed_preserved'
            state = dict(status=status, scientific_fingerprint=p['fingerprint'], execution_fingerprint=v['fingerprint'], controller=science.identity(), active=[dict(task=k[0],dataset=k[1],worker=w)for k,w in active.items()], pending=[list(k)for k in pending], complete=done, failures=failures, paused=paused)
            science.atomic(D/'parallel_v2_state.json',state)
            science.atomic(D/'state.json',state)
            science.atomic(D/'parallel_v2_guard_latest.json',dict(time=time.time(),available_RAM_bytes=science.mem(),Vid=current,baseline=baseline,paused=paused,active_groups=list(active.values()),CG_MAP_group=cgmap))
            if active or pending:
                time.sleep(5)
        assert not failures, 'Failures preserved, no automatic retry'
    except BaseException as e:
        science.atomic(D/'parallel_v2_failure.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc(),active=active and [dict(job=list(k),worker=w)for k,w in active.items()]))
        raise
    finally:
        if paused:
            for w in [*active.values(),cgmap]:
                science.group_signal(w,signal.SIGCONT)

if __name__ == '__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','queue']);args=a.parse_args()
    dict(prepare=prepare,queue=queue)[args.command]()
