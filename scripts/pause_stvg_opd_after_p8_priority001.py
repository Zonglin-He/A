"""Reversible saved-query handoff to the next already authorized experiment."""
import os,signal,sys,time,hashlib,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.pause_stvg_opd_for_geometry_v1 import process
BASE=ROOT/'artifacts/stvg_opd_claim_strengthening_v1/after_P8_priority001'
S=ROOT/'artifacts/stvg_opd_supplement_v1/baselines'
def run():
    from scripts.stvg_opd_expanded_state_common_v1 import verify
    verify();assert read(BASE/'CPU_STORAGE_CONTRACTS.json')['status']=='pass'
    assert read(ROOT/'artifacts/stvg_opd_support_factorization_v1/CPU_FACTORIAL_FINAL_COMPLETION.json')['status']=='complete'
    current=read(S/'STATUS.json');assert current['status']=='running' and current['job']=='formal'
    cp,wp=int(current['controller_pid']),int(current['worker_pid']);identities=[process(wp),process(cp)]
    assert 'run_stvg_opd_baseline_supplement_v1.py formal' in identities[0]['command']
    assert 'continue_stvg_opd_baseline_supplement_v1.py' in identities[1]['command']
    stage=S/'formal'/current['dataset']/current['candidate'];old=read(stage/'STATUS.json')
    originals=BASE/'original_scheduling_metadata';originals.mkdir(exist_ok=False)
    write(originals/'BASELINE_CONTROLLER_STATUS.json',current);write(originals/'BASELINE_STAGE_STATUS.json',old)
    os.kill(cp,signal.SIGSTOP)
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            now=read(stage/'STATUS.json')
            if now['done']>old['done']:break
            time.sleep(.02)
        else:raise RuntimeError('No new accepted query within90s; no baseline worker was paused')
        os.kill(wp,signal.SIGSTOP);time.sleep(.5)
        assert all(process(p['pid'])['state']=='T' and process(p['pid'])['start_ticks']==p['start_ticks'] for p in identities)
        prefix={}
        for p in sorted(stage.glob('*/*.pt')):
            receipt=p.with_suffix('.json')
            if receipt.exists():
                rc=read(receipt);h=sha(p);assert rc['sha256']==h and not rc['GT_read']
                prefix[str(p.relative_to(ROOT))]=dict(sha256=h,bytes=p.stat().st_size,receipt_sha256=sha(receipt))
        assert len(prefix)>=now['done']>old['done']
        write(BASE/'BASELINE_ACCEPTED_PREFIX.json',dict(status='pass',scope='entire currently accepted formal baseline prefix and complete per-query saved optimizer/parameter witnesses remain immutable; no lost-memory bitwise claim',files=prefix,accepted_prefix_count=len(prefix),GT_read=False,time=time.time()))
        dependencies=[('claim',ROOT/'artifacts/stvg_opd_claim_strengthening_v1/STATUS.json'),('LN',ROOT/'artifacts/stvg_opd_mechanism_review_v1/cross_parent_LN/STATUS.json'),('S_root',ROOT/'artifacts/stvg_opd_supplement_v1/actual_root/postseal_continuation006/STATUS.json'),('cost_isolation',ROOT/'artifacts/stvg_opd_supplement_v1/cost/CPU_profile_isolation002/STATUS.json')]
        waiters=[]
        for label,path in dependencies:
            d=read(path);assert 'wait' in d['status'].lower() and not d.get('worker_pid'),(label,d['status'])
            pid=int(d['controller_pid']);p=process(pid);assert p['state'] in ['S','R','D'],(label,p['state'])
            assert '.py' in p['command'];write(originals/(label+'_STATUS.json'),d)
            os.kill(pid,signal.SIGSTOP);time.sleep(.1);assert process(pid)['state']=='T' and process(pid)['start_ticks']==p['start_ticks'];waiters.append(dict(label=label,identity=p,original_status_sha256=sha(path)))
        write(BASE/'PRIORITY_PAUSE_RECEIPT.json',dict(status='same_original_baseline_and_idle_dependency_waiters_reversibly_paused_for_next_experiment',scope='latest human after P8 skip further baseline tuning/proceed to next experiment; incomplete formal baseline kept in reversible standby, not completed or scored',baseline_processes=identities,dependency_waiters=waiters,accepted_before=old['done'],accepted_stage_snapshot=now,prefix_sha256=sha(BASE/'BASELINE_ACCEPTED_PREFIX.json'),original_baseline_optimizer_memory_preserved=True,original_predictions_code_and_ordinary_GPU_lease_preserved=True,baseline82710_global_exists=(S/'FORMAL_GLOBAL_BARRIER.json').exists(),baseline_incomplete=True,CPU_only=True,GT_read=False,time=time.time()))
        own=['protocols/stvg_opd_after_p8_priority001.md','scripts/pause_stvg_opd_after_p8_priority001.py','scripts/run_stvg_opd_expanded_after_p8_priority001.py','scripts/continue_stvg_opd_expanded_after_p8_priority001.py']
        from scripts.stvg_opd_temporal_guided_p7_common_v1 import O
        pins={**read(O/'frozen_mask_cache012/REVISION_RUNTIME.json')['pins'],**{n:sha(ROOT/n) for n in own}}
        write(BASE/'REVISION_RUNTIME.json',dict(status='locked_before_actual_next_GPU_qualification',pins=pins,pause_sha256=sha(BASE/'PRIORITY_PAUSE_RECEIPT.json'),CPU_storage_contract_sha256=sha(BASE/'CPU_STORAGE_CONTRACTS.json'),original_expanded_science_runtime_sha256=sha(ROOT/'artifacts/stvg_opd_claim_strengthening_v1/expanded_state/RUNTIME_LOCK.json'),qualification_arrivals=24,actual_complete_repeat_GPU_fits=48,new_formal_fits=5430,scientific_source_or_hyperparameters_or_roster_changed=False,GT_read=False,time=time.time()))
        print('ACTUAL_NEXT_EXPERIMENT_PRIORITY_PAUSED_BASELINE',now['done'],wp,cp,flush=True)
    except BaseException:
        # If preparation failed before launching any new GPU, restore only
        # original processes we paused; never leave an unexplained halt.
        for p in locals().get('waiters',[]):
            q=p['identity']
            if process(q['pid'])['state']=='T':os.kill(q['pid'],signal.SIGCONT)
        for q in identities:
            if process(q['pid'])['state']=='T':os.kill(q['pid'],signal.SIGCONT)
        raise
if __name__=='__main__':run()
