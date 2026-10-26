"""One deferred gamma GPU owner, gated on actual CG exits, audits and root publication."""
from gamma_full_common import *
import subprocess,traceback
def gate():
    cg=R/'outputs/shared_corrected_roi_CG_v1';owner=read(cg/'controller.json')
    assert owner==dict(pid=3374443,startticks=252798286)
    if not selected.exited(owner):return None
    state=read(cg/'state.json');assert state['status']=='both_CG_domains_scored_audited_pending_root_analysis_publication','CG predecessor failed; preserve waiting queue'
    out={}
    for ds in COUNTS:
        for stage in ['GPU','CPU']:assert read(cg/f'{ds}_{stage}_exit.json')['code']==0
        a=cg/f'{ds}_reporting_receipt.json';audit=read(a)
        assert audit['status']=='verified_whole_domain'and audit['all_means_recomputed']and audit['all_compact_arrays_cold_rederived_exact']
        pub=cg/f'{ds}_publication.json'
        if not pub.exists():return None
        publication=read(pub);assert publication['status']=='complete_published'and publication['all_remote_verified']
        cp=Path(publication['completion_path']);assert sha(cp)==publication['completion_sha256']
        assert read(cp)['status']=='verified_all_remote_tree_files_bytes_SHA_ancestors'
        out[ds]=dict(audit_path=str(a),audit_sha256=sha(a),publication_sha256=sha(pub),completion_sha256=sha(cp))
    # A paused corruption owner is neither restored nor treated as completed.
    pause=read(R/'outputs/maintenance/corruption_paused_by_user_20261026/owned_exit.json')
    assert pause['status']=='paused_by_user'and pause['owned_controller_actual_exit']
    return out
def run():
    validate();assert read(D/'preflight_receipt.json')['status']=='passed_four_actual_cached_inputs'
    assert not(D/'controller.json').exists();who=identity();atomic(D/'controller.json',who)
    jobs=[]
    try:
        atomic(D/'state.json',dict(status='waiting_CG_successful_exit_full_audits_root_publication',controller=who,model_calls=0,corruption_paused_by_user=True))
        while True:
            dependency=gate()
            if dependency is not None:break
            time.sleep(15)
        atomic(D/'dependency_gate.json',dependency)
        for ds in COUNTS:
            validate()
            with(D/f'{ds}_GPU.log').open('ab')as log:
                c=subprocess.Popen([sys.executable,'-B',str(S/'worker.py'),'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
                atomic(D/f'{ds}_GPU_child.json',identity(c.pid));atomic(D/'state.json',dict(status='gamma_GPU_'+ds,controller=who,corruption_paused_by_user=True));rc=c.wait()
            atomic(D/f'{ds}_GPU_exit.json',dict(code=rc));assert rc==0,'Preserve failed paid state; no retry'
            log=(D/f'{ds}_CPU.log').open('ab')
            c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'report.py'),'domain','--dataset',ds],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            atomic(D/f'{ds}_CPU_child.json',identity(c.pid));jobs.append((ds,c,log));assert len(jobs)<=2
        for ds,c,log in jobs:
            rc=c.wait();log.close();atomic(D/f'{ds}_CPU_exit.json',dict(code=rc));assert rc==0
        atomic(D/'state.json',dict(status='both_gamma_domains_scored_audited_pending_root_curves_analysis_publication',controller=who,corruption_paused_by_user=True))
    except BaseException:atomic(D/'state.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc()));raise
if __name__=='__main__':run()
