"""Serial HC2 grid-best locking. No next coordinate exists before this lock."""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_hc2_coordinate_common_v1 import *

def step(job,label,cpu=False,allow_numeric_invalid=False):
    verify();budget();command=[str(PYTHON),'-B',*job];log=BASE/'logs'/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as out:
        p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''} if cpu else None)
        status(BASE/'STAGE.json',dict(status='running',controller_pid=os.getpid(),worker_pid=p.pid,
            label=label,command=command,log=str(log.relative_to(ROOT)),CPU=cpu,time=time.time()))
        code=p.wait()
    if code:
        text=log.read_text()
        expected='AssertionError: Policy requires an invertible native box chart; do not silently clamp'
        if allow_numeric_invalid and text.rstrip().endswith(expected):return False
        raise RuntimeError(f'Engineering failure {label}, exit {code}; preserve original log {log}')
    return True

def run():
    verify();lease=(BASE/'controller.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (BASE/'QUALIFICATION.json').exists():
        qs=[]
        for cfg in [START,{**START,'lr':.01,'sigma':.025,'tau':.5,'steps':3,'writeback':0.}]:
            uid=ensure_trial(cfg,True);step(['scripts/run_stvg_opd_hc2_coordinate_v1.py',uid],uid)
            qs.append(read(BASE/'trials'/uid/'QUALIFICATION.json'))
        assert all(x['status']=='pass' and x['actual_backwards']>0 for x in qs)
        assert len(qs[0]['parity'])==2
        write(BASE/'QUALIFICATION.json',dict(status='pass',actual_GPU_fits=4,records=qs,
            original_selected_configuration_two_box_and_state_bitwise=True,synthetic_is_not_GPU_qualification=True,time=time.time()))
    incumbent=START.copy();previous_selection=None
    for index,coordinate in enumerate(COORDINATES):
        dest=BASE/'coordinates'/f'{index:02}_{coordinate}';selection=dest/'SELECTION.json'
        if selection.exists():
            selected=read(selection);assert selected['incumbent']==incumbent and selected['previous_selection_sha256']==previous_selection
            incumbent=selected['config'];previous_selection=sha(selection);continue
        if not (dest/'CONFIG.json').exists():
            candidates=[dict(trial=ensure_trial(cfg),config=cfg,ordinal=j) for j,cfg in enumerate(candidate_configs(incumbent,coordinate))]
            write(dest/'CONFIG.json',dict(index=index,coordinate=coordinate,incumbent=incumbent,candidates=candidates,
                previous_selection_sha256=previous_selection,time=time.time()))
        c=read(dest/'CONFIG.json');assert c['incumbent']==incumbent and c['previous_selection_sha256']==previous_selection
        dispositions={}
        for v in c['candidates']:
            uid=v['trial'];trial=BASE/'trials'/uid
            if not (trial/'PREDICTION_BARRIER.json').exists() and not (trial/'INVALID_DISPOSITION.json').exists():
                if not seal_exact_alias(uid):
                    ok=step(['scripts/run_stvg_opd_hc2_coordinate_v1.py',uid],uid,allow_numeric_invalid=True)
                    if not ok:
                        paths=sorted(trial.glob('*/*.pt'));evidence=trial/'failure/traceback.txt'
                        write(trial/'INVALID_DISPOSITION.json',dict(status='unscored_numerical_invalid',
                            reason='noninvertible native Gaussian box chart; no clamp or optimizer substitution',
                            config=v['config'],preserved_receipted_prefix=len(paths),
                            prefix={str(p.relative_to(ROOT)):sha(p) for p in paths},
                            failure_sha256=sha(evidence),score_read=False,time=time.time()))
            name='PREDICTION_BARRIER.json' if (trial/'PREDICTION_BARRIER.json').exists() else 'INVALID_DISPOSITION.json'
            dispositions[uid]={name:sha(trial/name)}
            status(BASE/'SEARCH_PROGRESS.json',dict(status='coordinate_candidates_running',coordinate=coordinate,index=index,
                sealed_or_invalid_candidates=len(dispositions),total_candidates=len(c['candidates']),GT_read_this_coordinate=False,time=time.time()))
        if not (dest/'PREDICTION_BARRIER.json').exists():
            write(dest/'PREDICTION_BARRIER.json',dict(status='all_candidates_sealed_or_unscored_numerical_invalid',
                coordinate=coordinate,index=index,config_sha256=sha(dest/'CONFIG.json'),dispositions=dispositions,
                GT_read_this_coordinate=False,time=time.time()))
        coordinate_barrier(index);ranked=[]
        for v in c['candidates']:
            uid=v['trial'];trial=BASE/'trials'/uid
            if (trial/'INVALID_DISPOSITION.json').exists():continue
            step(['scripts/score_stvg_opd_hc2_coordinate_v1.py',str(index),uid],uid+'_score',cpu=True)
            summary=read(PUB/'trials'/uid/'SUMMARY.json')
            ranked.append((rank(summary,incumbent,v['ordinal']),v))
        assert ranked,'All candidates invalid; preserve evidence for root, never invent a winner'
        winner=max(ranked,key=lambda x:x[0])[1]
        for k in incumbent:assert k==coordinate or winner['config'][k]==incumbent[k]
        write(selection,dict(status='coordinate_locked',coordinate=coordinate,index=index,incumbent=incumbent,
            config=winner['config'],trial=winner['trial'],previous_selection_sha256=previous_selection,
            coordinate_barrier_sha256=sha(dest/'PREDICTION_BARRIER.json'),
            score_receipts={v['trial']:sha(BASE/'trials'/v['trial']/'CPU_COMPLETION.json') for _,v in ranked},
            selection_on_exposed_development_only=True,time=time.time()))
        incumbent=winner['config'];previous_selection=sha(selection)
        status(BASE/'SEARCH_PROGRESS.json',dict(status='coordinate_locked',coordinate=coordinate,index=index,
            selected_trial=winner['trial'],selected_config=incumbent,completed_coordinates=index+1,total_coordinates=5,time=time.time()))
        print('HC2_COORDINATE_LOCKED',coordinate,winner['trial'],incumbent,flush=True)
    if not (BASE/'SELECTION_BARRIER.json').exists():
        write(BASE/'SELECTION_BARRIER.json',dict(status='all_five_coordinates_locked',config=incumbent,selected_trial=winner['trial'] if 'winner' in locals() else read(selection)['trial'],
            coordinate_selections={str((BASE/'coordinates'/f'{i:02}_{k}'/'SELECTION.json').relative_to(ROOT)):sha(BASE/'coordinates'/f'{i:02}_{k}'/'SELECTION.json') for i,k in enumerate(COORDINATES)},
            development_selected=True,independent_confirmation=False,VidSTG_unchanged=True,time=time.time()))
    step(['scripts/finalize_stvg_opd_hc2_coordinate_v1.py'],'independent_root_finalizer',cpu=True)
    status(BASE/'STATUS.json',dict(status='search_complete_pending_actual_root_visual_publication_and_registration',
        selected_config=incumbent,root_must_finish=True,original_paper_resume_after_closing_authorized=True,
        paper_suite_complete=False,time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException:
        d=BASE/'failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True);(d/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved',evidence=str(d.relative_to(ROOT)),time=time.time()))
        raise
