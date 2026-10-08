"""Root-only closure after actual plot/remote verification, then authorized paper resume."""
import os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_hc2_coordinate_common_v1 import *

def run():
    verify();closing=BASE/'ROOT_CLOSING_RECEIPT.json';assert not closing.exists()
    visuals=read(BASE/'ROOT_VISUAL_REVIEW.json');remote=read(BASE/'FINAL_GITHUB_RECEIPT.json')
    assert visuals['status']==remote['status']=='pass'
    audit=read(BASE/'ROOT_AUDIT_COMPLETION.json');assert audit['statistics_and_chain']=='pass'
    assert read(BASE/'ROOT_BYTE_READBACK.json')['status']=='pass'
    assert read(BASE/'PUBLIC_SCALAR_AUDIT.json')['status']=='pass'
    cfg=read(BASE/'SELECTION_BARRIER.json')['config'];new=read(BASE/'NEW_CONFIGS.json');old=read(BASE/'PREVIOUS_CONFIGS.json')
    file=ROOT/'methods/decota_spatial_opd_v1/configs.json'
    assert sha(file)==sha(BASE/'PREVIOUS_CONFIGS.json')
    assert new['datasets']['hc2']['config']==cfg and new['datasets']['vidstg']==old['datasets']['vidstg']
    registry_before=sha(ROOT/'methods/CURRENT_METHOD.json');oldhash=sha(file);newhash=sha(BASE/'NEW_CONFIGS.json')
    temp=file.with_suffix('.coordinate-register.tmp');shutil.copy2(BASE/'NEW_CONFIGS.json',temp);temp.replace(file)
    assert sha(file)==newhash and sha(ROOT/'methods/CURRENT_METHOD.json')==registry_before
    write(BASE/'REGISTRATION_RECEIPT.json',dict(status='selected_HC2_configuration_registered',
        previous_config_file_sha256=oldhash,new_config_file_sha256=newhash,HC2_config=cfg,
        VidSTG_entry_unchanged=True,CURRENT_registry_bytes_unchanged=True,
        public_production_configuration_bytes_verified=True,commit=remote['commit'],time=time.time()))
    write(closing,dict(status='complete',scope='HC2_32_exposed_development_sequential_coordinate_tuning_only',
        selected_config=cfg,independent_source_bootstrap=10000,all_candidates_and_negative_results_retained=True,
        real_GPU_qualification_fits=4,actual_visual_review_sha256=sha(BASE/'ROOT_VISUAL_REVIEW.json'),
        independent_audit_sha256=sha(BASE/'ROOT_AUDIT_COMPLETION.json'),
        opaque_readback_sha256=sha(BASE/'ROOT_BYTE_READBACK.json'),
        final_GitHub_receipt_sha256=sha(BASE/'FINAL_GITHUB_RECEIPT.json'),
        registration_receipt_sha256=sha(BASE/'REGISTRATION_RECEIPT.json'),
        commit=remote['commit'],paper_suite_complete=False,independent_confirmation=False,
        original_paper_resume_authorized_by_human=True,time=time.time()))
    from scripts.stvg_opd_paper_hc2_revision_common_v2 import prepare,BASE as revision
    prepare()
    from scripts.prepare_stvg_opd_revision_later_v2 import run as prepare_later
    prepare_later()
    command=[str(PYTHON),'-B','scripts/continue_stvg_opd_paper_hc2_revision_v2.py','P0']
    log=revision/'logs/controller_P0.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as out:p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    write(revision/'LAUNCH.json',dict(status='actual_started_after_HC2_tuning_root_closure',controller_pid=p.pid,
        command=command,log=str(log.relative_to(ROOT)),runtime_sha256=sha(revision/'RUNTIME_LOCK.json'),
        tuning_root_receipt_sha256=sha(closing),single_GPU_serial=True,GT_read=False,time=time.time()))
    status(revision/'STATUS.json',dict(status='revised_P0_actual_started',controller_pid=p.pid,
        original_paper_continuation=True,HC2_config=cfg,VidSTG_unchanged=True,paper_suite_complete=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='HC2_coordinate_tuning_closed_original_paper_actual_resumed',
        selected_config=cfg,root_closing_sha256=sha(closing),paper_controller_pid=p.pid,
        paper_BASE=str(revision.relative_to(ROOT)),paper_suite_complete=False,time=time.time()))
    print('HC2_COORDINATE_CLOSED_PAPER_RESUMED',cfg,p.pid,remote['commit'])

if __name__=='__main__':run()
