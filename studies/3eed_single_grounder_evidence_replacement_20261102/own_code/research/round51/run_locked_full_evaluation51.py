"""One-shot fixed full evaluation, after sealed evidence, with no search.

Every processing/scoring payload is invoked through manual_r51_guard. This
driver only builds hash-pinned plans for the preselected five-arm experiment.
No timer, preference creation, method selection, retry or TTA is implemented.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, time

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'research/round51/experiment'
PY=ROOT/'crossplatform/env/bin/python'
MANIFEST='research/round8/experiment/full5912_gpu_inputs.json'
DEV='research/round24/input/ALL336_PRIVATE.json'
ARMS=['source','external_G','replace_all','no_relation','no_instance_association']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);assert not p.exists();p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();grant=read(a.plan)
    assert grant['state']=='ROOT_AUTHORIZED_ONESHOT_FROZEN_FULL_EVALUATION_NO_SEARCH'
    assert grant['program_sha256']==sha(__file__)
    freeze='research/round51/experiment/METHOD_FREEZE_BEFORE_FULL5912.json'
    assert sha(ROOT/freeze)==grant['method_freeze_sha256']
    assert read(ROOT/freeze)['TTA_will_execute'] is False
    for p,h in grant['programs_sha256'].items():assert sha(ROOT/p)==h,p
    safe='crossplatform/round51/grouped5912_attempt001'
    na='crossplatform/round51/noassociation5912_attempt001'
    prep='crossplatform/round51/prepared5912_attempt001'
    ref='crossplatform/round51/reference5912_headspan_attempt001'
    old='crossplatform/round50/full5912_attempt001/predicted5912_attempt001'
    source='crossplatform/round8/full5912_attempt001'
    pred='crossplatform/round51/predicted5912_attempt001'
    audit='research/round51/independent_audit/full5912_attempt001'
    score='research/round51/score_full5912_attempt001'
    saved='research/round51/independent_audit/saved_full5912_attempt001'
    start=time.perf_counter()
    while not all((ROOT/p/'SEALED.json').is_file() for p in (safe,na)):
        assert time.perf_counter()-start<grant['wait_timeout_seconds']
        assert not any(list((ROOT/p).glob('FAILED*.json')) for p in (safe,na,prep,ref))
        time.sleep(5)
    while (ROOT/'research/round51/RUNNING_CPU.json').exists():time.sleep(.2)
    def pins(paths):return {p:sha(ROOT/p) for p in paths}
    def programs(paths):
        pp={p:grant['programs_sha256'][p] for p in paths}
        for p,h in pp.items():assert sha(ROOT/p)==h
        return pp
    def run(label,script,p,state,extra,timeout=7200):
        pp=D/(label+'_PLAN_LOCK.json');aa=D/(label+'_AUTHORIZATION.json')
        dump(pp,p);dump(aa,{'state':state,'program_sha256':sha(ROOT/script),'plan_sha256':sha(pp),**extra})
        print('START_FIXED_EVALUATION_STAGE',label,flush=True)
        code=subprocess.call([str(PY),'tools/manual_r51_guard.py','--resource','cpu','--timeout-seconds',str(timeout),'--',str(PY),script,'--plan',str(pp.relative_to(ROOT)),'--authorization',str(aa.relative_to(ROOT))],cwd=ROOT)
        assert code==0,(label,code,'No automatic retry or method mutation')
        return pp
    p={'state':'ROOT_LOCKED_R51_SELECTED_GROUPED_FULL_FIVE_READOUT_ASSEMBLY','n':5912,'manifest':MANIFEST,
       'arm_names':ARMS,'selected_method':'grouped_replace_all_c05','selected_factor':.5,
       'method_selected_before_full_run':True,'new_full_calibration':False,'method_freeze':freeze,
       'safe_root':safe,'noassociation_root':na,'r50_prediction_root':old,'output_root':pred,
       'programs_sha256':programs(['crossplatform/round51/grouped_full_frames51.py']),
       'inputs_sha256':pins([MANIFEST,freeze]+[p+'/SEALED.json' for p in (safe,na,old)])}
    run('GROUPED_FULL_FIVE_FRAMES','crossplatform/round51/grouped_full_frames51.py',p,
        'ROOT_AUTHORIZED_R51_SELECTED_GROUPED_FULL_FIVE_READOUT_ASSEMBLY_NO_GT',{'method_freeze_sha256':sha(ROOT/freeze)})
    p={'state':'ROOT_LOCKED_R51_INDEPENDENT_FULL_GROUP_MEDOID_AUDIT','n':5912,'manifest':MANIFEST,
       'no_GT_models_or_optimizer':True,'selected_factor':.5,'rawsupport_recompute_scope':'HASH_MIN_N_PER_PLATFORM',
       'rawsupport_n_per_platform':2,'rawsupport_hash_salt':'R51_full_grouped_raw2perplatform_before_GT_v1',
       'dev_overlap_bridge':False,'prediction_root':pred,'safe_root':safe,'noassociation_root':na,
       'prepared_root':prep,'reference_root':ref,'r50_prediction_root':old,'canonical_source_root':source,
       'arm_names':ARMS,'arm_roles':dict(zip(['source','external','measured','no_relation','no_association'],ARMS)),
       'r50_external_arm':'wideentry_oldW','output_directory':audit,
       'programs_sha256':programs(['research/round51/audit_grouped_full51.py','research/round51/audit51.py','research/round51/audit_full51.py']),
       'inputs_sha256':pins([MANIFEST]+[p+'/SEALED.json' for p in (pred,safe,na,prep,ref,old,source)])}
    run('GROUPED_FULL_PREDICTION_AUDIT','research/round51/audit_grouped_full51.py',p,
        'ROOT_AUTHORIZED_R51_INDEPENDENT_FULL_GROUP_MEDOID_AUDIT_NO_GT',{'prediction_seal_sha256':sha(ROOT/pred/'SEALED.json')})
    base=read(D/'DEV336_SINGLE_POSTSEAL_TASK_SCORING_PLAN_LOCK.json')
    p={k:base[k] for k in ['sample_relative_template','dev_split_by_id','independent_geometry_core','scoring_index','scoring_index_preparation_receipt']}
    p.update(n=5912,no_protected_FINAL_specific_artifacts=True,arm_names=ARMS,reference_arms=['source','external_G'],
             prediction_directory=pred,prediction_seal=pred+'/SEALED.json',independent_prediction_audit=audit+'/AUDIT_RESULT.json',
             readout_status_table=pred+'/READOUT_STATUS_PRIVATE.json',cohort_registration=MANIFEST,dev_registration=DEV,
             calibration=None,output_directory=score)
    oldrows='research/round50/independent_score/full5912_attempt001/ALL5912_TASK_ROWS_PRIVATE.json'
    p['reused_arms']={a:{'registration':MANIFEST,'saved_rows':oldrows,'saved_arm':b} for a,b in [('source','source'),('external_G','wideentry_oldW')]}
    p['programs_sha256']=programs(['research/round51/score51.py','research/round5/audit/independent_core_v2.py','tools/obb_iou.py'])
    p['inputs_sha256']=pins([MANIFEST,DEV,p['dev_split_by_id'],p['prediction_seal'],p['independent_prediction_audit'],p['readout_status_table'],p['scoring_index'],p['scoring_index_preparation_receipt'],oldrows,freeze])
    scoreplan=run('FULL5912_SINGLE_POSTSEAL_TASK_SCORE','research/round51/score51.py',p,
        'ROOT_AUTHORIZED_R51_SINGLE_POSTSEAL_TASK_SCORE',{'prediction_seal_sha256':sha(ROOT/pred/'SEALED.json'),'independent_prediction_audit_sha256':sha(ROOT/audit/'AUDIT_RESULT.json')},600)
    p={'state':'ROOT_LOCKED_R51_SAVED_SCALAR_AUDIT','n':5912,'scoring_directory':score,
       'scoring_plan':str(scoreplan.relative_to(ROOT)),'output_directory':saved,
       'no_GT_geometry_scoring_media_models_or_prediction_arrays':True,
       'programs_sha256':programs(['research/round51/audit_saved_scalars51.py']),
       'inputs_sha256':pins([score+'/SEALED.json',str(scoreplan.relative_to(ROOT)),MANIFEST,DEV,base['dev_split_by_id'],oldrows])}
    run('FULL5912_SAVED_SCALAR_AUDIT','research/round51/audit_saved_scalars51.py',p,
        'ROOT_AUTHORIZED_R51_SAVED_SCALAR_AUDIT_NO_GT',{},600)
    dump(ROOT/'research/round51/FULL_EVALUATION_ONESHOT_RECEIPT.json',{
        'state':'COMPLETE_LOCKED_FULL_FIVE_ARM_INDEPENDENTLY_AUDITED_SINGLE_SCORE',
        'n':5912,'arm_names':ARMS,'no_search_retry_TTA_or_protected_pool_access':True,
        'score_seal_sha256':sha(ROOT/score/'SEALED.json'),'saved_audit_sha256':sha(ROOT/saved/'AUDIT_RESULT.json'),
        'seconds_including_wait':time.perf_counter()-start,'method_freeze_sha256':sha(ROOT/freeze)})

if __name__=='__main__':main()
