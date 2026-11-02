"""UNEXECUTED root-only constructor for four independently authorized modes.

This writes plans/grants, never runs prediction, audits, models or scoring.
All future evidence reads are inside main AFTER a dedicated builder grant.
The constructor does not read raw media, XYZ, task GT or metadata containers.
It never creates the single scoring grant before both independent audits PASS.
"""
from pathlib import Path
import argparse,hashlib,json,time,datetime

ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'research/round50/experiment'
RUN='crossplatform/round50/full5912_attempt001'
MANIFEST='research/round8/experiment/full5912_gpu_inputs.json'
PRED=RUN+'/predicted5912_attempt001'
ARMS=['source','old_G','wideentry_oldW']
FREEZE='research/round50/experiment/FULL_METHOD_FREEZE_BEFORE_FULL_MEDIA.json'
GATE='research/round50/experiment/FULL_GATE_AND_TTA_ADVANCE_BEFORE_FULL_GT.json'
SOURCE_PLAN='research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'
SOURCE_SEAL='crossplatform/round8/full5912_attempt001/SEALED.json'
DEV_ROWS='research/round24/input/ALL336_PRIVATE.json'
DEV_PRED='crossplatform/round50/predicted336_attempt001'
DEV_SCORE='research/round50/independent_score/dev336_attempt001'
BRIDGE='research/round50/independent_review/full_prepared_dev336_bridge_attempt001/AUDIT_RESULT.json'
ARITH='research/round50/independent_review/full5912_audit_attempt001/AUDIT_RESULT.json'
COST='research/round50/independent_review/full_evidence_cost_audit_attempt001/AUDIT_RESULT.json'
PRED_PROGRAM='crossplatform/round50/full_predict50.py'
COST_PROGRAM='research/round50/independent_review/audit_full_evidence_cost50_draft.py'
ARITH_PROGRAM='research/round50/independent_review/audit_full50_draft.py'
SCORE_PROGRAM='research/round50/independent_review/score_full50_draft.py'
PRODUCERS={s:f'research/round50/experiment/FULL_{s.upper()}_EVIDENCE_PLAN_LOCK.json' for s in ('dino','numeric')}
COMMON_FIXED=[MANIFEST,FREEZE,GATE,SOURCE_PLAN,SOURCE_SEAL,DEV_ROWS,
    DEV_PRED+'/SEALED.json',DEV_SCORE+'/SEALED.json',
    'research/round50/experiment/FULL_DEV_GATE_PASS.json',
    'research/round50/experiment/FULL_FRONTEND_PLAN_LOCK.json',
    'research/round50/experiment/FULL_PREPARED_DEV_BRIDGE_PLAN_LOCK.json',BRIDGE,
    'research/round50/independent_scalar_recheck/dev336_attempt001/AUDIT_RESULT.json',
    'research/round15/experiment/SEMANTIC_PLAN_V4_LOCK.json',
    'research/round15/experiment/RUNTIME_ADAPTER_LOCK.json',
    'crossplatform/round20/assets/DINO_ASSET_RECEIPT.json',*PRODUCERS.values()]
ORIGINAL_PROGRAMS=['crossplatform/round23/rank23.py','crossplatform/round20/detect.py',
    'crossplatform/round7/regression_adapter.py','research/round49/independent_audit/reference49.py']
EXPLICIT_OLD_INPUTS={SOURCE_PLAN,SOURCE_SEAL,DEV_ROWS,
    'research/round15/experiment/SEMANTIC_PLAN_V4_LOCK.json',
    'research/round15/experiment/RUNTIME_ADAPTER_LOCK.json',
    'crossplatform/round20/assets/DINO_ASSET_RECEIPT.json',
    'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json',
    'crossplatform/round5/baseline_engineering/PREPARATION_RECEIPT.json'}
HASHES={}
JSON_CACHE={}


def local(rel):
    p=Path(rel)
    assert not p.is_absolute() and '..' not in p.parts
    assert 'FINAL' not in p.as_posix() and 'round46' not in p.as_posix(),rel
    result=(ROOT/p).resolve();assert result.is_relative_to(ROOT)
    return result


def sha(path):
    p=Path(path)
    if p not in HASHES:
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
        HASHES[p]=h.hexdigest()
    return HASHES[p]


def read(rel):
    if rel not in JSON_CACHE:JSON_CACHE[rel]=json.loads(local(rel).read_text())
    return JSON_CACHE[rel]


def pin(target,rel):target[rel]=sha(local(rel))


def dump_exclusive(rel,value):
    path=local(rel);assert path.parent==EXP and not path.exists(), ('Refuse overwrite',rel)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def context(args):
    authpath=Path(args.authorization).resolve();assert authpath.is_relative_to(EXP)
    auth=json.loads(authpath.read_text())
    assert auth['state']=='ROOT_AUTHORIZED_R50_POSTSEAL_PLAN_CONSTRUCTION_ONLY'
    assert auth['builder_program_sha256']==sha(__file__) and auth['mode']==args.mode
    assert auth['no_model_audit_or_scoring_execution'] is True
    fixed=auth['fixed_inputs_sha256'];assert set(COMMON_FIXED)<=set(fixed)
    for rel,h in fixed.items():
        assert rel in EXPLICIT_OLD_INPUTS or rel==MANIFEST or rel.startswith(('research/round50/','crossplatform/round50/'))
        assert sha(local(rel))==h,rel
    freeze=read(FREEZE);gate=read(GATE);source=read(SOURCE_PLAN)
    assert freeze['state']=='ROOT_FROZEN_R50_FULL5912_METHOD_BEFORE_FULL_MEDIA_OR_NEW_PREDICTION'
    assert freeze['selected_configuration']=='wideentry_oldW' and freeze['comparators']==ARMS
    assert freeze['full_manifest']==MANIFEST and freeze['n']==5912 and freeze['lambda_VPR']==[2,2,2]
    devgate=read('research/round50/experiment/FULL_DEV_GATE_PASS.json')
    dsrel='research/round50/independent_scalar_recheck/dev336_attempt001/AUDIT_RESULT.json';ds=read(dsrel)
    assert devgate['state']=='PASS_R50_LOCKED_DEV_GATE_FOR_FULL' and devgate['selected_configuration']=='wideentry_oldW'
    assert devgate['independent_scalar_audit_sha256']==sha(local(dsrel))
    assert ds['state']=='PASS_R50_INDEPENDENT_SAVED_DEV_SCALARS_AND_FIXED_SELECTION_REAGGREGATION'
    assert ds['n']==336 and ds['arms']==7 and ds['full_gate_PASS'] is True
    assert gate['state']=='ROOT_LOCKED_R50_FULL_ADVANCE_RULE_BEFORE_ANY_FULL_GT'
    assert gate['scientific_freeze_sha256']==sha(local(FREEZE))
    assert gate['registered_full_gate']=={'candidate':'wideentry_oldW','scope':'ALL5912',
        'metric':'two_platform_equal_weight_mIoU_percent','references':['old_G','source'],'strict_minimum_gain_pp':1e-10}
    assert gate['platform_or_case_zero_harm_gate'] is False and gate['no_new_configuration_selection_after_full_GT'] is True
    assert source['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
    assert sha(local(DEV_PRED+'/SEALED.json'))==freeze['DEV_prediction_seal_sha256']
    assert sha(local(DEV_SCORE+'/SEALED.json'))==freeze['DEV_taskseal_sha256']
    bridge=read(BRIDGE)
    assert bridge['state']=='PASS_R50_INDEPENDENT_FULL_PREPARED_DEV336_CHEAP_BRIDGE_NO_GT'
    assert bridge['scientific_freeze_sha256']==sha(local(FREEZE)) and bridge['n']==336 and bridge['full_n']==5912
    programs={}
    requested=auth['programs_sha256']
    assert set(ORIGINAL_PROGRAMS)<=set(requested)
    for rel,h in requested.items():assert sha(local(rel))==h;programs[rel]=h
    pin(programs,Path(__file__).relative_to(ROOT).as_posix())
    inputs=dict(fixed)
    assert programs.get('crossplatform/round50/pool_weights50.py')==freeze['algorithm_sha256']
    inputs[authpath.relative_to(ROOT).as_posix()]=sha(authpath)
    return auth,inputs,programs


def completed_stages(inputs,programs):
    """Descriptor/seal/index checks, not a substitute for cost arithmetic audit."""
    stages={};sealed={};ids=None
    for stage in ('dino','numeric'):
        producer=read(PRODUCERS[stage]);assert producer['selected_configuration']=='wideentry_oldW'
        assert producer['output_root']==RUN and producer['no_GT_or_parameter_updates'] is True
        assert len(producer['chunks'])==24 and all(j['stage']==stage for j in producer['chunks'].values())
        for rel,h in producer['programs_sha256'].items():
            assert sha(local(rel))==h;assert rel not in programs or programs[rel]==h;programs[rel]=h
        folder=RUN+('/dino_chunks' if stage=='dino' else '/evidence_chunks')
        ap=folder+'/ALL_CHUNKS_SEALED.json';ip=folder+'/CASE_INDEX_PRIVATE.json'
        aggregate=read(ap);index=read(ip)
        assert aggregate['state']=='R50_FULL_'+stage.upper()+'_ALL5912_CHUNKS_COMPLETE'
        assert aggregate['n']==len(index)==5912 and aggregate['chunks']==24
        assert aggregate['evidence_plan_sha256']==sha(local(PRODUCERS[stage]))
        assert aggregate['case_index_sha256']==sha(local(ip))
        assert aggregate['GT_access'] is False and aggregate['optimizer_steps']==0
        pin(inputs,ap);pin(inputs,ip)
        if ids is None:ids=set(index)
        else:assert ids==set(index)
        expected_roots={RUN+'/'+j['output_subdir'] for j in producer['chunks'].values()}
        assert set(aggregate['sealed_chunks'])==expected_roots
        stage_n=0
        for root,h in aggregate['sealed_chunks'].items():
            assert sha(local(root)/'SEALED.json')==h
            seal=read(root+'/SEALED.json')
            assert seal['state']=='R50_FULL_FROZEN_'+stage.upper()+'_EVIDENCE_NO_GT'
            assert seal['plan_sha256']==sha(local(PRODUCERS[stage]));sealed[root]=h
            assert seal['generated_tokens']==seal['optimizer_steps']==0 and seal['model_unchanged'] is True
            stage_n+=seal['n']
        assert stage_n==5912
        for ref in index.values():
            path=local(ref['file']);roots=[r for r in expected_roots if path.is_relative_to(local(r))]
            assert len(roots)==1
            seal=read(roots[0]+'/SEALED.json');member=path.relative_to(local(roots[0])).as_posix()
            assert seal['files'][member]==ref['sha256']
        work=RUN+'/worklists/'+stage;wp=work+'/SEALED.json'
        pin(inputs,wp);workseal=read(wp)
        assert workseal['n']==5912 and workseal['chunks']==24
        for rel,h in producer['sealed_roots'].items():
            assert sha(local(rel)/'SEALED.json')==h
            assert rel not in sealed or sealed[rel]==h;sealed[rel]=h
        stages[stage]={'evidence_plan_file':PRODUCERS[stage],'worklist_root':work,
            'worklist_seal_sha256':sha(local(wp)),'chunk_seals_sha256':aggregate['sealed_chunks'],
            'model_recipe':producer['model_recipes'][stage],'model_assets':producer['model_assets'][stage]}
    for sub in ('frames','text','prepared'):
        root=RUN+'/'+sub;assert root in sealed;pin(inputs,root+'/SEALED.json')
    return stages,sealed


def emit(planname,grantname,plan,grant):
    pp='research/round50/experiment/'+planname;gp='research/round50/experiment/'+grantname
    assert not local(pp).exists() and not local(gp).exists()
    dump_exclusive(pp,plan)
    digest=sha(local(pp))
    for key in ('plan_sha256','audit_plan_sha256','independent_plan_sha256','scoring_plan_sha256'):
        if key in grant and grant[key] is None:grant[key]=digest
    dump_exclusive(gp,grant)
    return {'plan':pp,'plan_sha256':digest,'grant':gp,'grant_sha256':sha(local(gp)),
            'no_prediction_audit_or_score_executed':True}


def prediction(inputs,programs):
    assert not local(PRED).exists()
    assert {PRED_PROGRAM,'crossplatform/round50/full_common50.py','crossplatform/round50/pool_weights50.py'}<=set(programs)
    _,sealed=completed_stages(inputs,programs)
    plan={'state':'ROOT_LOCKED_R50_FULL5912_THREE_FIXED_BOX_READOUTS','n':5912,'arms':ARMS,
        'no_GT_or_model_or_optimizer':True,'runroot':RUN,'manifest':MANIFEST,'output':PRED,
        'evidence_case_index':RUN+'/evidence_chunks/CASE_INDEX_PRIVATE.json',
        'selected_configuration':'wideentry_oldW','lambda_VPR':[2,2,2],
        'scientific_freeze_sha256':sha(local(FREEZE)),'registered_full_gate_sha256':sha(local(GATE)),
        'programs_sha256':programs,'inputs_sha256':inputs,'sealed_roots':sealed}
    return emit('FULL_PREDICTION_PLAN_LOCK.json','FULL_PREDICTION_AUTHORIZATION.json',plan,
        {'state':'ROOT_AUTHORIZED_R50_FULL5912_THREE_FIXED_BOX_READOUTS','plan_sha256':None,
         'program_sha256':sha(local(PRED_PROGRAM)),'no_GT_or_model_or_optimizer':True})


def cost(inputs,programs):
    assert COST_PROGRAM in programs and not local('research/round50/independent_review/full_evidence_cost_audit_attempt001').exists()
    stages,sealed=completed_stages(inputs,programs)
    plan={'state':'ROOT_LOCKED_R50_INDEPENDENT_FULL_EVIDENCE_COST_AUDIT_NO_GT','n':5912,
        'no_GT_rawmedia_models_or_optimization':True,'runroot':RUN,
        'expected_target_pool_counts':{'old':9459,'wide':44093},
        'expected_DINO_whole_case_reuse':{'cases':224,'logical_uses':857},
        'source_capture_plan':SOURCE_PLAN,'programs_sha256':programs,'inputs_sha256':inputs,
        'sealed_roots_sha256':sealed,'stages':stages}
    return emit('FULL_EVIDENCE_COST_AUDIT_PLAN_LOCK.json','ROOT_INDEPENDENT_FULL_EVIDENCE_COST_AUDIT_AUTHORIZATION.json',plan,
        {'state':'ROOT_AUTHORIZED_R50_INDEPENDENT_FULL_EVIDENCE_COST_AUDIT_NO_GT',
         'program_sha256':sha(local(COST_PROGRAM)),'audit_plan_sha256':None})


def sealed_predictions(inputs):
    pp='research/round50/experiment/FULL_PREDICTION_PLAN_LOCK.json'
    pa='research/round50/experiment/FULL_PREDICTION_AUTHORIZATION.json'
    producer=read(pp);grant=read(pa)
    assert producer['output']==PRED and producer['manifest']==MANIFEST and producer['arms']==ARMS
    assert producer['n']==5912 and producer['no_GT_or_model_or_optimizer'] is True
    assert grant['plan_sha256']==sha(local(pp))
    seal=read(PRED+'/SEALED.json')
    assert seal['state']=='R50_FULL5912_THREE_FIXED_BOX_READOUTS_BEFORE_TASK_SCORE'
    assert seal['n']==5912 and seal['arms']==ARMS and seal['DEV336_bitexact_bridges']==336
    assert seal['plan_sha256']==sha(local(pp))
    for rel in (pp,pa,PRED+'/SEALED.json'):pin(inputs,rel)
    return pp,producer,seal


def arithmetic(inputs,programs):
    assert ARITH_PROGRAM in programs and not local('research/round50/independent_review/full5912_audit_attempt001').exists()
    pp,_,_=sealed_predictions(inputs)
    plan={'state':'ROOT_LOCKED_R50_INDEPENDENT_FULL5912_ARITHMETIC_AUDIT','n':5912,'arms':ARMS,
        'no_GT_rawXYZ_or_models':True,'producer_prediction_plan':pp,
        'programs_sha256':programs,'inputs_sha256':inputs}
    return emit('FULL_INDEPENDENT_AUDIT_PLAN_LOCK.json','ROOT_INDEPENDENT_FULL_AUDIT_AUTHORIZATION.json',plan,
        {'state':'ROOT_AUTHORIZED_R50_INDEPENDENT_FULL5912_ARITHMETIC_AUDIT_NO_GT',
         'program_sha256':sha(local(ARITH_PROGRAM)),'independent_plan_sha256':None,
         'prediction_seal_sha256':sha(local(PRED+'/SEALED.json'))})


def scoring(auth,inputs,programs):
    assert {SCORE_PROGRAM,'research/round5/audit/independent_core_v2.py','tools/obb_iou.py'}<=set(programs)
    assert not local('research/round50/independent_score/full5912_attempt001').exists()
    sealed_predictions(inputs)
    arith=read(ARITH);cost_result=read(COST)
    assert arith['state']=='PASS_R50_INDEPENDENT_FULL5912_FIXED_SOURCE_THREE_READOUT_AUDIT_NO_GT'
    assert arith['n']==5912 and arith['arms']==3 and arith['DEV336_three_arm_score_query_box_bridges_bitexact']==336
    assert arith['prediction_seal_sha256']==sha(local(PRED+'/SEALED.json'))
    assert arith['program_sha256']==sha(local(ARITH_PROGRAM))
    assert cost_result['state']=='PASS_R50_INDEPENDENT_FULL_FROZEN_EVIDENCE_CALL_COST_AND_RAW_ARITHMETIC_AUDIT_NO_GT'
    assert cost_result['n']==5912 and cost_result['program_sha256']==sha(local(COST_PROGRAM))
    assert cost_result['audit_plan_sha256']==sha(local('research/round50/experiment/FULL_EVIDENCE_COST_AUDIT_PLAN_LOCK.json'))
    for rel in (ARITH,COST,'research/round50/experiment/FULL_INDEPENDENT_AUDIT_PLAN_LOCK.json',
        'research/round50/experiment/ROOT_INDEPENDENT_FULL_AUDIT_AUTHORIZATION.json',
        'research/round50/experiment/FULL_EVIDENCE_COST_AUDIT_PLAN_LOCK.json',
        'research/round50/experiment/ROOT_INDEPENDENT_FULL_EVIDENCE_COST_AUDIT_AUTHORIZATION.json',
        'research/round50/independent_review/full_evidence_cost_audit_attempt001/SEALED.json',
        'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json',
        'crossplatform/round5/baseline_engineering/PREPARATION_RECEIPT.json'):
        pin(inputs,rel)
    devseal=read(DEV_SCORE+'/SEALED.json')
    for name,h in devseal['files'].items():
        rel=DEV_SCORE+'/'+name;assert sha(local(rel))==h;pin(inputs,rel)
    explanation=auth.get('DEV_seed_mismatch_explanation')
    if arith['DEV336_metadata_seed_mismatch_n']:
        assert isinstance(explanation,str) and explanation.strip(), 'Root explanation required before GT, no inferred IDs'
    plan={'state':'ROOT_LOCKED_R50_SINGLE_POSTSEAL_FULL5912_TASK_SCORE','n':5912,'arms':ARMS,
        'source_manifest':MANIFEST,'prediction_directory':PRED,'independent_prediction_audit':ARITH,
        'independent_evidence_cost_audit':COST,'scientific_freeze':FREEZE,'registered_full_gate':GATE,
        'prediction_schema':{'arm_names':'arm_names','scores':'scores','queries':'queries',
            'selected_boxes':'selected_original_boxes','source_boxes':'source_boxes','source_scores':'source_scores'},
        'programs_sha256':programs,'inputs_sha256':inputs,'no_new_method_selection_after_full_GT':True}
    if explanation:plan['DEV_seed_mismatch_explanation']=explanation
    return emit('FULL_SCORING_PLAN_LOCK.json','ROOT_SINGLE_FULL_SCORING_AUTHORIZATION.json',plan,
        {'state':'ROOT_AUTHORIZED_R50_SINGLE_POSTSEAL_FULL5912_TASK_SCORE',
         'program_sha256':sha(local(SCORE_PROGRAM)),'scoring_plan_sha256':None,
         'prediction_seal_sha256':sha(local(PRED+'/SEALED.json')),
         'independent_prediction_audit_sha256':sha(local(ARITH)),
         'independent_evidence_cost_audit_sha256':sha(local(COST))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prediction','cost','arithmetic','scoring'),required=True)
    parser.add_argument('--authorization',required=True);args=parser.parse_args()
    auth,inputs,programs=context(args)
    if args.mode=='prediction':result=prediction(inputs,programs)
    elif args.mode=='cost':result=cost(inputs,programs)
    elif args.mode=='arithmetic':result=arithmetic(inputs,programs)
    else:result=scoring(auth,inputs,programs)
    print(json.dumps({'mode':args.mode,**result},indent=2))


if __name__=='__main__':main()
