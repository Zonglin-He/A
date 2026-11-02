"""UNEXECUTED independent reaggregation of 2016 already-saved episodic IoUs.

Stdlib only; no author/evaluator imports, NumPy, GT, raw metadata, manifest,
capture/fit/closure tensors, predictions, models, optimizer or network.
All future reads occur after a separate root grant and an exact input list.
This is a saved-scalar audit, never a second scoring/evaluation pass.
"""
from pathlib import Path
import collections,csv,hashlib,itertools,json,math,statistics,time

ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/round50/experiment'
PLAN=EXP/'EPISODIC_SAVED_SCALAR_AUDIT_PLAN_LOCK.json'
AUTH=EXP/'ROOT_INDEPENDENT_EPISODIC_SAVED_SCALAR_AUDIT_AUTHORIZATION.json'
SCORE='research/round50/independent_score/episodic336_attempt001'
OUT='research/round50/independent_scalar_recheck/episodic336_attempt001'
OLD='research/round50/independent_score/dev336_attempt001'
SCORE_PLAN='research/round50/experiment/EPISODIC_SCORING_PLAN_LOCK.json'
SCORE_AUTH='research/round50/experiment/ROOT_EPISODIC_SINGLE_SCORE_EXECUTION_AUTHORIZATION.json'
SCORER='research/round50/score_episodic50_draft.py'
ARMS=['source','literal_A_wideG','group_A_wideG','sameinfo_consensus0','sameinfo_consensus1','sameinfo_consensus3']
SPLITS=['ALL336','D208','V64','M64']
RULE='ALL336_balanced_max_step1_or3_exact_tie1_strictgain_vs_consensus0_and_literalAwideG'
FILES=['ALL336_TASK_ROWS_PRIVATE.json','TASK_SUMMARY.json','SCENE_SUMMARIES_PRIVATE.json',
    'ALL_ORDERED_NEGATIVE_COMPARISONS.json','DEV_STEP_SELECTION_AND_GATE.json',
    'ALL2016_ANONYMOUS_TASK_SCALARS.csv','ALL_ANONYMOUS_ORDERED_NEGATIVES.csv',
    'CONDITION_COUNTS.json','SCORING_RECEIPT.json']
BASE_INPUTS=[SCORE+'/SEALED.json',*[SCORE+'/'+n for n in FILES],
    OLD+'/TASK_SUMMARY.json',OLD+'/ALL336_TASK_ROWS_PRIVATE.json',OLD+'/SEALED.json',
    SCORE_PLAN,SCORE_AUTH]
ERROR=0.;COMPARISONS=0;READS=[];ALLOWED=set()


def local(rel):
    p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
    assert p.as_posix().startswith('research/round50/')
    assert 'FINAL' not in p.as_posix() and 'round46' not in p.as_posix()
    return ROOT/p


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(rel):
    assert rel in ALLOWED, ('Saved-scalar input not prelocked',rel)
    READS.append(rel);return json.loads(local(rel).read_text())


def dump(folder,name,value):
    (folder/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def same(a,b,path='$'):
    global ERROR,COMPARISONS
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
        for k in a:same(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),(path,'length')
        for j,(x,y) in enumerate(zip(a,b)):same(x,y,path+f'[{j}]')
    elif type(a) in (int,float):
        assert type(b) in (int,float) and math.isfinite(a) and math.isfinite(b),path
        delta=abs(a-b);ERROR=max(ERROR,delta);COMPARISONS+=1
        assert delta<=1e-10,(path,a,b,delta)
    else:assert type(a) is type(b) and a==b,(path,a,b)


def average(values):return statistics.fmean(values)


def aggregate(tasks):
    summary={};scenes=[];effects={}
    for split in SPLITS:
        rr=tasks if split=='ALL336' else [r for r in tasks if r['split']==split]
        assert len(rr)=={'ALL336':336,'D208':208,'V64':64,'M64':64}[split]
        part={}
        for platform in ('drone','quad'):
            pr=[r for r in rr if r['platform']==platform]
            assert len(pr)=={'ALL336':168,'D208':104,'V64':32,'M64':32}[split]
            part[platform]={}
            for arm in ARMS:
                values=[r['IoU'][arm] for r in pr]
                d0=[r['IoU'][arm]-r['IoU']['sameinfo_consensus0'] for r in pr]
                dg=[r['IoU'][arm]-r['IoU']['literal_A_wideG'] for r in pr]
                part[platform][arm]={'n':len(pr),'mIoU_percent':average(values)*100,
                    'Acc25_percent':sum(v>.25 for v in values)/len(pr)*100,
                    'Acc50_percent':sum(v>.5 for v in values)/len(pr)*100,
                    'gain_vs_consensus0_pp':average(d0)*100,'gain_vs_literal_A_wideG_pp':average(dg)*100,
                    'improved_vs_consensus0':sum(v>1e-10 for v in d0),
                    'worsened_vs_consensus0':sum(v< -1e-10 for v in d0),
                    'improved_vs_literal_A_wideG':sum(v>1e-10 for v in dg),
                    'worsened_vs_literal_A_wideG':sum(v< -1e-10 for v in dg),
                    'query_changes_vs_consensus0':sum(r['queries'][arm]!=r['queries']['sameinfo_consensus0'] for r in pr),
                    'accepted_episode_n':sum(r['accepted'] for r in pr),
                    'scene_clusters':len({r['scene_PRIVATE'] for r in pr})}
            for scene in sorted({r['scene_PRIVATE'] for r in pr}):
                sr=[r for r in pr if r['scene_PRIVATE']==scene]
                scenes.append({'split':split,'platform':platform,'scene_PRIVATE':scene,'n':len(sr),
                    'mIoU_percent':{arm:average(r['IoU'][arm] for r in sr)*100 for arm in ARMS}})
        part['balanced_mIoU_percent']={arm:average(part[p][arm]['mIoU_percent'] for p in ('drone','quad')) for arm in ARMS}
        summary[split]=part;effects[split]={}
        for platform in ('drone','quad','balanced'):
            m=part['balanced_mIoU_percent'] if platform=='balanced' else {a:part[platform][a]['mIoU_percent'] for a in ARMS}
            effects[split][platform]={
                'literal_G_minus_source_pp':m['literal_A_wideG']-m['source'],
                'group_A_minus_literal_G_pp':m['group_A_wideG']-m['literal_A_wideG'],
                'sameinfo0_minus_literal_G_pp':m['sameinfo_consensus0']-m['literal_A_wideG'],
                'step1_minus_sameinfo0_pp':m['sameinfo_consensus1']-m['sameinfo_consensus0'],
                'step3_minus_sameinfo0_pp':m['sameinfo_consensus3']-m['sameinfo_consensus0'],
                'step1_minus_literal_G_pp':m['sameinfo_consensus1']-m['literal_A_wideG'],
                'step3_minus_literal_G_pp':m['sameinfo_consensus3']-m['literal_A_wideG']}
    negative=[]
    for r in tasks:
        for arm,reference in itertools.permutations(ARMS,2):
            delta=r['IoU'][arm]-r['IoU'][reference]
            if delta< -1e-10:negative.append({'row':r['row'],'platform':r['platform'],'split':r['split'],
                'arm':arm,'reference':reference,'delta_pp':delta*100})
    return summary,scenes,negative,effects


def check_csv(rel,fields,expected):
    assert rel in ALLOWED;READS.append(rel)
    with local(rel).open(newline='') as stream:
        reader=csv.DictReader(stream);assert reader.fieldnames==fields
        rows=list(reader)
    assert len(rows)==len(expected),(rel,'rowcount')
    for j,(got,want) in enumerate(zip(rows,expected)):
        assert set(got)==set(want),(rel,j,'columns')
        for name,value in want.items():
            if type(value) is bool:assert got[name]==str(value),(rel,j,name)
            elif type(value) is int:assert str(value)==got[name],(rel,j,name)
            elif type(value) is float:
                parsed=float(got[name]);assert math.isfinite(parsed) and parsed==value,(rel,j,name)
            else:assert got[name]==value,(rel,j,name)


def main():
    global ALLOWED
    start=time.perf_counter()
    # Only these two registration documents precede closed-list validation.
    auth=json.loads(AUTH.read_text());plan=json.loads(PLAN.read_text())
    assert auth['state']=='ROOT_AUTHORIZED_R50_INDEPENDENT_SAVED_EPISODIC_SCALAR_REAGGREGATION_NO_GT'
    assert auth['program_sha256']==sha(__file__) and auth['audit_plan_sha256']==sha(PLAN)
    assert plan['n']==336 and plan['arms']==ARMS and plan['task_values']==2016
    assert plan['saved_IoU_only_no_GT_or_new_evaluation'] is True
    assert plan['score_root']==SCORE and plan['output']==OUT
    numeric_result=plan['independent_numeric_audit'];numeric_root=Path(numeric_result).parent.as_posix()
    numeric_plan=plan['independent_numeric_audit_plan'];numeric_program=plan['independent_numeric_audit_program']
    assert Path(numeric_result).name=='AUDIT_RESULT.json' and numeric_result.startswith('research/round50/independent_review/')
    assert numeric_plan.startswith('research/round50/experiment/') and numeric_plan.endswith('.json')
    assert numeric_program.startswith('research/round50/independent_review/') and numeric_program.endswith('.py')
    for rel in (numeric_result,numeric_plan,numeric_program):local(rel)
    inputs=BASE_INPUTS+[numeric_result,numeric_root+'/SEALED.json',numeric_plan]
    assert len(inputs)==len(set(inputs))==18
    assert plan['original_DEV_score_root']==OLD and plan['selection_rule']==RULE
    assert set(plan['inputs_sha256'])==set(inputs), 'Exact closed saved-input list required'
    assert set(plan['programs_sha256'])=={SCORER,numeric_program}
    ALLOWED=set(inputs)
    for rel,h in {**plan['inputs_sha256'],**plan['programs_sha256']}.items():assert sha(local(rel))==h,rel
    assert auth['score_seal_sha256']==sha(local(SCORE+'/SEALED.json'))
    scoreseal=read(SCORE+'/SEALED.json');assert set(scoreseal['files'])==set(FILES)
    assert scoreseal['state']=='PASS_R50_SINGLE_DEV336_EPISODIC_TASK_SCORE'
    for name,h in scoreseal['files'].items():assert sha(local(SCORE+'/'+name))==h
    numerical=read(numeric_result);numericseal=read(numeric_root+'/SEALED.json')
    assert numericseal['files']['AUDIT_RESULT.json']==sha(local(numeric_result))
    assert numerical['state']=='PASS_R50_INDEPENDENT_EPISODIC_NUMERIC_AND_CONTRACT_AUDIT_NO_GT'
    assert numerical['n']==336 and numerical['arms']==6 and numerical['no_GT_read'] is True
    assert numerical['program_sha256']==plan['programs_sha256'][numeric_program]
    assert numerical['plan_sha256']==sha(local(numeric_plan))
    scoringplan=read(SCORE_PLAN);scoringauth=read(SCORE_AUTH);numericplan=read(numeric_plan)
    assert scoringplan['selection_rule']==RULE and scoringplan['n']==336 and scoringplan['arms']==ARMS
    assert scoringplan['task_values']==2016 and scoringplan['max_GT_scoring_passes']==1
    for registered in (scoringplan,numericplan):
        assert registered['full_advance_gate']['state']=='PASS_R50_FULL_GATE_FOR_SEPARATE_DEV_EPISODIC_PLANS'
        assert registered['full_advance_gate']['both_reduction_predicates_agree_and_PASS'] is True
    assert scoringauth['state']=='ROOT_AUTHORIZED_R50_SINGLE_DEV_EPISODIC_POSTSEAL_TASK_SCORE'
    assert scoringauth['plan_sha256']==sha(local(SCORE_PLAN)) and scoringauth['program_sha256']==plan['programs_sha256'][SCORER]
    assert scoringplan['independent_audit_result']==numeric_result
    assert scoringauth['independent_audit_sha256']==sha(local(numeric_result))
    assert scoringauth['prediction_seal_sha256']==numerical['prediction_seal_sha256']
    changed=numerical['nonzero_parameter_element_count'];assert type(changed) is int and changed>=0
    if changed:
        assert numerical['installed_closure_requirement_satisfied'] is True
        assert numerical['actual_adapted_closure_verified'] is True and numerical['actual_adapted_closure_episodes']>0
        assert numerical['TTA_effect_actually_tested'] is True
        measurement='ACTUAL_NONZERO_EPISODIC_STATES_WITH_INSTALLED_CLOSURE'
    else:
        assert scoringplan['allow_verified_no_parameter_update_branch'] is True
        assert numericplan['allow_verified_no_parameter_update_branch'] is True
        assert numerical['all_saved_heads_source_bytes_identical'] is True and numerical['all_no_update_outputs_equal_s0'] is True
        assert numerical['installed_closure_state']=='NOT_APPLICABLE_VERIFIED_NO_PARAMETER_UPDATE'
        assert numerical['installed_closure_requirement_satisfied'] is False and numerical['actual_adapted_closure_verified'] is False
        assert numerical['TTA_effect_actually_tested'] is False
        measurement='VERIFIED_NO_PARAMETER_UPDATE_TTA_NOT_ACTUALLY_TESTED'
    oldseal=read(OLD+'/SEALED.json')
    for name in ('TASK_SUMMARY.json','ALL336_TASK_ROWS_PRIVATE.json'):
        assert oldseal['files'][name]==sha(local(OLD+'/'+name))
    oldrows=read(OLD+'/ALL336_TASK_ROWS_PRIVATE.json');oldby={r['id_PRIVATE']:r for r in oldrows}
    assert len(oldrows)==len(oldby)==336
    tasks=read(SCORE+'/ALL336_TASK_ROWS_PRIVATE.json')
    assert len(tasks)==336 and len({r['row'] for r in tasks})==len({r['id_PRIVATE'] for r in tasks})==336
    assert set(oldby)=={r['id_PRIVATE'] for r in tasks}
    assert collections.Counter(r['platform'] for r in tasks)=={'drone':168,'quad':168}
    assert collections.Counter(r['split'] for r in tasks)=={'D208':208,'V64':64,'M64':64}
    assert {p:len({r['scene_PRIVATE'] for r in tasks if r['platform']==p}) for p in ('drone','quad')}=={'drone':2,'quad':3}
    for row in tasks:
        assert set(row['queries'])==set(row['IoU'])==set(ARMS)
        assert all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in row['IoU'].values())
        assert all(type(q) is int and 0<=q<256 for q in row['queries'].values())
        assert type(row['accepted']) is bool and isinstance(row['reason'],str)
        assert set(row['nonzero_parameter_updates'])=={'1','3'} and all(type(x) is bool for x in row['nonzero_parameter_updates'].values())
        prior=oldby[row['id_PRIVATE']]
        for name in ('platform','split','scene_PRIVATE'):assert row[name]==prior[name]
        for arm,oldarm in [('source','source'),('literal_A_wideG','wideentry_oldW')]:
            assert row['queries'][arm]==prior['queries'][oldarm]
            assert abs(row['IoU'][arm]-prior['IoU'][oldarm])<1e-12
        assert row['queries']['group_A_wideG']==row['queries']['literal_A_wideG']
        assert row['IoU']['group_A_wideG']==row['IoU']['literal_A_wideG']
        for step in ('1','3'):
            nonzero=row['nonzero_parameter_updates'][step]
            if not row['accepted']:assert not nonzero
            if not nonzero:
                assert row['queries']['sameinfo_consensus'+step]==row['queries']['sameinfo_consensus0']
                assert row['IoU']['sameinfo_consensus'+step]==row['IoU']['sameinfo_consensus0']
    accepted=sum(r['accepted'] for r in tasks)
    nonzero={s:sum(r['nonzero_parameter_updates'][s] for r in tasks) for s in ('1','3')}
    assert numerical['counts'].get('accepted',0)==accepted
    assert numerical['counts'].get('optimizer_steps',0)==3*accepted
    for s in ('1','3'):assert numerical['counts'].get('nonzero_s'+s,0)==nonzero[s]
    assert bool(changed)==bool(nonzero['1']+nonzero['3'])
    destination=local(OUT);assert not destination.exists();destination.mkdir(parents=True)
    try:
        summary,scenes,negative,effects=aggregate(tasks)
        same(summary,read(SCORE+'/TASK_SUMMARY.json'),'TASK_SUMMARY')
        same(scenes,read(SCORE+'/SCENE_SUMMARIES_PRIVATE.json'),'SCENES')
        same(negative,read(SCORE+'/ALL_ORDERED_NEGATIVE_COMPARISONS.json'),'NEGATIVES')
        oldsummary=read(OLD+'/TASK_SUMMARY.json')
        for split in SPLITS:
            for platform in ('drone','quad'):
                for arm,oldarm in [('source','source'),('literal_A_wideG','wideentry_oldW')]:
                    for metric in ('n','mIoU_percent','Acc25_percent','Acc50_percent','scene_clusters'):
                        same(summary[split][platform][arm][metric],oldsummary[split][platform][oldarm][metric],
                             'OLD_DEV_BRIDGE.'+split+'.'+platform+'.'+arm+'.'+metric)
        m=summary['ALL336']['balanced_mIoU_percent']
        step=1 if m['sameinfo_consensus1']>=m['sameinfo_consensus3'] else 3
        arm='sameinfo_consensus'+str(step);d0=m[arm]-m['sameinfo_consensus0'];dg=m[arm]-m['literal_A_wideG']
        gate={'state':'R50_EXPOSED_DEV_EPISODIC_SHARED_STEP_SELECTION_ALL_TRIALS_RETAINED',
            'selected_step':step,'exact_tie_priority':1,'selected_arm':arm,'accepted_episodes':accepted,'all336_denominator':336,
            'gain_vs_sameinfo_consensus0_pp':d0,'gain_vs_literal_A_wideG_pp':dg,
            'gate_PASS':bool(accepted>0 and changed>0 and d0>0 and dg>0),
            'nonzero_parameter_element_count':changed,'measurement_mode':measurement,
            'all_zero_parameters_force_STOP':changed==0,'all_six_arms_and_negatives_retained':True,
            'selection_is_exposed_DEV_label_based':True,'no_GT_classspan_or_feedback_filtering_or_R8_replacement':True,
            'full_TTA_not_authorized_by_this_receipt':True}
        # Exact step/positive-gain decisions remain exact; the comparison
        # tolerance for independently summed percentages is not a new gate.
        same(gate,read(SCORE+'/DEV_STEP_SELECTION_AND_GATE.json'),'STEP_GATE')
        if not changed:assert gate['gate_PASS'] is False and step==1 and d0==0
        counts={'n':336,'rejected_n':336-accepted,'accepted_n':accepted,
                'nonzero_s1_n':nonzero['1'],'nonzero_s3_n':nonzero['3']}
        same(counts,read(SCORE+'/CONDITION_COUNTS.json'),'CONDITIONS')
        receipt=read(SCORE+'/SCORING_RECEIPT.json')
        assert receipt['state']=='PASS_R50_SINGLE_DEV336_EPISODIC_TASK_SCORE' and receipt['n']==336 and receipt['arms']==6
        assert receipt['task_values']==2016 and receipt['historical_source_and_literalG_replayed_n']==336
        assert receipt['ordered_negative_comparisons']==len(negative) and receipt['gate_PASS']==gate['gate_PASS']
        assert receipt['program_sha256']==plan['programs_sha256'][SCORER]
        assert receipt['prediction_seal_sha256']==numerical['prediction_seal_sha256']
        assert receipt['independent_audit_sha256']==sha(local(numeric_result))
        assert receipt['measurement_mode']==measurement and receipt['nonzero_parameter_element_count']==changed
        assert receipt['single_GT_scoring_pass'] is True and receipt['no_method_or_model_changes'] is True
        scalar=[{'row':r['row'],'platform':r['platform'],'split':r['split'],'arm':a,'IoU':r['IoU'][a],
            'accepted':r['accepted'],'nonzero_s1':r['nonzero_parameter_updates']['1'],
            'nonzero_s3':r['nonzero_parameter_updates']['3']} for r in tasks for a in ARMS]
        check_csv(SCORE+'/ALL2016_ANONYMOUS_TASK_SCALARS.csv',
                  ['row','platform','split','arm','IoU','accepted','nonzero_s1','nonzero_s3'],scalar)
        check_csv(SCORE+'/ALL_ANONYMOUS_ORDERED_NEGATIVES.csv',
                  ['row','platform','split','arm','reference','delta_pp'],negative)
        dump(destination,'RECOMPUTED_TASK_SUMMARY.json',summary)
        dump(destination,'RECOMPUTED_SCENE_SUMMARIES_PRIVATE.json',scenes)
        dump(destination,'RECOMPUTED_ALL_ORDERED_NEGATIVES.json',negative)
        dump(destination,'RECOMPUTED_SAMEINFO_AND_LITERAL_G_EFFECTS.json',effects)
        dump(destination,'FIXED_STEP_SELECTION_AND_MEASUREMENT_RECHECK.json',gate)
        dump(destination,'RECOMPUTED_CONDITION_COUNTS.json',counts)
    except BaseException as error:
        dump(destination,'FAILED.json',{'error_type':type(error).__name__,'error':str(error),
            'no_GT_evaluator_or_new_IoU':True,'no_automatic_score_rerun_or_selection_change':True})
        raise
    assert set(READS)<=set(inputs)
    result={'state':'PASS_R50_INDEPENDENT_SAVED_EPISODIC_SCALARS_SELECTION_AND_MEASUREMENT_REAGGREGATION',
        'n':336,'arms':6,'IoU_values_reaggregated':2016,'known_platform_counts':{'drone':168,'quad':168},
        'known_scene_counts':{'drone':2,'quad':3},'scene_summary_rows_reproduced':len(scenes),
        'ordered_negative_comparisons_reproduced':len(negative),'accepted_episodes':accepted,
        'nonzero_parameter_element_count':changed,'measurement_mode':measurement,
        'TTA_effect_actually_tested':changed>0,'verified_allzero_is_NOT_TTA_and_forces_STOP':changed==0,
        'selected_step':step,'selected_arm':arm,'gate_PASS':gate['gate_PASS'],
        'source_literalG_and_groupG_scalar_bridges_reproduced':336,
        'fixed_step_selection_exact_tie1_not_a_new_parameter_grid':True,
        'step_and_gate_predicates_agree_between_independent_fmean_and_saved_numpy_summary':True,
        'all336_all_trials_rejections_and_negatives_retained':True,
        'sameinfo_B_is_used_in_fitting_not_unseen_validation':True,
        'exposed_DEV_selection_not_new_independent_confirmation':True,
        'numeric_comparisons':COMPARISONS,'max_abs_scalar_error':ERROR,
        'score_seal_sha256':sha(local(SCORE+'/SEALED.json')),
        'independent_numeric_audit_result':numeric_result,
        'independent_numeric_audit_sha256':sha(local(numeric_result)),
        'no_GT_metadata_manifest_media_NPZ_model_author_import_or_IoU_recomputation':True,
        'closed_input_read_count':len(set(READS)),'program_sha256':sha(__file__),
        'audit_plan_sha256':sha(PLAN),'seconds':time.perf_counter()-start}
    dump(destination,'AUDIT_RESULT.json',result)
    dump(destination,'SEALED.json',{'state':result['state'],
        'files':{p.name:sha(p) for p in destination.iterdir() if p.is_file()}})
    print(json.dumps(result))


if __name__=='__main__':main()
