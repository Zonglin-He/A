"""DRAFT: seal seven unique fixed-box DEV336 readouts after visual evidence.

No task GT, RGB, raw XYZ, model, optimizer, fallback imputation or selection.
Future visual data must be complete, sealed, and root-pinned before execution.
"""
from pathlib import Path
import json,hashlib,time,datetime,collections,re
import numpy as np
from pool_weights50 import five_readouts,exact
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'research/round50/experiment'
PREP=ROOT/'crossplatform/round50/prepared336_attempt001'
VISUAL=ROOT/'crossplatform/round50/visual336_attempt001'
OLD=ROOT/'crossplatform/round49/stage336_attempt001'
OUT=ROOT/'crossplatform/round50/predicted336_attempt001'
ARMS=['source','old_G','wideentry_oldW','oldentry_newW_k3','wideentry_newW_k3','oldentry_newW_k10','wideentry_newW_k10']
SEALS={};BINDINGS={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def bound(p,folder):
    p=Path(p);folder=Path(folder);sp=folder/'SEALED.json'
    if folder not in SEALS:SEALS[folder]=read(sp);BINDINGS[sp.relative_to(ROOT).as_posix()]=sha(sp)
    h=sha(p);assert SEALS[folder]['files'][p.relative_to(folder).as_posix()]==h
    BINDINGS[p.relative_to(ROOT).as_posix()]=h
    return h

def check():
    plan=read(EXP/'PREDICT_PLAN_LOCK.json');auth=read(EXP/'PREDICT_EXECUTION_AUTHORIZATION.json')
    assert auth['state']=='ROOT_AUTHORIZED_R50_DEV336_SEVEN_READOUT_PREDICTION'
    assert auth['plan_sha256']==sha(EXP/'PREDICT_PLAN_LOCK.json')
    assert plan['arm_names']==ARMS and plan['n']==336 and plan['no_GT_model_or_optimizer'] is True
    for group in ('programs_sha256','inputs_sha256'):
        for p,h in plan[group].items():assert sha(ROOT/p)==h,p
    assert sha(VISUAL/'SEALED.json')==plan['visual_seal_sha256']
    vs=read(VISUAL/'SEALED.json');assert vs['n']==336 and vs['state']==plan['visual_seal_expected_state']
    # Pin the inventory provenance, not a newly self-consistent replacement seal.
    inv=ROOT/'research/round49/inventory_attempt001';oldseal=read(inv/'SEALED.json')
    for p,h in oldseal['files'].items():assert sha(inv/p)==h
    return plan,read(inv/'SOURCE_BINDINGS_PRIVATE.json')

def checked_views(views):
    assert isinstance(views,list) and len(views)==2
    deltas=[];full=[];gray=[]
    for v in views:
        assert isinstance(v,dict) and {'full','gray','visual_evidence'}<=set(v)
        for name in ('full','gray'):
            obj=v[name];value=obj['log_odds_yes_no']
            assert type(value) in (int,float) and np.isfinite(value)
            assert obj['generated_tokens']==0 and type(obj['input_tokens']) is int and 0<obj['input_tokens']<=8192
            for key in ('image_sha256','prompt_sha256'):assert re.fullmatch('[0-9a-f]{64}',obj[key])
        assert type(v['visual_evidence']) in (int,float)
        delta=v['full']['log_odds_yes_no']-v['gray']['log_odds_yes_no']
        assert delta==v['visual_evidence'] and np.isfinite(delta), 'full_minus_gray_must_be_original_exact_subtraction'
        deltas.append(v['visual_evidence']);full.append(v['full']['log_odds_yes_no']);gray.append(v['gray']['log_odds_yes_no'])
    # Preserve original operation: average the per-view differences, not the
    # potentially differently rounded difference of full/gray view means.
    return float(np.mean(deltas)),float(np.mean(full)),float(np.mean(gray))

def old_core_views_equal(old,new):
    assert len(old)==len(new)==2
    for a,b in zip(old,new):
        for key in ('full','gray','visual_evidence'):assert a[key]==b[key], 'old_numeric_views_must_be_reused_without_regeneration'

def main():
    plan,legacybindings=check();assert not OUT.exists();OUT.mkdir()
    for name in ('samples_PRIVATE','cases_PRIVATE'):(OUT/name).mkdir()
    rr=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');assert len(rr)==336 and len({r['id'] for r in rr})==336
    counts=collections.Counter();records=[];start=time.perf_counter()
    try:
        for i,r in enumerate(rr):
            sid=r['id'];ap=PREP/'arrays_PRIVATE'/(sid+'.npz');cp=PREP/'cases_PRIVATE'/(sid+'.json')
            vp=VISUAL/'cases_PRIVATE'/(sid+'.json');sp=OLD/'samples_PRIVATE'/(sid+'.npz')
            for p,f in ((ap,PREP),(cp,PREP),(vp,VISUAL),(sp,OLD)):bound(p,f)
            a=npz(ap);c=read(cp);v=read(vp);old=npz(sp)
            assert c['id_PRIVATE']==v['id_PRIVATE']==sid
            assert v['prep_array_sha256']==sha(ap) and v['prep_case_sha256']==sha(cp)
            assert v['actual_subject_query_sha256']==c['actual_subject_query_sha256']
            assert v['wide_pool_PRIVATE']==a['wide_pool'].tolist()
            textpath=ROOT/c['actual_text_source_PRIVATE'];assert sha(textpath)==legacybindings[c['actual_text_source_PRIVATE']]
            BINDINGS[c['actual_text_source_PRIVATE']]=sha(textpath)
            text=read(textpath);assert hashlib.sha256(text['subject_query'].encode()).hexdigest()==c['actual_subject_query_sha256']
            base=ROOT/'crossplatform'/('round24' if r['stage']=='M' else 'round23');suffix='M64' if r['stage']=='M' else '272'
            oldprep=base/('prepared'+suffix+'_attempt001')/'cases_PRIVATE'/(sid+'.json')
            oldevidence=base/('evidence'+suffix+'_attempt001')/'cases_PRIVATE'/(sid+'.json')
            for path in (oldprep,oldevidence):
                digest=sha(path);rel=path.relative_to(ROOT).as_posix();assert legacybindings[rel]==digest;BINDINGS[rel]=digest
            framebinding=read(oldprep)['bindings'];oldtargets=read(oldevidence)['targets_PRIVATE']
            assert v['image_sha256']==framebinding['image_sha256'] and v['mask_cache_sha256']==framebinding['mask_cache_sha256']
            wide=a['wide_pool'];poolkeys={str(m) for m in wide}
            assert set(v['targets_PRIVATE'])==set(v['mask_sha256_by_index_PRIVATE'])==poolkeys
            e=a['image_e'].copy();full=a['image_full'].copy();gray=a['image_gray'].copy()
            for m in wide:
                key=str(m);assert v['mask_sha256_by_index_PRIVATE'][key]==str(a['mask_sha256'][m])
                ev,ef,eg=checked_views(v['targets_PRIVATE'][key])
                if key in oldtargets:old_core_views_equal(oldtargets[key],v['targets_PRIVATE'][key])
                e[m]=ev;full[m]=ef;gray[m]=eg
            assert np.isfinite(e[wide]).all(), 'No source fallback for missing new mask evidence'
            oldpool=a['old_pool'];exact(e[oldpool],a['image_e'][oldpool],'oldmask_e_mean')
            exact(full[oldpool],a['image_full'][oldpool],'oldmask_full_mean');exact(gray[oldpool],a['image_gray'][oldpool],'oldmask_gray_mean')
            exact(a['source_boxes'],old['source_boxes'],'original_source_boxpool');exact(a['source_scores'],old['source_scores'],'original_source_scores')
            oldvpr={key:old[key] for key in ('vision','position','relation')}
            args=(a['source_scores'],a['source_boxes'],oldpool,a['crop_valid'],a['mask_counts'],a['allvisible'],a['unassigned'],a['ambiguous'],
                a['legacy_support'],a['legacy_coverage'],a['legacy_projection_iou'],e,a['raw_mask_PR'])
            result3=five_readouts(*args,3,expected_old_weights=a['old_weights'],expected_old_vpr=oldvpr)
            result10=five_readouts(*args,10,expected_old_weights=a['old_weights'],expected_old_vpr=oldvpr)
            for key in ('scores','queries','selected_original_boxes'):exact(result3[key][:3],result10[key][:3],'k_independent_fixed_rows_'+key)
            exact(result3['scores'][1],old['arm_scores'][7],'oldG_fullscore_endpoint')
            assert int(result3['queries'][1])==int(old['queries'][7])
            scores=np.concatenate([result3['scores'],result10['scores'][3:]],0)
            queries=np.concatenate([result3['queries'],result10['queries'][3:]],0)
            boxes=a['source_boxes'][queries];exact(boxes,np.concatenate([result3['selected_original_boxes'],result10['selected_original_boxes'][3:]],0),'all7_original_boxes')
            newstates=[result3['states'][3],result3['states'][4],result10['states'][3],result10['states'][4]]
            states=result3['states']+result10['states'][3:]
            payload={'arm_names':np.array(ARMS),'scores':scores,'queries':queries,'selected_original_boxes':boxes,
                'source_boxes':a['source_boxes'],'source_scores':a['source_scores'],'old_pool':oldpool,'wide_pool':wide,
                'image_e':e,'image_full':full,'image_gray':gray,'raw_mask_PR':a['raw_mask_PR'],
                'rho':np.stack([state['rho'] for state in newstates]),
                'new_association_arm_names':np.array([ARMS[i] for i in (3,4,5,6)]),
                'vision':np.stack([state.get('vision',np.zeros(256)) for state in states]),
                'position':np.stack([state.get('position',np.zeros(256)) for state in states]),
                'relation':np.stack([state.get('relation',np.zeros(256)) for state in states]),
                'conditional_vision_clipped':np.stack([state['conditional_vision_clipped'] for state in newstates]),
                'conditional_position_clipped':np.stack([state['conditional_position_clipped'] for state in newstates]),
                'conditional_relation_clipped':np.stack([state['conditional_relation_clipped'] for state in newstates]),
                'mapped':np.stack([state.get('mapped',np.zeros(256,bool)) for state in states])}
            for k,result in ((3,result3),(10,result10)):
                for tag,key in (('old','oldentry_new'),('wide','wideentry_new')):
                    for field in ('conditional_association','rho','unknown_mass','unknown_prior_mass','out_of_pool_foreground_mass','unassigned_mass','ambiguous_mass'):
                        value=result[key][field];exact(value,a[f'{tag}_k{k}_{field}'],'prepared_'+tag+str(k)+field)
                        payload[f'{tag}_k{k}_{field}']=value
            np.savez_compressed(OUT/'samples_PRIVATE'/(sid+'.npz'),**payload)
            record={'id_PRIVATE':sid,'split':c['split'],'platform':r['platform'],
                'old_pool_count':len(oldpool),'wide_pool_count':len(wide),'old_and_source_bitexact':True,
                'old_visual_records_reused_exact':True,'all_required_new_visual_values_present':True,
                'all_visual_full_minus_gray_two_crop_numeric_exact':True,
                'actual_subject_query_sha256':c['actual_subject_query_sha256'],'image_sha256':v['image_sha256'],
                'mask_cache_sha256':v['mask_cache_sha256'],'visual_case_sha256':sha(vp),
                'new_anchor':[{ 'arm':ARMS[j], 'q0_mapped':state['source_q0_mapped'],
                    'fallback_original_maskmedian':state['anchor_fallback_original_mask_median'],
                    'anchor_v':state['anchor_v'],'anchor_PR':None if state['anchor_r'] is None else state['anchor_r'].tolist()}
                    for j,state in zip((3,4,5,6),newstates)],
                'maximum_rho_scaled_increment_abs':float(max(np.max(abs(state[key])) for state in newstates for key in ('vision','position','relation'))),
                'no_GT_model_or_optimizer':True,'no_parameter_or_box_change':True}
            dump(OUT/'cases_PRIVATE'/(sid+'.json'),record);records.append(record);counts['n']+=1;counts['task_readouts']+=7
            if (i+1)%16==0:print('PREDICT50',i+1,flush=True)
    except Exception as exc:
        dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':counts['n'],
            'STOP_no_prediction_seal_no_taskscore':True,'no_case_skip_no_imputation':True});raise
    assert counts['n']==336 and counts['task_readouts']==2352
    dump(OUT/'ROWS_PRIVATE.json',records);dump(OUT/'BINDINGS_PRIVATE.json',BINDINGS);dump(OUT/'COUNTS.json',dict(counts))
    dump(OUT/'SEALED.json',{'state':'R50_DEV336_SEVEN_UNIQUE_FIXED_BOX_READOUTS_BEFORE_TASK_SCORE',
        'n':336,'arms':ARMS,'files':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        'plan_sha256':sha(EXP/'PREDICT_PLAN_LOCK.json'),'seconds':time.perf_counter()-start,
        'no_GT_model_optimizer_or_selection':True,'no_parameter_or_box_change':True,
        'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print('PREDICT50_SEALED',dict(counts),flush=True)

if __name__=='__main__':main()
