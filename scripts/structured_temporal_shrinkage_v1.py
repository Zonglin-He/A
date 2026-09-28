"""F45 finite two-phase execution. Selection reads development only.

Phase A has no backward. Every parameter path prediction is reinserted into
the actual full model. Phase B is inaccessible before an explicit A decision.
"""
import argparse
import collections
import copy
import fcntl
import gc
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
from scripts.dense_support_tuning_v1 import keyfile,METRICS
from scripts.dense_support_generalization_v1 import cache_for,a4_for

OUT=ROOT/'artifacts/decota_structured_shrinkage_v1'
PARENT=ROOT/'artifacts/decota_structured_calibration_v1'
OWN=['vg_tta/structured_temporal_shrinkage_v1.py','scripts/structured_temporal_shrinkage_v1.py',
     'tests/test_structured_temporal_shrinkage_v1.py']


def record(path,value):
    save(path,value)
    write(path.with_suffix('.json'),dict(key=value.get('key'),sha256=sha(path),
        completed=time.time(),code_pins={f:sha(ROOT/f) for f in OWN}))


def recorded(path):
    if not path.exists():return False
    assert sha(path)==read(path.with_suffix('.json'))['sha256'],path
    return True


def prepare():
    from vg_tta.structured_temporal_shrinkage_v1 import ETAS
    p=read(PARENT/'LOCK.json');seal=read(PARENT/'COMPLETION.json')
    protected={**p['protected_pins'],**seal['files'],**seal['code_pins'],
        str((PARENT/'COMPLETION.json').relative_to(ROOT)):sha(PARENT/'COMPLETION.json')}
    for f,h in protected.items():assert sha(ROOT/f)==h,f
    assert read(PARENT/'AUDIT.json')['status']=='pass'
    rows=copy.deepcopy(p['rows'])
    for rr in rows.values():
        for r in rr:r['f45_role']=r['f44_role']
    cfg=read(PARENT/'SELECTION.json')['choices']
    lock=dict(name='F45_R54_Last_Temporal_Shrinkage_then_Soft_Structured_Teacher',created=time.time(),
        user_authority='Current user explicitly reopens F44 only for this bounded two-phase final iteration',
        rows=rows,parent_results=str(PARENT/'ALL_SOURCE_RESULTS.json'),F44_selection=cfg,
        fixed=p['fixed'],numerical=p['numerical'],spatial=p['spatial'],etas=list(ETAS),
        teacher_tau=1.,mode_margin=.2,gamma=1.,beta=.1,steps=5,
        selection=dict(unit='one global eta across BOTH directions, not per dataset or query',
            role='64 development sources only; 32 per direction',
            criterion='maximize two-direction equal-source mean parameter-path delta_v',
            tie='within1e-12 prefer eta1 (parent), then larger eta',
            logit='same eta for matched control; all five curves reported, no second selection',
            reasonable_interior='only selected global eta in (0,1) with positive development macro delta_v',
            phase_b='soft and soft+locked eta (only when reasonable interior); choose best development macro, tie prefer unshrunk soft',
            validation='49 historically exposed sources; no evaluation reselection; all prespecified A curve values retained'),
        acceptance=dict(practical_macro_delta_v=.001,positive_macro_delta_t=True,
            tail='fewer >5pp-loss sources than F44 for mixed-domain tradeoff; not more for two-positive-domain case',
            two_positive_domain='both domain delta_v > .001, not a CI requirement',
            good_cases='HC928 and Vid9637 remain >.001 above native; all other good cases and negative tails also reported',
            controls='equal-direction mean correct minus wrong_query and time_shift >0',
            minimum_eta='eta0 cannot count as successful parameter TTA',
            scope='practical bounded resource decision, not a confirmatory significance test'),
        caps=dict(new_fits=400,new_backwards=2000,full_model_predictions=1500,wall_seconds=14400,
                  new_DINO=0,new_spatial_fits=0),
        closed=['A4 edits','alpha search','lambda search','LR search','steps search','temperature search',
            'reliability','trust guard','decision-aware selection','ASA','crop','warp','coverage','boundary',
            'shared TTA','new expert','new backbone','query contrastive teacher','EMA','memory','automatic next round'],
        historical_exposure=True,independent_test=False,GT_online=False,automation=False,
        registry_changes=False,protected_pins=protected,code_pins={f:sha(ROOT/f) for f in OWN})
    write(OUT/'LOCK.json',lock)
    write(OUT/'PARENT_AUDIT.json',dict(status='pass',protected_files=len(protected)))
    status(OUT/'STATUS.json',dict(stage='prepared',finished=False))
    print('F45 prepared',len(protected),'protected files',flush=True)


def plan():
    p=read(OUT/'LOCK.json')
    for f,h in p['code_pins'].items():assert sha(ROOT/f)==h,f
    return p


def f44_rows():return {r['key']:r for r in read(PARENT/'ALL_SOURCE_RESULTS.json')['rows']}
def eta_id(eta):return str(eta)
def jsonfile(key):return keyfile(key).replace('.pt','.json')


def full_point(model,frames,r,cache,a4,point):
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    st={**a4['state'],**point['state']}
    return full_prediction(model,frames,cache['frame_ids'],r['input'],r['subject'],
                           st,point['parameter'])['audit']


def run_a(role,controls=False):
    import torch
    from scripts.run_closure_v1 import model_for
    from scripts.run_spatial10_components_v1 import capture_timed
    from vg_tta.dense_support_tuning_v1 import HeadReplay
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.structured_temporal_shrinkage_v1 import path_point
    from scripts.analyze_spatial10_components_v1 import labels_for,checked_score
    p=plan();old=f44_rows();selected=read(OUT/'A_SELECTION.json') if controls or role!='development' else None
    stage='A_controls' if controls else 'A_'+role;total=113 if controls else (64 if role=='development' else 49)
    done=0;started=time.time()
    for cohort,rr in p['rows'].items():
        rr=[r for r in rr if controls or r['f45_role']==role]
        gt=labels_for(rr);model=model_for(cohort);digest=state_digest(model)
        for r in rr:
            dest=OUT/stage/keyfile(r['key'])
            if recorded(dest):done+=1;continue
            cache=cache_for(r);_,a4=a4_for(r);head=HeadReplay(cache)
            frames,_,_,views,qa,cost=capture_timed(model,r);assert qa==cache['qa']
            kinds=['wrong_query','time_shift'] if controls else ['P1']
            etas=[selected['eta']] if controls else p['etas']
            points={};arms={};intervals={};tick=time.perf_counter()
            for kind in kinds:
                ref=old[r['key']]['fit_references'][kind]['path']
                assert sha(ref)==read(Path(ref).with_suffix('.json'))['sha256']
                tr=load(ref);points[kind]={}
                for eta in etas:
                    item=path_point(head,tr,cache['records'],cache['frame_ids'],eta)
                    item['full_model_audit']=full_point(model,frames,r,cache,a4,item)
                    points[kind][eta_id(eta)]=item
                    for method in ['parameter','logit']:
                        arm=f'{kind}_{method}_{eta_id(eta)}';out=item[method]
                        arms[arm]=checked_score(out['boxes'],gt[r['key']],cache['frame_ids'],out['indices'])[0]
                        intervals[arm]=out['physical_interval']
                assert state_digest(model)==digest
            x=dict(key=r['key'],cohort=cohort,source=r['source'],group=r['group'],role=r['f45_role'],
                points=points,arms=arms,intervals=intervals,qa=qa,capture_cost=cost,
                replay_seconds=time.perf_counter()-tick,new_backwards=0,
                full_model_predictions=len(kinds)*len(etas),source_restored=True,spatial_invariant=True,GT_online=False)
            record(dest,x);done+=1
            write(OUT/stage/jsonfile(r['key']).replace('.json','.scores.json'),{k:v for k,v in x.items() if k!='points'})
            print('F45',stage,done,total,r['key'],'elapsed',round(time.time()-started,1),flush=True)
            status(OUT/'STATUS.json',dict(stage=stage,completed=done,total=total,finished=False))
            del frames,views,head,x,points,cache;gc.collect()
        del model;gc.collect();torch.cuda.empty_cache()
    write(OUT/(stage+'_COMPLETED.json'),dict(sources=done,seconds=time.time()-started,new_backwards=0))


def score_rows(stage):return [read(f) for f in sorted((OUT/stage).glob('*.scores.json'))]


def mean_delta(rows,arm,base='T0'):
    old=f44_rows();return sum(r['arms'][arm]['vIoU_corrected']-old[r['key']]['arms'][base]['vIoU_corrected'] for r in rows)/len(rows)


def select_a():
    p=plan();assert (OUT/'A_development_COMPLETED.json').exists()
    rows=score_rows('A_development');assert len(rows)==64
    cells=[]
    for eta in p['etas']:
        d={c:mean_delta([r for r in rows if r['cohort']==c],f'P1_parameter_{eta_id(eta)}') for c in p['rows']}
        cells.append(dict(eta=eta,per_direction=d,macro_delta_v=sum(d.values())/2))
    best=max(x['macro_delta_v'] for x in cells)
    choice=min([x for x in cells if best-x['macro_delta_v']<=1e-12],key=lambda x:(x['eta']!=1.,-x['eta']))
    write(OUT/'A_SELECTION.json',dict(**choice,menu=cells,created=time.time(),development_sources=64,
        reasonable_interior=0<choice['eta']<1 and choice['macro_delta_v']>0,
        independent_test=False,no_evaluation_reselection=True))
    print('F45 globally locked eta',choice,flush=True)


def acceptance(rows,arm,wrong,shift,eta=1.):
    p=plan();cohorts=list(p['rows']);old=f44_rows();details={}
    for c in cohorts:
        rr=[r for r in rows if r['cohort']==c]
        ds=[r['arms'][arm]['vIoU_corrected']-old[r['key']]['arms']['T0']['vIoU_corrected'] for r in rr]
        dt=[r['arms'][arm]['tIoU']-old[r['key']]['arms']['T0']['tIoU'] for r in rr]
        od=[old[r['key']]['arms']['P1']['vIoU_corrected']-old[r['key']]['arms']['T0']['vIoU_corrected'] for r in rr]
        details[c]=dict(delta_v=sum(ds)/len(ds),delta_t=sum(dt)/len(dt),
            negative_gt5pp=sum(x<-.05 for x in ds),F44_negative_gt5pp=sum(x<-.05 for x in od),
            minus_wrong=sum(r['arms'][arm]['vIoU_corrected']-r['arms'][wrong]['vIoU_corrected'] for r in rr)/len(rr),
            minus_shift=sum(r['arms'][arm]['vIoU_corrected']-r['arms'][shift]['vIoU_corrected'] for r in rr)/len(rr))
    macro={k:sum(x[k] for x in details.values())/2 for k in ['delta_v','delta_t','minus_wrong','minus_shift']}
    tail=sum(x['negative_gt5pp'] for x in details.values());oldtail=sum(x['F44_negative_gt5pp'] for x in details.values())
    bykey={r['key']:r for r in rows};good={}
    for key in ['hcstvg1_test:000928','vidstg_test:009637']:
        assert key in bykey,key
        good[key]=bykey[key]['arms'][arm]['vIoU_corrected']-old[key]['arms']['T0']['vIoU_corrected']
    both=all(x['delta_v']>.001 for x in details.values())
    conditions=dict(nonzero_eta=eta>0,macro_delta_v=macro['delta_v']>.001,
        macro_delta_t=macro['delta_t']>0,tail=(tail<=oldtail if both else tail<oldtail),
        good_cases=all(x>.001 for x in good.values()),wrong=macro['minus_wrong']>0,shift=macro['minus_shift']>0)
    return dict(accepted=all(conditions.values()),conditions=conditions,per_direction=details,macro=macro,
        good_cases=good,negative_gt5pp=tail,F44_negative_gt5pp=oldtail,both_directions_positive=both,
        CI_not_used_as_gate=True,independent_test=False)


def decision_a():
    p=plan();sel=read(OUT/'A_SELECTION.json');eta=eta_id(sel['eta'])
    rows=score_rows('A_second_validation');cc={r['key']:r for r in score_rows('A_controls')}
    for r in rows:r['arms'].update(cc[r['key']]['arms'])
    d=acceptance(rows,'P1_parameter_'+eta,'wrong_query_parameter_'+eta,'time_shift_parameter_'+eta,sel['eta'])
    write(OUT/'A_DECISION.json',dict(**d,created=time.time(),eta=sel['eta'],
        run_phase_b=not d['accepted'],selection_sha256=sha(OUT/'A_SELECTION.json'),
        role='historically exposed second validation; decision to stop or execute the sole preauthorized B',
        not_unique_cause='shrinkage gain alone cannot prove supervision correct or all failure due to overshoot'))
    print('F45 A decision',d,flush=True)


def run_b(role):
    import torch
    from scripts.run_closure_v1 import model_for
    from scripts.run_spatial10_components_v1 import capture_timed
    from scripts.analyze_spatial10_components_v1 import labels_for,checked_score
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.dense_support_tuning_v1 import HeadReplay,move
    from vg_tta.structured_temporal_shrinkage_v1 import fit_soft,path_point
    p=plan();assert read(OUT/'A_DECISION.json')['run_phase_b']
    sel=read(OUT/'A_SELECTION.json');etas=[1.]+([sel['eta']] if sel['reasonable_interior'] else [])
    if role!='development':assert (OUT/'B_SELECTION.json').exists()
    total=64 if role=='development' else 49;done=0;started=time.time()
    for c,rr in p['rows'].items():
        rr=[r for r in rr if r['f45_role']==role];labels=labels_for(rr)
        model=model_for(c);digest=state_digest(model);cfg=p['F44_selection'][c]['config']
        for r in rr:
            dest=OUT/('B_'+role)/keyfile(r['key'])
            if recorded(dest):done+=1;continue
            cache=cache_for(r);head=HeadReplay(cache);_,a4=a4_for(r)
            frames,_,_,views,qa,cost=capture_timed(model,r);assert qa==cache['qa']
            points={};refs={};arms={};intervals={};backwards=0
            for kind in ['correct','wrong_query','time_shift']:
                teacher=head.teacher if kind=='correct' else move(cache['wrong_teacher' if kind=='wrong_query' else 'rolled_teacher'],'cuda')
                path=OUT/'soft_fits'/role/kind/keyfile(r['key'])
                if recorded(path):tr=load(path)
                else:
                    tr=fit_soft(head,cache['records'],cache['frame_ids'],teacher,cfg,**p['numerical'][c])
                    tr.update(key=r['key'],kind=kind,role=role);record(path,tr)
                refs[kind]=str(path);points[kind]={};backwards+=tr['backwards']
                for eta in etas:
                    point=path_point(head,tr,cache['records'],cache['frame_ids'],eta)
                    point['full_model_audit']=full_point(model,frames,r,cache,a4,point)
                    points[kind][eta_id(eta)]=point
                    for method in ['parameter','logit']:
                        arm=f'{kind}_{method}_{eta_id(eta)}';v=point[method]
                        arms[arm]=checked_score(v['boxes'],labels[r['key']],cache['frame_ids'],v['indices'])[0]
                        intervals[arm]=v['physical_interval']
            assert state_digest(model)==digest
            x=dict(key=r['key'],cohort=c,source=r['source'],group=r['group'],role=role,
                arms=arms,intervals=intervals,points=points,fit_references=refs,qa=qa,capture_cost=cost,
                new_backwards=backwards,full_model_predictions=3*len(etas),source_restored=True,
                spatial_invariant=True,GT_online=False)
            record(dest,x);write(OUT/('B_'+role)/jsonfile(r['key']).replace('.json','.scores.json'),
                                {k:v for k,v in x.items() if k!='points'})
            done+=1;print('F45 B',role,done,total,r['key'],'elapsed',round(time.time()-started,1),flush=True)
            status(OUT/'STATUS.json',dict(stage='B_'+role,completed=done,total=total,finished=False))
            del frames,views,head,x,points,cache;gc.collect()
        del model;gc.collect();torch.cuda.empty_cache()
    write(OUT/('B_'+role+'_COMPLETED.json'),dict(sources=done,seconds=time.time()-started))


def select_b():
    p=plan();assert read(OUT/'A_DECISION.json')['run_phase_b']
    assert (OUT/'B_development_COMPLETED.json').exists()
    rows=score_rows('B_development');sel=read(OUT/'A_SELECTION.json')
    etas=[1.]+([sel['eta']] if sel['reasonable_interior'] else []);cells=[]
    for eta in etas:
        ds={c:mean_delta([r for r in rows if r['cohort']==c],'correct_parameter_'+eta_id(eta)) for c in p['rows']}
        cells.append(dict(eta=eta,per_direction=ds,macro_delta_v=sum(ds.values())/2))
    best=max(x['macro_delta_v'] for x in cells);choice=min([x for x in cells if best-x['macro_delta_v']<=1e-12],key=lambda x:x['eta']!=1.)
    write(OUT/'B_SELECTION.json',dict(**choice,menu=cells,created=time.time(),tau=1.,
        role='development only; no evaluation reselection',independent_test=False))
    print('F45 B locked',choice,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','a-dev','a-select','a-validation','a-controls','a-decision',
        'b-dev','b-select','b-validation']);a=ap.parse_args()
    cpu={'prepare':prepare,'a-select':select_a,'a-decision':decision_a,'b-select':select_b}
    if a.stage in cpu:cpu[a.stage]();return
    from scripts.run_decota_refine_v1 import configure
    configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if a.stage=='a-dev':run_a('development')
        elif a.stage=='a-validation':run_a('second_validation')
        elif a.stage=='a-controls':run_a(None,True)
        elif a.stage=='b-dev':run_b('development')
        elif a.stage=='b-validation':run_b('second_validation')
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':main()
