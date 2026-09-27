"""Finite label-free inference for the final subtractive round.

Only the offline companion may read GT or choose a cross-episode configuration.
All saved episodes are atomic and hash receipted. Parents are immutable.
"""
import argparse
import collections
import copy
import fcntl
import gc
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha

OUT=ROOT/'artifacts/decota_final_simplification_v1'
PARENT=ROOT/'artifacts/decota_final_freeze_v1'
F44=ROOT/'artifacts/decota_structured_calibration_v1'
DIRECTIONS={'hcstvg1_test':'vid_to_hc1','vidstg_test':'hc2_to_vid'}
OWN=['vg_tta/final_simplification_v1.py','scripts/run_final_simplification_v1.py',
     'scripts/analyze_final_simplification_v1.py','tests/test_final_simplification_v1.py']


def prepare():
    from scripts.run_decota_final_freeze_v1 import verify as parent_verify
    parent=parent_verify(); rows=read(F44/'LOCK.json')['rows']
    if (OUT/'LOCK.json').exists():
        verify(); return
    assert read(PARENT/'STATUS.json')['stage']=='paused_by_user'
    pause=PARENT/'checkpoints/user_pause_20260915_v1/CHECKPOINT.json'
    cp=read(pause); assert sha(cp['checkpoint_path'])==cp['checkpoint_sha256']
    for cohort,rr in rows.items():
        assert sum(r['f44_role']=='development' for r in rr)==32
        lookup={r['key']:r for r in parent['rows'][cohort]}
        for r in rr:
            assert r['input']==lookup[r['key']]['input']
            r['cohort']=cohort
            r['ordinal']=lookup[r['key']]['ordinal']
            if 'cache_reference' not in r:
                r['cache_reference']=str(ROOT/'artifacts/decota_dense_support_generalization_v1/cache'/
                    (r['key'].replace(':','_')+'.pt'))
            for field in ['cache_reference']:
                r[field+'_sha256']=sha(r[field])
            assert sha(r['A4_reference']['path'])==r['A4_reference']['sha256']
    p=dict(name='Final_Simplification_Round',created=time.time(),rows=rows,
        full_rows=parent['rows'],labels=parent['labels'],labels_sha256=parent['labels_sha256'],
        code_pins={name:sha(ROOT/name) for name in OWN},parent_code_pins=parent['code_pins'],
        parent_lock_sha256=sha(PARENT/'LOCK.json'),checkpoint_manifest_sha256=sha(pause),
        protocol_sha256=sha(OUT/'PROTOCOL.md'),registries=cp['registry_pins'],
        historical_exposure=True,independent_test=False,GT_online=False,
        source_configs=read(ROOT/'methods/decota_final_v1/configs.json'),
        directions=DIRECTIONS,full_caps=dict(seconds=48*3600,min_free_bytes=50*2**30),
        development_queries=64,development_sources=64,transfer_queries=49,
        temporal_eta=.25,spatial_steps=10,temporal_steps=5,no_recurring_task=True)
    write(OUT/'LOCK.json',p)
    status(OUT/'STATUS.json',dict(stage='ready_for_spatial_subtraction',complete=False,updated=time.time()))
    print('PREPARED Final Simplification Round:64 development,49 descriptive transfer,full locked pools',flush=True)


def verify():
    p=read(OUT/'LOCK.json')
    for name,h in {**p['parent_code_pins'],**p['code_pins'],**p['registries']}.items():
        assert sha(ROOT/name)==h,('locked dependency changed',name)
    assert sha(PARENT/'LOCK.json')==p['parent_lock_sha256']
    assert sha(OUT/'PROTOCOL.md')==p['protocol_sha256']
    return p


def episode_path(stage,row):
    return OUT/stage/row['cohort']/f'{row["ordinal"]:06d}.pt'


def existing(path):
    rp=path.with_suffix('.json')
    if rp.exists():
        r=read(rp); assert sha(path)==r['sha256'] and r['lock_sha256']==sha(OUT/'LOCK.json')
        return r
    if path.exists():
        raise RuntimeError('Unreceipted output preserved: '+str(path))
    return None


def commit(path,x):
    x['lock_sha256']=sha(OUT/'LOCK.json')
    save(path,x)
    r=dict(key=x['key'],path=str(path),sha256=sha(path),lock_sha256=x['lock_sha256'],
        completed=time.time(),status=x.get('status','available'),source=x['input']['source'])
    write(path.with_suffix('.json'),r)
    return r


def lease():
    f=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a')
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return f


def dev_rows(p,cohort):
    return [r for r in p['rows'][cohort] if r['f44_role']=='development']


def checked_cache(row):
    assert sha(row['cache_reference'])==row['cache_reference_sha256']
    x=load(row['cache_reference']); assert x['key']==row['key'] and x['frame_ids']==row['input']['frame_ids']
    return x


def capture(predictor,frames,ids,q,subject):
    import torch
    from vg_tta.decota_tastvg_episode_v1 import make_batch
    from vg_tta.shared_state_v1 import capture_shared
    from vg_tta.parametric_observation_v1 import ObservationReplay
    from vg_tta.structured_temporal_shrinkage_v1 import prediction
    _,_,records,_,_,views=capture_shared(predictor.model,make_batch(frames,ids,q,subject,predictor.model))
    spatial=ObservationReplay(predictor.model,views,len(ids),'spatial')
    with torch.no_grad():zero=spatial.values()
    native=prediction(zero['logits'],zero['boxes'],records,ids)
    return records,views,spatial,native


def run_space(cohort,limit=0):
    import torch
    from methods.decota_final_v1 import FinalDeCoTAPredictor
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.final_simplification_v1 import spatial_fit
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    p=verify(); gpu=lease(); predictor=FinalDeCoTAPredictor.from_pretrained(DIRECTIONS[cohort])
    digest=state_digest(predictor.model); done=0
    for row in dev_rows(p,cohort):
        dest=episode_path('space',row)
        if existing(dest):continue
        tick=time.perf_counter(); q=row['input']; assert sha(q['video_path'])==q['video_sha256']
        frames,ids=decode(q); torch.cuda.reset_peak_memory_stats()
        ref=row['A4_reference']; assert sha(ref['path'])==ref['sha256']; old=load(ref['path'])
        records,views,it,native=capture(predictor,frames,ids,q,row['subject'])
        assert torch.equal(native['boxes'],old['native_boxes'])
        assert all(torch.equal(a,b) for a,b in zip(native['logits'],old['native_logits']))
        anchors=copy.deepcopy(old['expert']['anchors']['single4'])
        fits={}; reinsertions={}; order=['A','B','C'] if done%2==0 else ['C','B','A']
        for name in order:
            fit=spatial_fit(it,records,ids,anchors,lr=predictor.config['lr']*(.5 if name=='C' else 1.),
                backtracking=name=='A',selection='best')
            if name=='A':
                prior=old['fits'][ref['arm']]
                assert all(torch.equal(v,prior['state'][n]) for n,v in fit['state'].items())
                assert [s['loss'] for s in fit['path']]==[s['loss'] for s in prior['path']]
            replay=full_prediction(predictor.model,frames,ids,q,row['subject'],fit['state'],fit['final'])
            fits[name]=fit; reinsertions[name]=replay['audit']
        assert state_digest(predictor.model)==digest
        x=dict(key=row['key'],input=q,cohort=cohort,group=row['group'],subject=row['subject'],
            frame_ids=ids,records=records,native=native,fits=fits,order=order,
            actual_observation_positions=old['expert']['observed4'],anchors=anchors,
            observations_reference=ref,source_restored=True,GT_online=False,
            full_reinsertions=reinsertions,new_DINO=0,historical_DINO=len(old['expert']['observed4']),
            seconds=time.perf_counter()-tick,peak_memory_bytes=torch.cuda.max_memory_allocated())
        commit(dest,x);done+=1
        status(OUT/'STATUS.json',dict(stage='S1',cohort=cohort,key=row['key'],new=done,complete=False,updated=time.time()))
        print('S1',cohort,row['key'],{n:round(fits[n]['measured_fit_seconds'],3) for n in fits},flush=True)
        del x,fits,it,views,old,frames;gc.collect();torch.cuda.empty_cache()
        if limit and done>=limit:break
    gpu.close()


def spatial_selected(row):
    sel=read(OUT/'SPATIAL_SELECTION.json')
    x=load(episode_path('space',row));fit=x['fits'][sel['base_arm']]
    if sel['selection']=='last':
        from vg_tta.final_simplification_v1 import use_last
        fit=use_last(fit)
    return x,fit


def time_spec(stage):
    if stage=='AB':return dict(backtracking=True,selection='best',beta=.1,gamma=1.)
    if stage=='C':return dict(backtracking=False,selection='last',beta=.1,gamma=1.)
    if stage=='D':
        d=read(OUT/'TEMPORAL_OPTIMIZER_SELECTION.json')['config']
        return {**d,'beta':0.}
    if stage=='E':
        d=read(OUT/'TEMPORAL_KL_SELECTION.json')['config']
        return {**d,'gamma':0.}
    raise ValueError(stage)


def run_time(cohort,stage,limit=0):
    import torch
    from vg_tta.dense_support_tuning_v1 import HeadReplay,move
    from vg_tta.final_simplification_v1 import temporal_fit,use_last
    from vg_tta.structured_temporal_shrinkage_v1 import path_point
    from methods.decota_final_v1.predictor import temporal_config
    p=verify();gpu=lease();done=0
    from scripts.run_decota_refine_v1 import configure
    configure()
    spec=time_spec(stage)
    if stage=='C': assert read(OUT/'TEMPORAL_B_DECISION.json')['try_C']
    for row in dev_rows(p,cohort):
        dest=episode_path('time_'+stage,row)
        if existing(dest):continue
        cache=checked_cache(row);sp,space=spatial_selected(row)
        cache=copy.deepcopy(cache);cache['zero']['boxes']=space['final']['boxes']
        it=HeadReplay(cache);cfg=temporal_config(DIRECTIONS[cohort]);fits={};points={}
        if stage=='C':
            fit=use_last(load(episode_path('time_AB',row))['fits']['B'])
            fits['C']=fit;points['C']=path_point(it,fit,cache['records'],cache['frame_ids'],.25)
        else:
            for name in (['A','B'] if stage=='AB' else [stage]):
                s={**spec,'backtracking':name=='A'} if stage=='AB' else spec
                conf={**cfg,'beta':s['beta'],'gamma':s['gamma']}
                fit=temporal_fit(it,cache['records'],cache['frame_ids'],it.teacher,conf,
                    lr=cfg['lr'],steps=5,backtracking=s['backtracking'],selection=s['selection'])
                fits[name]=fit;points[name]=path_point(it,fit,cache['records'],cache['frame_ids'],.25)
        x=dict(key=row['key'],input=row['input'],cohort=cohort,group=row['group'],
            frame_ids=cache['frame_ids'],records=cache['records'],fits=fits,points=points,
            spatial_state_sha256=__import__('vg_tta.dense_support_tuning_v1',fromlist=['state_hash']).state_hash(space['state']),
            native={**sp['native'],'boxes':space['final']['boxes']},
            source_restored=True,GT_online=False,new_DINO=0,new_backwards=sum(f['backwards'] for f in fits.values()) if stage!='C' else 0)
        commit(dest,x);done+=1
        status(OUT/'STATUS.json',dict(stage='T1_'+stage,cohort=cohort,key=row['key'],new=done,complete=False,updated=time.time()))
        print('T1',stage,row['key'],{n:p['parameter']['indices'] for n,p in points.items()},flush=True)
        del it,x,fits,points,cache;gc.collect();torch.cuda.empty_cache()
        if limit and done>=limit:break
    gpu.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','space','time'])
    ap.add_argument('--cohort',choices=list(DIRECTIONS));ap.add_argument('--stage',default='AB',choices=['AB','C','D','E'])
    ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='space':run_space(a.cohort,a.limit)
    else:run_time(a.cohort,a.stage,a.limit)


if __name__=='__main__':main()
