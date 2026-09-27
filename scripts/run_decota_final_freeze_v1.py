"""F46: freeze once, evaluate the complete historical pool once, never retune.

Finite resumable job, not a scheduler. No labels are read by GPU workers.
Old full-scale/causal/corruption queues are never called.
"""
import argparse
import collections
import copy
import fcntl
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha

OUT=ROOT/'artifacts/decota_final_freeze_v1'
METHOD=ROOT/'methods/decota_final_v1'
SOURCE=ROOT/'artifacts/stvg_fullscale_diagnostics_v1'
F45=ROOT/'artifacts/decota_structured_shrinkage_v1'
PY=ROOT/'.conda/tubedetr/bin/python'
DIRECTIONS={'hcstvg1_test':'vid_to_hc1','vidstg_test':'hc2_to_vid'}
ARMS=['Frozen','A4','Temporal_final','Final_DeCoTA','Spatial_interpolation',
      'Direct_projection','Old_DeCoTA']
SUPPLEMENT=['Interpolation_native_time','F39_framewise','Full_update_eta1','Logit_damping_eta025']
OWN=['methods/decota_final_v1/__init__.py','methods/decota_final_v1/predictor.py',
     'methods/decota_final_v1/configs.json','methods/decota_final_v1/METHOD_CARD.md',
     'scripts/run_decota_final_freeze_v1.py','scripts/score_decota_final_freeze_v1.py',
     'scripts/verify_decota_final_v1.py','tests/test_decota_final_v1.py',
     'tests/test_decota_final_statistics_v1.py']


def verify(parents=False):
    p=read(OUT/'LOCK.json')
    for name,h in p['code_pins'].items():
        assert sha(ROOT/name)==h,('F46 locked code changed',name)
    assert sha(METHOD/'FINAL_METHOD.json')==p['method_manifest_sha256']
    assert sha(SOURCE/'lock.json')==p['source_lock_sha256']
    assert sha(ROOT/'methods/CURRENT_WORKING_METHOD.json')==p['working_registry_sha256']
    if parents:
        for name,h in p['protected_pins'].items():
            assert sha(ROOT/name)==h,('parent changed',name)
    return p


def prepare():
    if (OUT/'LOCK.json').exists():
        verify(True);print('F46 already frozen',flush=True);return
    from scripts.run_stvg_fullscale_v1 import plan as source_plan
    source=source_plan()
    api=read(OUT/'API_VERIFICATION.json');assert api['status']=='pass' and api['sources']==4
    f45=read(F45/'LOCK.json');seal=read(F45/'COMPLETION.json')
    protected={**f45['protected_pins'],**seal['files'],**seal['code_pins']}
    protected[str((F45/'COMPLETION.json').relative_to(ROOT))]=sha(F45/'COMPLETION.json')
    for name,h in protected.items():assert sha(ROOT/name)==h,name
    before=read(ROOT/'methods/CURRENT_WORKING_METHOD.json')
    prior_bytes=(ROOT/'methods/CURRENT_WORKING_METHOD.json').read_text()
    write(OUT/'REGISTRY_BEFORE.json',dict(working=before,working_exact_text=prior_bytes,
        working_sha256=sha(ROOT/'methods/CURRENT_WORKING_METHOD.json'),
        formal=read(ROOT/'methods/CURRENT_METHOD.json'),formal_sha256=sha(ROOT/'methods/CURRENT_METHOD.json')))
    # Only the user-selected research working registry is intentionally revised.
    protected.pop('methods/CURRENT_WORKING_METHOD.json',None)
    pins={name:h for name,h in protected.items()
          if name.endswith('.py') and name.startswith(('methods/','vg_tta/','external/TA-STVG/','scripts/'))}
    pins.update({name:sha(ROOT/name) for name in OWN})
    pins['methods/decota_spatial4_v1/WORKING_METHOD.json']=sha(ROOT/'methods/decota_spatial4_v1/WORKING_METHOD.json')
    pins['methods/decota_spatial4_v1/configs.json']=sha(ROOT/'methods/decota_spatial4_v1/configs.json')
    method=dict(version='decota_final_v1',status='final_design_locked_not_independently_confirmed',
        created=time.time(),authority='./private_authorization_notes/d449d50e-c980-4f8c-b7d2-6b9556b56e93/pasted-text.txt',
        definition='methods/decota_final_v1/METHOD_CARD.md',config=read(METHOD/'configs.json'),
        code_pins=pins,api_verification_sha256=sha(OUT/'API_VERIFICATION.json'),
        F45_is_all_development=True,formal_registry_changed=False,new_method_search=False)
    write(METHOD/'FINAL_METHOD.json',method)
    working=dict(version='decota_final_v1',role='user_selected_final_design_for_frozen_evaluation',
        status='design_locked_for_fixed_evaluation',selected_date='2026-09-15',
        user_authority=method['authority'],entrypoint='methods.decota_final_v1.FinalDeCoTAPredictor',
        definition='methods/decota_final_v1/METHOD_CARD.md',configuration='methods/decota_final_v1/configs.json',
        dependency_lock='methods/decota_final_v1/FINAL_METHOD.json',
        spatial='exact A4,1792 query/LN parameters,max10 steps,4 original DINO observations',
        temporal='F44 hard structured teacher,66306 actual head parameters,max5 steps,source-anchored eta=.25',
        F45_exposure='all113 sources are now development; no independent confirmation claim',
        evaluation='F46 full-pool retrospective evaluation after iterative development',
        evaluation_status_file='artifacts/decota_final_freeze_v1/STATUS.json',
        previous_working='decota_spatial4_v1; unchanged and retained as spatial-only control',
        historical_production_registry='methods/CURRENT_METHOD.json',historical_production_registry_changed=False,
        GT_online=False,algorithm_search=False,automatic_recurring_task=False)
    status(ROOT/'methods/CURRENT_WORKING_METHOD.json',working)
    rows=source['rows'];old_receipts={};exposure={}
    for c,rr in rows.items():
        barpath=SOURCE/'method/tastvg'/c/'barrier.json';bar=read(barpath)
        assert bar['all_available_complete'] and bar['queries']==len(rr)
        old_receipts[c]={r['key']:r for r in bar['receipts']}
        assert set(old_receipts[c])=={r['key'] for r in rr}
        exposure[c]=dict(nominal_queries=len(rr),sources=len({r['input']['source'] for r in rr}),
            previous_full_evaluation_barrier=str(barpath),barrier_sha256=sha(barpath),
            previous_method_inputs=len(bar['receipts']),input_unavailable=sum(r['input_unavailable'] is not None for r in rr),
            untouched_sources_claimed=0,independent_test=False,
            conclusion='All available queries were already run and analyzed in the previous full-pool study.')
    f45keys=[r['key'] for rr in f45['rows'].values() for r in rr]
    exposure['F45_reclassification']=dict(keys=f45keys,queries=len(f45keys),
        old_roles_preserved_in_F45=True,new_use='iterative development and explicit final eta=.25 design selection',
        original_dev_winner=1.,final_posthoc_design_choice=.25,
        later_panel_not_independent_confirmation=True,old_results_not_overwritten=True)
    write(OUT/'DATA_EXPOSURE.json',exposure)
    p=dict(name='F46_Final_Freeze_Evaluation',created=time.time(),run_state='planned',
        nature='full-pool retrospective evaluation after iterative development',independent_test=False,
        source_lock_sha256=sha(SOURCE/'lock.json'),rows=rows,counts=source['counts'],
        labels=source['labels'],labels_sha256=source['labels_sha256'],old_receipts=old_receipts,
        f45_development_keys=f45keys,code_pins=pins,protected_pins=protected,
        method_manifest_sha256=sha(METHOD/'FINAL_METHOD.json'),
        working_registry_sha256=sha(ROOT/'methods/CURRENT_WORKING_METHOD.json'),
        main_arms=ARMS,supplementary_arms=SUPPLEMENT,
        f39_fixed_config=dict(lr=.01,beta=1.,steps=5),
        statistics=dict(primary='source-macro paired corrected vIoU',also='query macro; fixed-GT-frame sIoU, physical tIoU',
            bootstrap=10000,seed=20260915,neutral_pp=.1,negative_tail_pp=[5,10],
            no_drop_of_noop_or_negative_cases=True,no_per_sample_GT_selection=True),
        fixed_cost_policy='share same original input/teacher/spatial and temporal fits across matched arms; retain actual component times; not separate standalone benchmarks',
        historical_method_precision='Old released method is reused with its original numerical pipeline; new main Frozen is matched FP32. Do not claim old/new difference is pure module ablation.',
        caps=dict(max_wall_seconds=48*3600,min_free_bytes=50*2**30,
            max_new_DINO=4*sum(len(x) for x in rows.values()),new_backbones=0,new_corruptions=0),
        method_changes_after_lock=False,GT_online=False,recurring_automation=False,
        missing_input_policy='preserve known published unusable input; verify unchanged bytes; any unexpected failure aborts without silent sample exclusion')
    write(OUT/'LOCK.json',p)
    write(OUT/'FREEZE_RECEIPT.json',dict(created=time.time(),method='decota_final_v1',
        method_manifest_sha256=p['method_manifest_sha256'],lock_sha256=sha(OUT/'LOCK.json'),
        working_registry_changed=True,formal_registry_changed=False,
        final_eta=.25,all_F45_development=True,full_pool_finished=False))
    status(OUT/'STATUS.json',dict(stage='frozen_ready_for_full_pool',finished=False,
        nominal_queries=sum(len(x) for x in rows.values()),new_algorithm_search=False))
    verify(True)
    print('F46 FROZEN',sum(len(x) for x in rows.values()),'nominal queries, no tuning',flush=True)


def supplemental(predictor,result,frames,ids,q,with_replay=True):
    """Matched pre-locked controls; no labels and no effect on final method."""
    import torch
    from methods.decota_refine_uniform_v1.api import reconstruct
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    from vg_tta.dense_support_tuning_v1 import HeadReplay,compact_fit
    from vg_tta.dense_support_temporal_v1 import fit
    preds=result['predictions'];tick=time.perf_counter()
    boxes,ia=reconstruct(result['native_boxes'],result['spatial']['anchors'],ids,mode='absolute')
    preds['Spatial_interpolation']=dict(boxes=boxes,indices=list(preds['Final_DeCoTA']['indices']),
        spatial_parameter_updates=False,same_accepted_A4_observations=True,temporal_origin='Final_DeCoTA')
    preds['Interpolation_native_time']=dict(boxes=boxes,indices=list(preds['Frozen']['indices']),
        spatial_parameter_updates=False,same_accepted_A4_observations=True,temporal_origin='Frozen')
    interpolation_s=time.perf_counter()-tick
    head=HeadReplay(result['temporal_cache']);tick=time.perf_counter()
    f39=fit(head,result['records'],ids,head.teacher,lr=.01,beta=1.,steps=5)
    f39=compact_fit(f39);f39['final']['boxes']=result['boxes']
    torch.cuda.synchronize();fit_s=time.perf_counter()-tick
    tick=time.perf_counter()
    if with_replay:
        live=full_prediction(predictor.model,frames,ids,q,result['parses']['subject'],
            {**result['spatial']['state'],**f39['state']},f39['final'])
        preds['F39_framewise']={**f39['final'],**live}
    else:
        preds['F39_framewise']=f39['final']
    torch.cuda.synchronize()
    result['supplementary']=dict(F39=f39,interpolation=ia,
        F39_fit_seconds=fit_s,F39_full_replay_seconds=time.perf_counter()-tick,
        interpolation_seconds=interpolation_s,GT_online=False)
    return result


def run(cohort,limit=0):
    import numpy as np
    import torch
    from methods.decota_final_v1 import FinalDeCoTAPredictor
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.foreground_runtime import state_digest
    p=verify();dest=OUT/'runs'/cohort;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'BARRIER.json').exists():return
    lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    predictor=FinalDeCoTAPredictor.from_pretrained(DIRECTIONS[cohort])
    digest=state_digest(predictor.model);verified_media={};pixels=collections.OrderedDict()
    receipts=[];done_new=0;locksha=sha(OUT/'LOCK.json');started=time.time()
    try:
        for row in p['rows'][cohort]:
            path=dest/f'{row["ordinal"]:06d}.pt';rp=path.with_suffix('.json');q=row['input']
            if rp.exists():
                r=read(rp);assert sha(path)==r['sha256'] and r['lock_sha256']==locksha
                receipts.append(r);continue
            if path.exists():raise RuntimeError('Unreceipted prediction preserved: '+str(path))
            if time.time()-started>p['caps']['max_wall_seconds']:raise TimeoutError('Finite worker time cap; receipts preserved')
            if shutil.disk_usage(ROOT).free<p['caps']['min_free_bytes']:raise RuntimeError('Free-disk guard; no files deleted')
            media=q['video_path']
            if media not in verified_media:
                assert sha(media)==q['video_sha256'],('input media changed',media)
                verified_media[media]=q['video_sha256']
            assert verified_media[media]==q['video_sha256']
            if row['input_unavailable'] is not None:
                result=dict(key=row['key'],status='known_input_unavailable',input=q,
                    reason=row['input_unavailable'],lock_sha256=locksha,GT_online=False)
            else:
                tick=time.perf_counter();pixelkey=(q['video_sha256'],tuple(q['frame_ids']),q['width'],q['height'])
                cached=pixelkey in pixels
                if cached:frames,ids=pixels.pop(pixelkey)
                else:frames,ids=decode(q)
                pixels[pixelkey]=(frames,ids)
                while len(pixels)>2:pixels.popitem(last=False)
                decode_s=time.perf_counter()-tick;torch.cuda.reset_peak_memory_stats()
                result=predictor.predict(frames,ids,q)
                result=supplemental(predictor,result,frames,ids,q)
                oldref=p['old_receipts'][cohort][row['key']]
                assert sha(oldref['path'])==oldref['sha256']
                old=load(oldref['path']);assert old['input']==q and old['frame_ids']==ids
                result['predictions']['Old_DeCoTA']=old['predictions']['mymethod']
                result['historical_reference']=dict(path=oldref['path'],sha256=oldref['sha256'],
                    seconds=old['seconds'],original_numerical_pipeline=True)
                assert set(result['predictions'])==set(ARMS+SUPPLEMENT)
                result.update(key=row['key'],input=q,cohort=cohort,group=q['source'],
                    query_type=row['query_type'],lock_sha256=locksha,decode_seconds=decode_s,
                    decoded_pixel_cache_hit=cached,seconds=time.perf_counter()-tick,
                    peak_memory_bytes=torch.cuda.max_memory_allocated(),historically_exposed=True)
                # Small source head inputs/weights are retained; no video or visual-feature cache is written.
                assert state_digest(predictor.model)==digest
            save(path,result)
            r=dict(key=row['key'],path=str(path),sha256=sha(path),source=q['source'],
                status=result.get('status','predicted'),lock_sha256=locksha,completed=time.time(),
                seconds=result.get('seconds',0.),method_seconds=result.get('cost',{}).get('pipeline_seconds'),
                DINO=result.get('expert',{}).get('new_DINO',0),GT_online=False)
            write(rp,r);receipts.append(r);done_new+=1
            status(dest/'PROGRESS.json',dict(done=len(receipts),total=len(p['rows'][cohort]),last=row['key'],
                last_seconds=r['seconds'],updated=time.time(),finished=False))
            print('F46',cohort,len(receipts),len(p['rows'][cohort]),row['key'],round(r['seconds'],2),flush=True)
            del result;gc.collect();torch.cuda.empty_cache()
            if limit and done_new>=limit:break
        assert state_digest(predictor.model)==digest
        verify()
        if len(receipts)==len(p['rows'][cohort]):
            write(dest/'BARRIER.json',dict(receipts=receipts,nominal_queries=len(receipts),
                available_queries=sum(r['status']=='predicted' for r in receipts),
                unavailable_queries=sum(r['status']!='predicted' for r in receipts),
                all_available_complete=True,source_restored=True,GT_online=False,lock_sha256=locksha,
                finished=time.time(),configuration_changes=0))
    except BaseException as e:
        status(dest/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),time=time.time(),
            current_key=row['key'] if 'row' in locals() else None,receipts_preserved=True,
            no_silent_exclusion=True,no_algorithm_change=True))
        raise
    finally:
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


def batch():
    p=verify(True);lease=open(OUT/'queue.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();deadline=start+p['caps']['max_wall_seconds']
    for cohort in DIRECTIONS:
        for stage in ['run','score']:
            if stage=='run' and (OUT/'runs'/cohort/'BARRIER.json').exists():continue
            if stage=='score' and (OUT/'analysis'/cohort/'COMPLETION.json').exists():continue
            if stage=='run':
                cmd=['bash','scripts/with_local_cuda.sh',str(PY),'-B','-u',__file__,'run','--cohort',cohort]
            else:cmd=[str(PY),'-B','scripts/score_decota_final_freeze_v1.py','--cohort',cohort]
            logpath=OUT/'logs'/f'{cohort}_{stage}.log';logpath.parent.mkdir(parents=True,exist_ok=True)
            with logpath.open('ab',buffering=0) as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                try:
                    while child.poll() is None:
                        progress=OUT/'runs'/cohort/'PROGRESS.json'
                        status(OUT/'STATUS.json',dict(stage=stage,cohort=cohort,child_pid=child.pid,
                            progress=read(progress) if progress.exists() else {},updated=time.time(),
                            finished=False,log=str(logpath),finite_not_recurring=True))
                        if time.time()>deadline:raise TimeoutError('48-hour finite evaluation cap reached')
                        if shutil.disk_usage(ROOT).free<p['caps']['min_free_bytes']:raise RuntimeError('50GiB disk guard')
                        time.sleep(15)
                    if child.returncode:raise RuntimeError(f'F46 child failed: {logpath}')
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
    verify(True)
    subprocess.run([str(PY),'-B','scripts/score_decota_final_freeze_v1.py','--finalize'],cwd=ROOT,check=True)


def launch():
    verify(True)
    existing=OUT/'PROCESS.json'
    if existing.exists():
        old=read(existing);proc=Path('/proc')/str(old['pid'])/'cmdline'
        if proc.exists() and str(Path(__file__).resolve()).encode() in proc.read_bytes() and b'batch' in proc.read_bytes():
            print('F46_ALREADY_RUNNING',old['pid']);return
    cmd=[str(PY),'-B','-u',str(Path(__file__).resolve()),'batch']
    with (OUT/'queue.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    status(existing,dict(pid=child.pid,command=cmd,started=time.time(),finite_not_recurring=True,
        log=str(OUT/'queue.log'),max_wall_seconds=48*3600,auto_retune=False))
    print('F46_STARTED_FINITE_EVALUATION',child.pid,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','verify','run','batch','launch'])
    ap.add_argument('--cohort',choices=list(DIRECTIONS));ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    if args.stage=='run':run(args.cohort,args.limit)
    elif args.stage=='verify':verify(True);print('F46 integrity pass')
    else:
        try:globals()[args.stage]()
        except BaseException as e:
            status(OUT/'STATUS.json',dict(stage='failed',error=str(e),traceback=traceback.format_exc(),
                updated=time.time(),finished=False,receipts_preserved=True,auto_retune=False))
            raise
