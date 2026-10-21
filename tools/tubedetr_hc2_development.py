"""Bounded, isolated HC128 numerical development after the original study.

The controller never performs model inference. Worker subprocesses cannot read
labels. Old predictions/configuration are immutable; selected full evaluation
has its own output directory and reuses only demonstrably identical caches.
"""
import argparse
from collections import defaultdict
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import digest, implementation_hash, population_hash, save_prediction, write

OUT=ROOT/'outputs/tubedetr_hc2_development'
PUBLIC=ROOT/'results/tubedetr_hc2_development'
OLD=ROOT/'outputs/tubedetr/vid_to_hc2'
PLAN=PUBLIC/'plan.json'
GAMMAS=[0.,.0125,.025,.05,.1,.25,.5,.75,1.]
BASE=dict(initialization_gamma=.5,lr=.014,tau=.14,steps=40)


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def parameters(gamma=.5,lr=.014,tau=.14,steps=40):
    return dict(initialization_gamma=float(gamma),lr=float(lr),tau=float(tau),steps=int(steps))


def pid(params):
    return 'p_'+digest(params)[:12]


def pick(values, baseline=BASE):
    # One global profile, never a query-specific oracle. Exact ties retain the
    # original settings where possible, then the shorter numerical budget.
    return max(values,key=lambda p:(values[p]['score'], values[p]['params']==baseline,
                                   -values[p]['params']['steps'],p))


def parent_identity(row):
    # Canonical HC parent: clip-number prefix is not an independent video.
    name=Path(row['metadata']['original_video_id']).stem
    prefix,separator,parent=name.partition('_')
    if not separator or not prefix.isdigit() or not parent:
        raise ValueError('HC clip identity must contain a numeric clip prefix and parent video ID')
    return parent


def select_inputs(records,count=128,seed=41):
    groups=defaultdict(list)
    for row in records:
        groups[parent_identity(row)].append(row)
    def rank(text):return hashlib.sha256(f'TubeDETR-HC128-v1|{seed}|{text}'.encode()).hexdigest()
    parents=sorted(groups,key=lambda p:(rank(p),p))[:count]
    if len(parents)!=count:raise ValueError('Insufficient independent parents')
    return sorted([min(groups[p],key=lambda r:(rank(r['id']),r['id']))for p in parents],key=lambda r:r['id'])


def prepare():
    records=[json.loads(s)for s in (ROOT/'data/evaluation/hc2.jsonl').read_text().splitlines()]
    chosen=select_inputs(records)
    manifest=ROOT/'data/evaluation/tubedetr_hc2_dev128.jsonl'
    text=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n'for r in chosen)
    if manifest.exists() and manifest.read_text()!=text:
        raise ValueError('Development population changed; do not select a new panel after seeing candidate scores')
    manifest.write_text(text) if not manifest.exists() else None
    # Selection above uses input identity only; labels are copied afterwards.
    ids={r['id']for r in chosen}
    labels=ROOT/'data/evaluation/tubedetr_hc2_dev128_labels.jsonl'
    source=[json.loads(s)for s in (ROOT/'data/evaluation/hc2_labels.jsonl').read_text().splitlines()]
    selected=[r for r in source if r['id']in ids]
    if len(selected)!=128 or len({r['parent']for r in selected})!=128:raise ValueError('Invalid dev128 pairing')
    label_text=''.join(json.dumps(r,separators=(',',':'))+'\n'for r in selected)
    if labels.exists() and labels.read_text()!=label_text:raise ValueError('Development labels changed')
    labels.write_text(label_text) if not labels.exists() else None
    old=json.loads((OLD/'run.json').read_text())
    plan=dict(status='frozen_bounded_HC128_development_plan',backbone='TubeDETR',dataset='hc2',seed=41,
        queries=128,parents=128,manifest_sha256=sha(manifest),labels_sha256=sha(labels),
        population_sha256=population_hash(chosen),implementation_sha256=implementation_hash(),
        baseline_run_sha256=sha(OLD/'run.json'),baseline_prediction_seal_sha256=sha(OLD/'sealed.json'),
        configuration_sha256=sha(ROOT/'configs/default.json'),adapter_configuration_sha256=sha(ROOT/'configs/tubedetr.json'),
        input_selection='new input-hash fixed128: one query per canonical parent, removing numeric clip prefix; seed41; no outcome-based selection',
        legacy_HC128_not_reused=True,not_claimed_identical_to_historical_HC128=True,
        full_target_results_previously_seen=True,development_exposure_registered_before_candidates=True,
        original_study_completion_required=['results/tubedetr/final_seal.json','results/tubedetr/ablation_seal.json',
                                             'results/tubedetr_vid_development/final_seal.json'],
        baseline=BASE,baseline_configuration=old['configuration'],
        diagnosis_sha256=sha(ROOT/'results/tubedetr/hc2_diagnosis.json'),
        pre_candidate_amendment=dict(prior_plan_sha256=sha(PUBLIC/'plan_v1_before_spatial_followup.json'),
            reason='User requests deeper spatial analysis; weaker gamma edge checked before any HC candidates'),
        development_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [ROOT/'tools/tubedetr_hc2_development.py',ROOT/'tools/complete_tubedetr_hc2_development.py',ROOT/'tools/analyze_tubedetr_spatial.py']},
        stages=[dict(name='initialization',gamma=GAMMAS,objective='WithoutOPD complete vIoU; gamma0 control only',
                     reason='Frozen HC initialization accounts for -1.877188pp whereas OPD adds only0.025177pp'),
                dict(name='learning_rate',lr=[.0035,.007,.014,.028],steps=40,tau=.14,
                     reason='Measure update sensitivity both below and above the transferred LR; do not assume overly large LR'),
                dict(name='temperature',tau=[.07,.14,.28],
                     reason='Inspect candidate entropy/ESS and actual localization; test sharpening and flattening'),
                dict(name='steps',steps=[20,40,80],
                     reason='Check under-updating/over-updating; saved prefixes must match independently completed40-step outputs'),
                dict(name='gamma_neighbours',max_new_profiles=2,
                     reason='Recheck initialization/optimizer coupling around the initialization-only choice')],
        max_new_development_fit_paths=9*128+5,one_global_profile=True,output_alpha=1.,
        unchanged=['source checkpoint','single ReferDINO','same eight observations','Strong time output',
                   'network weights frozen','per-query reset','policy scales','E calibration'],
        selection_objective='maximum dev128 query-average Full m_vIoU including the original Full baseline',
        no_full_population_parameter_search=True,selected_profile_full_evaluations=1,
        baseline_retention_policy='If original profile wins, reuse its already-complete full predictions exactly',
        new_full_outputs='outputs/tubedetr_hc2_development/full_selected',
        old_results_overwritten=False,labels_read_only_by_offline_scoring_or_controller=True)
    if PLAN.exists() and json.loads(PLAN.read_text())!=plan:raise ValueError('Frozen development plan changed')
    write(PLAN,plan) if not PLAN.exists() else None
    return plan


def guard_plan(plan):
    if (implementation_hash()!=plan['implementation_sha256']
            or sha(ROOT/'configs/default.json')!=plan['configuration_sha256']
            or sha(ROOT/'configs/tubedetr.json')!=plan['adapter_configuration_sha256']
            or sha(OLD/'run.json')!=plan['baseline_run_sha256']
            or sha(OLD/'sealed.json')!=plan['baseline_prediction_seal_sha256']
            or sha(ROOT/'results/tubedetr/hc2_diagnosis.json')!=plan['diagnosis_sha256']):
        raise ValueError('Frozen model, configuration or baseline identity changed')
    if any(sha(ROOT/p)!=h for p,h in plan['development_code_sha256'].items()):
        raise ValueError('Development execution code changed after the plan was frozen')


def identity(manifest,params,scope,plan):
    return dict(scope=scope,configuration={**plan['baseline_configuration'],**params},
        manifest_sha256=sha(manifest),implementation_sha256=plan['implementation_sha256'],
        upstream_prediction_seal_sha256=plan['baseline_prediction_seal_sha256'],plan_sha256=sha(PLAN))


def saved(out,row,ident):
    path=out/'predictions'/f"{row['id']}.npz"
    receipt=path.with_suffix('.json')
    if not path.exists() and not receipt.exists():return False
    if not path.exists() or not receipt.exists():raise ValueError('Incomplete immutable prediction pair')
    r=json.loads(receipt.read_text())
    if (r['prediction_sha256']!=sha(path) or r['identity_sha256']!=digest(ident)
            or r['request_sha256']!=digest(row)):raise ValueError('Cached development prediction changed')
    return True


def seal(out,records,ident):
    files={}
    for row in records:
        if not saved(out,row,ident):raise ValueError('Cannot seal a partial candidate population')
        p=out/'predictions'/f"{row['id']}.npz";r=p.with_suffix('.json')
        files[row['id']]=dict(path=str(p.relative_to(out)),sha256=sha(p),
            receipt_path=str(r.relative_to(out)),receipt_sha256=sha(r))
    value=dict(status='sealed_before_offline_scoring',queries=len(records),
        manifest_sha256=ident['manifest_sha256'],identity=ident,files=files,labels_read=False)
    path=out/'sealed.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Candidate seal changed')
    write(path,value) if not path.exists() else None
    write(out/'run.json',ident) if not (out/'run.json').exists() else None


def from_original(row,seal_value):
    e=seal_value['files'][row['id']];p=OLD/e['path'];r=OLD/e['receipt_path']
    if sha(p)!=e['sha256'] or sha(r)!=e['receipt_sha256']:raise ValueError('Original cache changed')
    with np.load(p,allow_pickle=False)as z:value={k:z[k].copy()for k in z.files}
    return value,json.loads(r.read_text())


def unscaled_initialization(value,receipt,plan):
    from stvg.consensus import apply
    from stvg.predictor import output_writeback
    raw=apply(value['Source'],np.asarray(receipt['initialization']['delta'],dtype=np.float64))
    restored=output_writeback(value['Source'],raw,plan['baseline_configuration']['initialization_gamma'])
    if not np.array_equal(restored,value['E_Ref']):
        raise ValueError('Saved consensus delta does not reproduce the baseline gamma0.5 initialization')
    return raw


def initialization_candidates(manifest,plan):
    from stvg.predictor import output_writeback
    records=[json.loads(s)for s in Path(manifest).read_text().splitlines()]
    oldseal=json.loads((OLD/'sealed.json').read_text())
    for gamma in GAMMAS:
        params=parameters(gamma=gamma);out=OUT/'initialization'/pid(params)
        ident=identity(manifest,params,'exact initialization-only counterfactual; no OPD or parameter fitting',plan)
        for row in records:
            if saved(out,row,ident):continue
            value,r=from_original(row,oldseal)
            raw=unscaled_initialization(value,r,plan)
            value['E_Ref']=output_writeback(value['Source'],raw,gamma)
            value['Live_raw']=value['Live']=value['E_Ref'].copy()
            value.update(cost=dict(extra_GPU_episodes=0),checkpoint=r['checkpoint'],
                         initialization={**r['initialization'],'initialization_gamma':gamma})
            save_prediction(out/'predictions'/f"{row['id']}.npz",value,
                dict(identity_sha256=digest(ident),request_sha256=digest(row)))
        seal(out,records,ident)


def worker(request):
    # No labels/metrics are passed to this process; the runtime audit guard also
    # forbids annotation paths, including symlinks. Acquire the shared GPU lease.
    import torch
    from stvg.runtime import setup,lease
    from stvg.tubedetr import load_source,native_forward,TubeReplay
    from stvg.initialization import initialize,evidence_hash
    from stvg.predictor import output_writeback
    from stvg.policy import fit
    from stvg.tube_decode import decode
    from stvg.inputs import validate_input
    from stvg.tensors import state_hash
    plan=json.loads(PLAN.read_text());guard_plan(plan);setup();lock=lease()
    write(OUT/'status.json',dict(status=request['scope'],new_GPU_work_started=True,as_of=now()))
    manifest=Path(request['manifest']);records=[json.loads(s)for s in manifest.read_text().splitlines()]
    oldseal=json.loads((OLD/'sealed.json').read_text());teacher=None
    if sha(manifest) not in {plan['manifest_sha256'],oldseal['manifest_sha256']}:
        raise ValueError('Worker manifest is not a frozen development/full population')
    model=load_source(plan['baseline_configuration']);before=state_hash(model.state_dict())
    profiles=request['profiles'];base=Path(request['output_base'])
    groups=defaultdict(list)
    for p in profiles:groups[(p['initialization_gamma'],p['lr'],p['tau'])].append(p)
    for params in profiles:
        out=base/pid(params);ident=identity(manifest,params,request['scope'],plan)
        path=out/'run.json'
        if path.exists() and json.loads(path.read_text())!=ident:raise ValueError('Existing profile run identity changed')
        write(path,ident) if not path.exists() else None
    completed=[];verified_original=False;new_fits=0
    try:
        for index,row in enumerate(records):
            original,prior=from_original(row,oldseal)
            work=[]
            for params in profiles:
                out=base/pid(params);ident=identity(manifest,params,request['scope'],plan)
                if not saved(out,row,ident):work.append(params)
            if not work:continue
            frames,fids=decode(row['metadata'],'hc2');validate_input(frames,fids,row['metadata'])
            native=native_forward(model,frames,fids,row['metadata']['caption'],224)
            if not np.array_equal(native['source'].numpy(),original['Source']) or native['interval']!=original['Native_interval'].tolist():
                raise ValueError('Fresh native output differs from the frozen cache; interface issue, not tuning evidence')
            positions=original['positions8'].tolist()
            cache=OUT/'teacher_cache'/f"{row['id']}.npz";meta=cache.with_suffix('.json')
            rgb_hash=hashlib.sha256(frames[positions].tobytes()).hexdigest()
            if cache.exists():
                entry=json.loads(meta.read_text())
                if entry['request_sha256']!=digest(row) or entry['RGB_sha256']!=rgb_hash or entry['sha256']!=sha(cache):
                    raise ValueError('Expert cache input identity changed')
                with np.load(cache,allow_pickle=False)as z:boxes=z['boxes'].copy()
            else:
                torch.cuda.empty_cache()
                if teacher is None:
                    from stvg.teacher_service import PersistentRefer
                    teacher=PersistentRefer(OUT/'teacher.log')
                answer=teacher(frames[positions],row['metadata']['caption']);boxes=np.asarray(answer['boxes'])
                cache.parent.mkdir(parents=True,exist_ok=True);np.savez(cache,boxes=boxes)
                write(meta,dict(request_sha256=digest(row),RGB_sha256=rgb_hash,sha256=sha(cache)))
            if evidence_hash(boxes)!=prior['initialization']['evidence_sha256']:
                raise ValueError('Recomputed single-expert evidence differs from the frozen query')
            raw,init=initialize(native['source'].double().numpy(),positions,boxes,plan['baseline_configuration']['E_calibration'])
            baseline_E=output_writeback(original['Source'],raw,plan['baseline_configuration']['initialization_gamma'])
            if not np.array_equal(baseline_E,original['E_Ref']):raise ValueError('Original gamma0.5 initialization is not bitexact')
            if not np.array_equal(raw,unscaled_initialization(original,prior,plan)):
                raise ValueError('Recomputed unscaled initialization differs from sealed consensus delta')
            if not verified_original:
                rp=TubeReplay(model,native,baseline_E,positions)
                replay=fit(rp,boxes,row['key'],'Live',41,plan['baseline_configuration'])
                if not np.array_equal(replay['final'].numpy(),original['Live_raw']):
                    raise ValueError('Original OPD replay is not bitexact; stop before parameter comparisons')
                verified_original=True;del replay,rp
            todo=defaultdict(list)
            for p in work:todo[(p['initialization_gamma'],p['lr'],p['tau'])].append(p)
            for (gamma,lr,tau),items in todo.items():
                max_steps=max(p['steps']for p in items);cfg={**plan['baseline_configuration'],**parameters(gamma,lr,tau,max_steps)}
                E=output_writeback(original['Source'],raw,gamma);rp=TubeReplay(model,native,E,positions)
                result=fit(rp,boxes,row['key'],'Live',41,cfg)
                new_fits+=1
                if rp.state.count_nonzero() or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                    raise ValueError('Per-query reset or network freezing failed')
                # Existing40-step candidate must equal the saved40 prefix of80.
                previous=base/pid(parameters(gamma,lr,tau,40))/'predictions'/f"{row['id']}.npz"
                if max_steps>40 and previous.exists():
                    with np.load(previous,allow_pickle=False)as z:
                        if not np.array_equal(z['Live_raw'],result['path'][40]['boxes'].numpy()):
                            raise ValueError('Saved-prefix equivalence failed')
                entropies=[];ess=[];gradients=[]
                for round_value in result['rounds']:
                    w=round_value['rollout']['weights'].numpy()
                    entropies.append(float(-(w*np.log(np.maximum(w,1e-300))).sum(-1).mean()/np.log(32)))
                    ess.append(float((1/(w*w).sum(-1)).mean()))
                    g=round_value['gradient']
                    gradients.append(float(g.norm()) if g is not None else 0.)
                for params in items:
                    out=base/pid(params);ident=identity(manifest,params,request['scope'],plan)
                    live=result['path'][params['steps']]['boxes'].numpy()
                    value={**original,'E_Ref':E,'Live_raw':live,'Live':live.copy(),
                        'checkpoint':model.load_audit,'initialization':{**init,'initialization_gamma':gamma},
                        'cost':dict(shared_fit_seconds=result['seconds'],executed_shared_updates=result['actual_updates'],
                            selected_step=params['steps'],prefix_readout=params['steps']!=max_steps,
                            pseudo_loss_before=result['path'][0]['anchor_loss'],
                            pseudo_loss_after=result['path'][params['steps']]['anchor_loss'],
                            mean_weight_entropy=float(np.mean(entropies[:params['steps']])),
                            mean_ESS=float(np.mean(ess[:params['steps']])),
                            mean_state_gradient_norm=float(np.mean(gradients[:params['steps']])),
                            Strong_reused=True,not_fresh_end_to_end_timing=True)}
                    save_prediction(out/'predictions'/f"{row['id']}.npz",value,
                        dict(identity_sha256=digest(ident),request_sha256=digest(row)))
                del result,rp
            del native,frames
            write(OUT/'worker_progress.json',dict(complete=index+1,total=len(records),scope=request['scope'],labels_read=False))
            print(f"Development worker {index+1}/{len(records)}",flush=True)
        if state_hash(model.state_dict())!=before:raise ValueError('Frozen network state changed')
        for params in profiles:seal(base/pid(params),records,identity(manifest,params,request['scope'],plan))
        write(Path(request['receipt']),dict(status='complete',queries=len(records),
            original_default_OPD_replay_bitexact=verified_original,new_fit_paths=new_fits,
            frozen_network_hash_unchanged=True,labels_read=False,profiles=profiles))
    finally:
        if teacher is not None:teacher.close()
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker-request')
    args=p.parse_args()
    if args.worker_request:worker(json.loads(Path(args.worker_request).read_text()))
    elif args.prepare_only:print(json.dumps(prepare(),indent=2))
    else:raise SystemExit('Use --prepare-only or an input-only --worker-request')
