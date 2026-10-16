"""Versioned six-coordinate development tuning; sealed v1 remains untouched.

Predictions/candidate caches never consume labels. Development GT is used only
by the outer Optuna scorer. Evaluation is opened after all four config locks.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, load, save, sha

OUT = ROOT / 'artifacts/decota_refine_tuning_v1'
STORY = ROOT / 'artifacts/decota_story_v1'
REFINE = ROOT / 'artifacts/decota_refine_v1'
BACKBONES = ['tastvg', 'tubedetr']
GROUPS = ['hc_to_vid', 'vid_to_hc']
SPATIAL = dict(keyframes=8, phrase_threshold=.35, distinct_margin=.05, nms_iou=.5)


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def prepare():
    if (OUT/'lock.json').exists():
        return
    from scripts.decota_refine_protocol_v1 import verify_code
    previous = read(REFINE/'lock.json'); verify_code(previous)
    s = read(STORY/'lock.json')
    old = read(ROOT/'artifacts/decota_s_v1/lock.json')
    excluded = [r['input'] for part in ['development', 'evaluation'] for rows in old[part].values() for r in rows]
    excluded += [r['input'] for rows in previous['new'].values() for r in rows]
    source_ex = {q['source'] for q in excluded}; hash_ex = {q['video_sha256'] for q in excluded}
    p = dict(version='decota_refine_tuning_v1', created_unix=time.time(), seed=20260910,
        objective='source-macro corrected vIoU of final temporal TTA + residual interpolation',
        protocol='Two coordinate sweeps, development-only labels. All four selections lock before evaluation.',
        old_method_unchanged=True, spatial_backward_calls=0,
        fixed=dict(gamma=0., optimizer='AdamW', optimizer_eps=1e-4, weight_decay=0.),
        lr_bounds=[1e-7, 10.], lr_log_grid_count=25, lr_tpe_extra=8,
        steps_grid=[1,2,3,5,8,10,15,20,30],
        spatial_grid=dict(keyframes=[0,2,4,6,8,12,16],
            phrase_threshold=[.10,.20,.25,.30,.35,.40,.45,.50,.60,.75],
            distinct_margin=[0.,.01,.025,.05,.075,.10,.15,.20],
            nms_iou=[.2,.3,.4,.5,.6,.7,.8]),
        zero_keyframes='Explicit development-selected spatial-off option; temporal parameter TTA remains active.',
        tie_break='Within 1e-10 utility: fewer steps, fewer expert calls, lower lr, conservative gates.',
        fine_lr_trials=13, fine_gate_trials=9, max_coordinate_sweeps=2,
        expert_cache='All sampled development positions cached once; deployment only calls selected K. Offline search cost reported separately.',
        expert_snapshot=previous['expert_snapshot'], expert_sha256=previous['expert_sha256'],
        development=old['development'], selected_start=previous['temporal_configs'],
        dev_label_manifest=s['dev_label_manifest'],dev_label_manifest_sha256=s['dev_label_manifest_sha256'],
        evaluation={}, evaluation_label_specs=previous['labels'],
        evaluation_historically_untouched=False,
        evaluation_note='Disjoint from declared development, previous 128 evaluation sources and later 128 refinement sources. Other historical experiment exposure is possible.',
        evaluation_no_retune=True, reference_lock_sha256=sha(REFINE/'lock.json'),
        native_caches={}, head_files={}, protected_pins=previous['pins'])
    for b in BACKBONES:
        p['native_caches'][b]={}; p['head_files'][b]={}
        for g in GROUPS:
            f=STORY/f'runs/{b}/{g}/development/clean/barrier.json'
            p['native_caches'][b][g]=dict(path=str(f),sha256=sha(f))
            h=STORY/f'heads/{b}_{g}.pt';p['head_files'][b][g]=dict(path=str(h),sha256=sha(h))
    from vg_tta.foreground_runtime import QuerySubjectParser
    from vg_tta.tg_spatial_tta_v1 import visual_query
    parser=QuerySubjectParser(ROOT/'.cache/stanza')
    for g in GROUPS:
        rows=p['development'][g]
        assert len(rows)==len({r['input']['source'] for r in rows})==32
        candidates={}
        for parent,v in s['parents'].items():
            if v['group']!=g:continue
            for q in v['spec']['queries']:
                if q['source'] in source_ex or q['video_sha256'] in hash_ex or not Path(q['video_path']).is_file():continue
                candidates.setdefault(q['source'],[]).append((parent,q))
        chosen=sorted(candidates,key=lambda k:digest(['refine_tune_eval_v1',k]))[:64]
        evalrows=[]
        for i,source in enumerate(chosen):
            parent,q=min(candidates[source],key=lambda x:digest([x[0],x[1]['index'],x[1]['caption']]))
            evalrows.append(dict(ordinal=i,input=q,parent=parent,student_subject=parser(q['caption']),visual_query=visual_query(parser,q['caption'])))
        p['evaluation'][g]=evalrows
        print('ROSTER',g,'dev',32,'evaluation',len(evalrows),'available',len(candidates),flush=True)
    p['pins']={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    write(OUT/'lock.json',p)


def plan():
    p=read(OUT/'lock.json')
    from scripts.decota_refine_protocol_v1 import verify_code
    verify_code(read(REFINE/'lock.json'))
    for f,h in p['pins'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('New tuning code differs from lock: '+f)
    return p


def configure():
    import numpy as np, torch
    torch.set_num_threads(4);torch.manual_seed(20260910);np.random.seed(20260910)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def cache_expert():
    import gc, torch
    from vg_tta.dense_expansion_data_v1 import decode_raw
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from vg_tta.foreground_runtime import state_digest
    p=plan();configure()
    if (OUT/'expert_barrier.json').exists():return
    assert sha(Path(p['expert_snapshot'])/'model.safetensors')==p['expert_sha256']
    expert=SpatialExpert(p['expert_snapshot']);before=state_digest(expert.model);receipts=[]
    for g in GROUPS:
        for j,r in enumerate(p['development'][g]):
            q=r['input'];f=OUT/'expert'/g/f"{q['index']:06d}.pt"
            if not f.exists():
                started=time.perf_counter();frames,ids=decode_raw(q);v=r['visual_query'];probes=[]
                if v['phrase']:
                    for pos in range(len(ids)):
                        z=expert(frames[pos],v['phrase'],v['entity'])
                        z.update(position=pos,frame_id=int(ids[pos]));probes.append(z)
                save(f,dict(input=q,frame_ids=list(ids),visual_query=v,probes=probes,GT_used=False,
                    expert_frozen=True,wall_seconds=time.perf_counter()-started,lock_sha256=sha(OUT/'lock.json')))
                del frames,probes;gc.collect()
            receipts.append(dict(group=g,index=q['index'],path=str(f),sha256=sha(f)))
            status(OUT/'progress.json',dict(stage='expert_cache',group=g,done=j+1,total=32,unix=time.time()))
            print('EXPERT_CACHE',g,j+1,32,flush=True)
    assert before==state_digest(expert.model)
    write(OUT/'expert_barrier.json',dict(receipts=receipts,state_unchanged=True,GT_used=False))


def imports(backbone):
    if backbone=='tubedetr':
        from vg_tta.tubedetr_runtime import add_repo_to_path
        add_repo_to_path(ROOT/'external/TubeDETR')
    else:
        from scripts import run_tastvg_span_tta as ta
        ta.official_imports()


def temporal_replay(head, rows, backbone, cfg):
    from scripts.run_decota_story_v1 import fitted_result
    intervals=[];audits=[]
    for x in rows:
        base=dict(indices=x['predictions']['frozen']['indices'],logits=x['native_logits'],
            boxes=x['predictions']['frozen']['boxes'],ids=x['frame_ids'],views=x.get('native_views'))
        r=fitted_result(head,[h.cuda() for h in x['head_inputs']],base,backbone,
            dict(lr=cfg['lr'],steps=cfg['steps'],gamma=0.),ablations=False)
        intervals.append(list(r['predictions']['decota']['indices']));audits.append(r['audits']['decota'])
    return dict(intervals=intervals,audits=audits,GT_used_for_fit=False)


def evidence(x,b,meta):
    if b=='tastvg':return meta['native_evidence']
    from methods.decota_refine_v1.predictor import tube_inclusion
    return tube_inclusion(x['native_logits'][0])


def anchors(probes,positions,cfg,cache=None):
    from vg_tta.tg_spatial_tta_v1 import choose_candidate
    bypos={z['position']:z for z in probes};out=[]
    for pos in positions:
        if pos not in bypos:continue
        z=bypos[pos];key=(pos,float(cfg['nms_iou']))
        if cache is not None and key in cache:c=cache[key]
        else:
            # Cache NMS only. Score and distinct-instance gates are evaluated
            # every trial, including the valid-box condition in choose_candidate.
            c=choose_candidate(z['all_boxes'],z['all_phrase_scores'],
                dict(nms_iou=cfg['nms_iou'],phrase_threshold=0.,distinct_margin=0.))
            if cache is not None:cache[key]=c
        if c['accepted'] and c['score']>=cfg['phrase_threshold'] and c['margin']>=cfg['distinct_margin']:
            out.append(dict(position=pos,frame_id=z['frame_id'],box=c['box'],score=c['score'],margin=c['margin']))
    return out


def score_boxes(boxes,targets,ids,gt,ij):
    from vg_tta.st_component_diagnostics_v1 import frame_iou,physical_score
    quality,_=frame_iou(boxes,targets,ids,gt)
    return physical_score(quality,ids,gt,(ids[ij[0]],ids[ij[1]]+1))


def tune(b,g):
    import numpy as np,torch,optuna
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.tg_spatial_tta_v1 import keyframes
    from methods.decota_refine_v1.api import refine
    p=plan();configure();dest=OUT/'tuning'/b/g
    if (dest/'selected.json').exists():return
    imports(b);h=p['head_files'][b][g];assert sha(h['path'])==h['sha256']
    head=load(h['path']).cuda().eval().requires_grad_(False);before=state_digest(head)
    br=p['native_caches'][b][g];assert sha(br['path'])==br['sha256'];bar=read(br['path'])
    rows=[]
    for rr in bar['receipts']:
        assert sha(rr['path'])==rr['sha256'];x=load(rr['path']);assert not x['GT_used'];rows.append(x)
    bymeta={r['input']['index']:r for r in p['development'][g]}
    assert len(rows)==len({x['source'] for x in rows})==32
    assert {x['source'] for x in rows}=={r['input']['source'] for r in bymeta.values()}
    eb=read(OUT/'expert_barrier.json');ers={r['index']:r for r in eb['receipts'] if r['group']==g}
    experts=[];ev=[]
    for x in rows:
        rr=ers[x['index']];assert sha(rr['path'])==rr['sha256'];e=load(rr['path'])
        assert not e['GT_used'] and e['frame_ids']==x['frame_ids'];experts.append(e['probes'])
        ev.append(evidence(x,b,bymeta[x['index']]))
    assert sha(p['dev_label_manifest'])==p['dev_label_manifest_sha256']
    manifest=read(p['dev_label_manifest']);lm={r['index']:r for r in manifest['groups'][g]};labels=[]
    for x in rows:
        rr=lm[x['index']];assert sha(rr['labels_path'])==rr['label_sha256'];lab=load(rr['labels_path'])
        assert lab['video_target']['frames_id']==x['frame_ids'] and lab['identity']==rr['identity']
        labels.append((lab['targets'],tuple(lab['annotation'][k] for k in ['tube_start_frame','tube_end_frame'])))
    nms_caches=[{} for _ in rows];timing_start=time.time();all_trials=[]
    def evaluate(cfg):
        tk=digest(dict(lr=cfg['lr'],steps=cfg['steps']));tf=dest/'temporal'/f'{tk}.json'
        if not tf.exists():
            result=temporal_replay(head,rows,b,cfg)
            assert state_digest(head)==before
            write(tf,dict(config={k:cfg[k] for k in ['lr','steps']},indices_in_development_order=[x['index'] for x in rows],**result))
        temporal=read(tf)
        f=dest/'scores'/f'{digest(cfg)}.json'
        if f.exists():return read(f)['utility']
        metrics=[];counts=[]
        for j,(x,ij) in enumerate(zip(rows,temporal['intervals'])):
            pos=keyframes(x['frame_ids'],ij,x['predictions']['frozen']['indices'],ev[j],cfg['keyframes'])
            pp=anchors(experts[j],pos,cfg,nms_caches[j]);boxes,_=refine(x['predictions']['frozen']['boxes'],pp,x['frame_ids'])
            tar,gt=labels[j];metrics.append(score_boxes(boxes,tar,x['frame_ids'],gt,ij))
            counts.append(dict(called=len(pos) if experts[j] else 0,accepted=len(pp)))
        value=float(np.mean([m['vIoU_corrected'] for m in metrics]));assert np.isfinite(value)
        write(f,dict(config=cfg,utility=value,metrics=metrics,counts=counts,
            temporal_prediction_sha256=sha(tf),selection_used_development_GT=True,
            query_indices=[x['index'] for x in rows],sources=[x['source'] for x in rows]))
        return value
    cfg={**p['selected_start'][b][g],**SPATIAL};cfg.pop('gamma',None)
    original=dict(cfg);initial=evaluate(cfg)
    storage='sqlite:///'+str((dest/'optuna.sqlite3').resolve())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def stage(name,param,grid,base,extra=0,bounds=None,log=False):
        grid=list(dict.fromkeys([base[param]]+list(grid)))
        sampler=optuna.samplers.TPESampler(seed=p['seed'],n_startup_trials=len(grid))
        study=optuna.create_study(study_name=name,direction='maximize',storage=storage,load_if_exists=True,sampler=sampler)
        if not study.trials:
            for v in grid:study.enqueue_trial({param:v})
        def objective(t):
            c=dict(base)
            c[param]=t.suggest_float(param,*bounds,log=log) if bounds else t.suggest_categorical(param,grid)
            start=time.monotonic()
            try:v=evaluate(c)
            except (ValueError,FloatingPointError,AssertionError) as e:
                t.set_user_attr('failure',repr(e));t.set_user_attr('no_samples_dropped',True)
                raise optuna.TrialPruned(repr(e))
            t.set_user_attr('config',c);t.set_user_attr('seconds',time.monotonic()-start)
            print('TRIAL',b,g,name,t.number,round(v*100,5),c,flush=True)
            status(OUT/'progress.json',dict(stage='tuning',backbone=b,group=g,coordinate=name,trial=t.number,utility=v,unix=time.time()))
            return v
        remaining=len(grid)+extra-len([t for t in study.trials if t.state.is_finished()])
        if remaining>0:study.optimize(objective,n_trials=remaining)
        valid=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
        if not valid:raise RuntimeError('No complete full-development trial')
        bestval=max(t.value for t in valid)
        close=[t for t in valid if t.value>=bestval-1e-10]
        best=min(close,key=lambda t:(t.user_attrs['config']['steps'],t.user_attrs['config']['keyframes'],t.user_attrs['config']['lr'],
            -t.user_attrs['config']['phrase_threshold'],-t.user_attrs['config']['distinct_margin'],t.user_attrs['config']['nms_iou']))
        all_trials.extend(dict(stage=name,number=t.number,state=t.state.name,value=t.value,params=t.params,**t.user_attrs) for t in study.trials)
        return best.user_attrs['config']
    cfg=stage('lr_wide','lr',np.logspace(-7,1,p['lr_log_grid_count']).tolist(),cfg,p['lr_tpe_extra'],p['lr_bounds'],True)
    cfg=stage('steps_wide','steps',p['steps_grid'],cfg)
    for k,grid in p['spatial_grid'].items():cfg=stage(k+'_wide',k,grid,cfg)
    # Second sweep starts with local LR refinement AFTER selecting steps and
    # spatial gates; it is predeclared, not triggered by evaluation results.
    lo=max(1e-7,cfg['lr']/10);hi=min(10.,cfg['lr']*10)
    cfg=stage('lr_fine','lr',np.geomspace(lo,hi,p['fine_lr_trials']).tolist(),cfg,0,[lo,hi],True)
    cfg=stage('steps_second','steps',p['steps_grid'],cfg)
    cfg=stage('keyframes_second','keyframes',p['spatial_grid']['keyframes'],cfg)
    for k,width,maximum in [('phrase_threshold',.10,.9),('distinct_margin',.05,.4),('nms_iou',.1,.9)]:
        low=max(0. if k!='nms_iou' else .1,cfg[k]-width);high=min(maximum,cfg[k]+width)
        grid=[round(float(v),6) for v in np.linspace(low,high,p['fine_gate_trials'])]
        cfg=stage(k+'_fine',k,grid,cfg)
    assert state_digest(head)==before
    final=evaluate(cfg);assert final>=initial-1e-10
    result=dict(config=cfg,initial_config=original,initial_utility=initial,development_utility=final,
        trials=len(all_trials),unique_scores=len(list((dest/'scores').glob('*.json'))),
        queries=32,sources=32,head_state_unchanged=True,temporal_GT_used=False,spatial_gradient_steps=0,
        configuration_selection_used_development_GT=True,evaluation_GT_used=False,
        seconds=time.time()-timing_start,lock_sha256=sha(OUT/'lock.json'),created_unix=time.time())
    write(dest/'trials.json',all_trials);write(dest/'selected.json',result)
    print('SELECTED',b,g,result,flush=True)


def seal():
    p=plan();configs={}
    for b in BACKBONES:
        configs[b]={}
        for g in GROUPS:
            f=OUT/f'tuning/{b}/{g}/selected.json';x=read(f)
            assert x['evaluation_GT_used'] is False
            configs[b][g]=dict(path=str(f),sha256=sha(f),config=x['config'])
    write(OUT/'selection_barrier.json',dict(configs=configs,GT_used_only_for_development_selection=True,created_unix=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','expert','tune','seal'])
    ap.add_argument('--backbone',choices=BACKBONES);ap.add_argument('--group',choices=GROUPS)
    a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='expert':cache_expert()
    elif a.stage=='tune':tune(a.backbone,a.group)
    else:seal()
