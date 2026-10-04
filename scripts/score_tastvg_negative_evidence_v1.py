"""CPU, post-global-seal scores for isolated writes and a separate reset-u stream.

Importing this module reads neither labels nor results. Raw boxes, states,
captions and frame identifiers are retained only in the private sealed inputs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, time, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tastvg_negative_evidence_common_v1 import (
    BASE, PUB, POOL, ARMS, DATASETS, BUNDLES, read, write, sha, key, plan,
    checked, oldcell, local_payload_path, reset_payload_path, verify_seal,
)
from scripts.audit_tastvg_dta_oracle_r1_v1 import summary as paired_summary

METRICS = ('v', 't', 's')
BLOCKS = ('query', 'norm1', 'norm3', 'norm4')
CONTRASTS = [(a, 'pre') for a in ARMS] + [(a, 'rank_native') for a in ARMS[1:]] + [('negative_local', 'negative_global')]


def array(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def metadata(c):
    return dict(cell_key=key(c), dataset=c['dataset'], split=c['split'],
                source_id=c['parent'], condition=c['condition'], order=c['order'],
                arrival=c['arrival'], expert_scheduled=bool(c['scheduled']))


def fixed_interval(c, row, old):
    a, b = old['final_indices']
    return [row['frame_ids'][a], row['frame_ids'][b] + 1]


def dense(pred, row, truth, span, ds, interval=None):
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube
    return DenseTube(array(pred['boxes']), row, truth, span, ds == 'hc2').score(
        pred['physical_interval'] if interval is None else interval)


def sampled_iou(boxes, row, truth, ds):
    """GT only at actually sampled annotated event frames; unknown stays NaN."""
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    b = xyxy(array(boxes), row['input']['width'], row['input']['height'])
    if ds == 'hc2':
        b = np.maximum(b, 0)
    out = np.full(len(b), np.nan)
    for j, fid in enumerate(row['frame_ids']):
        if int(fid) in truth:
            out[j] = float(box_iou(b[j], truth[int(fid)]))
    return out


def observed_summary(boxes, row, truth, ds, positions, span):
    i = sampled_iou(boxes, row, truth, ds)
    eligible = np.isfinite(i) & (np.asarray(row['frame_ids']) >= span[0]) & (np.asarray(row['frame_ids']) < span[1])
    observed = np.zeros(len(i), bool)
    observed[np.asarray(positions, int)] = True
    obs, unseen = eligible & observed, eligible & ~observed
    return dict(observed_GT_frames=int(obs.sum()), unobserved_GT_frames=int(unseen.sum()),
                observed_GT_IoU=float(i[obs].mean()) if obs.any() else None,
                unobserved_GT_IoU=float(i[unseen].mean()) if unseen.any() else None)


def tails(rows, a, b):
    d = np.array([r[f'{a}_minus_{b}_v'] for r in rows])
    return dict(gain=int((d > 1e-12).sum()), harm=int((d < -1e-12).sum()),
                severe_harm_gt5pp=int((d < -.05).sum()), severe_harm_gt20pp=int((d < -.20).sum()),
                baseline_good_destroyed_at_03=sum(r[b+'_v'] > .3 and r[a+'_v'] <= .3 for r in rows),
                baseline_bad_rescued_at_03=sum(r[b+'_v'] <= .3 and r[a+'_v'] > .3 for r in rows),
                baseline_good_destroyed_at_05=sum(r[b+'_v'] > .5 and r[a+'_v'] <= .5 for r in rows),
                baseline_bad_rescued_at_05=sum(r[b+'_v'] <= .5 and r[a+'_v'] > .5 for r in rows))


def add_differences(r, contrasts):
    for a, b in contrasts:
        for m in METRICS:
            r[f'{a}_minus_{b}_{m}'] = r[a+'_'+m] - r[b+'_'+m]
        d = r[f'{a}_minus_{b}_v']
        r[f'{a}_gross_gain_{b}'], r[f'{a}_gross_loss_{b}'] = max(d, 0.), max(-d, 0.)


def summarize(rows, fields, contrasts, *, reset=False, target_cluster=False):
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for sp in ('search', 'confirm'):
            rr = [r for r in rows if r['dataset'] == ds and r['split'] == sp]
            predicates = dict(corrupt=lambda r:r['condition'] != 'clean', clean=lambda r:r['condition'] == 'clean', all=lambda r:True)
            if reset:
                predicates.update(expert_corrupt=lambda r:r['expert_scheduled'] and r['condition'] != 'clean',
                                  nonexpert_corrupt=lambda r:not r['expert_scheduled'] and r['condition'] != 'clean')
            panels = {}
            for name, test in predicates.items():
                q = [r for r in rr if test(r)]
                if target_cluster:
                    q = [dict(r, source_id=r['target_source_id']) for r in q]
                z = paired_summary(q, fields)
                z['cluster_unit'] = 'target_source' if target_cluster else ('arrival_source' if reset else 'write_source')
                z['negative_tails'] = {a+'_minus_'+b:tails(q, a, b) for a,b in contrasts}
                z['orders'] = {o:paired_summary([r for r in q if r['order'] == o], fields) for o in ('order1', 'order2')}
                panels[name] = z
            out[ds][sp] = panels
    return out


def scalar_step(c, arm, step, row, truth, span, ds, expert_positions, first_interval):
    """Anonymous support diagnostics; no geometry coordinates leave private storage."""
    r = dict(metadata(c), arm=arm, step=step['step'], lr=float(step['lr']),
             teacher_temperature=float(step['teacher_temperature']),
             loss=float(step['loss']), loss_after=float(step['loss_after']),
             eta_gradient_norm=float(step['eta_gradient_norm']),
             gradient_block_norms={k:float(v) for k,v in step['gradient_block_norms'].items()},
             parameter_displacement=float(np.sqrt(sum(np.square(array(step['post_state'][n]).astype(float)-array(v).astype(float)).sum() for n,v in step['pre_state'].items()))))
    positions = np.asarray(array(step['valid_positions']), int)
    assert np.array_equal(positions, expert_positions)
    e = np.asarray(array(step['e_jk']), float)
    p0, q, pa = [np.exp(np.asarray(array(step[x]), float)) for x in ('logp0', 'logq', 'logp_after')]
    evidence = e if arm == 'negative_local' else (e.mean(0) if len(e) else np.zeros(9))
    for name, p in [('p0',p0), ('q',q), ('p_after',pa)]:
        r[name] = p.tolist()
    r['e_jk'] = e.tolist()
    r['valid_observations'] = len(positions)
    r['negative_entries'] = int((e[:,1:] > 0).sum())
    r['undecided_entries'] = int((e[:,1:] == 0).sum())
    r['negative_evidence_mass'] = float(e[:,1:].sum())
    lp0, lq, la = [np.asarray(array(step[x]), float).reshape(-1,9) for x in ('logp0','logq','logp_after')]
    ev = np.asarray(evidence).reshape(-1,9)
    target_odds, actual_odds = [], []
    for j, ee in enumerate(ev):
        for a in range(9):
            for b in range(a+1,9):
                if ee[a] == 0 and ee[b] == 0:
                    target_odds.append(abs((lq[j,a]-lq[j,b])-(lp0[j,a]-lp0[j,b])))
                    actual_odds.append(abs((la[j,a]-la[j,b])-(lp0[j,a]-lp0[j,b])))
    r.update(undecided_odds_pairs=len(target_odds),
             target_undecided_log_odds_max_error=max(target_odds,default=None),
             actual_undecided_log_odds_max_change=max(actual_odds,default=None))
    ids = np.asarray(row['frame_ids'])
    eligible = (ids[positions] >= span[0]) & (ids[positions] < span[1])
    gt = np.stack([sampled_iou(z['boxes'],row,truth,ds)[positions] for z in step['candidates']],1)
    known = eligible & np.isfinite(gt).all(1)
    neg, better = e[known,1:] > 0, gt[known,1:] > gt[known,:1] + 1e-12
    weight = e[known,1:]
    r.update(observed_GT_frames=int(known.sum()), unknown_or_outside_event_observations=int((~known).sum()),
             known_negative_entries=int(neg.sum()), negative_GT_better_entries=int((neg & better).sum()),
             known_negative_evidence_mass=float(weight.sum()), negative_GT_better_evidence_mass=float(weight[better].sum()))
    # A global contradiction is also checked against full-event dense spatial quality.
    sg = np.array([dense(z,row,truth,span,ds,first_interval)['s'] for z in step['candidates']])
    eg = e.mean(0) if len(e) else np.zeros(9)
    r.update(full_event_GT_spatial_candidate_quality=sg.tolist(),
             global_negative_candidates=int((eg[1:] > 0).sum()),
             global_negative_GT_better_candidates=int(((eg[1:] > 0) & (sg[1:] > sg[0]+1e-12)).sum()),
             global_negative_evidence_mass=float(eg[1:].sum()),
             global_negative_GT_better_evidence_mass=float(eg[1:][sg[1:] > sg[0]+1e-12].sum()))
    if arm == 'negative_local':
        shift = pa - p0
        r['GT_better_probability_change'] = float((shift[known,1:] * better).sum(1).mean()) if known.any() else None
    else:
        r['GT_better_probability_change'] = float((pa[1:]-p0[1:])[sg[1:] > sg[0]+1e-12].sum())
    for name in ('pre_prediction','post_prediction'):
        m = dense(step[name],row,truth,span,ds,first_interval)
        r.update({name+'_'+k:v for k,v in m.items()})
        r.update({name+'_'+k:v for k,v in observed_summary(step[name]['boxes'],row,truth,ds,positions,span).items()})
    # Spatial output gradient is distinct from induced movement after a parameter write.
    valid = np.zeros(len(ids), bool); valid[positions] = True
    gradient = np.asarray(array(step['output_gradient']),float)
    movement = np.asarray(array(step['post_prediction']['boxes']),float)-np.asarray(array(step['pre_prediction']['boxes']),float)
    r.update(output_gradient_observed_norm=float(np.linalg.norm(gradient[valid])),
             output_gradient_unobserved_norm=float(np.linalg.norm(gradient[~valid])),
             output_movement_observed_norm=float(np.linalg.norm(movement[valid])),
             output_movement_unobserved_norm=float(np.linalg.norm(movement[~valid])),
             native_interval_changed=step['pre_prediction']['indices'] != step['post_prediction']['indices'])
    return r


def score():
    import torch
    torch.set_num_threads(2)
    tick=time.monotonic();barrier=verify_seal();score_start=time.time()
    assert score_start > barrier['time'] and not torch.cuda.is_initialized()
    lock=read(BASE/'RUNTIME_LOCK.json');cells=read(BASE/'COHORT.json')['cells']
    assert len(cells)==1152 and sum(bool(c['scheduled']) for c in cells)==288
    # Private label pins are checked only here, after every deployable prediction is sealed.
    gt_pins=lock['GT_inputs'];labels={}
    for ds in DATASETS:
        labels[ds]={}
        for sp in ('search','confirm'):
            p=POOL/ds/f'GT_LABELS_{sp}.json'
            assert sha(p)==gt_pins[str(p.relative_to(ROOT))]
            labels[ds][sp]=read(p)
    write(BASE/'GT_EXPOSURE.json',dict(time=score_start,global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
         posthoc_only=True,teacher_GT_used=False,target_retuning=False,GT_inputs=gt_pins))
    current=[];future=[];reset=[];traces=[];plans={ds:plan(ds) for ds in DATASETS}
    for c in cells:
        ds=c['dataset'];row=plans[ds]['rows'][c['parent']];g=labels[ds][c['split']][str(c['parent'])]
        truth={int(k):v for k,v in g['truth'].items()};span=g['span'];old=oldcell(c);interval=fixed_interval(c,row,old)
        oldpred=dict(old['slow'],physical_interval=interval)
        z=checked(reset_payload_path(c));r=metadata(c)
        for name,p,ii in [('A',oldpred,interval),('reset_u',z['result']['output_prediction'],None),
                          ('reset_u_fixedA',z['result']['output_prediction'],interval)]:
            r.update({name+'_'+m:v for m,v in dense(p,row,truth,span,ds,ii).items()})
        add_differences(r,[('reset_u','A'),('reset_u_fixedA','A')]);reset.append(r)
        if not c['scheduled']:continue
        x=checked(local_payload_path(c));r=metadata(c);positions=np.flatnonzero(np.asarray(array(x['evidence']['valid']),bool))
        r.update({"pre_"+m:v for m,v in dense(x['pre_prediction'],row,truth,span,ds).items()})
        r.update(observed_summary(x['pre_prediction']['boxes'],row,truth,ds,positions,span))
        fc=x['future_cell'];fr=plans[ds]['rows'][fc['parent']];fg=labels[ds][fc['split']][str(fc['parent'])]
        ft={int(k):v for k,v in fg['truth'].items()};f=metadata(c)
        f.update(future_cell_key=key(fc),target_source_id=fc['parent'],future_arrival=fc['arrival'],single_write_transfer=True)
        f.update({'pre_'+m:v for m,v in dense(x['future_baseline'],fr,ft,fg['span'],ds).items()})
        for arm in ARMS:
            a=x['arms'][arm]
            r.update({arm+'_'+m:v for m,v in dense(a['post_prediction'],row,truth,span,ds).items()})
            r.update({arm+'_'+k:v for k,v in observed_summary(a['post_prediction']['boxes'],row,truth,ds,positions,span).items()})
            r[arm+'_parameter_displacement']=float(np.sqrt(sum(np.square(array(a['post_state'][n]).astype(float)-array(v).astype(float)).sum() for n,v in x['pre_state'].items())));r[arm+'_steps']=len(a['steps']);r[arm+'_loss_before']=float(a['steps'][0]['loss']);r[arm+'_loss_after']=float(a['steps'][-1]['loss_after'])
            for b in BLOCKS:
                bm=dense(a['block_counterfactual_predictions'][b],row,truth,span,ds)
                r.update({arm+'_'+b+'_'+m:v for m,v in bm.items()})
                r[arm+'_'+b+'_minus_pre_v']=bm['v']-r['pre_v']
            f.update({arm+'_'+m:v for m,v in dense(a['future_prediction'],fr,ft,fg['span'],ds).items()})
            first_interval=interval
            traces.extend(scalar_step(c,arm,s,row,truth,span,ds,positions,first_interval) for s in a['steps'])
        add_differences(r,CONTRASTS);add_differences(f,CONTRASTS);current.append(r);future.append(f)
        if len(current)%24==0:print('CPU_SCORE',len(current),288,flush=True)
    assert len(current)==len(future)==288 and len(reset)==1152
    fields=[a+'_'+m for a in ['pre']+ARMS for m in METRICS]
    fields += [a+'_minus_'+b+'_'+m for a,b in CONTRASTS for m in METRICS]
    fields += [a+'_gross_'+p+'_'+b for a,b in CONTRASTS for p in ('gain','loss')]
    currentfields=fields+[a+'_'+b+'_minus_pre_v' for a in ARMS for b in BLOCKS]
    resetcontrasts=[('reset_u','A'),('reset_u_fixedA','A')]
    resetfields=[a+'_'+m for a in ('A','reset_u','reset_u_fixedA') for m in METRICS]
    resetfields += [a+'_minus_'+b+'_'+m for a,b in resetcontrasts for m in METRICS]
    resetfields += [a+'_gross_'+p+'_'+b for a,b in resetcontrasts for p in ('gain','loss')]
    s=dict(current=summarize(current,currentfields,CONTRASTS),future=summarize(future,fields,CONTRASTS),
           future_target_source_sensitivity=summarize(future,fields,CONTRASTS,target_cluster=True),
           reset_u=summarize(reset,resetfields,resetcontrasts,reset=True))
    PUB.mkdir(parents=True,exist_ok=True)
    cfg=dict(arms=ARMS,bundles=BUNDLES,lambda_value=1.,student_temperature=1.,parameters=1792,
         geometry_coefficients=dict(vidstg=[5,3],hc2=[5,4]),bootstrap_draws=10000,bootstrap_seed=20261004,
         local_expert_cells=288,local_corrupt_cells=240,local_clean_cells=48,reset_arrivals=1152,
         development_sources_per_dataset=32,confirmation_sources_per_dataset=16,historically_exposed=True,
         current_interval='sealed original A final_indices',step_interval='sealed original A final_indices; same as arrival contrast',
         future_cluster='write source',future_target_source_sensitivity=True,
         future_scope='isolated one-write transfer; not a new persistent online trajectory',
         reset_u_scope='separate actual online Rank stream with zero query residual per arrival and persistent LN',
         predictions_sealed_before_GT=True,teacher_GT_used=False,target_retuning=False,production_promoted=False,
         main_contrasts=['negative_global_minus_rank_native','negative_local_minus_rank_native','negative_local_minus_negative_global'],
         cross_bundle_is_not_individual_parameter_causality=True,
         no_evidence_is_not_correctness=True,geometric_distribution_is_not_native_tube_likelihood=True,
         global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'))
    for name,value in [('CONFIG',cfg),('ROWS',current),('FUTURE_ROWS',future),('RESET_U_ROWS',reset),('STEP_DIAGNOSTICS',traces),('SUMMARY',s)]:write(PUB/(name+'.json'),value)
    publicbar=dict(status='sealed',time=barrier['time'],GT_read=False,
        files=len(barrier['files']),global_barrier_sha256=cfg['global_barrier_sha256'],
        runtime_lock_sha256=cfg['runtime_lock_sha256'],score_start_time=score_start)
    write(PUB/'PREDICTION_BARRIER.json',publicbar)
    resources={ds:{stage:read(BASE/ds/(stage+'_RESOURCES.json')) for stage in ('local','reset_u')} for ds in DATASETS}
    resources.update(score_CPU_wall_seconds=time.monotonic()-tick,score_new_GPU_calls=0,
        score_new_backbone_calls=0,score_new_expert_calls=0,GPU_initialized=torch.cuda.is_initialized(),
        score_time_excludes_development_audit_report=True,time=time.time())
    write(PUB/'RESOURCES.json',resources)
    write(BASE/'SCORE_COMPLETION.json',dict(status='scored_pending_independent_audit',time=time.time(),
        score_start_time=score_start,current_cells=288,future_cells=288,reset_cells=1152,
        files={n:sha(PUB/n) for n in ['CONFIG.json','ROWS.json','FUTURE_ROWS.json','RESET_U_ROWS.json','STEP_DIAGNOSTICS.json','SUMMARY.json']}))
    print('NEGATIVE_EVIDENCE_CPU_SCORE_COMPLETE',flush=True)


if __name__=='__main__':
    assert len(sys.argv)==1 or sys.argv[1]=='score'
    score()
