"""Posthoc CPU GT diagnosis of sealed quick logs; no inference or method edits."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import json
import math
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from scripts.tastvg_best_quick_common_v1 import BASE, read, sha, load, loadz
from scripts.score_tastvg_best_quick_v1 import truth, iou
from vg_tta.box_stability_diagnostics_v1 import overlap
OUT = ROOT / 'artifacts/tastvg_quick_deep_diagnosis_cpu_v1'


def quantiles(values):
    return dict(n=len(values), values=np.quantile(values, [0, .25, .5, .75, 1]).tolist()) if values else dict(n=0, values=[])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(ds):
    start = time.time(); b = BASE / ds; out = OUT / ds
    assert read(BASE / 'FINAL_COMPLETION.json')['status'] == 'completed_verified_published'
    assert read(b / 'ROOT_READBACK.json')['status'] == 'pass'
    bar = read(b / 'PREDICTION_BARRIER.json')
    assert sha(b / 'PREDICTION_BARRIER.json') == read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')['datasets'][ds]
    p = read(b / 'PLAN.json'); rows = read(b / 'ROWS.json'); steps = read(b / 'SPATIAL_STEP_ROWS.json')
    dense, spans, provenance = truth(ds, p)
    write(out / 'GT_EXPOSURE.json', dict(time=time.time(), previous_global_seal_sha256=sha(BASE / 'GLOBAL_PREDICTION_BARRIER.json'),
                                       sources=provenance, scope='Same sealed quick cohort; posthoc CPU only; no reselection'))
    step_index = {(s['condition'], s['order'], s['arrival'], s['step']): s for s in steps}
    detailed = []; references = []; temporal = []; payloads = 0
    for r in rows:
        if not r['expert_scheduled']:
            continue
        rf = b / 'online' / r['condition'] / r['order'] / f"{r['arrival']:05}.pt.json"
        assert sha(rf) == bar['files'][str(rf.relative_to(b))]
        assert sha(rf.with_suffix('.gz')) == read(rf)['sha256']
        x = loadz(rf.with_suffix('.gz')); payloads += 1
        row = p['rows'][r['parent']]; gt = dense[r['parent']]; span = spans[r['parent']]
        evidence = {}
        for stage in ('spatial', 'temporal'):
            er = read(b / 'experts' / stage / r['condition'] / f"{r['parent']:05}.json")
            f = b / 'experts' / er['cache']; assert sha(f) == er['cache_sha256']
            evidence[stage] = load(f)
        e = evidence['spatial']; valid = np.asarray(e['valid'], bool)
        sampled = np.flatnonzero(valid); inside = [i for i in sampled if row['frame_ids'][i] in gt]
        ref_iou = []
        for i in inside:
            a = np.asarray(gt[row['frame_ids'][i]], float)
            a /= [row['input']['width'], row['input']['height']] * 2
            target = np.r_[(a[:2] + a[2:]) / 2, a[2:] - a[:2]]
            ref_iou.append(float(overlap(e['boxes'][i], target)))
        references.append(dict(source_id=r['source_id'], parent=r['parent'], condition=r['condition'], order=r['order'], arrival=r['arrival'],
                               valid_frames=len(sampled), inside_GT_frames=len(inside), reference_GT_IoU=float(np.mean(ref_iou)) if ref_iou else None))
        for j, trace in enumerate(x['update_steps']):
            s = step_index[r['condition'], r['order'], r['arrival'], j]; u = trace['update']
            z = {k:s[k] for k in ('source_id','parent','condition','order','arrival','step','pre_v','post_v','delta_update','valid_frames','flat_rewards','loss_decreased')}
            if u is not None:
                reward = np.asarray(trace['rewards']); ranked = np.sort(reward)[::-1]
                pp = np.asarray(u['p'], float); qq = np.asarray(u['q'], float)
                q_entropy = float(-np.sum(qq[qq > 0]*np.log(qq[qq > 0])))
                correct = wrong = tied = 0
                for a in range(len(reward)):
                    for c in range(a+1, len(reward)):
                        g = s['candidate_v'][a] - s['candidate_v'][c]
                        if abs(g) <= 1e-10: continue
                        v = reward[a] - reward[c]
                        if abs(v) <= 1e-12: tied += 1
                        elif (v > 0) == (g > 0): correct += 1
                        else: wrong += 1
                preference_gt_gain = s['teacher_top_best_v'] - s['pre_v']
                z.update(updated=trace['updated'], reward_range=float(np.ptp(reward)), top_gap=float(ranked[0]-ranked[1]),
                         q_top=float(qq.max()), q_entropy=q_entropy, p_entropy=float(-np.sum(pp[pp>0]*np.log(pp[pp>0]))),
                         p_native_is_max=bool(pp[0] >= pp.max()-1e-7), global_gradient_norm=u['global_gradient_norm'],
                         step_norm=float(math.sqrt(sum(float(((trace['post_state'][n]-v).double()**2).sum()) for n,v in trace['pre_state'].items()))),
                         teacher_top_GT_gain=preference_gt_gain, useful_top_but_harm=bool(preference_gt_gain>1e-10 and s['delta_update'] < -1e-10),
                         teacher_pairs_correct=correct, teacher_pairs_wrong=wrong, teacher_pairs_tie=tied,
                         loss_before=u['loss_before'],loss_after=u['loss_after'])
            else:
                z['updated'] = False
            detailed.append(z)
        e = evidence['temporal']; td = x['temporal']; selected = td['selected']; c = td['candidates'][selected]
        contributors = [iou(c['physical_interval'], a)*v for a,v in zip(e['proposals'], e['proposal_confidence'])]
        wi = int(np.argmax(contributors)) if contributors else None
        temporal.append(dict(source_id=r['source_id'],parent=r['parent'],condition=r['condition'],order=r['order'],arrival=r['arrival'],
                             pre_v=r['slow_v'],post_v=r['final_v'],delta=r['delta_temporal_rerank'],
                             oracle_v=r['temporal_best_v'],best_teacher_t=r['best_teacher_t'],winning_teacher_t=r['winning_teacher_t'],
                             winner_confidence=e['proposal_confidence'][wi] if wi is not None else None,
                             selected=selected,candidate_scores=td['scores'],native_score=td['scores'][0],
                             all_candidates_v=r['temporal_candidate_v'],all_candidates_t=r['temporal_candidate_t']))
        if payloads % 200 == 0: print('CPU_LOGS',ds,payloads,flush=True)
    summary = {}
    for group in ('corruption', 'clean'):
        keep = lambda r: (r['condition'] != 'clean') == (group == 'corruption')
        ss = [r for r in detailed if keep(r) and 'reward_range' in r]; first = [r for r in ss if r['step']==0]
        rr = [r for r in references if keep(r)]; tt = [r for r in temporal if keep(r)]
        stats = dict(scheduled=len(rr), updated_steps=len(ss), first_updated_steps=len(first),
                     references=dict(empty=sum(r['valid_frames']==0 for r in rr),nonempty_but_outside_GT=sum(r['valid_frames']>0 and r['inside_GT_frames']==0 for r in rr),
                                     within_GT_IoU=quantiles([r['reference_GT_IoU'] for r in rr if r['reference_GT_IoU'] is not None])),
                     first_step=dict(reward_range=quantiles([r['reward_range'] for r in first]), top_gap=quantiles([r['top_gap'] for r in first]),
                                     q_top=quantiles([r['q_top'] for r in first]), native_is_p_max=sum(r['p_native_is_max'] for r in first),
                                     useful_teacher_top_but_GT_harm=sum(r['useful_top_but_harm'] for r in first)),
                     flat_steps=dict(count=sum(r['flat_rewards'] for r in ss),nonzero_gradient=sum(r['flat_rewards'] and r['global_gradient_norm']>0 for r in ss),
                                     GT_harm=sum(r['flat_rewards'] and r['delta_update']< -1e-10 for r in ss),
                                     net_GT_delta_pp=100*sum(r['delta_update'] for r in ss if r['flat_rewards'])),
                     weak_reward_bands={str(eps):dict(first_steps=sum(r['reward_range']<=eps for r in first),
                                                    harm=sum(r['reward_range']<=eps and r['delta_update']< -1e-10 for r in first)) for eps in (.001,.01,.05)},
                     teacher_pair_counts={k:sum(r[k] for r in first) for k in ('teacher_pairs_correct','teacher_pairs_wrong','teacher_pairs_tie')},
                     temporal={str(t):dict(destroyed=sum(r['pre_v']>t>=r['post_v'] for r in tt),
                                           good_proposal_but_bad_winner=sum(r['pre_v']>t>=r['post_v'] and r['best_teacher_t']>.5>=r['winning_teacher_t'] for r in tt),
                                           correct_t_candidate_but_none_correct_joint=sum(max(r['all_candidates_t'])>t and r['oracle_v']<=t for r in tt)) for t in (.3,.5)})
        rr = [r for r in rows if keep(r)]
        stats['arrival_quartiles'] = []
        for q in range(4):
            arr = [r for r in rr if min(3,4*r['arrival']//p['sources']) == q]
            stats['arrival_quartiles'].append(dict(quartile=q+1,cells=len(arr),inherit_box_delta_pp=100*np.mean([r['delta_inherited_boxes'] for r in arr]),
                                                   harm=sum(r['delta_inherited_boxes']< -1e-10 for r in arr)))
        summary[group] = stats
    cases = {}
    for name, seq, field in [('temporal',temporal,'delta'),('spatial',detailed,'delta_update')]:
        seq = [r for r in seq if r['condition']!='clean']
        cases[name] = dict(negative=sorted(seq,key=lambda r:r[field])[:4],positive=sorted(seq,key=lambda r:-r[field])[:4])
    assert not torch.cuda.is_initialized()
    write(out/'DEEP_STEPS.json',detailed);write(out/'REFERENCE_ROWS.json',references);write(out/'TEMPORAL_ROWS.json',temporal)
    write(out/'SUMMARY.json',summary);write(out/'CASES.json',cases)
    write(out/'COMPLETION.json',dict(status='cpu_analysis_complete',verified_payloads=payloads,rows=len(rows),steps=len(steps),GPU_initialized=False,
                                    model_forward_calls=0,parameter_updates=0,new_predictions=0,seconds=time.time()-start,
                                    input_hashes={n:sha(b/n) for n in ('ROWS.json','SPATIAL_STEP_ROWS.json','PREDICTION_BARRIER.json','ROOT_READBACK.json')},
                                    script_sha256=sha(Path(__file__)),time=time.time()))
    print(ds,json.dumps(summary),flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
