"""CPU-only reading of sealed predictions/traces and already published scores.

No model, RGB, annotation, new GT score, fitting, or parameter selection.
All geometric comparisons are between existing predictions, not against GT.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json, hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'artifacts/stvg_spatial_strong_bidirectional256_v1'
OLD = ROOT / 'results/stvg_error_state/2026-10-20/spatial_strong_bidirectional256'
OUT = ROOT / 'results/stvg_error_state/2026-10-20/vid_spatial_failure_cached'

def array(x):
    return x.detach().cpu().numpy().astype(float) if torch.is_tensor(x) else np.asarray(x, dtype=float)

def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def xy(b):
    return np.concatenate((b[..., :2]-b[..., 2:]/2, b[..., :2]+b[..., 2:]/2), -1)

def iou(a, b):
    a, b = xy(a), xy(b)
    inter = np.maximum(np.minimum(a[..., 2:], b[..., 2:])-np.maximum(a[..., :2], b[..., :2]), 0).prod(-1)
    union = np.maximum(a[..., 2:]-a[..., :2], 0).prod(-1)+np.maximum(b[..., 2:]-b[..., :2], 0).prod(-1)-inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

def cost(a, b):
    ax, bx = xy(a), xy(b)
    inter = np.maximum(np.minimum(ax[..., 2:], bx[..., 2:])-np.maximum(ax[..., :2], bx[..., :2]), 0).prod(-1)
    area = lambda x: np.maximum(x[..., 2:]-x[..., :2], 0).prod(-1)
    union = area(ax)+area(bx)-inter
    enclosure = np.maximum(np.maximum(ax[..., 2:], bx[..., 2:])-np.minimum(ax[..., :2], bx[..., :2]), 0).prod(-1)
    giou = inter/np.maximum(union, 1e-7)-(enclosure-union)/np.maximum(enclosure, 1e-7)
    return 5*np.abs(a-b).sum(-1)+2*(1-giou)

def group(mask, delta, overlap, ratio):
    if not mask.any():
        return dict(nodes=0)
    return dict(nodes=int(mask.sum()), center_shift_E_wh_mean=float(delta[mask].mean()),
                center_shift_E_wh_max=float(delta[mask].max()),
                self_IoU_to_E_mean=float(overlap[mask].mean()),
                median_width_height_ratio=ratio[mask].median(0).tolist() if torch.is_tensor(ratio) else np.median(ratio[mask], axis=0).tolist())

def analyze(row):
    alias = row['alias']
    predfile = INPUT / f'predictions/vidstg/{alias}.pt'
    tracefile = INPUT / f'traces/vidstg/{alias}.pt'
    p, tr = load(predfile), load(tracefile)
    fit = tr['fits']['Live_41']
    E, final, target = array(p['E']), array(fit['final']), array(fit['target'])
    n = row['GT_supported_actual_nodes']
    old_E = row['values']['Without_spatial_adaptation']['fixed_actual_sIoU']
    positions = np.asarray(fit['positions'], int)
    ids = np.asarray(p['frame_ids'], int)
    observed = np.zeros(len(E), bool); observed[positions] = True
    strong = (ids >= p['intervals']['Strong'][0]) & (ids < p['intervals']['Strong'][1])
    path = []
    cost_errors = []
    for item in fit['path']:
        b = array(item['boxes'])
        c = cost(b[positions], target)
        cost_errors.append(abs(float(c.mean())-item['anchor_loss']))
        delta = np.linalg.norm((b[:, :2]-E[:, :2])/E[:, 2:], axis=-1)
        overlap, ratio = iou(b, E), b[:, 2:]/E[:, 2:]
        lower = max(0., old_E+float(np.sort(overlap)[:n].mean())-1)
        upper_path = min(1., 1-old_E+float(np.sort(overlap)[-n:].mean()))
        path.append(dict(step=item['step'], anchor_cost=float(c.mean()),
                         per_anchor_cost=c.tolist(), state_norm=float(np.linalg.norm(array(item['state']))),
                         actual_GT_IoU_conservative_lower_bound=lower,
                         actual_GT_IoU_conservative_upper_bound=upper_path,
                         observed=group(observed, delta, overlap, ratio),
                         unobserved=group(~observed, delta, overlap, ratio),
                         Strong_support=group(strong, delta, overlap, ratio),
                         outside_Strong=group(~strong, delta, overlap, ratio)))
    assert max(cost_errors) < 1e-10, (alias, max(cost_errors))
    rounds = []
    for rr in fit['rounds']:
        w = array(rr['rollout']['weights'])
        rounds.append(dict(step=rr['step'], gradient_norm=float(np.linalg.norm(array(rr['gradient']))),
                           state_step_norm=float(np.linalg.norm(array(rr['state_step']))),
                           ESS=float(np.mean(1/(w*w).sum(-1))), max_weight=float(w.max(-1).mean())))
    # A conservative bound, NOT a newly scored teacher prediction. Jaccard distance
    # is a metric, so IoU(T,G) <= 1-IoU(E,G)+IoU(T,E). The old scorer saved only
    # the number n of GT-supported nodes, not their identities. Maximize over all
    # n-node subsets to remain valid without reading the GT mask or annotations.
    # T here is explicitly the hypothetical linear interpolation of the existing
    # eight selected references, NOT a claim about each raw observed reference.
    interpolated = np.column_stack([np.interp(ids, ids[positions], target[:, k]) for k in range(4)])
    teacher_E = iou(interpolated, E)
    upper = min(1., 1-old_E+float(np.sort(teacher_E)[-n:].mean()))
    metrics = row['values']
    # Existing final metrics must satisfy the independently derived bounds.
    assert path[-1]['actual_GT_IoU_conservative_lower_bound']-1e-12 <= metrics['Full']['fixed_actual_sIoU'] <= path[-1]['actual_GT_IoU_conservative_upper_bound']+1e-12
    return dict(alias=alias, intervals=p['intervals'], actual_nodes=len(E),
                GT_supported_actual_nodes=n, observed_GT_supported_nodes=n-row['unobserved_actual_nodes'],
                existing_metrics=metrics,
                Full_minus_E_Strong_pp=100*(metrics['Full']['v']-metrics['Without_spatial_adaptation']['v']),
                actual_node_delta_pp=100*(metrics['Full']['fixed_actual_sIoU']-old_E),
                unobserved_delta_pp=100*(metrics['Full']['unobserved_actual_sIoU']-metrics['Without_spatial_adaptation']['unobserved_actual_sIoU']),
                teacher_E_anchor_IoU=iou(target, E[positions]).tolist(),
                final_teacher_anchor_IoU=iou(final[positions], target).tolist(),
                teacher_selected_confidence=array(tr['teacher']['selected_frame_scores']).reshape(len(positions), -1).max(-1).tolist(),
                teacher_min_width_height=target[:, 2:].min(0).tolist(),
                target_at_all8_even_if_low_confidence_or_degenerate=True,
                initial_anchor_costs=path[0]['per_anchor_cost'], final_anchor_costs=path[-1]['per_anchor_cost'],
                interpolated_teacher_actual_GT_IoU_upper_bound=upper,
                bound_scope='hypothetical linear reference interpolation; unknown GT support maximized; raw anchor correctness not measured',
                trace=path, round_diagnostics=rounds,
                stored_seed_geometry={str(s):dict(mean_self_IoU_E=float(iou(array(b), E).mean()),
                    mean_center_shift_E_wh=float(np.linalg.norm((array(b)[:, :2]-E[:, :2])/E[:, 2:], axis=-1).mean())) for s,b in p['Live'].items()},
                cost_reconstruction_max_error=max(cost_errors), configuration=fit['configuration'],
                input_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [predfile, tracefile]})

def main():
    torch.set_num_threads(2)
    saved = json.loads((OLD/'RESULTS_PUBLIC.json').read_text())
    rows = saved['directions']['vidstg']['rows']
    details = [analyze(r) for r in rows]
    ordered = sorted(details, key=lambda x:x['Full_minus_E_Strong_pp'])
    failures = [x for x in details if x['Full_minus_E_Strong_pp'] < -1e-10]
    positives = [x for x in details if x['Full_minus_E_Strong_pp'] > 1e-10]
    def median(group_, fun):
        return float(np.median([fun(x) for x in group_]))
    summary = dict(status='completed_readonly_saved_evidence_analysis', parents=128, seed=41,
        no_new_model_or_GT_scoring=True, GPU_forwards=0, fitting_updates=0, RGB_reads=0, annotation_reads=0,
        existing_score_source=str((OLD/'RESULTS_PUBLIC.json').relative_to(ROOT)),
        negatives=len(failures), positives=len(positives), unchanged=128-len(failures)-len(positives),
        full_minus_E_Strong_pp=float(np.mean([x['Full_minus_E_Strong_pp'] for x in details])),
        worst_three_mean_contribution_pp=sum(x['Full_minus_E_Strong_pp'] for x in ordered[:3])/128,
        remainder125_mean_pp=float(np.mean([x['Full_minus_E_Strong_pp'] for x in ordered[3:]])),
        failure_median_teacher_cost_drop=median(failures,lambda x:x['trace'][0]['anchor_cost']-x['trace'][-1]['anchor_cost']),
        failure_cost_improved=sum(x['trace'][-1]['anchor_cost']<x['trace'][0]['anchor_cost'] for x in failures),
        failure_same_Native_Strong_interval=sum(x['intervals']['Native']==x['intervals']['Strong'] for x in failures),
        negative_median_unobserved_center_shift=median(failures,lambda x:x['trace'][-1]['unobserved']['center_shift_E_wh_mean']),
        positive_median_unobserved_center_shift=median(positives,lambda x:x['trace'][-1]['unobserved']['center_shift_E_wh_mean']),
        negative_median_ESS=median(failures,lambda x:np.mean([r['ESS']for r in x['round_diagnostics']])),
        positive_median_ESS=median(positives,lambda x:np.mean([r['ESS']for r in x['round_diagnostics']])),
        score_limitations=['No saved per-frame GT IoUs or teacher-reference GT IoUs in these caches.',
          'E+Native is absent from the five saved rows; general E-by-adaptation temporal interaction cannot be exactly decomposed.',
          'Saved path geometry and teacher cost do not establish task quality of earlier stopping steps.',
          'HC-selected hyperparameters were transferred to Vid without a Vid parameter comparison.'],
        bounds_definition=dict(distance='1-IoU, the Jaccard metric for positive-area boxes',
          lower='max(0, saved_mean_IoU(E,G) + mean_n_smallest_IoU(E,X) - 1)',
          upper='min(1, 1 - saved_mean_IoU(E,G) + mean_n_largest_IoU(E,X))',
          support='n is the old saved GT-supported-node count; maximize/minimize over all n-node subsets; no GT mask accessed',
          role='conservative bounds, not newly scored predictions, optimized parameters, or an early-stop policy'),
        decisive_cases={x['alias']:x for x in ordered[:8]+ordered[-2:]},
        all_parent_summaries=[{k:x[k]for k in ['alias','Full_minus_E_Strong_pp','actual_node_delta_pp','unobserved_delta_pp','intervals','initial_anchor_costs','final_anchor_costs']}for x in details],
        independent_cost_readback_max_abs=max(x['cost_reconstruction_max_error']for x in details),
        protected_inputs_sha256={k:v for x in details for k,v in x['input_sha256'].items()})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'ANALYSIS_PUBLIC.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    # This checksum document also makes immutable-input verification repeatable.
    for f,h in summary['protected_inputs_sha256'].items():
        assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
    print(json.dumps({k:v for k,v in summary.items()if k not in ['decisive_cases','all_parent_summaries','protected_inputs_sha256']},indent=2))
    for x in ordered[:3]+ordered[-2:]:
        print(x['alias'], x['Full_minus_E_Strong_pp'], 'cost',x['trace'][0]['anchor_cost'],x['trace'][-1]['anchor_cost'],
              'observed/unobserved motion',x['trace'][-1]['observed']['center_shift_E_wh_mean'],x['trace'][-1]['unobserved']['center_shift_E_wh_mean'],
              'teacher bound',x['interpolated_teacher_actual_GT_IoU_upper_bound'])

if __name__=='__main__':
    main()
