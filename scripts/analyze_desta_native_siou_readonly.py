"""Read saved DESTA Dev16 scores and sealed endpoints; never run a model/scorer.

No media/annotation pool is opened. GT intervals/counts below are read from the
already published scoring report. Coordinate distances compare predictions to
saved expert evidence, not to GT boxes. Run on one CPU thread with CUDA hidden.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/desta3d_v3/latent_oracle_v1/dual_expert_native_v1'
OUT = D/'siou_readonly_diagnosis_v1'
ARM = 'K1_R0.03_T2'
pins = {}


def pin(p):
    p = Path(p)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    pins[str(p.relative_to(ROOT))] = h
    return h


def read(p):
    pin(p)
    return json.loads(p.read_text())


def sealed(p):
    manifest = read(p.parent/'COMPLETE.json')['files']
    assert pin(p) == manifest[p.name], p
    return torch.load(p, map_location='cpu', weights_only=False)


def xyxy(pred):
    result = {}
    for j, pos in enumerate(pred['positions']):
        if bool(pred['geometry_valid'][j]):
            b = pred['boxes_cxcywh'][j].double().numpy()
            result[pos] = np.r_[b[:2]-b[2:]/2, b[:2]+b[2:]/2]
    return result


def distribution(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values)) if values else dict(n=0)


def main():
    cfg = read(D/'CONFIG.json')
    report = read(D/'scores/dev16/REPORT.json')
    roster = read(D/'DEV64.json')
    registration = read(D/'REGISTRATION.json')['pins']
    for p in (D/'CONFIG.json',D/'DEV64.json',D/'QC.pt'):
        assert pin(p) == registration[str(p)]
    assert pin(D/'scores/dev16/REPORT.json') == read(D/'scores/dev16/COMPLETE.json')['pins'][str(D/'scores/dev16/REPORT.json')]
    base_metrics = {r['key']: r['metrics'] for r in report['rows']['B1']}
    pin(D/'QC.pt')
    q = torch.load(D/'QC.pt', map_location='cpu', weights_only=False).double().numpy()
    gram = q.T@q
    inv = np.linalg.inv(gram)
    rows = []
    max_projection_crosscheck = 0.
    for n, score in enumerate(report['rows'][ARM], 1):
        idx = next(i for i, r in enumerate(roster) if r['key'] == score['key'])
        assert idx in cfg['dev16_indices']
        ep = D/'native'/f'{idx:02}'
        base = sealed(ep/'identity/B1.pt')
        after = sealed(ep/ARM/'PREDICTION.pt')
        step = sealed(ep/ARM/'STEP01.pt')
        se = read(D/'experts/spatial'/f'{idx:02}'/'EVIDENCE.json')
        te = read(D/'experts/temporal'/f'{idx:02}'/'EVIDENCE.json')
        for branch in ('spatial','temporal'):
            p = D/'experts'/branch/f'{idx:02}'/'EVIDENCE.json'
            assert pin(p) == read(p.parent/'COMPLETE.json')['files'][p.name]
        m0, m1 = base_metrics[score['key']], score['metrics']
        assert base['key'] == after['key'] == score['key'] == se['key'] == te['key']
        assert base['frame_ids'] == after['frame_ids'] == se['frame_ids'] == te['frame_ids']
        assert base['adapter_sha'] == after['adapter_sha'] and base['support'] == after['support']
        boxes0, boxes1 = xyxy(base), xyxy(after)
        common = sorted(set(boxes0)&set(boxes1))
        distances = []
        for p in common:
            expert = se['boxes_by_frame'].get(str(base['frame_ids'][p]))
            if expert is not None:
                distances.append(dict(position=p, physical_frame=base['frame_ids'][p],
                    before=float(np.abs(boxes0[p]-expert).mean()),
                    after=float(np.abs(boxes1[p]-expert).mean())))
        gt_begin, gt_end = m0['gt_interval_physical']
        # This is only interval membership. Do not invent missing GT box flags.
        event_positions = [p for p, fid in enumerate(base['frame_ids']) if gt_begin <= fid < gt_end]
        all_event_frames_box_valid = len(event_positions) == m0['spatial_support_frames']
        projection = {}
        for branch in ('event', 'spatial'):
            p = step['projected'][branch].double().numpy().reshape(-1, 16)
            full_norm = float(step['full_norms'][branch])
            energy = float(np.sum((p@inv)*p))
            pt = torch.from_numpy(p)
            alternate = float((torch.linalg.solve(torch.from_numpy(gram), pt.T).T*pt).sum())
            max_projection_crosscheck = max(max_projection_crosscheck, abs(energy-alternate))
            rec = step['records'][branch]
            projection[branch] = dict(full_norm=full_norm, projected_energy=energy,
                energy_fraction=energy/full_norm**2 if full_norm else None,
                norm_fraction=energy**.5/full_norm if full_norm else None,
                missing_or_disabled=bool(rec.get('missing_or_disabled',False)),
                actions=rec.get('actions', 0), loss=rec.get('loss'), classes=rec.get('classes'))
        rows.append(dict(case=f'Q{n:02}', index=idx, key=score['key'], source=score['source'],
            before=m0, after=m1, delta_pp={k:100*(m1[k]-m0[k]) for k in ('tIoU','sIoU','vIoU')},
            reference_equal=base['readout']['spatial_reference_token_ids']==after['readout']['spatial_reference_token_ids'],
            interval_positions=[base['interval'],after['interval']], positions=[base['positions'],after['positions']],
            format=[bool(base['format_ok']),bool(after['format_ok'])],
            invalid_boxes=[int((~base['geometry_valid']).sum()),int((~after['geometry_valid']).sum())],
            observed_event_positions_from_saved_interval=event_positions,
            saved_GT_count_equals_all_event_frames=all_event_frames_box_valid,
            boxes_on_all_observed_event_frames=[sorted(set(event_positions)&set(boxes0)), sorted(set(event_positions)&set(boxes1))],
            common_valid_expert_coordinate_L1=distances,
            common_valid_expert_coordinate_L1_mean=[statistics.mean(x['before'] for x in distances),statistics.mean(x['after'] for x in distances)] if distances else None,
            expert_interval=te['interval_physical'], expert_anchor=se['anchor'],
            expert_box_area_distribution=distribution([(b[2]-b[0])*(b[3]-b[1]) for b in se['boxes_by_frame'].values() if b]),
            initial_projection=projection))
    ds = [r['delta_pp']['sIoU'] for r in rows]
    worst = min(range(len(rows)), key=lambda i: ds[i])
    marginals = {}
    for factor in ('steps','radius','temporal_weight'):
        marginals[factor] = {}
        for value in sorted({x[factor] for x in cfg['grid'].values()}):
            arms = [a for a,c in cfg['grid'].items() if c[factor]==value]
            marginals[factor][str(value)] = {m:100*statistics.mean(z['metrics'][m] for a in arms for z in report['rows'][a]) for m in ('tIoU','sIoU','vIoU')}
    summary = dict(scope='CPU readback only; no model, backward, optimizer, native inference, new GT scoring, video, annotation pool or fresh data',
        selected_arm=ARM, selection='historical Dev16 vIoU-best; no new selection',
        samples=16, parents=16, exposure='previously exposed source validation; offline GT selection; not fresh evaluation',
        siou_delta_pp=distribution(ds), positive=sum(x>0 for x in ds), negative=sum(x<0 for x in ds), unchanged=sum(x==0 for x in ds),
        worst_case=rows[worst]['case'], worst_contribution_to_macro_pp=ds[worst]/16,
        worst_fraction_of_net_decline=ds[worst]/sum(ds),
        excluding_worst_descriptive_only_pp=statistics.mean(x for i,x in enumerate(ds) if i!=worst),
        reference_unchanged=sum(r['reference_equal'] for r in rows),
        interval_changed=sum(r['interval_positions'][0]!=r['interval_positions'][1] for r in rows),
        format_valid_before_after=[sum(r['format'][j] for r in rows) for j in (0,1)],
        invalid_boxes_before_after=[sum(r['invalid_boxes'][j] for r in rows) for j in (0,1)],
        all27_sIoU_above_B1=sum(statistics.mean(z['metrics']['sIoU'] for z in report['rows'][a])>statistics.mean(z['metrics']['sIoU'] for z in report['rows']['B1']) for a in cfg['grid']),
        factorial_marginal_percent=marginals, initial_unique_query_projection={b:{f:distribution([r['initial_projection'][b][f] for r in rows if r['initial_projection'][b][f] is not None]) for f in ('energy_fraction','norm_fraction')} for b in ('event','spatial')},
        projection_formula='||P_col(Q) g||^2 = sum((p @ inv(Q.T @ Q))*p), p=g@Q; initial only, one entry per query, not full trajectories',
        missing_branch_cases={b:[r['case'] for r in rows if r['initial_projection'][b]['missing_or_disabled']] for b in ('event','spatial')},
        projection_numpy_torch_max_abs=max_projection_crosscheck)
    OUT.mkdir(exist_ok=True)
    for name,value in [('ROOT_READBACK.json',dict(summary=summary, rows=rows)),('INPUT_HASHES.json',pins)]:
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
