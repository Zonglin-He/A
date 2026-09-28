"""CPU interpretation of sealed, read-only source-training query diagnostics."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta3d_v2_event_query_audit import OUT, ARMS, read, sha, write
from scripts import score_desta3d_v2_reference_audit as scalar
from scripts.desta3d_source_fit_v1 import _frozen_prediction, _tube_metric


def relative(a, b):
    a, b = a.float(), b.float()
    return float((a-b).norm() / ((a.norm()+b.norm())/2).clamp_min(1e-12))


def main():
    torch.set_num_threads(2)
    done = read(OUT/'COMPLETE.json'); assert done['status']=='sealed_readonly_source_predictions'
    assert done['seal_sha'] == sha(OUT/'PREDICTIONS_SEAL.json')
    seal=read(OUT/'PREDICTIONS_SEAL.json');assert len(seal['pins'])==16
    for path,digest in seal['pins'].items():assert sha(Path(path))==digest
    grid=read(OUT/'PHYSICAL_GRID_AUDIT.json');assert grid['actual_pixels_visual_grid_physical_times_equal_within_pairs']
    rows=read(OUT/'INPUTS.json');assert all(r['split']=='train' for r in rows)
    preds={a:[torch.load(OUT/'predictions'/a/f'{i:03}.pt',map_location='cpu',weights_only=False) for i in range(4)] for a in ARMS}
    for a,ps in preds.items():
        for r,p in zip(rows,ps):
            assert p['prediction']['key']==r['key'] and p['prediction']['adapter_sha']==seal['adapter_hashes'][a]
            assert p['prediction']['frame_ids']==r['input']['frame_ids']
    write(OUT/'PRE_LABEL_AUDIT.json',{'time':time.time(),'sealed_files':16,'source_queries':4,'source_parents':2,
          'pixel_grid_identity':True,'all_prediction_and_adapter_identities_passed':True,'target_GT_read':False})
    labels=read(ROOT/'artifacts/desta3d_v1/SOURCE_LABELS_TRAINING_ONLY.json')
    barrier=read(ROOT/'artifacts/desta3d_v1/SOURCE_FULL_FEATURE_BARRIER.json')
    baseline=[]
    for r in rows:
        p=_frozen_prediction(r,barrier)
        baseline.append(scalar.score_tube_independently(p,labels[r['key']]))
    output=[]; max_geometry_error=0.
    for pair_index in range(2):
        i=2*pair_index;ra,rb=rows[i:i+2]
        ya,yb=[bool(v) for v in labels[ra['key']]['event_active']],[bool(v) for v in labels[rb['key']]['event_active']]
        assert len(ya)==len(yb) and not any(x and y for x,y in zip(ya,yb))
        pair={'source':ra['source'],'keys':[ra['key'],rb['key']],'sampled_event_active_indices':[[j for j,y in enumerate(ys) if y] for ys in (ya,yb)],
              'frozen_tube_metrics':baseline[i:i+2], 'arms':{}}
        for arm in ARMS:
            pa,pb=preds[arm][i:i+2];ea,eb=pa['evidence'],pb['evidence']
            la,lb=ea['event_logits'].float().reshape(-1),eb['event_logits'].float().reshape(-1)
            pr_a,pr_b=la.sigmoid(),lb.sigmoid()
            corr=float(np.corrcoef(la.numpy(),lb.numpy())[0,1]) if la.std()>0 and lb.std()>0 else None
            native=[]
            for p,r in ((pa,ra),(pb,rb)):
                score=scalar.score_tube_independently(p['prediction'],labels[r['key']]);native.append(score)
                second=_tube_metric(p['prediction'],labels[r['key']])
                for m in scalar.METRICS:max_geometry_error=max(max_geometry_error,abs(score[m]-second[m]))
            own=(scalar.binary_bce(ya,la.tolist())+scalar.binary_bce(yb,lb.tolist()))/2
            swapped=(scalar.binary_bce(ya,lb.tolist())+scalar.binary_bce(yb,la.tolist()))/2
            pair['arms'][arm]={'probability_mean_abs_difference':float((pr_a-pr_b).abs().mean()),
                'probability_max_abs_difference':float((pr_a-pr_b).abs().max()), 'logit_curve_correlation':corr,
                'event_text_pool_relative_change':relative(ea['z_event'],eb['z_event']),
                'spatial_text_pool_relative_change':relative(ea['z_spatial'],eb['z_spatial']),
                'referent_probability_mean_abs_difference':float((ea['referent_logits'].float().sigmoid()-eb['referent_logits'].float().sigmoid()).abs().mean()),
                'own_curve_BCE':own,'swapped_curve_BCE':swapped,'swapped_minus_own_BCE':swapped-own,
                'own_event_AUROC':[scalar.binary_auc(ya,la.tolist()),scalar.binary_auc(yb,lb.tolist())],
                'native_predicted_intervals':[pa['prediction']['interval'],pb['prediction']['interval']],
                'native_tube_metrics':native, 'delta_relative_norm':[ea['delta_relative_norm'],eb['delta_relative_norm']]}
        output.append(pair)
    assert max_geometry_error<2e-6
    result={'time':time.time(),'status':'completed_source_diagnostic','scope':'two metadata-selected source-training parents/four queries; all16 fixed-state read-only outputs',
        'selection_not_by_effect':True,'optimizer_steps':0,'target_GT_read':False,
        'interpretation':'Disjoint annotation intervals do not imply semantically exclusive captions. Swapped-curve BCE is a descriptive sensitivity diagnostic, not negative-label training or a deployment criterion.',
        'pairs':output,'maximum_scalar_vs_tensor_geometry_error':max_geometry_error,
        'seal_sha':done['seal_sha'],'script_sha':sha(Path(__file__))}
    write(OUT/'REPORT.json',result)
    lines=['# Same-video query sensitivity (source diagnostic)','',result['interpretation'],'',
        '| Parent | State | Mean probability change | Event pool relative change | Swapped minus own BCE | Native intervals |',
        '|---|---|---:|---:|---:|---|']
    for p in output:
        for a,z in p['arms'].items():lines.append(f"|{p['source']}|{a}|{z['probability_mean_abs_difference']:.6f}|{z['event_text_pool_relative_change']:.4f}|{z['swapped_minus_own_BCE']:+.6f}|{z['native_predicted_intervals']}|")
    lines += ['', 'All actual pixels, visual-grid tensors and physical time grids matched within each pair. No weights changed. Four source-training queries are a small diagnostic, not an independent efficacy sample.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
