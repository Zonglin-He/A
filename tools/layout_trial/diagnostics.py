"""GT is used only here, after sealing diagnostic candidates and predictions."""
import json
import numpy as np
from scipy.stats import spearmanr
from stvg.runtime import ROOT,sha
from stvg.metrics import box_iou
from stvg.metrics_base import xyxy
from stvg.tube_runner import write
from tools.layout_trial.worker import checked_plan,records


def paired_ranking(reward,iou):
    delta=iou[:,None]-iou[None,:];rd=reward[:,None]-reward[None,:]
    mask=np.triu(np.abs(delta)>1e-6,1)
    accuracy=float(np.mean(np.where(rd[mask]*delta[mask]>0,1.,np.where(rd[mask]==0,.5,0.))))if mask.any()else None
    correlation=float(spearmanr(reward,iou).statistic)if np.ptp(iou)>1e-12 and np.ptp(reward)>1e-12 else None
    return dict(pair_accuracy=accuracy,spearman=correlation,pairs=int(mask.sum()))


def mean(values):
    a=[x for x in values if x is not None and np.isfinite(x)];return float(np.mean(a))if a else None


def diagnose(plan_path):
    p=checked_plan(plan_path);out=ROOT/p['output'];seal=json.loads((out/'diagnostic_seal.json').read_text())
    if seal['queries']!=128:raise ValueError('No partial reward diagnostic')
    for e in seal['files'].values():
        if sha(out/e['path'])!=e['sha256']:raise ValueError('Diagnostic candidates changed')
    for arm in p['adaptation_arms']:
        if not(out/'predictions'/arm/'sealed.json').exists():raise ValueError('Seal adaptations before reading GT')
    labels_path=ROOT/p['labels'];assert sha(labels_path)==p['labels_sha256']
    labels={x['id']:x for x in map(json.loads,labels_path.read_text().splitlines())};details=[]
    for row in records(p):
        truth={int(f):v for f,v in labels[row['id']]['boxes'].items()};per={k:[]for k in ['layout3','layout1','geometry']};r3r1=[]
        with np.load(out/seal['files'][row['id']]['path'],allow_pickle=False)as z:
            for i,f in enumerate(z['observed_frame_ids']):
                if int(f)not in truth:continue
                pix=xyxy(z['boxes'][i],row['metadata']['width'],row['metadata']['height']);iou=box_iou(pix,np.asarray(truth[int(f)]))
                for key in per:per[key].append(paired_ranking(z[key+'_reward'][i],iou))
                kinds=z['candidate_kinds'].tolist();t=kinds.index('teacher');shifts=[n for n,k in enumerate(kinds)if k.startswith(('shift','scale','aspect'))]
                r3r1.append(dict(layout3_self_minus_perturbed=float(z['layout3_reward'][i,t]-z['layout3_reward'][i,shifts].mean()),
                    layout1_self_minus_perturbed=float(z['layout1_reward'][i,t]-z['layout1_reward'][i,shifts].mean())))
        details.append(dict(id=row['id'],valid_GT_observations=len(per['layout3']),rewards={k:{field:mean([v[field]for v in vals])for field in ['pair_accuracy','spearman']}for k,vals in per.items()},
            self_minus_perturbed={key:mean([v[key]for v in r3r1])for key in ['layout3_self_minus_perturbed','layout1_self_minus_perturbed']}))
    result=dict(status='complete_offline_diagnostic',queries=128,eligible_queries=sum(v['valid_GT_observations']>0 for v in details),
        rewards={key:{field:mean([v['rewards'][key][field]for v in details])for field in ['pair_accuracy','spearman']}for key in ['layout3','layout1','geometry']},
        self_minus_perturbed={key:mean([v['self_minus_perturbed'][key]for v in details])for key in ['layout3_self_minus_perturbed','layout1_self_minus_perturbed']},
        GT_only_for_offline_diagnostic=True,not_independent_confirmation=True,
        aggregation='equal frames within each query then equal eligible queries; no-GT queries excluded with denominator',
        alternative_queries_not_claimed_distinct_target_identities=True,cross_attention_not_implemented=True,no_adversarial_robustness_claim=True)
    write(out/'reward_diagnostics_private.json',details);write(out/'reward_diagnostics.json',result);return result
