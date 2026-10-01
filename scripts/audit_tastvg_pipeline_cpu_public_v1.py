"""Independent anonymous-scalar checks, no GT, predictions, model, or torch."""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np

def audit(base):
    rows=json.loads((base/'ROWS.json').read_text());summ=json.loads((base/'SUMMARY.json').read_text());n=0
    def eq(a,b):
        nonlocal n
        np.testing.assert_allclose(a,b,atol=1e-10,rtol=0);n+=np.asarray(a).size
    for r in rows:
        eq(r['delta_total'],r['final_v']-r['frozen_v'])
        eq(r['delta_inherited_total'],r['slow_v']-r['frozen_v'])
        eq(r['delta_inherited_boxes']+r['delta_inherited_interval'],r['delta_inherited_total'])
        eq(r['delta_temporal_rerank']+r['delta_inherited_total'],r['delta_total'])
        if r['expert_scheduled']:
            cm=r['candidate_metrics'];eq(len(cm),r['candidate_count']);eq(np.argmax(r['critic_scores']),r['selected'])
            for k in ['v','t']:
                eq(cm[0][k],r['slow_'+k]);eq(cm[r['selected']][k],r['final_'+k]);eq(max(c[k] for c in cm),r['candidate_best_'+k]);eq(r['candidate_best_'+k]-r['final_'+k],r['delta_temporal_oracle_'+k])
        else: eq(r['slow_v'],r['final_v'])
    for group in summ:
        for sub,z in summ[group].items():
            rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
            eq(len(rr),z['cells']);eq(len({r['parent'] for r in rr}),z['sources'])
            for field,m in z['metrics'].items():
                buckets=defaultdict(lambda:defaultdict(list))
                for r in rr:buckets[r['parent']][r['order']].append(r[field])
                a=np.array([np.mean([np.mean(v) for _,v in sorted(buckets[s].items())]) for s in sorted(buckets)])
                eq(a.mean(),m['mean']); rng=np.random.default_rng(20260930)
                boot=np.concatenate([a[rng.integers(0,len(a),(500,len(a)))].mean(axis=1) for _ in range(20)])
                eq(np.quantile(boot,[.025,.975]),m['ci95'])
            for stage,a,b in [('inherited_boxes','frozen_v','boxes_only_v'),('inherited_interval','boxes_only_v','slow_v'),('inherited_total','frozen_v','slow_v'),('temporal_rerank','slow_v','final_v'),('total','frozen_v','final_v')]:
                t=z['transitions'][stage];d=np.array([r[b]-r[a] for r in rr]);eq(sum(d< -1e-10),t['degraded']);eq(sum(d>1e-10),t['improved']);eq(-np.minimum(d,0).mean()*100,t['gross_loss_pp']);eq(np.maximum(d,0).mean()*100,t['gross_gain_pp'])
                for threshold in [.3,.5]:
                    before=np.array([r[a]>threshold for r in rr]);after=np.array([r[b]>threshold for r in rr]);c=t[str(threshold)]
                    eq(sum(before),c['correct_before']);eq(sum(before&~after),c['correct_to_wrong']);eq(sum(~before&after),c['wrong_to_correct'])
                    if before.any():eq(sum(before&~after)/sum(before),c['conditional_damage_rate'])
            if sub=='expert':
                for k in ['v','t']:
                    a=np.array([r['candidate_best_'+k] for r in rr]);b=np.array([r['final_'+k] for r in rr]);c=z['temporal_selection'][k]
                    eq(sum(a-b>1e-10),c['suboptimal_cells']);eq((a-b).mean()*100,c['oracle_gap_pp'])
                    for threshold in [.3,.5]:
                        h=c['thresholds'][str(threshold)];eq(sum(a>threshold),h['support_has_correct']);eq(sum((a>threshold)&(b<=threshold)),h['missed_correct'])
                        if (a>threshold).any():eq(sum((a>threshold)&(b<=threshold))/sum(a>threshold),h['miss_rate'])
    if (base/'CRITIC_ROWS.json').exists():
        cr=json.loads((base/'CRITIC_ROWS.json').read_text());cs=json.loads((base/'CRITIC_SUMMARY.json').read_text())
        lookup={(r['parent'],r['order'],r['condition'],r['arrival']):r for r in rows}
        for r in cr:
            original=lookup[r['parent'],r['order'],r['condition'],r['arrival']]
            for k in ['slow_v','final_v','slow_t','final_t','candidate_best_v','candidate_best_t','selected']:eq(r[k],original[k])
            eq(r['native_score'],original['critic_scores'][0]);eq(r['selected_score'],original['critic_scores'][original['selected']]);eq(r['oracle_v_score'],original['critic_scores'][r['oracle_v_index']])
        for group in cs:
            rr=[r for r in cr if (r['condition']!='clean')==(group=='corruption')]
            subsets={'scheduled':rr,'native_correct_destroyed_v03':[r for r in rr if r['slow_v']>.3 and r['final_v']<=.3],'correct_candidate_missed_v03':[r for r in rr if r['candidate_best_v']>.3 and r['final_v']<=.3],'correct_candidate_missed_t05':[r for r in rr if r['candidate_best_t']>.5 and r['final_t']<=.5]}
            for key,seq in subsets.items():
                z=cs[group][key];eq(len(seq),z['cells']);eq(sum(r['oracle_at_top_tie'] for r in seq),z['oracle_top_tie']);eq(sum(r['best_teacher_tIoU']>.5 and r['winning_teacher_tIoU']<=.5 for r in seq),z['good_teacher_exists_but_bad_winner']);eq(sum(r['best_teacher_tIoU']<=.5 for r in seq),z['no_teacher_tIoU_above05'])
    return dict(status='pass',scalar_checks=int(n),cells=len(rows),GT_read=False,model_execution=False)

if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1])),indent=2))
