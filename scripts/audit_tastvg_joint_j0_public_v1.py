"""Recompute J0 paired/source effects, CIs and schedule from public scalars only."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.audit_tastvg_spatial_s06_s1_public_v1 import check_stats,means


def run(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text());rows=read('ROWS.json');summary=read('SUMMARY.json');checks=0
    for r in rows:
        assert r['expert_scheduled']==(r['arrival'] in [0,4,8,12]);assert r['current_spatial_pre_update']
        for metric in ['s','t','v']:
            for a,b in [('fast','frozen'),('slow','frozen'),('final','frozen'),('final','fast'),('final','slow')]:assert abs(r[f'{a}_minus_{b}_{metric}']-(r[a+'_'+metric]-r[b+'_'+metric]))<1e-14;checks+=1
            assert abs(r['interaction_'+metric]-(r['final_'+metric]-r['fast_'+metric]-r['slow_'+metric]+r['frozen_'+metric]))<1e-14;checks+=1
            if not r['expert_scheduled']:assert r['final_'+metric]==r['slow_'+metric] and r['fast_'+metric]==r['frozen_'+metric];checks+=2
        assert r['final_s']==r['slow_s'] and r['fast_s']==r['frozen_s']
        if r['expert_scheduled']:
            for field in ['temporal','fast_control']:
                d=r['temporal_diagnostics'][field];assert int(np.argmax(d['scores']))==d['selected'] and len(d['scores'])==d['candidate_count'];checks+=2
        if r['updated']:
            u=r['update_diagnostics'];q=np.exp(-np.asarray(u['rank']));q/=sum(q);d=np.asarray(u['distances']);p=np.exp(-d+d.min());p/=sum(p);np.testing.assert_allclose(q,u['q'],atol=1e-7,rtol=0);np.testing.assert_allclose(p,u['p'],atol=1e-7,rtol=0);assert abs(np.sum(p*np.log(p/q))-u['loss_before'])<1e-6;checks+=19
    for group,subsets in summary.items():
        rr=[r for r in rows if (r['condition']!='clean' if group=='corruption' else r['condition']==group)]
        for subset,s in subsets.items():
            seq=[r for r in rr if subset=='all' or (subset=='expert' and r['expert_scheduled']) or (subset=='nonexpert' and not r['expert_scheduled']) or (subset=='nonexpert_after_first_write' and not r['expert_scheduled'] and r['arrival']>4)]
            assert len(seq)==s['cells'] and len({r['parent'] for r in seq})==s['sources']
            for k,v in s['metrics'].items():checks+=check_stats(means(seq,k),v)
    for s in read('SOURCE_EFFECTS.json'):
        seq=[r for r in rows if r['parent']==s['parent'] and (r['condition']!='clean' if s['group']=='corruption' else r['condition']=='clean') and (s['subset']=='all' or r['expert_scheduled']==(s['subset']=='expert'))]
        for k,v in s.items():
            if '_minus_' in k or k.startswith('interaction_'):assert abs(np.mean([r[k] for r in seq])-v)<1e-14;checks+=1
    reference=read('TEMPORAL_REFERENCE_ROWS.json');diagnosis=read('TEMPORAL_DIAGNOSIS.json')
    for r in reference:
        assert int(np.argmax(r['candidate_scores']))==r['selected']
        for short,key in [('s','sIoU'),('t','tIoU'),('v','vIoU_corrected')]:
            native=r['candidate_metrics'][0][key];rerank=r['candidate_metrics'][r['selected']][key];oracle=max(v[key] for v in r['candidate_metrics']);assert abs(r['native_'+short]-native)<1e-14 and abs(r['rerank_'+short]-rerank)<1e-14
            assert abs(r['historical_full_rerank_gain_'+short]-(rerank-native))<1e-14 and abs(r['historical_candidate_oracle_gain_'+short]-(oracle-native))<1e-14;checks+=4
    for group,subsets in diagnosis['groups'].items():
        rr=[r for r in reference if (r['condition']!='clean' if group=='corruption' else r['condition']=='clean')]
        for subset,z in subsets.items():
            seq=[r for r in rr if subset=='all' or r['expert_scheduled']==(subset=='scheduled')];assert len(seq)==z['cells']
            for k,v in z['metrics'].items():checks+=check_stats(means(seq,k),v)
    return dict(status='pass',cells=len(rows),scalar_checks=checks,scope='Four-arm differences, per-source aggregation/bootstrap, online-vs-immediate output identities, teacher selection, spatial probabilities; raw parameter/GT metrics audited separately')

if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
