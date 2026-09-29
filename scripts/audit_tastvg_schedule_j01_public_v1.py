"""Public-only schedule, source statistics, equal-order variance and Tier0 reconstruction."""
import sys,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.audit_tastvg_spatial_s06_s1_public_v1 import check_stats,means

def run(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text());rows=read('ROWS.json');summary=read('SUMMARY.json');across=read('ACROSS_ORDERS.json');old=read('J0_REFERENCE.json');incl=read('INCLUDING_J0.json');orders=read('ORDERS.json')['orders'];checks=0
    for r in rows:
        expected=orders[r['order']]['sequence'][r['arrival']];got=r['parent'] if isinstance(r['parent'],str) else f"Q{r['parent']+1:02}"
        assert got==expected and r['expert_scheduled']==(r['arrival'] in [0,4,8,12]);assert r['current_spatial_pre_update']
        for m in ['s','t','v']:
            for a,b in [('fast','frozen'),('slow','frozen'),('final','frozen'),('final','fast'),('final','slow')]:assert abs(r[f'{a}_minus_{b}_{m}']-(r[a+'_'+m]-r[b+'_'+m]))<1e-14;checks+=1
            assert abs(r['interaction_'+m]-(r['final_'+m]-r['fast_'+m]-r['slow_'+m]+r['frozen_'+m]))<1e-14;checks+=1
            if not r['expert_scheduled']:assert r['final_'+m]==r['slow_'+m] and r['fast_'+m]==r['frozen_'+m];checks+=2
        assert r['final_s']==r['slow_s'] and r['fast_s']==r['frozen_s']
        if r['expert_scheduled']:
            for field in ['temporal','fast_control']:
                d=r['temporal_diagnostics'][field];assert int(np.argmax(d['scores']))==d['selected'];checks+=1
        if r['updated']:
            u=r['update_diagnostics'];q=np.exp(-np.asarray(u['rank']));q/=q.sum();d=np.array(u['distances']);p=np.exp(-d+d.min());p/=p.sum()
            np.testing.assert_allclose(q,u['q'],atol=1e-7,rtol=0);np.testing.assert_allclose(p,u['p'],atol=1e-7,rtol=0);assert abs(np.sum(p*np.log(p/q))-u['loss_before'])<1e-6;checks+=19
    for order,groups in summary.items():
        for group,subsets in groups.items():
            seq=[r for r in rows if r['order']==order and (r['condition']!='clean' if group=='corruption' else r['condition']==group)]
            for subset,s in subsets.items():
                rr=[r for r in seq if subset=='all' or r['expert_scheduled']==(subset=='expert')];assert len(rr)==s['cells']
                for key,st in s['metrics'].items():checks+=check_stats(means(rr,key),st)
    def desc(values,got):
        a=np.array(values);np.testing.assert_allclose(a,got['values'],atol=1e-14,rtol=0)
        for k,v in [('mean',a.mean()),('sample_std',a.std(ddof=1)),('min',a.min()),('max',a.max())]:assert abs(v-got[k])<1e-14
        for k,v in [('positive',int((a>0).sum())),('negative',int((a<0).sum())),('zero',int((a==0).sum()))]:assert v==got[k]
        return 8
    for group,subsets in across.items():
        for subset,metrics in subsets.items():
            for key,st in metrics.items():
                values=[summary[o][group][subset]['metrics'][key]['mean'] for o in orders];checks+=desc(values,st)
                if group in incl:checks+=desc(values+[old[group][subset]['metrics'][key]['mean']],incl[group][subset][key])
    source=read('TIER0_SOURCE_ROWS.json');subsets=read('TIER0_SUBSETS.json');tier=read('TIER0.json')['subset_distribution'];gains=[]
    for ids,s in zip(itertools.combinations(range(16),4),subsets):
        assert s['parents']==[source[i]['parent'] for i in ids];v=float(np.mean([source[i]['v_gain'] for i in ids]));assert abs(v-s['subset_v_gain'])<1e-14;gains.append(v);checks+=2
    gains=np.array(gains);assert len(gains)==1820
    for q,v in tier['quantiles'].items():assert abs(np.quantile(gains,float(q))-v)<1e-14;checks+=1
    assert int((gains<0).sum())==tier['negative_count'];assert abs(np.mean(gains<0)-tier['negative_fraction'])<1e-14
    current=tier['current_subset_v_gain'];assert abs(np.mean(gains<current)*100-tier['current_percentile_strict'])<1e-12;assert abs(np.mean(gains<=current)*100-tier['current_percentile_inclusive'])<1e-12
    margin=read('TIER0_MARGIN_ROWS.json')
    for group,g in read('TIER0.json')['margin_descriptive'].items():
        rr=[r for r in margin if group=='all96' or r['expert_scheduled']];x=np.array([r['margin'] for r in rr]);y=np.array([r['v_gain'] for r in rr]);q=np.quantile(x,.75)
        assert abs(np.corrcoef(x,y)[0,1]-g['pearson'])<1e-14 and abs(q-g['upper_quartile_margin'])<1e-14
        assert abs(y[x>=q].mean()-g['upper_quartile_v_gain'])<1e-14;checks+=3
    return dict(status='pass',cells=480,orders=5,scalar_checks=checks,scope='Fixed roster/order membership, four-arm differences, within-order source bootstrap, between-order descriptive variance, finite1820subsets and pooled margin diagnostics')

if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
