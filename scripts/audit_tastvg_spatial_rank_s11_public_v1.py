"""Public-only independent reconstruction of three-arm paired results and teachers."""
import sys,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.audit_tastvg_spatial_s06_s1_public_v1 import online,check_stats,means

def run(root,raw=None):
    root=Path(root);raw=Path(raw) if raw else root/'raw';read=lambda p:json.loads(Path(p).read_text());arms={a:read((raw if a=='raw' else root/a)/'ROWS.json') for a in ['raw','rank','norm']};paired=read(root/'ROWS.json');summary=read(root/'SUMMARY.json');checks=0
    for a,rows in arms.items():
        checks+=online(raw if a=='raw' else root/a)['scalar_checks']
        if a=='raw':continue
        for r in rows:
            if not r['updated']:continue
            u=r['update_diagnostics'];rew=np.array(u['reward']);order=np.argsort(-rew);ranks=np.empty(9);start=0
            while start<9:
                end=start+1
                while end<9 and rew[order[start]]-rew[order[end]]<=1e-12:end+=1
                ranks[order[start:end]]=np.mean(np.arange(start,end));start=end
            q=np.exp(-ranks);q/=sum(q);d=np.array(u['distances']);p=np.exp(-d+d.min());p/=sum(p)
            np.testing.assert_array_equal(ranks,u['rank']);np.testing.assert_allclose(q,u['q'],atol=1e-7,rtol=0);np.testing.assert_allclose(p,u['p'],atol=1e-7,rtol=0);assert abs(np.sum(p*np.log(p/q))-u['loss_before'])<1e-6;checks+=28
            if a=='norm':assert abs(r['step_displacement']-.015765801926049215)<2e-7;checks+=1
    for i,row in enumerate(paired):
        for a in arms:assert all(arms[a][i][k]==row[k] for k in ['parent','condition','arrival','expert_scheduled'])
        for key,val in row.items():
            if '_minus_' not in key:continue
            a,tail=key.split('_minus_');b,m=tail.rsplit('_',1)
            expected=arms[a][i]['online_'+m]-(arms[a][i]['frozen_'+m] if b=='frozen' else arms[b][i]['online_'+m]);assert abs(expected-val)<1e-14;checks+=1
    for group,subsets in summary.items():
        seq=[r for r in paired if (r['condition']!='clean' if group=='corruption' else r['condition']==group)]
        for subset,s in subsets.items():
            rows=[r for r in seq if subset=='all' or (subset=='nonexpert' and not r['expert_scheduled']) or (subset=='expert' and r['expert_scheduled']) or (subset=='nonexpert_after_first_write' and not r['expert_scheduled'] and r['arrival']>4)]
            assert len(rows)==s['cells'] and len({r['parent'] for r in rows})==s['sources']
            for key,st in s['metrics'].items():checks+=check_stats(means(rows,key),st)
    for r in read(root/'SOURCE_EFFECTS.json'):
        seq=[p for p in paired if p['parent']==r['parent'] and not p['expert_scheduled'] and (p['condition']!='clean' if r['group']=='corruption' else p['condition']=='clean')]
        for k,v in r.items():
            if '_minus_' in k:assert abs(np.mean([p[k] for p in seq])-v)<1e-14;checks+=1
    return dict(status='pass',scalar_checks=checks,scope='Public three-arm scalar differences, source bootstrap, rank distributions and step magnitude; raw-state/gradient reconstruction separately audited')

if __name__=='__main__':print(json.dumps(run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None),indent=2))
