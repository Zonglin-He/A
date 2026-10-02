"""Anonymous-only independent aggregation, selection and severe-tail validation."""
import json,sys,collections
from pathlib import Path
import numpy as np
def audit(directory):
    directory=Path(directory);count=0
    def read(f):return json.loads((directory/f).read_text())
    def check(x):
        nonlocal count
        assert x;count+=1
    def close(a,b):check(abs(a-b)<1e-10)
    def reproduce(rows,saved):
        check(len(rows)==saved['cells'])
        if not rows:check(saved['sources']==0);return
        fields=list(saved['metrics']);cells=collections.defaultdict(list)
        for r in rows:cells[r['source_id'],r['order'],r['condition']].append([r[k] for k in fields])
        ordercells=collections.defaultdict(list)
        for (s,o,c),v in cells.items():ordercells[s,o].append(np.mean(v,axis=0))
        sources=collections.defaultdict(list);orders=collections.defaultdict(list)
        for (s,o),v in ordercells.items():a=np.mean(v,axis=0);sources[s].append(a);orders[o].append(a)
        m=np.array([np.mean(sources[s],axis=0) for s in sorted(sources)]);rng=np.random.default_rng(20261001)
        boots=np.concatenate([m[rng.integers(0,len(m),(100,len(m)))].mean(1) for _ in range(100)])
        ci=np.quantile(boots,[.025,.975],axis=0);ov=np.array([np.mean(orders[o],axis=0) for o in sorted(orders)])
        check(len(m)==saved['sources'])
        for j,k in enumerate(fields):
            z=saved['metrics'][k];close(z['mean'],m[:,j].mean());close(z['query_macro'],np.mean([r[k] for r in rows]))
            for a,b in zip(z['ci95'],ci[:,j]):close(a,b)
            for a,b in zip(z['order_values'],ov[:,j]):close(a,b)
            if len(ov)>1:close(z['order_sample_SD'],ov[:,j].std(ddof=1))
            if k.startswith('delta_'):check(z['harm_gt5pp_sources']==int((m[:,j]<-.05).sum()))
    for stage in ['round1','round2']:
        for split in ['search','confirm']:
            prefix=f'{stage}/{split}';groups={}
            if not (directory/prefix).exists():continue
            for ds in ['vidstg','hc2']:
                base=prefix+'/'+ds;rows=read(base+'/ROWS.json');wr=read(base+'/WRITE_ROWS.json');summary=read(base+'/SUMMARY.json')
                cases=read(base+'/CASES.json')
                expected=384 if split=='search' else 192;check(len(rows)==expected and len(wr)==expected//4)
                check(len({(r['condition'],r['order'],r['arrival']) for r in rows})==expected)
                arms=list(summary['corruption']['all']['effects']);groups[ds]=rows
                for arm in arms:
                    rr=[r for r in rows if r['condition']!='clean'];field='delta_'+arm+'_v'
                    check(cases[arm]['positive']==sorted(rr,key=lambda r:-r[field])[:5])
                    check(cases[arm]['negative']==sorted(rr,key=lambda r:r[field])[:5])
                for r in rows:
                    check(r['expert_scheduled']==(r['arrival']%4==0))
                    for a in arms:
                        for metric in ['v','t','s']:check(0<=r[a+'_'+metric]<=1+1e-12)
                        close(r['delta_'+a+'_v'],r[a+'_v']-r['A_v'])
                        if not r['expert_scheduled']:close(r['delta_'+a+'_v'],0.)
                for w in wr:
                    for b,e in w['evidence'].items():
                        r=e['rewards'];sel=e['selection'];top=[] if r is None else np.flatnonzero(np.array(r)==max(r)).tolist()
                        check(sel['selected']==(top[0] if len(top)==1 else 0));check(sel['top_ties']==top)
                        close(e['selected_v'],w['probe_v'][sel['selected']]);check(0<=e['event_frame_precision']<=1)
                        pair=e['critic_pairwise'];accuracy=[]
                        for z in pair['pairs']:
                            i,j=z['i'],z['j'];a=0 if r is None else np.sign(r[i]-r[j]) if abs(r[i]-r[j])>1e-12 else 0
                            v=w['probe_v'][i]-w['probe_v'][j];g=np.sign(v) if abs(v)>1e-12 else 0
                            if g:accuracy.append(.5 if a==0 else float(a==g))
                        if accuracy:close(pair['pairwise_accuracy'],np.mean(accuracy))
                    if stage=='round2':
                        tr=w['temporal']['rule'];sc=np.array(tr['scores']);dv=sc-sc[:,:1];worst=dv.min(0)
                        np.testing.assert_allclose(dv,tr['differences'],atol=1e-12,rtol=0);np.testing.assert_allclose(worst,tr['worst'],atol=1e-12,rtol=0)
                        top=np.flatnonzero(worst==worst.max());k=int(top[0]) if len(top)==1 and worst[top[0]]>0 else 0
                        check(k==tr['selected']);check(k==0 or np.all(dv[:,k]>0))
                        r=next(r for r in rows if (r['condition'],r['order'],r['arrival'])==(w['condition'],w['order'],w['arrival']))
                        tm=w['temporal']
                        for tag,arm in [('old','A'),('new','T')]:
                            check(tm[tag+'_replaced']==(tm['original_selected' if tag=='old' else 'new_selected']!=0))
                            for metric in ['v','t']:
                                check(tm[tag+'_wrong_replacement_'+metric]==(tm[tag+'_replaced'] and r[arm+'_'+metric]<r['A_native_'+metric]-1e-12))
                for group in ['clean','corruption']:
                    for sub in ['all','expert','nonexpert']:
                        rr=[r for r in rows if (r['condition']=='clean')==(group=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
                        target=summary[group][sub];reproduce(rr,target)
                        for a,e in target['effects'].items():
                            f='delta_'+a+'_v';check(e['positive']==sum(r[f]>1e-12 for r in rr));check(e['negative']==sum(r[f]<-1e-12 for r in rr))
                            check(e['zero']==sum(abs(r[f])<=1e-12 for r in rr));check(e['harm_gt5pp']==sum(r[f]<-.05 for r in rr))
                            reproduce([dict(r,gross_gain=max(r[f],0.),gross_loss=max(-r[f],0.)) for r in rr],e['gross'])
                            for t in [.3,.5]:
                                c=e['correctness'][str(t)];check(c['rescued']==sum(r['A_v']<=t<r[a+'_v'] for r in rr));check(c['destroyed']==sum(r[a+'_v']<=t<r['A_v'] for r in rr))
            pooled=read(prefix+'/POOLED.json')
            for a,z in pooled.items():
                arrays=[]
                for ds,rr in groups.items():
                    rr=[r for r in rr if r['condition']!='clean'];m=np.array([np.mean([r['delta_'+a+'_v'] for r in rr if r['source_id']==s]) for s in sorted({r['source_id'] for r in rr})]);arrays.append(m)
                    close(z['dataset_means'][ds],m.mean())
                close(z['mean'],np.mean([m.mean() for m in arrays]));rng=np.random.default_rng(20261001);boots=[]
                for i in range(100):boots.extend(.5*sum(m[rng.integers(0,len(m),(100,len(m)))].mean(1) for m in arrays))
                for x,y in zip(z['ci95'],np.quantile(boots,[.025,.975])):close(x,y)
            if (directory/prefix/'CONTRASTS.json').exists():
                diag=read(prefix+'/CONTRASTS.json')
                for name,gg in diag['contrasts'].items():
                    contrast,metric=name.rsplit('_',1);left,right=contrast.split('_minus_')
                    for group,ss in gg.items():
                        for sub,saved in ss.items():
                            matrices=[]
                            for ds,rr in groups.items():
                                rr=[r for r in rr if (r['condition']=='clean')==(group=='clean') and
                                    (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
                                cells=collections.defaultdict(list)
                                for r in rr:cells[r['source_id'],r['order'],r['condition']].append(r[left+'_'+metric]-r[right+'_'+metric])
                                orders=collections.defaultdict(list)
                                for (s,o,c),vv in cells.items():orders[s,o].append(np.mean(vv))
                                sources=collections.defaultdict(list)
                                for (s,o),vv in orders.items():sources[s].append(np.mean(vv))
                                ids=sorted(sources);m=np.array([np.mean(sources[s]) for s in ids]);matrices.append(m)
                                z=saved['datasets'][ds];check(z['cells']==len(rr) and z['sources']==len(m));close(z['mean'],m.mean())
                                rng=np.random.default_rng(20261001);boots=np.concatenate([m[rng.integers(0,len(m),(100,len(m)))].mean(1) for _ in range(100)])
                                for x,y in zip(z['ci95'],np.quantile(boots,[.025,.975])):close(x,y)
                                for x,s,v in zip(z['source_effects'],ids,m):check(x['source_id']==s);close(x['delta'],v)
                                loo=(m.sum()-m)/(len(m)-1)
                                for x,y in zip(z['leave_one_source_out_range'],[min(loo),max(loo)]):close(x,y)
                                check(z['leave_one_source_out_sign_changes']==sum(v*m.mean()<0 for v in loo))
                            z=saved['pooled'];close(z['mean'],np.mean([m.mean() for m in matrices]));rng=np.random.default_rng(20261001);boots=[]
                            for _ in range(100):boots.extend(.5*sum(m[rng.integers(0,len(m),(100,len(m)))].mean(1) for m in matrices))
                            for x,y in zip(z['ci95'],np.quantile(boots,[.025,.975])):close(x,y)
    if (directory/'ROUND1_SELECTION.json').exists():
        s=read('ROUND1_SELECTION.json');e=s['effects'];order=['A','U_select','R_select','R_temp','Specific_temp']
        eligible=[a for a in order if e[a]['mean']>0 and min(e[a]['dataset_means'].values())>=0]
        check(s['eligible']==eligible);check(s['correction']==(max(eligible,key=lambda a:e[a]['mean']) if eligible else 'A'))
    if (directory/'FINAL_SELECTION.json').exists():
        s=read('FINAL_SELECTION.json');e=s['effects'];a=s['acquisition_effect'];valid=(s['correction'].startswith('R') or s['correction']=='Specific_temp') and a['mean']>0 and min(a['dataset_means'].values())>=0
        check(s['new_acquisition_eligible']==bool(valid));eligible=[k for k in s['candidates'] if e[k]['mean']>0 and min(e[k]['dataset_means'].values())>=0]
        check(s['final']==(max(eligible,key=lambda k:e[k]['mean']) if eligible else 'A'))
    if (directory/'COHORT_CONFIG.json').exists():
        cfg=read('COHORT_CONFIG.json')
        for ds,z in cfg.items():
            check(z['historical_exposure'] and z['confirmation_disjoint_within_batch'])
            ss=z['splits'];search=set(ss['search']['orders']['order1']);confirm=set(ss['confirm']['orders']['order1'])
            check(not(search&confirm));check(len(search)==32 and len(confirm)==16)
            for split,z in ss.items():
                check(z['anonymous_source_count']==len(z['orders']['order1']))
                check(set(z['orders']['order1'])==set(z['orders']['order2']))
                for stage in ['round1','round2']:
                    rr=read(f'{stage}/{split}/{ds}/ROWS.json')
                    check({r['source_id'] for r in rr}==set(z['orders']['order1']))
    print('Anonymous public audit PASS',count)
    return dict(status='pass',checks=count,private_inputs=False)
if __name__=='__main__':audit(sys.argv[1])
