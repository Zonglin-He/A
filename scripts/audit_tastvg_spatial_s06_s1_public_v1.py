"""Public scalar-only reconstruction of critic ordering and online aggregation."""
import sys,json,collections,itertools
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())

def check_stats(values,got):
    a=np.asarray(values,float);assert len(a)==got['n']
    if not len(a):assert got['mean'] is None;return 1
    assert abs(a.mean()-got['mean'])<1e-12
    boot=a[np.random.default_rng(20260929).integers(len(a),size=(10000,len(a)))].mean(1)
    np.testing.assert_allclose(np.quantile(boot,[.025,.975]),got['ci95'],atol=1e-12,rtol=0);return 4

def means(rows,key):
    by=collections.defaultdict(list)
    for r in rows:
        if r[key] is not None:by[r['parent']].append(r[key])
    return [np.mean(by[p]) for p in sorted(by)]

def critic(root):
    root=Path(root);rows=read(root/'ROWS.json');summary=read(root/'SUMMARY.json');cuts=read(root/'MARGIN_CUTS.json');saved=read(root/'PAIRS.json');expected=[];checks=0
    for r in rows:
        gt=np.asarray(r['gt_sIoU']);rew=r['rewards'];selection=int(np.argmax(rew)) if rew is not None else 0;assert r['selected']==selection
        assert abs(r['selected_gain']-(gt[selection]-gt[0]))<1e-12;checks+=2
        if rew is None:continue
        group='clean' if r['condition']=='clean' else 'corruption'
        for i,j in itertools.combinations(range(9),2):
            de=rew[i]-rew[j];dg=gt[i]-gt[j];es=np.sign(de) if abs(de)>1e-12 else 0;gs=np.sign(dg) if abs(dg)>1e-12 else 0;anti=(i,j) in [(1,2),(3,4),(5,6),(7,8)]
            acc=None if gs==0 else (.5 if es==0 else float(gs==es));p=dict(parent=r['parent'],condition=r['condition'],i=i,j=j,antithetic=anti,accuracy=acc,decisive_accuracy=float(es==gs) if es and gs else None,sign_agreement=float(es==gs),all_bin=['low','mid','high'][np.searchsorted(cuts[group]['all'],abs(de),side='right')],antithetic_bin=['low','mid','high'][np.searchsorted(cuts[group]['antithetic'],abs(de),side='right')] if anti else None);expected.append(p)
    assert len(expected)==len(saved)
    for a,b in zip(expected,saved):
        for k,v in a.items():assert b[k]==v;checks+=1
    def pair_check(ps,got):
        nonlocal checks
        assert len(ps)==got['pairs'];by=collections.defaultdict(list)
        for p in ps:by[p['parent'],p['condition']].append(p)
        for key,outkey in [('accuracy','accuracy'),('decisive_accuracy','decisive_accuracy'),('sign_agreement','literal_sign_agreement')]:
            rr=[]
            for (parent,cond),seq in by.items():
                vals=[p[key] for p in seq if p[key] is not None];rr.append(dict(parent=parent,value=float(np.mean(vals)) if vals else None))
            checks+=check_stats(means(rr,'value'),got[outkey])
    for group,s in summary.items():
        condition=lambda r:r['condition']!='clean' if group=='corruption' else r['condition']==group
        rr=[r for r in rows if condition(r)];pp=[p for p in expected if condition(p)];anti=[p for p in pp if p['antithetic']]
        pair_check(pp,s['all_pairs']);pair_check(anti,s['antithetic'])
        for b in ['low','mid','high']:pair_check([p for p in pp if p['all_bin']==b],s['all_margin'][b]);pair_check([p for p in anti if p['antithetic_bin']==b],s['antithetic_margin'][b])
        for k,(i,j) in enumerate([(1,2),(3,4),(5,6),(7,8)]):pair_check([p for p in anti if p['i']==i and p['j']==j],s['directions'][str(k+1)])
        for key,got in s['metrics'].items():checks+=check_stats(means(rr,key),got)
    return dict(status='pass',cells=96,pairs=len(expected),scalar_checks=checks,scope='Independent reward/GT scalar ordering, source aggregation and confidence intervals')

def online(root):
    root=Path(root);rows=read(root/'ROWS.json');summary=read(root/'SUMMARY.json');checks=0
    for r in rows:
        for m in ['s','t','v']:
            for name,a,b in [('online_minus_frozen','online','frozen'),('online_minus_budgeted','online','budgeted'),('post_minus_pre','post','online')]:
                v=None if r[a+'_'+m] is None or r[b+'_'+m] is None else r[a+'_'+m]-r[b+'_'+m];assert r[name+'_'+m]==v;checks+=1
    for group,subs in summary.items():
        rr=[r for r in rows if (r['condition']!='clean' if group=='corruption' else r['condition']==group)]
        for subset,s in subs.items():
            seq=[r for r in rr if subset=='all' or r['expert_scheduled']==(subset=='expert')];assert len(seq)==s['cells']
            for key,got in s['metrics'].items():checks+=check_stats(means(seq,key),got)
    return dict(status='pass',cells=96,scalar_checks=checks,scope='Independent prequential scalar aggregation; state and SGD verified separately')

if __name__=='__main__':print(json.dumps((critic if sys.argv[1]=='critic' else online)(sys.argv[2]),indent=2))
