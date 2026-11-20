"""Recalculate published paired metrics from all public rows and verify private prediction seals."""
import csv
import gzip
from common import ARMS, HERE, PREV, RESULTS, WORK, atomic, digest, np, read
from fit_duels import loaded


def read_rows(path):
    opener = gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',newline='') as file:
        return list(csv.DictReader(file))


def audit_panel(root,work):
    records=read_rows(root/'all_target_rows.csv.gz')
    assert len(records)==5528==len({r['row'] for r in records})
    arms=[name[:-4] for name in records[0] if name.endswith('_iou')]
    rows={t:[r for r in records if r['target']==t] for t in ('drone','quad')}
    assert [len(v) for v in rows.values()]==[2792,2736]
    groups={'waymo_to_'+k:v for k,v in rows.items()}
    groups['two_equal']=records
    for group in {r['recording_group'] for r in records}:
        groups[group]=[r for r in records if r['recording_group']==group]
    observed={}
    outcomes={}
    for group,items in groups.items():
        for a in arms:
            values=np.array([float(x[a+'_iou']) for x in items])
            assert np.isfinite(values).all() and (values>=0).all() and (values<=1).all()
            outcomes[group,a]=values
            observed[group,a]=100*np.array([(values>.25).mean(),(values>.50).mean(),values.mean()])
    for a in arms:
        observed['two_equal',a]=(observed['waymo_to_drone',a]+observed['waymo_to_quad',a])/2
    lines=read_rows(root/'directions.csv')
    assert len(lines)==len(groups)*len(arms)
    for r in lines:
        np.testing.assert_allclose(observed[r['scope'],r['arm']],[float(r[k]) for k in ('Acc25','Acc50','mIoU')],atol=1e-11,rtol=0)
    for r in read_rows(root/'paired_changes.csv'):
        g,a,b=r['scope'],r['arm'],r['baseline']
        v,base=outcomes[g,a],outcomes[g,b]
        for threshold,k in ((.25,'25'),(.50,'50')):
            assert ((v>threshold)&(base<=threshold)).sum()==int(r['rescues'+k])
            assert ((base>threshold)&(v<=threshold)).sum()==int(r['harms'+k])
        assert sum(x[a+'_choice']!=x[b+'_choice'] for x in groups[g])==int(r['switches'])
        np.testing.assert_allclose(observed[g,a]-observed[g,b],[float(r[k]) for k in ('Acc25_pp','Acc50_pp','mIoU_pp')],atol=1e-11,rtol=0)
    changed=[r for r in records if any(int(r[a+'_switch']) for a in arms)]
    assert changed==read_rows(root/'changed_rows.csv')
    assert digest(work/'predictions.npz')==read(work/'PREDICTIONS_SEAL.json')['file']
    return dict(rows=len(records),methods=len(arms),groups=len(groups),retained_changed_rows=len(changed))


def analytic_heads():
    d=loaded(WORK/'fit.npz')
    use=d['available'].all(1)
    old=loaded(PREV/'run_PRIVATE/heads.npz')['heads'][0].astype(float)
    observed=loaded(WORK/'heads.npz')
    perm=observed['permutation']
    x=d['h'][use].astype(float)
    x=np.concatenate((x,np.ones((*x.shape[:-1],1))),axis=-1)
    baseline=d['j'][use]-.75*np.tanh(x@old)
    ids=np.arange(len(x))[:,None]
    caps=np.arange(2)[None,:]
    slots=d['slots'][use]
    errors={}
    for i,arm in enumerate(ARMS[1:5],1):
        w=old.copy()
        avg=np.zeros_like(w)
        variance=np.zeros_like(w)
        for step in range(1,65):
            u=np.tanh(x@w)
            logits=baseline+.75*u
            ds=logits[ids,caps,slots]-logits[ids,caps,slots[:,::-1]]
            jac=.75*x*(1-u*u)[...,None]
            ds_jac=jac[ids,caps,slots]-jac[ids,caps,slots[:,::-1]]
            if arm in ('assignment','broken_pair'):
                p=perm if arm=='broken_pair' else np.arange(len(x))
                wgt=-1/(1+np.exp(ds[:,0]+ds[p,1]))
                grad=(wgt[:,None]*(ds_jac[:,0]+ds_jac[p,1])).mean(0)
            else:
                active=np.ones_like(ds) if arm=='independent' else (ds==ds.min(1,keepdims=True)).astype(float)
                coeff=-active/active.sum(1,keepdims=True)/(1+np.exp(ds))
                grad=(coeff[...,None]*ds_jac).sum(1).mean(0)
            avg=.9*avg+.1*grad
            variance=.999*variance+.001*grad*grad
            w-=.001*(avg/(1-.9**step))/(np.sqrt(variance/(1-.999**step))+1e-8)
        errors[arm]=float(np.max(np.abs(w-observed['heads'][i])))
        assert errors[arm]<2e-5
    return errors


def check_source(results,filename):
    rows=read_rows(results/filename)
    summary=read_rows(results/'source_summary.csv')
    for r in summary:
        group=[x for x in rows if all(x[k]==r[k] for k in ('split','probe','arm'))]
        assert len(group)==int(r.get('potential',r.get('n')))
        assert len(group)==len({x['pair'] for x in group})
        with_pair=[x for x in group if x['m_a']!='']
        assert len(with_pair)==int(r.get('covered',r.get('available')))
        assert sum(min(float(x['m_a']),float(x['m_b']))>0 for x in with_pair)==int(r['both'])
        assert sum(float(x['m_a'])+float(x['m_b'])>0 for x in with_pair)==int(r.get('true_assignment',r.get('assignment_correct')))
    return len(rows)


if __name__=='__main__':
    panel=audit_panel(RESULTS,WORK)
    panel.update(analytic_Adam_max_errors=analytic_heads(),verified_source_rows=check_source(RESULTS,'source_pairs.csv.gz'),
                 FINAL_access=False)
    atomic(RESULTS/'verification.json',panel)
    print(panel)
