"""Independent score, gradient and denominator checks using saved numeric outputs."""
import csv
import gzip
from paired import ARMS, HERE, PUBLIC, WORK, atomic, digest, initialize, np, read


def analytic_training(data, w0, arm, steps, lr):
    use = (data['slots']>=0).all(1)
    slots = data['slots'][use]
    x = data['h'][use].astype(np.float64)
    x = np.concatenate((x,np.ones((*x.shape[:-1],1))),axis=-1)
    orig = w0.astype(np.float64)
    fixed = data['j'][use].astype(np.float64) - .75*np.tanh(x@orig)
    weight = orig.copy()
    m1,m2 = np.zeros_like(orig),np.zeros_like(orig)
    idx = np.arange(len(x))[:,None]
    role = np.arange(2)[None,:]
    for step in range(1,steps+1):
        u = np.tanh(x@weight)
        score = fixed + .75*u
        marg = score[idx,role,slots]-score[idx,role,slots[:,::-1]]
        jac = .75*x*(1-u*u)[...,None]
        diff = jac[idx,role,slots]-jac[idx,role,slots[:,::-1]]
        selected = np.ones_like(marg) if arm=='mean' else (marg==marg.min(1,keepdims=True)).astype(float)
        coeff = -selected/selected.sum(1,keepdims=True)/(1+np.exp(marg))
        grad = np.einsum('ij,ijk->k',coeff,diff)/len(x)
        m1=.9*m1+.1*grad
        m2=.999*m2+.001*grad**2
        weight-=lr*(m1/(1-.9**step))/(np.sqrt(m2/(1-.999**step))+1e-8)
    return weight


def check_fit():
    cfg=read(HERE/'PROTOCOL.json')['optim']
    with np.load(WORK/'heads.npz') as z:
        heads=z['heads']
    with np.load(WORK/'fit.npz') as fit:
        diffs={}
        for idx,arm in enumerate(ARMS[1:],1):
            rebuilt=analytic_training(fit,heads[0],arm,cfg['steps'],cfg['lr'])
            diffs[arm]=float(np.max(np.abs(rebuilt-heads[idx])))
            np.testing.assert_allclose(rebuilt,heads[idx],rtol=0,atol=2e-5)
    return diffs


def rows(p):
    opener=gzip.open if p.name.endswith('.gz') else open
    with opener(p,'rt') as f:
        return list(csv.DictReader(f))


def check_source():
    s=read(PUBLIC/'source_summary.json')
    records=rows(PUBLIC/'source_pairs.csv')
    cover=rows(PUBLIC/'source_coverage.csv')
    assert len(cover)==257 and len(records)==771
    awards={}
    for split, arms in s.items():
        section=[r for r in cover if r['split']==split]
        available=sum(int(r['endpoints_in_four'])==2 for r in section)
        assert all(int(r['endpoints_in_A256'])>=int(r['endpoints_in_four']) for r in section)
        for arm in ARMS:
            select=[r for r in records if r['split']==split and r['arm']==arm]
            assert len(select)==len(section)==len({r['pair_id'] for r in select})
            measured=[r for r in select if r['both_pair_correct']!='']
            assert len(measured)==available==arms[arm]['both_available']
            signs={r['pair_id']:(float(r['m_a'])>0 and float(r['m_b'])>0) for r in measured}
            assert sum(signs.values())==arms[arm]['both_pair_correct']
            for r in section:
                if r[arm+'_both']!='':
                    assert signs[r['pair_id']]==bool(int(r[arm+'_both']))
            awards[f'{split}_{arm}']=sum(signs.values())
    return awards


def check_transfer():
    allrows=rows(PUBLIC/'target_all_rows.csv.gz')
    switch=rows(PUBLIC/'target_switches.csv')
    assert len(set(r['row'] for r in allrows))==len(allrows)==5528
    assert [r for r in allrows if int(r['mean_switch']) or int(r['worst_switch'])]==switch
    directions=rows(PUBLIC/'directions.csv')
    comp=rows(PUBLIC/'comparisons.csv')
    stats={}
    for t in ('drone','quad','pooled','two_equal'):
        rr=allrows if t in ('pooled','two_equal') else [r for r in allrows if r['target']==t]
        scope=t if t in ('pooled','two_equal') else 'waymo_to_'+t
        for a in ARMS:
            v=np.array([float(r[a+'_iou']) for r in rr])
            assert all(0<=v[i]<=float(r['four_max_iou'])+1e-12 for i,r in enumerate(rr))
            calc=([np.mean(v>.25)*100,np.mean(v>.50)*100,v.mean()*100] if t!='two_equal'
                  else np.mean([stats[k,a] for k in ('drone','quad')],axis=0))
            stats[t,a]=calc
            line=next(r for r in directions if r['scope']==scope and r['arm']==a)
            assert len(rr)==int(line['n'])
            np.testing.assert_allclose(calc,[float(line[k]) for k in ('Acc25','Acc50','mIoU')],rtol=0,atol=1e-10)
        for new,old in ((1,0),(2,0),(2,1)):
            arm,base=ARMS[new],ARMS[old]
            row=next(r for r in comp if r['scope']==scope and r['arm']==arm and r['baseline']==base)
            for threshold,tag in ((.25,'25'),(.50,'50')):
                a=np.array([float(r[arm+'_iou'])>threshold for r in rr])
                b=np.array([float(r[base+'_iou'])>threshold for r in rr])
                assert int(row['rescues'+tag])==int((a & ~b).sum())
                assert int(row['harms'+tag])==int((~a & b).sum())
                if t!='two_equal':
                    net=int(row['rescues'+tag])-int(row['harms'+tag])
                    assert abs(100*net/len(rr)-float(row['Acc'+tag+'_pp']))<1e-10
    return len(switch)


def main():
    initialize()
    delta=check_fit()
    pairs=check_source()
    n=check_transfer()
    target=read(PUBLIC/'target_audit.json')
    assert target['n_four_boxes_dual_scored']==4*5528 and target['max_archived_v1_and_oracle_error']<1e-10
    assert target['predictions_sealed_before_GT'] and not target['FINAL_access']
    files=[p for p in sorted(PUBLIC.iterdir()) if p.is_file() and p.name!='verify.json']
    atomic(PUBLIC/'verify.json',dict(checked_parameters_per_arm=97, independent_Adam_max_abs=delta,
             source_dual_summary=pairs,all_target_rows=5528,any_switch_rows=n,all_comparisons_verified=True,
             v1_registry_weights_and_implementation_verified=True,
             output_sha256={p.name:digest(p) for p in files},FINAL_access=False))
    print('Independent audit passed',delta,n,flush=True)


if __name__=='__main__':
    main()
