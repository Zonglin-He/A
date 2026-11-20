from intervene import HERE, RESULTS, WORK, atomic, digest, loaded, np, read
import csv
import gzip


def main():
    with gzip.open(RESULTS/'source_all.csv.gz','rt') as f:
        records=list(csv.DictReader(f))
    with (RESULTS/'source_summary.csv').open() as f:
        summary=list(csv.DictReader(f))
    model=loaded(WORK/'model.npz')
    lock=read(WORK/'LOCK.json')
    assert lock['model']==digest(WORK/'model.npz') and lock['code']==digest(HERE/'intervene.py')
    errs=[]
    design=[]
    labels=[]
    for split in ('fit','calibration','fresh'):
        z=loaded(WORK/(split+'.npz'))
        for s,delta,slots in zip(z['j'],z['effect'],z['slots']):
            assert np.all(np.diagonal(delta,axis1=-2,axis2=-1)==0)
            errs.append(np.max(np.abs(delta+delta.transpose(0,2,1))))
            if split=='fit' and min(slots)>=0:
                i,j=sorted(slots)
                for x in delta[:,i,j]:
                    design.append(x-model['center'][i,j])
                    labels.append((s[:,i]-s[:,j]).mean()-model['rank'][i,j])
    beta=np.linalg.lstsq(np.array(design)[:,None],labels,rcond=None)[0][0]
    np.testing.assert_allclose(beta,model['beta'],rtol=0,atol=1e-11)
    for r in summary:
        matches=[x for x in records if x['split']==r['split'] and x['arm']==r['arm']]
        assert len(matches)==len({x['pair'] for x in matches})==int(r['n'])
        for key in ('available','direct_edge_both','both','both_four'):
            assert sum(int(x[key]) for x in matches)==int(r[key])
        mse=np.mean([float(x['nuisance_sqerr']) for x in matches if x['nuisance_sqerr']])
        np.testing.assert_allclose(mse,float(r['nuisance_mse']),rtol=0,atol=1e-12)
    assert max(errs)==0
    atomic(RESULTS/'verification.json',dict(pair_rows=len(records),cohort_arm_rows=len(summary),
         max_edge_antisymmetry_error=max(errs),source_coefficient_independent=float(beta),
         precommitted_source_condition_passed=False,target_run=False,FINAL_access=False))


if __name__=='__main__':
    main()
