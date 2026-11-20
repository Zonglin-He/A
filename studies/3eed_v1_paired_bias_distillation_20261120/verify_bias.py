"""Audit the fitted correction, the matched-bias identity and both source/target panels."""
from bias import DUEL, HERE, RESULTS, WORK, atomic, digest, loaded, np, read
from verify_transfer import audit_panel, check_source


def check_distillation():
    d=loaded(DUEL/'run_PRIVATE/fit.npz')
    ii=np.flatnonzero(d['available'].all(1))
    x,y=[],[]
    max_error=0.
    for i in ii:
        a,b=d['slots'][i]
        s=d['j'][i].astype(float)
        h=d['h'][i].astype(float)
        delta=s[:,a]-s[:,b]
        nuisance=(delta[0]+delta[1])/2
        language=(delta[0]-delta[1])/2
        max_error=max(max_error,abs(min(delta[0]-nuisance,nuisance-delta[1])-language))
        for t in range(2):
            x.append(h[t,a]-h[t,b])
            y.append(nuisance)
    x,y=np.array(x),np.array(y)
    # The primal RMS-regularized solution also satisfies the augmented least-squares system.
    rms=np.sqrt((x*x).mean(0))
    design=np.vstack([x,np.sqrt(len(x))*np.diag(rms)])
    response=np.concatenate([y,np.zeros(x.shape[1])])
    estimate=np.linalg.lstsq(design,response,rcond=None)[0]
    learned=loaded(WORK/'model.npz')['weights'][1]
    norm=float(np.max(np.abs(learned-estimate)))
    assert norm<1e-8 and max_error<1e-12
    return dict(source_pair_count=len(ii),ridge_replay_max=norm,analytic_midpoint_error=float(max_error))


if __name__=='__main__':
    result=audit_panel(RESULTS,WORK)
    result.update(check_distillation())
    result.update(verified_source_rows=check_source(RESULTS,'source_all.csv.gz'),FINAL_access=False)
    lock=read(WORK/'FRESH_SEAL.json')
    assert lock['lock']==digest(WORK/'LOCK.json') and lock['predictions']==digest(WORK/'fresh_predictions.npz')
    result.update(new_source_recordings=lock['recordings'],new_source_pairs=lock['count'])
    atomic(RESULTS/'verification.json',result)
    print(result)
