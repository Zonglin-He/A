"""Independent conditional energy, normalized mass, chart mean, utility and metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha,load
from scripts.run_stvg_correction_distribution64_v1 import D,P,ARMS,verify


def main():
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,box_iou
    p=verify();assert read(D/'CPU_COMPLETION.json')['count']==64;assert sha(P/'RESULTS.json')==read(D/'CPU_COMPLETION.json')['result_sha256']
    result=read(P/'RESULTS.json');by={r['alias']:r for r in result['rows']};checks=collections.Counter();maxerr=0.
    targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets}
    for r in p['records']:
        z=load(D/'predictions'/(r['alias']+'.pt'));old=load(ROOT/r['prior_prediction']);base=z['outputs']['Source_MBR'];assert np.array_equal(base,old['outputs']['Source_no_correction'])and np.array_equal(z['outputs']['E_MBR'],old['outputs']['E_DINO_support']);checks['old_output_exact']+=2
        d=z['decision'];h=d['distribution'];energies=[];deltas=[];setting=z['setting'];sc=np.array(setting['scale']);cap=setting['truncation'];weights=np.zeros(len(h['deltas']))
        for rec in h['assignments']:
            residual=[];scorecost=0.;outliers=0
            for f,k in zip(z['frames'],rec['assignment']):
                if k<0:outliers+=1;continue
                b=f['boxes'][k];s=base[f['position']];residual.append(np.r_[b[:2]-s[:2],np.log(b[2:])-np.log(s[2:])]);v=np.maximum(np.array(f['scores']),1e-12);scorecost+=np.log(v.max()/v[k])
            delta=np.array(residual).mean(0)if residual else np.zeros(4)
            e=outliers*cap+sum(float(np.dot((x-delta)/sc,(x-delta)/sc))for x in residual)+cap/16*scorecost
            assert abs(e-rec['energy'])<1e-8 and np.max(abs(delta-rec['delta']))<1e-12;energies.append(e);deltas.append(delta);checks['conditional_assignments']+=1
        ee=np.array(energies);w=np.exp(-(ee-ee.min())/cap);w/=w.sum();assert np.max(abs(w-h['assignment_weights']))<1e-12
        for j,ids in enumerate(h['memberships']):weights[j]=sum(w[i]for i in ids)
        weights/=weights.sum();assert np.max(abs(weights-h['weights']))<1e-12;checks['working_weights']+=1
        md=np.sum(np.array(deltas)*w[:,None],0);expected=np.c_[base[:,:2]+md[:2],base[:,2:]*np.exp(md[2:])];err=float(np.max(abs(expected-z['outputs']['Correction_mean'])));assert err<1e-12;maxerr=max(maxerr,err);checks['mean_outputs']+=1
        tubes=d['candidate_tubes'];u=np.zeros((len(tubes),len(tubes)));ids=np.array(z['frame_ids']);iv=z['interval'];edges=np.r_[ids[0],(ids[:-1]+ids[1:])/2,ids[-1]+1];tw=np.maximum(0,np.minimum(edges[1:],iv[1])-np.maximum(edges[:-1],iv[0]));tw/=tw.sum()
        # A different rowwise vectorization checks every candidate pair and time.
        xy=np.concatenate((tubes[:,:,:2]-tubes[:,:,2:]/2,tubes[:,:,:2]+tubes[:,:,2:]/2),2)
        for i in range(len(tubes)):
            u[i]=np.sum(box_iou(xy[i][None],xy)*tw[None],axis=1)
        err=float(np.max(abs(u-d['utility'])));assert err<1e-12;maxerr=max(maxerr,err);checks['utility_entries']+=u.size
        eu=u@weights;assert np.max(abs(eu-d['expected_utilities']))<1e-12
        # Floating summation can differ below machine precision in ties; require
        # the selected expected utility to equal the independent maximum.
        assert abs(eu[d['selected_index']]-eu.max())<1e-12
        assert np.array_equal(z['outputs']['Correction_MBR'],tubes[d['selected_index']]);checks['actual_candidate_MBR']+=1
        ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q];mask=np.array([int(t)in gt for t in ids])&~np.isin(np.arange(len(ids)),z['planned_positions'])
        for a,b in z['outputs'].items():
            m=official(b,row,gt,span,iv,ds);saved=by[r['alias']]['values'][a]
            for k in ['v','t','s']:assert abs(m[k]-saved[k])<2e-10;checks['metrics']+=1
            pp=np.concatenate((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1)*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':pp=np.maximum(pp,0)
            if mask.any():assert abs(float(box_iou(pp[mask],np.array([gt[int(t)]for t in ids[mask]])).mean())-saved['unobserved_actual_s'])<2e-10
            else:assert saved['unobserved_actual_s']is None
            checks['unobserved_actual_support']+=1
        assert len({v['t']for v in by[r['alias']]['values'].values()})==1;checks['same_time_invariant']+=1
    for ds,g in result['summary'].items():
        rr=[r for r in result['rows']if r['dataset']==ds];assert len(rr)==32
        for a,eps in g['query_mean'].items():
            for ep,v in eps.items():
                vv=[r['values'][a][ep]for r in rr if r['values'][a][ep]is not None];assert v is None and not vv or v is not None and abs(v-sum(vv)/len(vv))<1e-12;checks['aggregates']+=1
        for pair,eps in g['paired'].items():
            a,c=pair.split('_minus_')
            for ep,saved in eps.items():
                v=np.array([100*(r['values'][a][ep]-r['values'][c][ep])for r in rr if r['values'][a][ep]is not None and r['values'][c][ep]is not None]);assert np.max(abs(v-saved['per_parent_pp']))<1e-12
                ix=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)));ci=np.quantile(v[ix].mean(1),[.025,.975]);assert np.max(abs(ci-saved['CI95']))<1e-12
                assert saved['severe_loss_below_minus5pp']==int((v< -5).sum());checks['paired_CIs_and_harm']+=1
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=64,checks=dict(checks),max_formula_error=maxerr,new_model_calls=0,new_DINO_calls=0,new_Student_updates=0,working_distribution_not_calibrated=True))
    put(D/'ROOT_READBACK_COMPLETION.json',dict(status='pass',count=64,actual_pixels_privacy_remote_archive_pending=True))


if __name__=='__main__':main()
