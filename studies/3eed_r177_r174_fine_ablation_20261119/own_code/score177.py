"""Score new selections with the independently audited, unchanged subject slots."""
from collections import Counter
from math import fsum

import numpy as np

from scope177 import (ARMS,N,ORDER,ORIGINAL,PLATFORMS,PRIOR,PRIVATE,R174,RESULTS,
                      json_atomic,npz_atomic,read,sha,update_state)
from predict177 import BENCHMARK,OUTPUT,verify
from score174 import write_csv
from model177 import OPTIONS

METRICS = ('Acc25','Acc50','mIoU')
COHORTS = ('all','direct_left_right','reference_mention','no_reference')
TRAINED = ('unary','self_only','cross_only','full')
EPS = 1e-12


def collect(identity, replay_seal):
    source, target = identity
    route = f'{source}_to_{target}'
    current,old = OUTPUT/route,ORIGINAL/route
    here = read(current/'SEAL.json')
    base = read(old/'SEAL.json')
    assert sha(old/'SEAL.json')==here['original_seal']
    scores = PRIOR/'score'/f'{route}.npz'
    assert sha(scores)==replay_seal['score_arrays'][route]
    with np.load(scores,allow_pickle=False) as a:
        paired = {k:a[k] for k in ('ordinal','ious','choice','slots','flags','recording')}
        assert str(a['predictions'])==here['original_seal']
    cases = []
    for name,digest in sorted(here['files'].items()):
        path, prior = current/name,old/name
        assert sha(path)==digest and sha(prior)==base['files'][name]
        with np.load(path,allow_pickle=False) as p,np.load(prior,allow_pickle=False) as q:
            indices = p['ordinals']
            assert str(p['archived_predictions'])==base['files'][name]
            choices = p['choices']
            np.testing.assert_array_equal(q['ordinals'],indices)
            np.testing.assert_array_equal(choices[:,0],q['choices'][:,-1])
            np.testing.assert_array_equal(q['choices'][...,0],paired['choice'][indices,1:])
            np.testing.assert_array_equal(q['reference_mention'],paired['flags'][indices,0])
            np.testing.assert_array_equal(q['direct_left_right'],paired['flags'][indices,1])
            i = np.arange(len(indices))[:,None]
            subjects = q['subject_slots'][i,choices[...,0]]
            references = q['reference_slots'][i,choices[...,1]]
            s_change = (subjects!=subjects[:,:1]).any(-1)
            r_change = (references!=references[:,:1]).any(-1)
            same_box = (subjects==references).all(-1)
            iou = paired['slots'][indices[:,None],choices[...,0]]
            np.testing.assert_array_equal(iou[:,0],paired['ious'][indices,-1])
            np.testing.assert_array_equal(iou[:,ARMS.index('without_terminal_messages')],iou[:,0])
            cases.append(dict(ordinal=indices,iou=iou,subject_change=s_change,
                              reference_change=r_change,same_box=same_box))
    result = {k:np.concatenate([g[k] for g in cases]) for k in cases[0]}
    np.testing.assert_array_equal(result['ordinal'],paired['ordinal'])
    result.update(flags=paired['flags'],recording=paired['recording'],trained=paired['ious'][:,1:])
    assert len(result['iou'])==here['n']
    out = PRIVATE/f'scored_{route}.npz'
    npz_atomic(out,**result,seal=sha(current/'SEAL.json'),prior_scores=sha(scores))
    return result,sha(out)


def cell(iou):
    return np.stack((iou>.25,iou>.5,iou),axis=-1).astype(np.float64)


def masks(row):
    reference,lr = row['flags'].T
    return dict(zip(COHORTS,(np.ones(len(reference),bool),lr,reference,~reference),strict=True))


def sign_description(delta):
    if all(x==0. for x in delta):
        return 'equal_aggregate'
    if all(x>=0. for x in delta):
        return 'all_nonnegative'
    if all(x<=0. for x in delta):
        return 'all_nonpositive'
    return 'mixed'


def render(data):
    endpoints,paired,trained,timing = [],[],[],[]
    numpy_means,python_means,history_means,counts = {},{},{},{}
    for d,z in data.items():
        selected = masks(z)
        x = z['iou']
        baseline = x[:,:1]
        delta = x-baseline
        values = cell(x)
        old_cells = cell(z['trained'])
        reference_active = z['flags'][:,0]
        arrays = dict(box_changes=z['subject_change'],
                      reference_changes=z['reference_change'],
                      reference_changes_when_present=z['reference_change']&reference_active[:,None],
                      identical_box_assignments=z['same_box']&reference_active[:,None],
                      iou_better=delta>EPS,iou_worse=delta<-EPS,iou_tie=abs(delta)<=EPS,
                      Acc25_gain=(x>.25)&(baseline<=.25),Acc25_loss=(x<=.25)&(baseline>.25),
                      Acc50_gain=(x>.50)&(baseline<=.50),Acc50_loss=(x<=.50)&(baseline>.50))
        for group,mask in selected.items():
            n = int(mask.sum())
            assert n > 0
            p = 100*values[mask].mean(0)
            checks = []
            for j in range(len(ARMS)):
                vector = x[mask,j].tolist()
                checks.append([100*sum(v>.25 for v in vector)/n,100*sum(v>.50 for v in vector)/n,
                               100*fsum(vector)/n])
            numpy_means[d,group] = p
            python_means[d,group] = np.asarray(checks)
            history_means[d,group] = 100*old_cells[mask].mean(0)
            quantities = {key:val[mask].sum(0) for key,val in arrays.items()}
            counts[d,group] = (n,quantities)
            for j,arm in enumerate(ARMS):
                differences = p[j]-p[0]
                endpoints.append(dict(direction=d,subset=group,n=n,variant=arm,
                                      **dict(zip(METRICS,p[j].tolist(),strict=True))))
                paired.append(dict(direction=d,subset=group,n=n,variant=arm,
                                   **dict(zip(METRICS,differences.tolist(),strict=True)),
                                   **{k:int(v[j]) for k,v in quantities.items()},
                                   endpoint_sign=sign_description(differences)))
            for j,arm in enumerate(TRAINED):
                row = history_means[d,group][j]
                trained.append(dict(direction=d,subset=group,n=n,variant=arm,
                                    **dict(zip(METRICS,row.tolist(),strict=True))))
        order = np.random.default_rng(1771+len(timing)).permutation(len(x))
        public = []
        for i,k in enumerate(order,1):
            change = {arm:float(delta[k,j]) if delta[k,j] else 0 for j,arm in enumerate(ARMS) if j}
            public.append(dict(record=f'{i:04d}',reference_mention=int(reference_active[k]),
                               direct_left_right=int(z['flags'][k,1]),
                               full_iou=float(x[k,0]),**change))
        write_csv(RESULTS/f'paired_{d}.csv',public)
        profile = read(OUTPUT/d/'TIMING.json')
        full_time = float(np.median(profile['milliseconds']['full']))
        for arm in BENCHMARK:
            op = OPTIONS[arm]
            t = float(np.median(profile['milliseconds'][arm]))
            timing.append(dict(direction=d,variant=arm,batch=profile['batch'],
                               device=profile['device'],decoder_ms=t,relative_to_full=t/full_time,
                               rounds=op.rounds,reference_slots=op.reference_slots,
                               decoded_slot_updates=op.rounds*(4+op.reference_slots),
                               terminal_message_projections=2*op.terminal_messages))
    errors = [float(abs(p-python_means[key]).max()) for key,p in numpy_means.items()]
    assert max(errors)<1e-10
    for group in COHORTS:
        get = [(d,group) for d in data]
        p = np.mean([numpy_means[key] for key in get],axis=0)
        old = np.mean([history_means[key] for key in get],axis=0)
        n = sum(counts[key][0] for key in get)
        totals = {k:sum(counts[key][1][k] for key in get) for k in counts[get[0]][1]}
        for j,arm in enumerate(ARMS):
            delta = p[j]-p[0]
            endpoints.append(dict(direction='six_equal',subset=group,n=n,variant=arm,
                                  **dict(zip(METRICS,p[j].tolist(),strict=True))))
            paired.append(dict(direction='six_equal',subset=group,n=n,variant=arm,
                               **dict(zip(METRICS,delta.tolist(),strict=True)),
                               **{k:int(v[j]) for k,v in totals.items()},
                               endpoint_sign=sign_description(delta)))
        for j,arm in enumerate(TRAINED):
            trained.append(dict(direction='six_equal',subset=group,n=n,variant=arm,
                                **dict(zip(METRICS,old[j].tolist(),strict=True))))
    return endpoints,paired,trained,timing,max(errors)


def recording_intervals(data):
    rng = np.random.default_rng(177)
    n_draws = 4000
    result = {}
    for subset in ('all','direct_left_right'):
        samples = []
        clusters = {}
        for target in PLATFORMS:
            near = [data[f'{s}_to_{target}'] for s in PLATFORMS if s!=target]
            valid = masks(near[0])[subset]
            for route in near[1:]:
                np.testing.assert_array_equal(near[0]['recording'],route['recording'])
                np.testing.assert_array_equal(valid,masks(route)[subset])
            recording,index = np.unique(near[0]['recording'][valid],return_inverse=True)
            n = len(recording)
            clusters[target] = n
            weights = rng.multinomial(n,np.repeat(1/n,n),size=n_draws)
            denominator = weights@np.bincount(index)
            for x in near:
                v = cell(x['iou'][valid])
                delta = v-v[:,:1]
                counts = np.zeros((n,len(ARMS),len(METRICS)))
                np.add.at(counts,index,delta)
                samples.append(100*np.einsum('dn,nak->dak',weights,counts)/denominator[:,None,None])
        provision = np.array(samples).mean(0)
        values = np.quantile(provision,[.025,.975],axis=0).transpose(1,2,0)
        result[subset] = dict(recordings=clusters,n_boot=n_draws,
                              deltas_pp={arm:dict(zip(METRICS,v.tolist(),strict=True))
                                         for arm,v in zip(ARMS,values,strict=True)})
    return result


def main():
    pred = verify()
    update_state('post_seal_scoring','Gather audited slot IoUs for sealed predictions and reconcile all paired outcomes.')
    older = read(PRIOR/'SCORING_SEAL.json')
    old_audit_path = R174/'results/independent_audit.json'
    older_audit = read(old_audit_path)
    assert older['n']==older_audit['n']==pred['n']==N
    assert not older['FINAL_access'] and not older_audit['FINAL_access']
    assert older_audit['scoring']==sha(PRIOR/'SCORING_SEAL.json')
    assert all(v['all_four_proposals_audited'] for v in older_audit['directions'].values())
    data,hashes = {},{}
    for s,t in ORDER:
        d = f'{s}_to_{t}'
        data[d],hashes[d] = collect((s,t),older)
        print('R177_SCORED',d,len(data[d]['iou']),flush=True)
    endpoints,paired,trained,timing,error = render(data)
    for name,rows in (('endpoints',endpoints),('paired_deltas',paired),
                      ('source_trained_controls',trained),('decoder_timing',timing)):
        write_csv(RESULTS/f'{name}.csv',rows)
    json_atomic(RESULTS/'recording_intervals.json',recording_intervals(data))
    benchmark = [r for r in paired if r['direction']=='six_equal' and r['subset']=='all']
    exact = [r['variant'] for r in benchmark if r['variant']!='full' and
             r['box_changes']==r['reference_changes']==0]
    assert 'without_terminal_messages' in exact
    r174_errors = [read(OUTPUT/d/'SEAL.json')['full_max_score_error'] for d in data]
    audit=dict(n=N,prediction_seal=sha(OUTPUT/'PREDICTION_SEAL.json'),
               previous_four_slot_geometry_audit=sha(old_audit_path),score_arrays=hashes,
               frozen_full_choice_disagreements=0,full_max_score_error=max(r174_errors),
               independent_scalar_max_error_pp=error,ablation_variants=len(ARMS)-1,
               exact_two_role_variants=exact,source_retraining=0,target_updates=0,FINAL_access=False)
    json_atomic(RESULTS/'independent_audit.json',audit)
    json_atomic(RESULTS/'DECISION.json',dict(n=N,main_method='R174_full',
                    exact_inference_simplifications=exact,
                    overall_all_nonnegative=[x['variant'] for x in benchmark
                                             if x['endpoint_sign']=='all_nonnegative'],
                    directional_recording_intervals='descriptive',
                    score_audit=sha(RESULTS/'independent_audit.json'),FINAL_access=False))
    output=dict(n=N,results={p.name:sha(p) for p in RESULTS.iterdir()},
                model_prediction=audit['prediction_seal'],FINAL_access=False)
    json_atomic(PRIVATE/'SCORING_SEAL.json',output)
    print('R177_COMPLETE',dict(Counter(r['endpoint_sign'] for r in benchmark)),exact,flush=True)


if __name__=='__main__':
    main()
