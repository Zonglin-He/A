"""Diagnose frozen score margins, then seal and score the source-calibrated readout."""

import argparse
import secrets

import numpy as np

from capture181 import HERE, TARGET, manifest, protocol, load_r180, write_npz
from readout181 import ARMS, K
from calibrate181 import OUTPUT, load_calibration, final_choices
from scope import PLATFORMS, evaluation_rows, file_sha, read, save
from evaluate import DIRECTIONS, score_boxes, sensor_truth, save_csv, recording_intervals
from train import metrics

TRUTH = HERE/'audited_iou_PRIVATE'
CHOICES = HERE/'calibrated_predictions_PRIVATE'
RESULTS = HERE/'results'


def load_terms():
    pin = read(TARGET/'SEALED.json')
    assert pin['n']==3072 and pin['implementation']==manifest() and not pin['FINAL_access']
    assert pin['earlier_seal_sha256']==protocol()['ordinary_evaluation']['r180_prediction_seal_sha256']
    assert not pin['target_labels_read']
    data = {}
    for source,target in DIRECTIONS:
        path=TARGET/f'{source}_to_{target}.npz'
        assert file_sha(path)==pin['files'][path.name]['sha256']
        with np.load(path,allow_pickle=False) as saved:
            block={key:saved[key] for key in saved.files}
        assert block['terms'].shape==(512,K,6) and block['scores'].shape==(512,K,3)
        assert np.isfinite(block['terms']).all() and np.isfinite(block['scores']).all()
        data[source,target]=block
    return data


def distribution(value):
    return dict(zip(('p10','median','p90'),np.quantile(value,[.1,.5,.9]).tolist())) if len(value) else None


def diagnose_margin(data):
    eligible = ((data['ious']>.25) & data['can_win'])
    cohort = np.flatnonzero((data['original_best']<=.25)&eligible.any(axis=1))
    good_rank = np.where(eligible[cohort],data['scores'][cohort,:,0],-np.inf)
    good = good_rank.argmax(axis=1)
    incumbent = data['incumbent'][cohort]
    picked = data['chosen'][cohort,0]
    features = data['terms'].astype(np.float64)
    f = data['scores'][:,:,0].astype(np.float64)
    dst = features[cohort,good]
    d_old = dst-features[cohort,incumbent]
    d_new = dst-features[cohort,picked]
    a,b = f[cohort,good]-f[cohort,incumbent],f[cohort,good]-f[cohort,picked]
    before_relation = d_old[:,0]+d_old[:,2]
    epsilon = 1e-6

    def n(condition):
        return int(np.count_nonzero(condition))

    selected_iou = data['ious'][cohort,picked]
    assert np.all(selected_iou<=.25) and np.all(b<=epsilon)
    variants={
        'prior_only':features[cohort,:,0],
        'omit_final_reference_marginal':features[cohort,:,0]+features[cohort,:,2],
        'omit_prior':features[cohort,:,2]+features[cohort,:,3],
    }
    flips={name:n(data['ious'][cohort,z.argmax(axis=1)]>.25) for name,z in variants.items()}
    flips['frozen_independent']=n(data['ious'][cohort,data['chosen'][cohort,1]]>.25)
    flips['frozen_mismatch']=n(data['ious'][cohort,data['chosen'][cohort,2]]>.25)
    return dict(n=len(cohort),reference_active=n(data['has_reference'][cohort]),
        prior_below_incumbent=n(d_old[:,0]<-epsilon),
        prior_above_incumbent=n(d_old[:,0]>epsilon),
        subject_residual_favors_good=n(d_old[:,2]>epsilon),
        final_relation_favors_good=n(d_old[:,3]>epsilon),
        best_good_still_behind_after_prior_plus_subject=n(before_relation < -epsilon),
        best_good_ahead_before_relation=n(before_relation > epsilon),
        relation_reverses_good_lead=n((before_relation>epsilon)&(a < -epsilon)),
        relation_creates_good_lead=n((before_relation < -epsilon)&(a>epsilon)),
        good_beats_incumbent_but_loses_to_another=n(a>epsilon),
        good_ties_incumbent=n(np.abs(a)<=epsilon),
        r180_changes_incumbent=n(incumbent!=picked),
        against_incumbent={
            'prior':distribution(d_old[:,0]),'subject_independent':distribution(d_old[:,1]),
            'subject_joint':distribution(d_old[:,2]),'relation_direct':distribution(d_old[:,3]),
            'reference_including_messages':distribution(d_old[:,2]-d_old[:,1]+d_old[:,3]),
            'full_score':distribution(a)},
        against_frozen_winner={
            'prior':distribution(d_new[:,0]),'subject_joint':distribution(d_new[:,2]),
            'relation_direct':distribution(d_new[:,3]),'full_score':distribution(b)},
        counterfactual_pass_counts=flips)


def diagnose():
    terms=load_terms()
    rows=evaluation_rows()
    earlier=load_r180(rows)
    TRUTH.mkdir(mode=0o700)
    RESULTS.mkdir()
    maximum=0.
    parts=[]
    hashes={}
    directions={}
    for (s,t),block in terms.items():
        old=earlier[s,t]
        np.testing.assert_array_equal(old['ordinals'],block['ordinals'])
        slate=old['slate'][:,6,:K]
        ii=np.arange(len(slate))
        np.testing.assert_array_equal(slate[ii,block['incumbent']],old['chosen'][:,0])
        np.testing.assert_array_equal(slate[ii,block['chosen'][:,0]],old['chosen'][:,6])
        np.testing.assert_array_equal(block['can_win'],old['can_win'][ii[:,None],slate])
        output=[]
        for i,ordinal in enumerate(block['ordinals']):
            row=rows[int(ordinal)]
            gt=sensor_truth(t,row['metapath'],row['caption'],row['suffix'])
            pool=old['pool'][i]
            boxes=np.concatenate((pool[slate[i]],pool[old['slate'][i,0,:4]]))
            val,err=score_boxes(boxes,gt)
            output.append(val)
            maximum=max(maximum,err)
        result=np.stack(output)
        name=f'{s}_to_{t}'
        path=TRUTH/(name+'.npz')
        scored=dict(ious=result[:,:K],original_best=result[:,K:].max(axis=1),
                    outside=~(slate[:,:,None]==old['slate'][:,0,None,:4]).any(axis=-1),
                    recordings=np.asarray([rows[int(i)]['scene_cluster'] for i in block['ordinals']]))
        write_npz(path,scored)
        hashes[path.name]=file_sha(path)
        parts.append(dict(block,**scored))
        directions[name]=diagnose_margin(parts[-1])
        print(name,directions[name]['n'],'prior-bound attainable rows',flush=True)
    merged={k:np.concatenate([a[k] for a in parts]) for k in parts[0]}
    overall=diagnose_margin(merged)
    assert overall['n']==381
    assert int(((merged['original_best']<=.25)&(merged['ious']>.25).any(axis=1)).sum())==664
    changed=(np.abs(merged['scores'][:,:,0]-merged['scores'][:,:,2])>1e-6).any(axis=1)
    save(RESULTS/'diagnosis.json',dict(overall=overall,directions=directions,
         mismatch_changes_scores=int(changed.sum()),
         reference_active=int(merged['has_reference'].sum()),inputs_already_exposed=True,
         target_labels_used_for_this_diagnostic_only=True,FINAL_access=False))
    save(TRUTH/'SEALED.json',dict(n=3072,files=hashes,terms_sha256=file_sha(TARGET/'SEALED.json'),
         code_sha256=file_sha(HERE/'score181.py'),max_independent_iou_error=maximum,
         exact_iou_checks=3072*(K+4),threshold_disagreements=0,FINAL_access=False))


def predict():
    terms=load_terms()
    calibration=load_calibration()
    assert tuple(protocol()['arms'])==ARMS
    CHOICES.mkdir(mode=0o700)
    files={}
    for (source,target),block in terms.items():
        p=CHOICES/f'{source}_to_{target}.npz'
        chosen=final_choices(block['terms'],block['chosen'],block['incumbent'],calibration['models_by_source'][source])
        assert (chosen>=0).all() and (chosen<K).all()
        write_npz(p,dict(chosen=chosen))
        files[p.name]=file_sha(p)
    save(CHOICES/'SEALED.json',dict(n=3072,files=files,arms=list(ARMS),
         terms_sha256=file_sha(TARGET/'SEALED.json'),calibration_sha256=file_sha(OUTPUT),
         code_sha256=file_sha(HERE/'score181.py'),target_labels_used=False,FINAL_access=False))


def transitions(block,index,threshold):
    before,after=(block['ious'][:,[0,index]]>threshold).T
    old_available=block['original_best']>threshold
    rescue=~before & after
    harm=before & ~after
    outside=block['outside'][:,index]
    return dict(rescued_original_four_miss=int((rescue & ~old_available).sum()),
         rescued_by_outside=int((rescue & outside).sum()),total_rescues=int(rescue.sum()),
         harms_by_outside=int((harm & outside).sum()),total_harms=int(harm.sum()),
         net=int(rescue.sum()-harm.sum()))


def score():
    pin=read(CHOICES/'SEALED.json')
    gt=read(TRUTH/'SEALED.json')
    assert pin['n']==gt['n']==3072 and pin['terms_sha256']==gt['terms_sha256']==file_sha(TARGET/'SEALED.json')
    assert pin['calibration_sha256']==file_sha(OUTPUT) and not pin['target_labels_used']
    assert pin['code_sha256']==gt['code_sha256']==file_sha(HERE/'score181.py')
    terms=load_terms()
    model=load_calibration()
    data={}
    anonymous=[]
    for s,t in DIRECTIONS:
        name=f'{s}_to_{t}.npz'
        assert file_sha(TRUTH/name)==gt['files'][name] and file_sha(CHOICES/name)==pin['files'][name]
        with np.load(TRUTH/name,allow_pickle=False) as box, np.load(CHOICES/name,allow_pickle=False) as chosen:
            ids=chosen['chosen']
            current=terms[s,t]
            np.testing.assert_array_equal(ids,final_choices(current['terms'],current['chosen'],current['incumbent'],
                                          model['models_by_source'][s]))
            rc=np.arange(len(ids))[:,None]
            data[f'{s}_to_{t}']=dict(ious=box['ious'][rc,ids],original_best=box['original_best'],
                  outside=box['outside'][rc,ids],recordings=box['recordings'],switches=ids!=ids[:,:1])
        block=data[f'{s}_to_{t}']
        for row in range(len(ids)):
            anonymous.append(dict(source=s,target=t,reference_active=int(terms[s,t]['has_reference'][row]),
                original_best=float(block['original_best'][row]),
                **{a:float(block['ious'][row,j]) for j,a in enumerate(ARMS)},
                **{f'outside_{a}':int(block['outside'][row,j]) for j,a in enumerate(ARMS)},
                **{f'switch_{a}':int(block['switches'][row,j]) for j,a in enumerate(ARMS)}))
    all_data={key:np.concatenate([b[key] for b in data.values()]) for key in next(iter(data.values()))}
    table=[]
    changes=[]
    for direction,block in (*data.items(),('six_equal',all_data)):
        for i,name in enumerate(ARMS):
            val=block['ious'][:,i]
            diff=val-block['ious'][:,0]
            table.append(dict(direction=direction,arm=name,**metrics(val),
                              switches=int(block['switches'][:,i].sum()),
                              iou_better=int((diff>1e-12).sum()),iou_worse=int((diff< -1e-12).sum())))
            for threshold in (.25,.50):
                changes.append(dict(direction=direction,arm=name,threshold=threshold,n=len(val),
                                    **transitions(block,i,threshold)))
    save_csv(RESULTS/'direction.csv',table)
    save_csv(RESULTS/'transitions.csv',changes)
    ci=recording_intervals(data,ARMS)
    ci['contrast']='method minus v1, percentage points'
    ci['intervals']={name:[[-upper,-lower] for lower,upper in vals] for name,vals in ci['intervals'].items()}
    save(RESULTS/'recording_intervals.json',ci)
    secrets.SystemRandom().shuffle(anonymous)
    save_csv(RESULTS/'anonymous_rows.csv',anonymous)
    save(RESULTS/'audit.json',dict(n=len(anonymous),endpoint_count=len(anonymous)*len(ARMS),
         exact_iou_checks=gt['exact_iou_checks'],max_independent_iou_error=gt['max_independent_iou_error'],
         threshold_disagreements=gt['threshold_disagreements'],
         frozen_terms_seal=gt['terms_sha256'],calibrated_prediction_seal=file_sha(CHOICES/'SEALED.json'),
         verified_R180_frozen_winners=read(TARGET/'SEALED.json')['identical_R180_frozen_winners'],
         recording_counts={t:len(np.unique(data[f'{s}_to_{t}']['recordings'])) for s,t in DIRECTIONS},
         source_only_calibration=True,target_updates=0,FINAL_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('diagnose','predict','score'))
    {'diagnose':diagnose,'predict':predict,'score':score}[parser.parse_args().phase]()
