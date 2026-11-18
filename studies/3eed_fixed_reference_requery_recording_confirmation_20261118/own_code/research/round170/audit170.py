"""Independently recompute critical matched claims from the evidence ledger."""
from collections import Counter
from math import fsum

from scope170 import ARMS, HERE, ORDER, OUT, PLATFORMS, VARIANTS, atomic_json, read, sha
from score165 import table


def value(r, arm, metric):
    iou=float(r[arm+'_iou'])
    return iou if metric=='mIoU' else float(iou>{'Acc25':.25,'Acc50':.5}[metric])


def groups(rows, label):
    if label=='pooled':
        return [rows] if rows else []
    if label=='six_equal':
        keys=ORDER
    elif label in PLATFORMS:
        keys=[(s,t) for s,t in ORDER if t==label]
    else:
        keys=[tuple(label.split('_to_'))]
    return [x for s,t in keys if (x:=[r for r in rows if (r['source'],r['target'])==(s,t)])]


def audit():
    input_proof=read(HERE/'INPUT_SEAL.json')
    inference=read(HERE/'PREDICTION_SEAL.json')
    label=read(HERE/'VISUAL_TRUTH_SEAL.json')
    evalp=read(HERE/'EVALUATION_AUDIT.json')
    scopes={'all','blind_pair','rewrite_supported'}
    metrics=('mIoU','Acc25','Acc50')
    assert not any(x['FINAL_access'] for x in (input_proof,inference,label,evalp))
    assert input_proof['n']==label['n']==inference['physical_frames']==48
    assert inference['n']==evalp['n']==96 and inference['r159_r158_pca_bit_exact']
    assert inference['source_fingerprints_unchanged'] and input_proof['recordings_overlap_with_D_selection']==0
    assert input_proof['sensor_overlap_with_R165_or_R168_source']==0
    assert len(inference['reconstruction_bridge'])==3 and all(x['bank_and_winners_bit_exact'] and x['max_score_error']<1.e-10
                                                              for x in inference['reconstruction_bridge'])
    assert evalp['source_prediction_seal']==sha(HERE/'PREDICTION_SEAL.json')
    assert evalp['blind_truth_seal']==sha(HERE/'VISUAL_TRUTH_SEAL.json')
    assert evalp['max_sensor_iou_error']<1.e-7
    assert all(sha(OUT/p)==v for p,v in evalp['files'].items())
    rows=table(OUT/'directions_and_cases.csv')
    witnesses=table(OUT/'mapping_cases.csv')
    assert len(rows)==len({(r['case'],r['source']) for r in rows})==96
    assert Counter(r['case'] for r in rows)==dict.fromkeys(dict.fromkeys(r['case'] for r in rows),2)
    assert len(witnesses)==len({(r['case'],r['source'],r['variant']) for r in witnesses})==192
    assert all(r['D_independent_iou']==r['C_independent_iou']==r['r158_pca_iou'] for r in rows)
    fallbacks=0
    for r in rows:
        for arm in ARMS[1:]:
            if r[arm+'_fallback']=='1':
                assert r[arm+'_iou']==r['r158_pca_iou'] and r[arm+'_has_reference']=='0'
                fallbacks+=1
    checked=0
    error=0.
    prior={}
    for r in table(OUT/'metrics.csv'):
        s,k,a=r['scope'],r['group'],r['arm']
        assert s in scopes
        if (s,k) not in prior:
            section=rows if s=='all' else ([h for h in rows if h['verified_pair']=='1'] if s=='blind_pair' else
                                           [h for h in rows if h['rewrite']=='supported'])
            prior[s,k]=groups(section,k)
        split=prior[s,k]
        selected=[x for g in split for x in g]
        assert len(selected)==int(r['n']) and len({x['case'] for x in selected})==int(r['frames'])
        assert sum(x['verified_pair']=='1' for x in selected)==int(r['both_labels_n'])
        if a!='r158_pca':
            assert sum(x[a+'_dual_correct']=='1' for x in selected)==int(r['dual_correct'])
        for name in metrics:
            mean=100*fsum(fsum(value(x,a,name) for x in z)/len(z) for z in split)/len(split)
            error=max(error,abs(mean-float(r[name])))
            checked+=1
    for row in table(OUT/'paired.csv'):
        sample=prior[row['scope'],row['group']]
        method,base=row['arm'],row['control']
        for key in metrics:
            diffs=[[(value(r,method,key)-value(r,base,key)) for r in part] for part in sample]
            mu=100*fsum(fsum(x)/len(x) for x in diffs)/len(diffs)
            error=max(error,abs(mu-float(row[key+'_pp'])))
            a=[v for g in diffs for v in g]
            assert sum(x>1.e-10 for x in a)==int(row[key+'_rescues'])
            assert sum(x< -1.e-10 for x in a)==int(row[key+'_harms'])
            checked+=3
    for m in table(OUT/'mapping_summary.csv'):
        cohort=[x for x in witnesses if x['variant']==m['variant'] and (m['target']=='pooled' or x['target']==m['target'])]
        verified=[x for x in cohort if x['reference_label']=='verified']
        assert len(cohort)==int(m['n']) and len(verified)==int(m['verified_reference_directions'])
        for view in ('native','mapped'):
            rank=[int(x[view+'_rank']) for x in verified]
            assert rank.count(1)==int(m[view+'_top1_correct'])
            assert sum(i>0 for i in rank)==int(m[view+'_coverage'])
            checked+=2
    for r in table(OUT/'recording_deletion.csv'):
        kept=groups([x for x in rows if x['recording']!=r['excluded_recording']],'six_equal')
        assert len(kept)==int(r['coverage_directions'])
        if len(kept)==6:
            for key in metrics:
                delta=100*fsum(fsum(value(r0,r['arm'],key)-value(r0,r['control'],key) for r0 in g)/len(g) for g in kept)/6
                error=max(error,abs(delta-float(r[key+'_pp'])))
                checked+=1
        else:
            assert all(r[k+'_pp']=='' for k in metrics)
    assert error<1.e-9
    p=dict(direction_rows=len(rows),physical_frames=len({r['case'] for r in rows}),
           recordings=len({r['recording'] for r in rows}),mapping_rows=len(witnesses),checked_scalars=checked,
           max_scalar_error=error,fallback_arm_outputs=fallbacks,all_negative_and_unknown_rows_retained=True,FINAL_access=False)
    atomic_json(HERE/'SCALAR_AUDIT.json',p)
    print(p,flush=True)
    return p


if __name__=='__main__':
    audit()
