"""Check the published summaries from sealed observations with scalar recomputation."""
from collections import Counter
from math import fsum

import numpy as np

from scope169 import HERE, N, ORDER, OUT, PANEL_N, QUERY_ARMS, R168, atomic_json, read, sha
from score165 import table, write_table
from score169 import sealed


def grouped(rows,label):
    if label=='pooled':
        return [rows]
    if label=='six_equal':
        keys=ORDER
    elif label.startswith('target_'):
        keys=[(a,b) for a,b in ORDER if b==label[7:]]
    else:
        keys=[tuple(label.split('_to_'))]
    return [front for a,b in keys if (front:=[x for x in rows if (x['source'],x['target'])==(a,b)])]


def summary(selected,arm,field):
    values=[float(r[arm+'_iou']) for r in selected]
    if field=='mIoU':
        return fsum(values)/len(values)
    if field=='subject_identity':
        return sum(x[arm+'_subject']=='1' for x in selected)/len(selected)
    cutoff={'Acc25':.25,'Acc50':.5}[field]
    return sum(x>cutoff for x in values)/len(values)


def cache_recheck():
    parents=read(HERE/'CACHE_EVALUATION_AUDIT.json')
    assert parents['prediction_seal']==sha(HERE/'CACHE_PREDICTION_SEAL.json')
    assert parents['n']==N and parents['evaluated_edges']==11378 and parents['exact_old_replay']
    assert not (parents['old_gate_used'] or parents['old_overlap_penalty_used'] or parents['FINAL_access'])
    assert all(s['max_sensor_error']<1.e-7 for s in parents['direction'].values())
    assert all(sha(OUT/p)==v for p,v in parents['files'].items())
    data=table(OUT/'cache_rows.csv')
    links=table(OUT/'cache_edges.csv')
    assert len(data)==len({(v['source'],v['target'],v['expression']) for v in data})==N
    assert len(links)==11378 and sum(r['visual']=='1' for r in links)==54
    old={(x['source'],x['target'],x['expression']):x for x in table(R168/'results/all_rows.csv')}
    assert all(r['independent_iou']==old[r['source'],r['target'],r['expression']]['independent_iou'] and
               r['old_left_iou']==old[r['source'],r['target'],r['expression']]['left_only_iou'] for r in data)
    groups={}
    metrics=('mIoU','Acc25','Acc50','subject_identity')
    checked=0
    err=0.
    for v in table(OUT/'cache_metrics.csv'):
        k=v['scope'],v['group']
        if k not in groups:
            z=data if k[0]=='all' else [r for r in data if r['camera_horizontal']=='1']
            groups[k]=grouped(z,k[1])
        q=groups[k]
        assert sum(map(len,q))==int(v['n'])
        for m in metrics:
            want=100*fsum(summary(part,v['arm'],m) for part in q)/len(q)
            err=max(err,abs(want-float(v[m])))
            checked+=1
    for v in table(OUT/'cache_paired.csv'):
        found=groups[v['scope'],v['group']]
        a,b=v['arm'],v['control']
        assert sum(map(len,found))==int(v['n'])
        for m in metrics:
            ref=100*fsum(summary(part,a,m)-summary(part,b,m) for part in found)/len(found)
            err=max(err,abs(ref-float(v[m+'_pp'])))
            checked+=1
        for m,t in (('Acc25',.25),('Acc50',.50)):
            z=[int(float(x[a+'_iou'])>t)-int(float(x[b+'_iou'])>t) for part in found for x in part]
            assert z.count(1)==int(v[m+'_rescue']) and z.count(-1)==int(v[m+'_harm'])
            checked+=2
    for t in table(OUT/'cache_reference.csv'):
        part=[r for r in links if (r['visual']=='1' if t['scope']=='visual' else r['prior_truth'] in ('visual_verified','annotation_proxy'))]
        assert len(part)==int(t['n'])
        label=t['scope']+'_'+t['arm']
        rank=[int(r[label+'_rank']) for r in part if r[label+'_rank']]
        assert len(rank)==int(t['rankable'])
        assert abs(fsum(1/i for i in rank)/len(rank)-float(t['mrr']))<1.e-12
        for name in ('independent_ref','joint_ref','joint_both','joint_own_subject'):
            assert sum(r[label+'_'+name]=='1' for r in part)==int(t[name])
            checked+=1
    assert err<1.e-9
    return dict(cache_scalars_rechecked=checked,max_scalar_error=err)


def panel_recheck():
    i=read(HERE/'QUERY_IDENTITY_AUDIT.json')
    j=read(HERE/'QUERY_JOINT_AUDIT.json')
    src=read(HERE/'SOURCE_IMAGE_AUDIT.json')
    assert all(sha(OUT/f)==v for f,v in {**i['files'],**j.get('files',{}),**src['files']}.items())
    identity=table(OUT/'query_identity_rows.csv')
    assert len(identity)==len({(r['panel'],r['arm']) for r in identity})==PANEL_N*len(QUERY_ARMS)
    byarm={a:[r for r in identity if r['arm']==a and r['rewrite_status']=='supported'] for a in QUERY_ARMS}
    for r in table(OUT/'query_identity_summary.csv'):
        subset=identity if r['scope']=='all_attempts' else byarm[r['arm']]
        x=[v for v in subset if v['arm']==r['arm'] and (r['target']=='pooled' or v['target']==r['target'])]
        assert len(x)==int(r['n']) and len({v['frame'] for v in x})==int(r['frames'])
        for kind in ('natural','mapped'):
            assert sum(v[kind+'_top']=='1' for v in x)==int(r[kind+'_correct'])
            rr=[1/int(v[kind+'_rank']) for v in x if v[kind+'_rank']!='0']
            assert len(rr)==int(r[kind+'_cover'])
            assert abs(fsum(rr)/len(x)-float(r[kind+'_mrr']))<1.e-12
    gated=[a for a in QUERY_ARMS[2:] if sum(r['natural_top']=='1' for r in byarm[a])>
           sum(r['natural_top']=='1' for r in byarm[QUERY_ARMS[1]]) and sum(r['mapped_top']=='1' for r in byarm[a])>
           max(sum(r['mapped_top']=='1' for r in byarm[b]) for b in QUERY_ARMS[:2])]
    assert gated==i['eligible_for_joint']==j['eligible']
    if not gated:
        return dict(panel_rows=len(identity),no_joint=True,files={})
    data=table(OUT/'query_joint_rows.csv')
    a={(r['source'],r['target'],r['expression']):r for r in table(R168/'results/all_rows.csv')}
    fallback=[]
    memory=Counter()
    for k,e,z in sealed():
        if e['status']=='supported':
            continue
        base=a[e['source'],e['target'],e['expression']]
        for arm in gated:
            idx=QUERY_ARMS.index(arm)
            assert not z['executed'][idx]
            np.testing.assert_array_equal(z['chosen'][idx,:,0],np.repeat(z['chosen'][0,0,0],2))
            np.testing.assert_array_equal(z['boxes'][idx,:,0],np.repeat(z['boxes'][0,0,0][None],2,axis=0))
            assert (z['chosen'][idx,:,1]==-1).all()
            fallback.append(dict(panel=str(k),arm=arm,frame=e['frame'],recording=e['recording'],subject_switch='0',
                                 independent_iou=base['independent_iou'],joint_iou=base['independent_iou'],
                                 independent_ref='-3',joint_ref='-3',independent_both='0',joint_both='0'))
    assert len(fallback)==(PANEL_N-i['paired_n'])*len(gated)
    stats=[]
    for arm in gated:
        for scope,sample in (('supported',data),('all_with_fallback',data+fallback)):
            x=[r for r in sample if r['arm']==arm]
            r=dict(arm=arm,scope=scope,n=len(x),frame_count=len({v['frame'] for v in x}),fallback=sum(v in fallback for v in x))
            for k in ('independent','joint'):
                ious=[float(v[k+'_iou']) for v in x]
                r.update({k+'_mIoU':100*fsum(ious)/len(ious),k+'_Acc25_count':sum(v>.25 for v in ious),
                          k+'_Acc50_count':sum(v>.50 for v in ious),k+'_both':sum(v[k+'_both']=='1' for v in x)})
            for name,t in (('Acc25',.25),('Acc50',.50)):
                gain=[v for v in x if float(v['joint_iou'])>t>=float(v['independent_iou'])]
                harm=[v for v in x if float(v['independent_iou'])>t>=float(v['joint_iou'])]
                r.update({name+'_rescues':len(gain),name+'_harms':len(harm),name+'_rescue_frames':len({v['frame'] for v in gain}),
                          name+'_rescue_recordings':len({v['recording'] for v in gain}),
                          name+'_rescues_with_verified_dual_identity':sum(v['joint_both']=='1' for v in gain)})
            stats.append(r)
    for published in table(OUT/'query_joint_summary.csv'):
        if published['target']=='pooled':
            rec=next(x for x in stats if x['arm']==published['arm'] and x['scope']=='supported')
            for key in ('independent_mIoU','joint_mIoU','independent_Acc25_count','joint_Acc25_count',
                        'independent_Acc50_count','joint_Acc50_count','n','Acc25_rescues','Acc50_rescues','Acc25_harms','Acc50_harms'):
                assert abs(float(published[key])-float(rec[key]))<1.e-10
    for v in table(OUT/'source_image_audit.csv'):
        memory[v['proxy_matches']]+=1
    assert (memory['1'],memory['0'],memory[''])==(src['confirmed'],src['contradicted'],src['unknown'])
    dest=OUT/'query_joint_verified_endpoints.csv'
    write_table(dest,stats)
    return dict(panel_rows=len(identity),fallback_rows=len(fallback),identity_gate_verified=True,files={dest.name:sha(dest)})


def audit():
    proof=dict(**cache_recheck(),**panel_recheck(),FINAL_access=False,
               all_negative_and_unknown_records_retained=True)
    atomic_json(HERE/'SCALAR_AUDIT.json',proof)
    print(proof,flush=True)
    return proof


if __name__=='__main__':
    audit()
