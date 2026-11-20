"""Post-fit source diagnosis, using the original cached proposals without any test-time change."""
from collections import Counter
from paired import (ARMS, HERE, PUBLIC, SOURCE, WORK, anon, atomic, digest, initialize, np, read)
from fit import load, write_csv
from source_pairs import catalog, source_identity
from scope import source_rows
from roles import inventory


def natural_anchors(split):
    root=SOURCE/'features_PRIVATE'
    whole=read(root/'SEALED.json')
    assert not whole['FINAL_access']
    part=root/f'waymo_{split}'
    seal=read(part/'SEAL.json')
    assert digest(part/'SEAL.json')==whole['splits'][part.name]['seal_sha256']
    assert seal['input_sha256']==digest(SOURCE/'inputs_PRIVATE'/f'{part.name}.json')
    for name, info in sorted(seal['files'].items()):
        p=part/name
        assert digest(p)==info['sha256']
        with np.load(p,allow_pickle=False) as z:
            real=z['origin']!=2
            for a,g in zip(z['anchors'][real],z['gt'][real,0]):
                yield a,g


def margin(d,c):
    i=np.arange(len(d))
    return np.stack([d[i,0,c[:,0]]-d[i,0,c[:,1]],d[i,1,c[:,1]]-d[i,1,c[:,0]]],axis=-1)


def main():
    initialize()
    assert read(WORK/'HEADS_LOCK.json')['heads']==digest(WORK/'heads.npz')
    seal=read(WORK/'SOURCE_SEAL.json')
    heads=np.load(WORK/'heads.npz')['heads']
    rows_out=[]
    totals={}
    for split in ('fit','calibration'):
        data=load(split,seal)
        raw={}
        for row,(anchors,primary_gt) in zip(source_rows('waymo',split),natural_anchors(split),strict=True):
            objects,pairs=catalog(row)
            if not pairs:
                continue
            annotated,_=inventory(row)
            np.testing.assert_array_equal(primary_gt,annotated[0])
            full,inv=source_identity(row,objects,anchors)
            for a,b in pairs:
                k=anon('pair',row['point_sha256'],a,b)
                primary_index = row['object_index']
                primary_side = 0 if primary_index==a else 1 if primary_index==b else -1
                raw[k]=(full[:,inv[[a,b]]], (full>.25).sum(1)==1, primary_side)
        assert set(raw)==set(data['key'])
        ok=(data['slots']>=0).all(-1)
        fixed=data['j']-.75*np.tanh(data['h']@heads[0,:-1]+heads[0,-1])
        maximum=margin(fixed,np.maximum(0,data['slots']))+1.5
        by_arm=[]
        for head in heads:
            scores=fixed+.75*np.tanh(data['h']@head[:-1]+head[-1])
            by_arm.append((margin(scores,np.maximum(0,data['slots']))>0).all(-1))
        cases=Counter()
        full_covered=0
        can=0
        prior_break=0
        primary_stats=Counter()
        for n,i in enumerate(np.argsort(data['key']),1):
            mat,unique,side=raw[data['key'][i]]
            for each, endpoint in zip(data['ious'][i],mat.T):
                assert np.max(np.min(np.abs(each[:,None]-endpoint),axis=1))<1e-9
            pool = ((mat>.25)&unique[:,None]).any(0).sum()
            local=(data['slots'][i]>=0).sum()
            assert pool>=local
            full_covered+=int(pool==2)
            if side>=0:
                in_full=((mat>.25)&unique[:,None]).any(0)
                in_four=data['slots'][i]>=0
                primary_stats['pairs_including_original_expression']+=1
                primary_stats['original_endpoint_in_four']+=int(in_four[side])
                primary_stats['other_endpoint_in_four']+=int(in_four[1-side])
                primary_stats['original_endpoint_in_A256']+=int(in_full[side])
                primary_stats['other_endpoint_in_A256']+=int(in_full[1-side])
            cause=('both_available' if local==2 else 'both_in_A256_lost_in_four' if pool==2
                   else 'A256_lacks_unambiguous_dual_coverage')
            cases[cause]+=1
            upper=maximum[i].min() if ok[i] else None
            if ok[i] and not by_arm[0][i]:
                can+=int(upper>0)
                prior_break+=int(upper<=0)
            rows_out.append(dict(split=split,pair_id=f'{split[0].upper()}{n:03d}',
                  endpoints_in_A256=int(pool),endpoints_in_four=int(local),coverage_stage=cause,
                  v1_both='' if not ok[i] else int(by_arm[0][i]),
                  mean_both='' if not ok[i] else int(by_arm[1][i]),
                  worst_both='' if not ok[i] else int(by_arm[2][i]),
                  upper_bound_worse_margin='' if upper is None else float(upper)))
        assert sum(cases.values())==len(ok)
        totals[split]=dict(potential_pairs=len(ok),both_A256=full_covered,both_four=int(ok.sum()),original_expression_asymmetry=dict(primary_stats),
                 eligible_pairs_with_nonzero_vote_support_on_all_four_under_both_sentences=int(
                     data['support'][ok].all(axis=(1,2)).sum()),
                 n_by_stage=dict(cases),existing_failures_beyond_subject_range=prior_break,
                 existing_failures_within_necessary_bound=can,
                 frozen_to_mean=dict(rescue=int((ok & ~by_arm[0] & by_arm[1]).sum()),
                                      harm=int((ok & by_arm[0] & ~by_arm[1]).sum())),
                 frozen_to_worst=dict(rescue=int((ok & ~by_arm[0] & by_arm[2]).sum()),
                                       harm=int((ok & by_arm[0] & ~by_arm[2]).sum())))
    write_csv(PUBLIC/'source_coverage.csv',rows_out)
    atomic(PUBLIC/'source_diagnosis.json',totals)
    print(totals,flush=True)


if __name__=='__main__':
    main()
