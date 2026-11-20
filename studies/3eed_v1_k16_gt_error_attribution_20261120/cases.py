"""Join the sealed case list to the frozen K16 and decompose scores on identical boxes."""
import csv
import gzip
import itertools
import math
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
SEED=HERE.parent/'3eed_v1_native_seed_code_decoding_20261120'
sys.path.insert(0,str(SEED))
from predict import (NATIVE, OUT, PARENT, ROOT, CandidateFrontend, atomic, digest, pinned, prior, read)
from seed_code import ARMS, FIELDS, focal_costs, key
from features import observation
from train import prepare_device
from roles import inventory
from crossplatform.candidate_frontend import orient_box
from crossplatform.round7.camera_projection import raw_to_model_points
sys.path[:0]=[str(ROOT/'research/round158'),str(ROOT/'research/round6/source_validation_scoring')]
from batch_yaw_iou import many_iou
from reader158 import overlaps
from grounding158 import select_grounding_box
from mechanisms import diverse_order

PRIV=HERE/'review_PRIVATE'
PUB=HERE/'results'
PARTS=('xy','z','size','yaw')
SLICES=([0,1],[2],[3,4,5],[6])
PF=('text','density','neighbors','clip')
NAMES=('prior','subject_independent','subject_joint','reference','subject_mismatch','reference_mismatch')


def write_csv(path,rows):
    cols=dict.fromkeys(k for row in rows for k in row)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def records():
    _,roster=pinned()
    p=read(HERE/'PROTOCOL.json')
    assert digest(OUT/'SEALED.json')==p['prediction_manifest_sha256']
    gtroot=SEED/'run_PRIVATE/postseal'
    assert digest(gtroot/'SEALED.json')==p['postseal_truth_sha256']
    audit=read(SEED/'results/audit.json')
    name='source_all_rows.csv.gz'
    assert digest(SEED/'results'/name)==audit['result_sha256'][name]
    with gzip.open(SEED/'results'/name,'rt') as f:
        logged=list(csv.DictReader(f))
    index=sorted(((s,i) for s in roster for i in roster[s]),key=lambda x:key('public',*x))
    assert len(index)==len(logged)==192
    result=[]
    for n,(r,(platform,i)) in enumerate(zip(logged,index),1):
        assert r['row_id']==f'S{n:03}' and r['platform']==platform
        f=float(r['frozen_joint_iou'])
        eligible=(f<=.5<float(r['pool_best_iou']))
        harm=(float(r['native_focal_abstain_iou'])<=.5<f)
        assert (r['recoverable50']=='True')==eligible and not(eligible and harm)
        if eligible or harm:
            result.append((r['row_id'],platform,i,'frozen_miss' if eligible else 'seed_harm'))
    assert len(result)==28 and sum(v[3]=='frozen_miss' for v in result)==18
    return result


def pick(values,indices,joint):
    return int(indices[np.lexsort((indices,-joint[indices],-values[indices]))[0]])


def geometry(good,bad,gt):
    f={}
    for flags in itertools.product((0,1),repeat=4):
        b=bad.copy()
        for active,ix in zip(flags,SLICES):
            if active:b[ix]=good[ix]
        f[flags]=float(many_iou(b[None],gt)[0])
    phi=np.zeros(4)
    for flags in f:
        for j in range(4):
            if flags[j]:continue
            plus=tuple(1 if t==j else x for t,x in enumerate(flags))
            m=sum(flags)
            phi[j]+=(f[plus]-f[flags])*math.factorial(m)*math.factorial(3-m)/24
    assert abs(phi.sum()-(f[(1,1,1,1)]-f[(0,0,0,0)]))<1e-10
    return dict(zip(PARTS,phi)),f


def inside(points,box):
    offset=points-box[:3]
    c,s=np.cos(box[6]),np.sin(box[6])
    x=offset[:,0]*c+offset[:,1]*s
    y=-offset[:,0]*s+offset[:,1]*c
    return (abs(x)<=box[3]/2)&(abs(y)<=box[4]/2)&(abs(offset[:,2])<=box[5]/2)


def raw_points(row):
    p=Path(row['pointpath'])
    assert digest(p)==row['point_sha256']
    pts=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4)
    return raw_to_model_points(pts[:,:3],row['platform'],row['suffix'] or 0)


def comparison_rows(terms,scores,ious,good,bad):
    out={}
    for k,n in enumerate(NAMES):
        out['gap_'+n]=float(terms[bad,k])-float(terms[good,k])
    for k,n in enumerate(('frozen','independent','reference_mismatch')):
        out['gap_'+n]=float(scores[bad,k])-float(scores[good,k])
        if n!='frozen':
            out[n+'_winner_iou']=float(ious[np.argmax(scores[:,k])])
    out['gap_reference_total']=out['gap_frozen']-out['gap_independent']
    assert abs(out['gap_prior']+out['gap_subject_joint']+out['gap_reference']-out['gap_frozen'])<8e-6
    return out


def frontend_terms(front,row,slate,boxes,expected,raw):
    obs=observation(row)
    args={k:obs[k] for k in ('target','points','text','seed','waymo_view')}
    scene=front.score_frame(**args)
    np.testing.assert_array_equal(raw[:,:6],scene.boxes)
    np.testing.assert_array_equal(scene.boxes[slate,:3],boxes[:,:3])
    kernel=overlaps(scene.boxes)
    _,utility=select_grounding_box(scene.boxes,scene.probability)
    prior0=np.log(np.maximum(utility,1.e-80))
    centered=prior0-prior0.max()
    registered=centered.clip(-20,0).astype(np.float32)
    np.testing.assert_array_equal(registered[slate],expected)
    prob=scene.probability
    assert (prob>0).all()
    d=(kernel*kernel).sum(1)
    own=prob/d
    total=kernel@own
    args3=np.stack((np.log(prob),-np.log(d),np.log(total/own)),axis=1)
    np.testing.assert_allclose(args3.sum(1)-args3.sum(1).max(),centered,rtol=0,atol=4e-11)
    terms=np.c_[args3,centered.clip(-20,0)-centered][slate]
    slots=np.array(diverse_order(utility,kernel,4))
    original=np.stack([orient_box(scene.sampled_points,scene.boxes[k]) for k in slots])
    return terms,slots,original


def case_rows():
    PRIV.mkdir(mode=0o700,exist_ok=True)
    PUB.mkdir(exist_ok=True)
    active=records()
    seal=read(OUT/'SEALED.json')
    gtseal=read(SEED/'run_PRIVATE/postseal/SEALED.json')
    rays=read(PARENT/'SEALED.json')
    outputs=[]
    for platform in ('waymo','drone','quad'):
        inputs=prior.source_inputs(platform)
        archived=prior.sealed_arrays(prior.R181,'SEALED.json','source_scores',platform,
                           ('ious','terms','scores','incumbent','chosen','has_reference'))
        path=SEED/'run_PRIVATE/postseal'/(platform+'.npz')
        assert digest(path)==gtseal['files'][path.name]
        with np.load(path) as src:
            lookup={int(v):k for k,v in enumerate(src['ordinal'])}
            gt_codes=src['gt_codes']
            audited=src['ious']
        # Only the exact frozen pre-vote probability needs replay. It wasn't in the seed cache.
        front=None
        for cid,p,i,cohort in active:
            if p!=platform:continue
            row=inputs[i]
            rel=f'{p}/{i:05}.npz'
            assert digest(OUT/rel)==seal['files'][rel] and digest(PARENT/rel)==rays['files'][rel]
            with np.load(PARENT/rel,allow_pickle=False) as rz:
                raw=rz['full'][-1]
            with np.load(OUT/rel,allow_pickle=False) as z:
                choices,joint,slate=z['choices'],z['joint'],z['slate']
                boxes=z['boxes'].astype(float)
                gtall,classes=inventory(row)
                gt=gtall[0].astype(float)
                ious=many_iou(boxes,gt)
                np.testing.assert_allclose(ious,audited[lookup[i]],rtol=0,atol=1e-7)
                np.testing.assert_allclose(ious,archived['ious'][i],rtol=0,atol=1e-7)
                terms=archived['terms'][i].astype(float)
                scores=archived['scores'][i].astype(float)
                np.testing.assert_array_equal(scores[:,0],joint)
                native=z['fields'][FIELDS.index('native_focal')]
                f,s=map(int,choices[[ARMS.index('frozen_joint'),ARMS.index('native_focal_abstain')]])
                assert f==int(choices[0]) and f==np.argmax(joint)
                passed=np.flatnonzero(ious>.5)
                failed=np.flatnonzero(ious<=.5)
                rank=joint if cohort=='frozen_miss' else native
                b=pick(rank,failed,joint)
                g=pick(rank,passed,joint)
                assert b==(f if cohort=='frozen_miss' else s)
                oracle=pick(ious,passed,joint)
                orig_gt=gt_codes[lookup[i]]
                local=PRIV/(cid+'_front.npz')
                if local.exists():
                    with np.load(local) as saved:
                        front_terms,slots4,boxes4=(saved[n] for n in ('terms','slots4','boxes4'))
                        # Keep the cached replay tied to this prediction and frozen model.
                        assert str(saved['prediction'])==seal['files'][rel]
                else:
                    if front is None:
                        front=CandidateFrontend(platform)
                    front_terms,slots4,boxes4=frontend_terms(front,row,slate,boxes,terms[:,0],raw)
                    np.savez_compressed(local,terms=front_terms,slots4=slots4,boxes4=boxes4,
                                        prediction=seal['files'][rel])
                fg=np.flatnonzero(slate==slots4[archived['incumbent'][i]])
                assert len(fg)==1 and fg[0]==f
                iou4=many_iou(boxes4,gt)
                codes=z['codes']
                lp,ln=focal_costs(z['logits'])
                w=ln-lp
                pos=codes[g]; neg=codes[b]
                pos_only=pos&~neg; neg_only=neg&~pos
                seed_parts=dict(
                    bad_code_credit=float(w[neg_only].clip(min=0).sum()),bad_code_penalty=float(w[neg_only].clip(max=0).sum()),
                    good_code_credit=float(w[pos_only].clip(min=0).sum()),good_code_penalty=float(w[pos_only].clip(max=0).sum()),
                    gap_saved_negative_cost=float(ln[neg].sum()-ln[pos].sum()),
                    gap_positive_cost=float(lp[neg].sum()-lp[pos].sum()),
                    extra_gt_code_contribution=float(w[neg_only&orig_gt].sum()-w[pos_only&orig_gt].sum()),
                    extra_non_gt_contribution=float(w[neg_only&~orig_gt].sum()-w[pos_only&~orig_gt].sum()),
                    bad_positive_credit=float(w[neg& (w>0)].sum()),good_positive_credit=float(w[pos&(w>0)].sum()))
                seed_gap=float(native[b]-native[g])
                assert abs(seed_parts['gap_saved_negative_cost']-seed_parts['gap_positive_cost']-seed_gap)<1e-9
                assert abs(seed_parts['bad_code_credit']+seed_parts['bad_code_penalty']-
                           seed_parts['good_code_credit']-seed_parts['good_code_penalty']-seed_gap)<1e-9
                geom,grid=geometry(boxes[g],boxes[b],gt)
                full_cloud=raw_points(row)
                target_hits=inside(full_cloud,gt)
                bad_hits=inside(full_cloud,boxes[b])
                good_hits=inside(full_cloud,boxes[g])
                inventory_ious=many_iou(gtall,boxes[b])
                other_idx=int(1+np.argmax(inventory_ious[1:])) if len(gtall)>1 else None
                d=dict(case_id=cid,cohort=cohort,platform=platform,good_iou=float(ious[g]),bad_iou=float(ious[b]),
                    old_winner_iou=float(ious[f]),seed_winner_iou=float(ious[s]),pool_max_iou=float(ious[oracle]),
                    four_max_iou=float(iou4.max()),four_has_hit=bool((iou4>.5).any()),
                    good_is_original_four=bool(slate[g] in slots4),bad_is_original_four=bool(slate[b] in slots4),
                    original_is_best_seed_good=bool(f==g) if cohort=='seed_harm' else None,
                    positive_seed_max=bool(native.max()>0),seed_actually_switched=bool(f!=s),has_reference=bool(archived['has_reference'][i]),
                    good_code_size=int(pos.sum()),bad_code_size=int(neg.sum()),codes_shared=int((pos&neg).sum()),
                    good_gt_code_hits=int((pos&orig_gt).sum()),bad_gt_code_hits=int((neg&orig_gt).sum()),
                    good_exact_gt_code=bool(np.array_equal(pos,orig_gt)),bad_exact_gt_code=bool(np.array_equal(neg,orig_gt)),
                    good_seed_score=float(native[g]),bad_seed_score=float(native[b]),gap_seed=seed_gap,
                    gap_seed_bad_minus_actual_frozen=float(native[b]-native[f]),
                    gap_joint_bad_minus_actual_frozen=float(joint[b]-joint[f]),
                    best_other_gt_iou=float(inventory_ious[other_idx]) if other_idx is not None else None,
                    annotated_object_count=len(gtall),bad_target_shared_points=int((bad_hits&target_hits).sum()),
                    bad_inside_points=int(bad_hits.sum()),target_inside_points=int(target_hits.sum()),
                    good_target_shared_points=int((good_hits&target_hits).sum()),good_inside_points=int(good_hits.sum()),
                    bad_other_shared_points=int(np.count_nonzero(bad_hits&inside(full_cloud,gtall[other_idx]))) if other_idx is not None else None,
                    center_error_xy_m=float(np.linalg.norm(boxes[b,:2]-gt[:2])),center_error_z_m=float(abs(boxes[b,2]-gt[2])),
                    bad_gt_long_axis_diff_degrees=float(abs(((boxes[b,6]+np.pi/2*(boxes[b,4]>boxes[b,3])-
                        gt[6]-np.pi/2*(gt[4]>gt[3])+np.pi/2)%np.pi)-np.pi/2)*180/np.pi),
                    **comparison_rows(terms,scores,ious,g,b),**seed_parts,**{'geometry_'+k:v for k,v in geom.items()},
                    **{'front_gap_'+k:float(front_terms[b,j]-front_terms[g,j]) for j,k in enumerate(PF)})
                assert abs(sum(d['front_gap_'+k] for k in PF)-d['gap_prior'])<3e-6
                # Compare the strongest failed and passing candidate over the entire K16.
                # The bounded slot score spans 1.5 and the normalized reference marginal spans 1.
                active_ref=bool(d['has_reference'])
                unary=terms[:,0]+terms[:,2]
                d['prior_gap_best_fail_minus_best_hit']=float(terms[failed,0].max()-terms[passed,0].max())
                d['prior_unary_gap_best_fail_minus_best_hit']=float(unary[failed].max()-unary[passed].max())
                d['beyond_full_correction_range']=d['prior_gap_best_fail_minus_best_hit']>1.5+int(active_ref)+1e-6
                d['beyond_direct_reference_range']=d['prior_unary_gap_best_fail_minus_best_hit']>int(active_ref)+1e-6
                for tag,vec in [('prior_only',terms[:,0]),('own_text_only',front_terms[:,0]),('unary_only',terms[:,2]),
                    ('remove_prior',terms[:,2]+terms[:,3]),('remove_direct_relation',terms[:,0]+terms[:,2]),
                    ('remove_subject_unary',terms[:,0]+terms[:,3]),('remove_neighbors',scores[:,0]-front_terms[:,2]),
                    ('remove_density',scores[:,0]-front_terms[:,1])]:
                    winner=int(np.argmax(vec))
                    d[tag+'_winner_iou']=float(ious[winner])
                    d[tag+'_pair_gap']=float(vec[b]-vec[g])
                priv=dict(case_id=cid,cohort=cohort,platform=platform,ordinal=i,data=row,raw_boxes=boxes.tolist(),
                    annotations=gtall.tolist(),classes=classes.tolist(),good=g,bad=b,previous=f,seed=s,oracle=oracle,
                    other=other_idx,ious=ious.tolist(),terms=terms.tolist(),scores=scores.tolist(),front_terms=front_terms.tolist(),
                    pca_indices=slate.tolist(),view=int(z['view']),
                    joint_good=pick(joint,passed,joint),joint_bad=pick(joint,failed,joint),
                    seed_good=pick(native,passed,joint),seed_bad=pick(native,failed,joint),
                    grid={''.join(map(str,k)):v for k,v in grid.items()})
                for k in ('joint_good','joint_bad','seed_good','seed_bad'):
                    d[k+'_iou']=float(ious[priv[k]])
                d['paired_good_frozen_rank']=1+int(np.count_nonzero(joint>joint[g]))
                d['paired_good_seed_rank']=1+int(np.count_nonzero(native>native[g])+np.count_nonzero((native==native[g])&(joint>joint[g])))
                atomic(PRIV/(cid+'.json'),priv)
                outputs.append(d)
        del front
    outputs.sort(key=lambda d:d['case_id'])
    write_csv(PUB/'paired_scores.csv',outputs)
    return outputs


if __name__=='__main__':
    prepare_device()
    results=case_rows()
    print('Paired cases',len(results))
