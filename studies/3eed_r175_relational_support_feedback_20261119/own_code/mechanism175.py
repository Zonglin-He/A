"""Post-seal reference routing probes with annotated subject and reference identities."""
from collections import defaultdict
import gc
import sys

import numpy as np
import torch

from scope175 import (HERE,N,PLATFORMS,PRIVATE,R165_DATA,RESULTS,camera_view,frame,json_atomic,
                      ordinary_rows,read,recording,sha)
from predict175 import BATCH,OUTPUT,PRED_ARMS,encoder,evidence_batch,models_for,verify
from roles175 import roles_batch
from score175 import write_csv
from train175 import load_split,select
from decoder import in_boxes
from score import labels
from batch_yaw_iou import many_iou

FACTOR=HERE.parent/'3eed_role_geometry_factorial_20261119'
sys.path.insert(0,str(FACTOR))
from reference_truth import library,phrase_key,truth_pins,verified_refs


def eligible(q,foil,kind,support):
    if not kind:
        return 'unsupported_predicate',None
    if min(q[0].max(),q[1].max())<=.1:
        return 'missing_labelled_candidate',None
    if not support:
        return 'unobserved_subject_or_reference',None
    ref=q[1].argmax()
    alternatives=np.flatnonzero(foil & (q[1]<.10) & (np.arange(len(foil))!=ref))
    if not len(alternatives):
        return 'no_same_category_candidate_foil',None
    return 'paired',(int(q[0].argmax()),int(ref),int(alternatives[0]))


@torch.inference_mode()
def mechanism(model,row,feat,q,truth,choice,metadata):
    xyz=feat['xyz']
    gt=torch.as_tensor(truth,dtype=torch.float32,device=xyz.device)[None]
    valid_samples=in_boxes(xyz,gt)
    gt_subject=valid_samples[0,0]&~valid_samples[0,1]
    selected,ref,wrong=choice
    def evaluate(bind):
        idx=torch.tensor([bind],dtype=torch.long,device='cuda')
        d=model(feat,reference_bind=idx)
        target=int(d['choices'][0,0])
        first=d['passes'][0][2][0,0,selected]
        last=d['passes'][-1][2][0,0,selected]
        return target,float(q[0,target]),float(first[gt_subject].sum()),float(last[gt_subject].sum())
    good,bad=evaluate(ref),evaluate(wrong)
    assert abs(good[2]-bad[2])<1e-7
    out=dict(source=metadata['source'],target=metadata['target'],status='paired',
             frame=frame(row),recording=recording(row),correct_iou=good[1],wrong_iou=bad[1],
             same_subject_candidate=int(good[0]==bad[0]),
             pre_feedback_mass=good[2],correct_mass=good[3],wrong_mass=bad[3])
    out.update({k:v for k,v in metadata.items() if k not in out})
    return out


def source_panels():
    ledger=[]
    for source in PLATFORMS:
        folder=PRIVATE/source
        rows=read(folder/'monitor_PRIVATE.json')
        data=load_split(source,'monitor')
        models=models_for(source)
        good=data['known_ref'].cpu().numpy()
        for i,row in enumerate(rows):
            if not good[i]:
                continue
            f=select(data,slice(i,i+1))
            q=f['quality'][0].cpu().numpy()
            foils=f['foil'][0,1].cpu().numpy()
            xyz=f['xyz']
            g=f['gt'][0]
            mask=in_boxes(xyz,g[None])[0]
            metadata={'source':source,'target':source}
            status,choice=eligible(q,foils,row['relation'],bool((mask & ~mask.flip(0)).any(-1).all()))
            if choice is not None:
                ledger.append(mechanism(models['full'],row,f,q,g.cpu().numpy(),choice,metadata))
            else:
                ledger.append(dict(**metadata,status=status,frame=frame(row),recording=recording(row)))
        del data,models
        gc.collect()
        torch.cuda.empty_cache()
    return ledger


def target_panels():
    _, inventory=library()
    audited=verified_refs()
    out=[]
    for source in PLATFORMS:
        # Build a model anew to make the absence of target adaptation explicit.
        loaded=encoder(source)
        models=models_for(source)
        for target in PLATFORMS:
            if source==target:
                continue
            direction=f'{source}_to_{target}'
            inputs=R165_DATA/direction
            routes=OUTPUT/direction
            source_seal=read(inputs/'SEAL.json')
            locked=read(routes/'SEAL.json')
            ordinary=ordinary_rows(target)
            for (p,i),truth in sorted(audited.items()):
                if p!=target:
                    continue
                row=ordinary[i]
                base=dict(source=source,target=target,ordinal=int(i),status='verified_role_unmatched',
                          frame=frame(row),recording=recording(row))
                out.append(base)
                context=roles_batch([row])[0]
                lo,hi=context['spans'][1]
                key=phrase_key(row['caption'][lo:hi])
                options={(j,n) for phrase,j,n,_ in truth if phrase and phrase==key}
                if len(options)!=1:
                    continue
                ref,n=next(iter(options))
                names,boxes=inventory(row['metapath'],target,camera_view(row),row['object_index'])
                assert len(names)==n and ref>0
                path=sorted(locked['files'])[i//128]
                f=routes/path
                assert sha(f)==locked['files'][path]
                with np.load(f,allow_pickle=False) as z:
                    j=int(np.searchsorted(z['ordinals'],i))
                    assert z['ordinals'][j]==i
                    candidates=z['slot_boxes'][j].copy()
                    relation=int(z['relation'][j])
                    native_slots=z['choice'][j].copy()
                    stored_seeds=z['xyz'][j].copy()
                assert context['relation']==relation
                target_gt,_=labels(row)
                truth=np.stack((target_gt,boxes[ref]))
                q=np.stack([many_iou(candidates[k],truth[k]) for k in (0,1)])
                real=np.asarray(names)==names[ref]
                real[[0,ref]]=False
                other=boxes[real]
                foil=np.asarray([many_iou(other,b).max(initial=0.)>.10 for b in candidates[1]])
                raw=torch.from_numpy(stored_seeds)[None]
                masks=in_boxes(raw,torch.as_tensor(truth,dtype=torch.float32)[None])[0]
                status,chosen=eligible(q,foil,relation,bool((masks & ~masks.flip(0)).any(-1).all()))
                base.update(status=status,normal_reference_iou={
                    arm:float(q[1,native_slots[k,1]]) for k,arm in enumerate(PRED_ARMS)},
                    normal_subject_iou={arm:float(q[0,native_slots[k,0]]) for k,arm in enumerate(PRED_ARMS)})
                if chosen is None:
                    continue
                p=inputs/path
                assert sha(p)==source_seal['files'][path]
                with np.load(p,allow_pickle=False) as z:
                    block=dict(z)
                k=(j//BATCH)*BATCH
                batch,slots=evidence_batch(source,target,block,loaded,k,min(k+BATCH,len(block['ordinals'])))
                np.testing.assert_array_equal(slots[j-k],candidates)
                feat=select(batch,slice(j-k,j-k+1))
                np.testing.assert_array_equal(feat['xyz'][0].cpu().numpy(),stored_seeds)
                base.update(mechanism(models['full'],row,feat,q,truth,chosen,base))
        del loaded,models
        gc.collect()
        torch.cuda.empty_cache()
    return out


def aggregate(both):
    fields=('correct_iou','wrong_iou','pre_feedback_mass','correct_mass','wrong_mass','same_subject_candidate')
    anon,summary=[],[]
    rng=np.random.default_rng(1759)
    for split,rows in both.items():
        for src in (*PLATFORMS,'pooled'):
            for key in ('all','paired'):
                records=[r for r in rows if (r['target']==src or src=='pooled')
                         and (key=='all' or r['status']=='paired')]
                if not records:
                    continue
                n=len(records)
                paired=[r for r in records if r['status']=='paired']
                effects=np.asarray([[r[k] for k in fields] for r in paired],float).reshape(-1,len(fields))
                known=[r for r in records if 'normal_reference_iou' in r]
                shift=effects[:,3]-effects[:,4]
                d=effects[:,0]-effects[:,1]
                endpoints={k:float(v.mean()) if len(v) else '' for k,v in zip(fields,effects.T,strict=True)}
                summary.append(dict(split=split,platform=src,cohort=key,n=n,paired_n=len(paired),
                    recordings=len({r['recording'] for r in records}),frames=len({r['frame'] for r in records}),
                    correct_ref_support_gain=int((shift>1e-7).sum()),correct_ref_support_loss=int((shift< -1e-7).sum()),
                    correct_ref_iou_gain=int((d>1e-10).sum()),correct_ref_iou_loss=int((d< -1e-10).sum()),
                    normal_known_n=len(known),normal_full_ref25=sum(r['normal_reference_iou']['full']>.25 for r in known),
                    normal_full_both25=sum(r['normal_reference_iou']['full']>.25 and
                                           r['normal_subject_iou']['full']>.25 for r in known),**endpoints))
        for j in rng.permutation(len(rows)):
            r=rows[j]
            anon.append(dict(split=split,source=r['source'],target=r['target'],status=r['status'],
                             **{k:r.get(k,'') for k in fields}))
    return summary,anon


def main():
    checked=verify()
    assert checked['n']==N
    both={'source':source_panels(),'target':target_panels()}
    summary,rows=aggregate(both)
    json_atomic(PRIVATE/'MECHANISM_PRIVATE.json',dict(rows=both,
                    target_label_sources=truth_pins(),prediction=sha(OUTPUT/'PREDICTION_SEAL.json')))
    a=RESULTS/'mechanism_summary.csv'
    b=RESULTS/'mechanism_rows.csv'
    write_csv(a,summary)
    write_csv(b,rows)
    detail={}
    for p,panel in both.items():
        notes=defaultdict(int)
        for r in panel:
            notes[r['status']]+=1
        detail[p]=dict(notes)
    json_atomic(RESULTS/'mechanism_status.json',dict(denominators=detail,sealed_before_diagnostic=True,
                              diagnostic_rows_hashed=sha(b),diagnostic_summary_hashed=sha(a),FINAL_access=False))
    print('R175_RELATIONAL_INTERVENTIONS',detail,flush=True)


if __name__=='__main__':
    main()
