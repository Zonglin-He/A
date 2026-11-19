"""Conditioned reference interventions, with source truth and a post-seal target panel."""
from collections import defaultdict
import gc
import sys

import numpy as np
import torch

from scope174 import (CELLS, FACTOR, ORDER, PLATFORMS, PRIVATE, R165_DATA, RESULTS, ROOT,
                      camera_view, json_atomic, ordinary_rows, read, recording, frame, sha)
from predict174 import BATCH, OUTPUT, native_encoder, target_evidence, trained, verify_predictions
from train174 import load_split, slice_fields
from batch_yaw_iou import many_iou
from score import labels, local_points, owner_for

sys.path.insert(0,str(FACTOR))
from reference_truth import library, role_phrase, truth_pins, verified_refs

CONDITIONS = ('natural_readout','correct_bound','wrong_descriptor_fixed_geometry','wrong_physical_reference')


def scene_inventory(row, annotation_inventory):
    # Synthetic source descriptions have a supervised box independently of the
    # original natural annotation order. Any starting index gives the full inventory.
    index=row.get('object_index',0)
    return annotation_inventory(row['metapath'],row['platform'],camera_view(row),index)


def choose_example(row, xyz, visibility, subject_truth, correct, annotation_inventory):
    names, boxes = scene_inventory(row,annotation_inventory)
    unique, first = np.unique(xyz,axis=0,return_index=True)
    inside = np.stack([local_points(unique, b) for b in boxes])
    exclusive = inside & (inside.sum(0)==1)
    counts = exclusive.sum(1)
    visible_count = (exclusive & visibility[first]).sum(1)
    total = len(names)
    eligible = (counts>=3)&(visible_count>=2)
    if not eligible[correct]:
        return 'reference_support_insufficient', None
    subject_match = many_iou(boxes,subject_truth)>.95
    candidates = [j for j in range(total) if j!=correct and not subject_match[j]
                  and names[j]==names[correct] and eligible[j]]
    if not candidates:
        return 'no_distinct_same_class_supported_foil', None
    def disparity(i):
        metric = abs(np.log1p(counts[i])-np.log1p(counts[correct]))
        metric += abs(np.log1p(visible_count[i])-np.log1p(visible_count[correct]))
        metric += np.abs(np.log(boxes[i,3:6]/boxes[correct,3:6])).mean()
        return metric, float(np.linalg.norm(boxes[i,:3]-boxes[correct,:3])), i
    foil = min(candidates, key=disparity)
    own = np.where(inside.sum(0)==1,inside.argmax(0),total)
    mask = np.zeros((2,len(xyz)),np.float32)
    for k,j in enumerate((correct,foil)):
        mask[k,first] = exclusive[j]/counts[j]
    return 'compared',dict(placements=boxes[[correct,foil]],weights=mask,
                           correct=correct,foil=foil,owner_points=unique,owners=own,inventory_size=total,
                           true_points=int(counts[correct]),foil_points=int(counts[foil]))


def save_table(path, rows):
    import csv
    with path.with_suffix('.tmp').open('w',encoding='utf-8',newline='') as f:
        d=csv.DictWriter(f,list(rows[0]),lineterminator='\n')
        d.writeheader()
        d.writerows(rows)
    path.with_suffix('.tmp').replace(path)


@torch.inference_mode()
def compare(models,feat,support,gt,record):
    arms = ('self_only','cross_only','full')
    coordinates = torch.as_tensor(support['placements'],dtype=torch.float32,device='cuda')
    weight = torch.as_tensor(support['weights'],dtype=torch.float32,device='cuda')
    interventions = [
        None,
        (coordinates[:1],weight[:1],weight[:1]),
        (coordinates[:1],weight[:1],weight[1:]),
        (coordinates[1:],weight[1:],weight[1:])]
    data = {}
    slots = feat['subject_boxes'][0].cpu().numpy()
    scene = (support['owner_points'],support['owners'],support['inventory_size'])
    for arm in arms:
        choices, iou, scores, owners = [], [], [], []
        for change in interventions:
            out = models[arm](feat,reference_override=change)
            coords = out['boxes'][0].cpu().numpy()
            choices.append(out['choices'][0].cpu().tolist())
            iou.append(float(many_iou(coords[:1],gt)[0]))
            owners.append(owner_for(coords[0],*scene))
            scores.append((out['subject_logits'][0]-out['subject_logits'][0,0]).cpu().numpy())
        if arm=='self_only':
            np.testing.assert_allclose(scores,np.repeat(np.asarray(scores[:1]),4,0),rtol=0.,atol=2e-6)
            assert all(ix[0]==choices[0][0] for ix in choices)
        selected_owner = owner_for(feat['reference_boxes'][0,choices[0][1]].cpu().numpy(),*scene)
        identity = 'unknown' if selected_owner<0 else ('correct' if selected_owner==support['correct'] else 'wrong')
        data[arm]=dict(iou=iou, slot=choices, native_reference_identity=identity,
                       subject_owner=owners, subject_score_max_change_correct_vs_foil=[
                           float(np.max(np.abs(scores[1]-s))) for s in scores[2:]])
    record.update(endpoint=data,quality=many_iou(slots,gt).tolist(),
                  true_points=support['true_points'],foil_points=support['foil_points'])
    return record


def source_truth_index(row, annotation_inventory):
    _, boxes = scene_inventory(row,annotation_inventory)
    gt = np.asarray(row['boxes'][1])
    iou = many_iou(boxes,gt)
    j=int(iou.argmax())
    return j if iou[j]>.95 else None


def source_panel():
    _, annotation_inventory=library()
    all_items=[]
    for source in PLATFORMS:
        features=load_split(source,'monitor')
        model=trained(source)
        from scope174 import OLD_PRIVATE
        path=OLD_PRIVATE/source/'monitor_PRIVATE.json'
        roster=read(path)
        for i,row in enumerate(roster):
            if not row['known_reference']:
                continue
            a=dict(source=source,target=source,ordinal=i,recording=recording(row),
                   frame=frame(row),status='inventory_reference_unknown')
            correct=source_truth_index(row,annotation_inventory)
            f=slice_fields(features,slice(i,i+1))
            if correct is not None:
                a['status'],probe=choose_example(row,f['xyz'][0].cpu().numpy(),
                    f['visible'][0].cpu().numpy(),np.asarray(row['boxes'][0]),correct,annotation_inventory)
                if probe is not None:
                    compare(model,f,probe,np.asarray(row['boxes'][0]),a)
            all_items.append(a)
        del features, model
        gc.collect()
        torch.cuda.empty_cache()
        print('SOURCE_MATCHED_REF',source,sum(r['source']==source for r in all_items),flush=True)
    return all_items


def target_panel():
    verify_predictions()
    _, inventory = library()
    catalog=verified_refs()
    by_target={t:sorted(i for p,i in catalog if t==p) for t in PLATFORMS}
    found=[]
    for source in PLATFORMS:
        native=native_encoder(source)
        model=trained(source)
        for s,t in ORDER:
            if s!=source:
                continue
            route=f'{s}_to_{t}'
            seal=read(OUTPUT/route/'SEAL.json')
            input_dir=R165_DATA/route
            upstream=read(input_dir/'SEAL.json')
            used=ordinary_rows(t)
            for i in by_target[t]:
                row=used[i]
                marks=catalog[t,i]
                unique={(j,n) for phrase,j,n,_ in marks if phrase and phrase==role_phrase(row)}
                rec=dict(source=s,target=t,ordinal=i,recording=recording(row),frame=frame(row),
                         status='verified_phrase_unmatched')
                found.append(rec)
                if len(unique)!=1:
                    continue
                correct,n=next(iter(unique))
                names,_=inventory(row['metapath'],t,camera_view(row),row['object_index'])
                assert correct>0 and len(names)==n
                # Direction blocks are ordered and pinned to the same R165 rows.
                name=sorted(seal['files'])[i//128]
                p=input_dir/name
                assert sha(p)==upstream['files'][name]
                with np.load(p,allow_pickle=False) as z:
                    orig=dict(z)
                j=int(np.searchsorted(orig['ordinals'],i))
                assert orig['ordinals'][j]==i
                start=(j//BATCH)*BATCH
                stop=min(start+BATCH,len(orig['ordinals']))
                batch=[used[int(i)] for i in orig['ordinals'][start:stop]]
                feat,_=target_evidence(s,batch,native,orig,start,stop)
                feat=slice_fields(feat,slice(j-start,j-start+1))
                assert bool(feat['has_reference'][0])
                truth,_=labels(row)
                rec['status'],probe=choose_example(row,feat['xyz'][0].cpu().numpy(),
                    feat['visible'][0].cpu().numpy(),truth,correct,inventory)
                if probe is not None:
                    compare(model,feat,probe,truth,rec)
                    # Check naturally decoded endpoint against the sealed outcome;
                    # batch-size numerical changes may only be assessed, never repaired by labels.
                    with np.load(OUTPUT/route/name,allow_pickle=False) as q:
                        nat=q['choices'][j]
                        exact={arm:rec['endpoint'][arm]['slot'][0]==nat[CELLS.index(arm)].tolist()
                               for arm in ('self_only','cross_only','full')}
                    rec['natural_replay_matches']=exact
                    rec['sealed_natural_iou']={arm:float(many_iou(
                        feat['subject_boxes'][0,nat[CELLS.index(arm),0]].cpu().numpy()[None],truth)[0])
                        for arm in ('self_only','cross_only','full')}
            print('POST_SEAL_VERIFIED_REF',route,len(by_target[t]),flush=True)
        del native,model
        gc.collect()
        torch.cuda.empty_cache()
    return found,dict(truth_pins=truth_pins(),prediction=sha(OUTPUT/'PREDICTION_SEAL.json'))


def findings(all_items, split):
    groups={p:[r for r in all_items if r['target']==p] for p in PLATFORMS}
    groups['pooled']=all_items
    result=[]
    for name, samples in groups.items():
        chosen=[r for r in samples if r['status']=='compared']
        for arm in ('self_only','cross_only','full'):
            matched=[r['endpoint'][arm] for r in chosen]
            y=np.array([p['iou'] for p in matched],np.float64).reshape(-1,4)
            r={}
            for k,m in zip(CONDITIONS,y.T,strict=True):
                r.update({f'{k}_mIoU':float(100*m.mean()) if len(m) else '',
                          f'{k}_Acc25':float(100*(m>.25).mean()) if len(m) else '',
                          f'{k}_Acc50':float(100*(m>.50).mean()) if len(m) else ''})
            for j,k in enumerate(('descriptor','physical'),2):
                d=y[:,1]-y[:,j]
                r.update({f'{k}_correct_improves':int((d>1e-10).sum()),
                          f'{k}_correct_worsens':int((d< -1e-10).sum()),
                          f'{k}_ties':int((abs(d)<=1e-10).sum()),
                          f'{k}_different_subject':sum(p['slot'][1][0]!=p['slot'][j][0] for p in matched),
                          f'{k}_score_shift_mean':float(np.mean([p['subject_score_max_change_correct_vs_foil'][j-2]
                                                                for p in matched])) if matched else ''})
            for threshold in (.25,.50):
                name25=str(int(threshold*100))
                missed=[sample for sample,p in zip(chosen,matched,strict=True)
                        if p['native_reference_identity']=='wrong'
                        and sample.get('natural_replay_matches',{}).get(arm,True)
                        and sample.get('sealed_natural_iou',{}).get(arm,p['iou'][0])<=threshold
                        and max(sample['quality'])>threshold]
                r[f'wrong_native_reference_and_missed_at{name25}']=len(missed)
                r[f'correct_binding_recovers_missed_at{name25}']=sum(
                    sample['endpoint'][arm]['iou'][1]>threshold for sample in missed)
            result.append(dict(split=split,target=name,arm=arm,n=len(samples),qualified=len(chosen),
                               recordings=len({p['recording'] for p in chosen}),
                               frames=len({p['frame'] for p in chosen}),
                               natural_reference_correct=sum(p['native_reference_identity']=='correct' for p in matched),
                               natural_reference_wrong=sum(p['native_reference_identity']=='wrong' for p in matched),
                               natural_reference_unknown=sum(p['native_reference_identity']=='unknown' for p in matched),
                               natural_replay_choice_mismatch=sum(not p.get('natural_replay_matches',{}).get(arm,True)
                                                                  for p in chosen),**r))
    return result


def anonymous_pairs(items):
    rng=np.random.default_rng(1743)
    fields=dict.fromkeys((*CONDITIONS,'max_candidate_iou','native_reference_identity'),'')
    rows=[]
    for i in rng.permutation(len(items)):
        item=items[i]
        base={k:item[k] for k in ('source','target','status')}
        if item['status']!='compared':
            rows.append(dict(**base,arm='',**fields))
            continue
        for arm,entry in item['endpoint'].items():
            values=dict(zip(CONDITIONS,entry['iou'],strict=True))
            values.update(max_candidate_iou=max(item['quality']),
                          native_reference_identity=entry['native_reference_identity'])
            rows.append(dict(**base,arm=arm,**values))
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('panel',choices=('source','target'))
    mode=p.parse_args().panel
    RESULTS.mkdir(exist_ok=True)
    if mode=='source':
        rows=source_panel()
        proof={'source_trained_heads':{s:sha(PRIVATE/s/'MODEL_SEAL.json') for s in PLATFORMS}}
    else:
        rows,proof=target_panel()
    assert rows
    json_atomic(PRIVATE/f'{mode}_identity_PRIVATE.json',dict(rows=rows, **proof, FINAL_access=False))
    counts=defaultdict(int)
    for r in rows:
        counts[r['status']]+=1
    proof.update(n=len(rows),status=dict(counts),FINAL_access=False, selection_used_target_labels=False,
                 descriptive_only=True)
    save_table(RESULTS/f'{mode}_reference_intervention.csv',findings(rows,mode))
    save_table(RESULTS/f'{mode}_reference_paired.csv',anonymous_pairs(rows))
    json_atomic(RESULTS/f'{mode}_reference_intervention_receipt.json',proof)
    print(mode,'REFERENCE_INTERVENTION',dict(counts),flush=True)


if __name__=='__main__':
    main()
