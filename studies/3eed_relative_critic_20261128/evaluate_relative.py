"""Score fixed candidates and audit competitive prototypes after prediction sealing."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata
import torch

from source_geometry import HERE,ROOT,RUN,PRIOR,read,digest,write
from predict_relative import ARMS,RANKS
from score_paradigms import ious
from competitive import heads,patch_ids
from physical_readout import axes


def hits(points,box):
    box=np.asarray(box,float)
    return (abs((points-box[:3])@axes(box[6]))<=box[3:6]/2+1.e-6).all(1)


def same_class(frame,label,target):
    from score import canonical_box,truth_for_slot

    items=read(label['metadata'])['ground_info']
    slot=label['slot']
    category=items[slot].get('class')
    if category is None:
        return []
    others=[]
    for i,v in enumerate(items):
        if i==slot or str(v.get('class')).casefold()!=str(category).casefold():
            continue
        raw=np.asarray(v.get('bbox_3d'),float)
        expected=(7,) if frame['target']=='waymo' else (9,)
        if raw.shape!=expected or not np.isfinite(raw).all() or min(raw[3:6])<=0 or not v.get('caption'):
            continue
        shadow=dict(frame,slot=i,text=v['caption'],calibration=label['metadata'])
        b=canonical_box(truth_for_slot(shadow,items))
        if float(ious([b],target)[0])<.50:
            others.append(b)
    return others


def pair_auc(field,positive,negative):
    p,n=field[positive],field[negative]
    gap=p[:,None]-n[None]
    return float(np.mean((gap>0)+.5*(gap==0)))


def separation(case,binding,visual,target,others):
    old=visual['legacy']
    stage=heads(case['base']['readout'],visual['stages'])
    region_data=binding.get('C') if stage is visual['stages'].get('C') else binding.get('H')
    if old is None or stage is None or not region_data:
        return None
    by={r['id']:r for r in region_data}
    measures={}
    primary=stage['ids'][0]
    arrays=[np.asarray(v,float) for v in (old['mask'],stage['compact'])]
    assert arrays[0].shape==arrays[1].shape
    height,width=arrays[0].shape
    positive=set(); proxy=set(); typed=set()
    alignment={}
    for rid in stage['ids']:
        region=by[rid]
        points=np.asarray(region['xyz'])
        pos=hits(points,target)
        pix=patch_ids(region['seeds'],case['size'],(height,width))
        alignment[rid]=dict(purity=float(pos.mean()),points=int(pos.sum()))
        positive.update(pix[pos].tolist())
        if rid!=primary:
            neg=~pos
            proxy.update(pix[neg].tolist())
            if others:
                rival=np.logical_or.reduce([hits(points,x) for x in others]) & neg
                typed.update(pix[rival].tolist())
    y,x=np.mgrid[:height,:width]
    uv=np.stack(((x+.5)*case['size'][0]/width,(y+.5)*case['size'][1]/height),-1).reshape(-1,2)
    xy=np.asarray(case['window'])
    support=(abs(uv-(xy[:2]+xy[2:])/2)/((xy[2:]-xy[:2])/2)).max(1)<=1.15
    for name,group in (('proxy',proxy),('same_class',typed)):
        p=np.array(sorted((positive-group)&set(np.flatnonzero(support))),int)
        n=np.array(sorted((group-positive)&set(np.flatnonzero(support))),int)
        if len(p) and len(n):
            measures[name]=[pair_auc(m.ravel(),p,n) for m in arrays]
    allowed=[(k,v) for k,v,permit in stage['admission'] if permit]
    if allowed:
        new=max(allowed,key=lambda x:x[1])[0]
        ok=lambda k:(alignment[k]['purity']>=.50 and alignment[k]['points']>=3)
        measures['binding']=[ok(primary),ok(new)]
    return measures


def rhofit(values,target):
    a,b=rankdata(values),rankdata(target)
    if a.std()>0 and b.std()>0:
        return float(np.corrcoef(a,b)[0,1])
    return None


def main():
    torch.set_num_threads(1)
    from shared import setup
    setup()
    from evaluate_structure import stat_panel
    from metric_image_common import inputs,METRIC
    from truth_score_inputs import truth_data

    out=RUN/'target'
    lock=read(out/'prediction-seal.json')
    pred=out/'boxes-private.json'
    assert lock['payload']==digest(pred) and lock['n']==873 and not lock['labels_read']
    assert lock['visual']==digest(out/'visual-seal.json') and lock['training']==digest(RUN/'source'/'training-seal.json')
    assert lock['code']==digest(HERE/'predict_relative.py')
    vis=read(out/'visual-seal.json')
    fields=out/'visual-private.json'; regs=out/'regions-private.json'
    assert vis['visual_sha']==digest(fields) and vis['regions_sha']==digest(regs)
    art=PRIOR/'critic-blind-private.json'
    assert lock['prior_candidate_pool']==digest(art)

    mono=ROOT/'runs/2d-pose-finalization-20261128'
    sys.path.insert(0,str(HERE.parent/'3eed_2d_pose_finalization_20261128'))
    from run_mono import labels_for_panel

    membership=read(PRIOR/'scores-private.json')['critic']
    assert digest(PRIOR/'scores-private.json')==read(PRIOR/'results.json')['offline_score_sha']
    roster={r['uid']:r for r in membership}
    raw=inputs()
    truth,_=truth_data({u:raw[u] for u,r in roster.items() if r['target']!='mono'})
    labels={r['uid']:r for r in read(METRIC/'labels.json')}
    external,sha=labels_for_panel(read(mono/'mono-input-seal.json'))
    assert sha==read(mono/'mono-results.json')['official_kitti_label_sha256']
    strict=set(read(mono/'mono-strict-roster-private.json'))
    bypool={r['uid']:r for r in read(art)}
    case={r['uid']:r for r in read(PRIOR/'inputs-private.json')}
    fields=read(fields); regs=read(regs)
    all_rows=[]
    old_max=0.
    for entry in read(pred):
        uid=entry['uid']
        if uid not in roster:
            continue
        old=roster[uid]
        target=(external if entry['target']=='mono' else truth)[uid]['box']
        predictions=entry['boxes']
        score=dict(zip(ARMS,ious([predictions[name] for name in ARMS],target),strict=True))
        source=score['Minimal-D']
        old_max=max(old_max,abs(source-old['iou']['Minimal-D']),abs(score['sim_iou']-old['iou']['critic_silhouette']),
                    abs(score['pool_aabb']-old['iou']['grid_aabb']))
        pool=bypool[uid].get('pool')
        ranks={}
        if pool:
            actual=ious(pool,target)
            score['candidate_oracle']=float(actual.max())
            ranks={k:rhofit(v,actual) for k,v in entry['rankings'].items()}
        else:
            score['candidate_oracle']=source
        row=dict(uid=uid,target=old['target'],recording=old['recording'],branch=old['branch'],active=entry['active'],
                 iou={k:float(v) for k,v in score.items()},changed=[a for a,b in predictions.items() if b!=predictions['Minimal-D']],
                 ranks=ranks,has_reference=entry.get('reference') is not None)
        if entry['compact'] and uid in labels:
            other=same_class(raw[uid]['frame'],labels[uid],target)
            row['separation']=separation(case[uid],regs[uid],fields[uid],target,other)
        all_rows.append(row)
    assert len(all_rows)==836 and old_max<1.e-9
    scopes=dict(all=all_rows,development=[r for r in all_rows if r['target']!='mono'],
                mono_strict=[r for r in all_rows if r['uid'] in strict],
                drone_quad=[r for r in all_rows if r['target'] in ('drone','quad')],
                target_transfer=[r for r in all_rows if r['target']!='waymo'])
    for name in ('waymo','drone','quad','kitti','mono'):
        scopes[name]=[r for r in all_rows if r['target']==name]
    scopes['development_competitive']=[r for r in scopes['development'] if r['active']]
    scopes['development_C_ray']=[r for r in scopes['development'] if r['branch']=='C_ray']
    summary={}
    ranking={}
    nvalues=(*ARMS,'candidate_oracle')
    for name,rows in scopes.items():
        summary[name]=stat_panel(rows,nvalues,'Minimal-D')
        switches={}
        for arm in ARMS[1:]:
            changes=[r['iou'][arm]-r['iou']['Minimal-D'] for r in rows if arm in r['changed']]
            switches[arm]=dict(replaced=len(changes),gain=sum(x>1.e-8 for x in changes),
                               harm=sum(x< -1.e-8 for x in changes),same=sum(abs(x)<=1.e-8 for x in changes))
        summary[name]['switches']=switches
        ranking[name]={}
        for arm in ('sim_iou',*RANKS,'pool_vit_old','pool_vit_compact','pool_vit_wide'):
            values=[r['ranks'][arm] for r in rows if r['ranks'].get(arm) is not None]
            ranking[name][arm]=dict(n=len(values),spearman=float(np.mean(values)) if values else None)
    controls=(('old_vit_select','competitive_2a'),('pool_vit_old','pool_vit_compact'),('pool_vit_compact','pool_vit_wide'),
              ('sim_iou','real_iou'),('real_iou','real_pair'),('real_pair','real_delta0'),
              ('real_delta0','joint_delta0'),('real_delta_val','joint_delta_val'),('pool_aabb','joint_delta_val'))
    effects={p:{y+'_vs_'+x:stat_panel(scopes[p],(x,y),x)['metrics'][y] for x,y in controls} for p in
             ('development','drone_quad','mono_strict','development_competitive') if scopes[p]}
    sep={}
    for key in ('proxy','same_class','binding'):
        parts=[r['separation'][key] for r in all_rows if r.get('separation') and key in r['separation']]
        sep[key]=dict(n=len(parts),old=float(np.mean([r[0] for r in parts])) if parts else None,
                     competitive=float(np.mean([r[1] for r in parts])) if parts else None,
                     improved=sum(a<b for a,b in parts),declined=sum(a>b for a,b in parts))
    payload=out/'scores-private.json'
    write(payload,all_rows)
    write(RUN/'results.json',dict(panels=summary,matched_effects=effects,rankings=ranking,separation=sep,
         source_features=read(RUN/'source'/'feature-seal.json'),training=read(RUN/'source'/'training-seal.json'),
         decode=read(RUN/'source'/'decode-seal.json'),visual=vis,sealed_predictions=lock,labels_after_seal=True,
         old_control_max_abs_delta=float(old_max),scores_sha=digest(payload),code=digest(__file__)))
    for k in ('development','drone_quad','mono_strict'):
        print(k,{a:(round(d['miou'],3),round(d['acc50'],3)) for a,d in summary[k]['metrics'].items()},flush=True)
    print('Separate instances',sep,flush=True)


if __name__=='__main__':
    main()
