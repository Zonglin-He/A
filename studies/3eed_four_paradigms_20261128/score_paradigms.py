"""Score sealed predictions for all four paradigms with the original offline references."""
from collections import Counter
import sys

import numpy as np
from scipy.stats import rankdata
import torch

from blind_experiments import ARMS, verified
from geometry_models import exact_iou
from prepare_blind import HERE, RUN, ROOT, TARGETS, digest, read, write


def ious(boxes, truth):
    values=np.zeros(len(boxes))
    valid=[i for i,b in enumerate(boxes) if b is not None]
    if valid:
        data=torch.tensor(np.array([boxes[i] for i in valid]),dtype=torch.float64)
        target=torch.tensor(np.asarray(truth),dtype=torch.float64)[None].expand_as(data)
        values[valid]=exact_iou(data,target).numpy()
    return values


def correlation(values, ref):
    x,y=rankdata(values), rankdata(ref)
    if x.std()>0 and y.std()>0:
        return float(np.corrcoef(x,y)[0,1])
    return None


def new_stats(rows,names,stat_panel):
    if not rows:
        return dict(n=0)
    return stat_panel(rows,names,'Minimal-D')


def main():
    torch.set_num_threads(1)
    inputs=verified('inputs-private.json','input-seal.json','evidence')
    stamps={mode:read(RUN/f'{mode}-prediction-seal.json') for mode in ARMS}
    assert any(stamps.values()) and all(not stamp['labels_read'] for stamp in stamps.values())
    assert all(digest(HERE/f)==sha for stamp in stamps.values() for f,sha in stamp['code'].items())
    bank={mode:{r['uid']:r for r in verified(f'{mode}-blind-private.json',f'{mode}-prediction-seal.json')} for mode in ARMS}
    for mode in ('flow','vit','critic'):
        assert len(bank[mode])==len(inputs)==873
    assert len(bank['fovea'])==160
    from shared import setup
    setup()
    from metric_image_common import inputs as original_inputs
    from truth_score_inputs import truth_data
    from evaluate_structure import stat_panel
    mono=ROOT/'runs/2d-pose-finalization-20261128'
    sys.path.insert(0,str(HERE.parent/'3eed_2d_pose_finalization_20261128'))
    from run_mono import labels_for_panel

    old=ROOT/'runs/minimal-pipeline-20261128'
    old_scores=old/'scores-private.json'
    assert read(old/'results.json')['score_sha256']==digest(old_scores)
    earlier={r['uid']:r for r in read(old_scores)}
    raw=original_inputs()
    gt,_=truth_data({u:raw[u] for u,v in earlier.items() if v['target']!='mono'})
    mono_truth,annotation_sha=labels_for_panel(read(mono/'mono-input-seal.json'))
    assert annotation_sha==read(mono/'mono-results.json')['official_kitti_label_sha256']
    strict=set(read(mono/'mono-strict-roster-private.json'))
    context={r['uid']:r for r in inputs}
    scored={}
    strict_rows=[]
    baseline_delta=0.
    profiles={}
    source_train=read(ROOT/'runs/waymo-posttraining-20261128/source-train.json')
    train_ids={r['recording'] for r in source_train}
    waymo_paths={row['frame']['rgb'] for row in raw.values() if row['target']=='waymo'}
    overlap={p for p in waymo_paths if any(piece in train_ids for piece in __import__('pathlib').Path(p).parts)}
    for mode,col in bank.items():
        scored[mode]=[]
        for uid,case in col.items():
            if uid not in earlier:
                continue
            platform=case['target']; comparison=earlier[uid]
            truth=(mono_truth if platform=='mono' else gt)[uid]['box']
            arms=('Minimal-D',*ARMS[mode]); candidates=[case['base'],*(case['boxes'][a] for a in ARMS[mode])]
            measure=dict(zip(arms,ious(candidates,truth).tolist(),strict=True))
            baseline_delta=max(baseline_delta,abs(measure['Minimal-D']-comparison['iou']['Full']))
            window=context[uid]['window']
            small=window is not None and min(window[2]-window[0],window[3]-window[1])<40
            row=dict(uid=uid,target=platform,recording=comparison['recording'],branch=comparison['branch'],
                     iou=measure,small_window=small,details=case.get('detail',{}),has_mask='data' in case)
            if mode=='vit' and case.get('strict_pair'):
                pair=case['strict_pair']
                outcomes=ious(pair['pair'],truth)
                values=dict(rect_strict=float(outcomes[pair['rect']]),mask_strict=float(outcomes[pair['mask']]),
                            pair_oracle=float(outcomes.max()),axis_fitted=float(outcomes[0]))
                strict_rows.append(dict(uid=uid,target=platform,recording=comparison['recording'],iou=values,
                     differing_pick=pair['rect']!=pair['mask'],decimal_distinct=abs(outcomes[0]-outcomes[1])>1.e-8,
                     either_correct=outcomes.max()>.50))
            if mode=='critic':
                if case.get('pool'):
                    true=ious(case['pool'],truth)
                    measure['candidate_oracle']=float(true.max())
                    detail=case['detail']
                    profiles[uid]=dict(pool=len(true),rho_geometry=correlation(detail['energy_geometry'],true),
                                       rho_silhouette=correlation(detail['energy_silhouette'],true),
                                       mae_geometry=float(np.mean(abs(np.asarray(detail['energy_geometry'])-true))),
                                       mae_silhouette=float(np.mean(abs(np.asarray(detail['energy_silhouette'])-true))))
                else:
                    measure['candidate_oracle']=measure['Minimal-D']
            scored[mode].append(row)
    assert baseline_delta<1.e-7,baseline_delta
    output={}
    for mode,rows in scored.items():
        keys=('Minimal-D',*ARMS[mode],*(['candidate_oracle'] if mode=='critic' else []))
        panels={
            'all':rows,
            'development':[r for r in rows if r['target']!='mono'],
            'mono_strict':[r for r in rows if r['uid'] in strict],
            'transfer':[r for r in rows if r['target']!='waymo'],
            'drone_quad':[r for r in rows if r['target'] in ('drone','quad')],
            'original_C_ray':[r for r in rows if r['branch']=='C_ray'],
            'original_fallback':[r for r in rows if r['branch']!='C_ray'],
        }
        if mode=='fovea':
            panels['small_window']=[r for r in rows if r['small_window'] and r['details'].get('crops',0)]
        if mode in ('flow','critic'):
            panels['original_C_ray_development']=[r for r in rows if r['branch']=='C_ray' and r['target']!='mono']
        panels.update({p:[r for r in rows if r['target']==p] for p in TARGETS})
        result={p:new_stats(rs,keys,stat_panel) for p,rs in panels.items()}
        headline={a:(round(v['miou'],3),round(v['acc50'],3)) for a,v in result['development']['metrics'].items()}
        print(mode,len(rows),headline,flush=True)
        output[mode]=dict(sealed=stamps[mode],panels=result)
    output['vit']['strict_pair']=stat_panel(strict_rows,('rect_strict','mask_strict','pair_oracle','axis_fitted'),'rect_strict')
    output['vit']['strict_pair'].update(changed_winners=sum(r['differing_pick'] for r in strict_rows),
                                  distinct_3d_scores=int(sum(r['decimal_distinct'] for r in strict_rows)))
    f1=[r.get('detail',{}) for r in bank['flow'].values()]
    output['flow']['audit']=dict(generated_queries=sum(d.get('proposed',0)>0 for d in f1),
        nonempty_support_queries=sum(d.get('nonempty_support',False) for d in f1),
        projected_particles=sum(d.get('projected',0) for d in f1),
        selection={k:sum(d['first_hits']['selection'][k] for d in f1 if 'first_hits' in d) for k in ('n','changed')},
        audit={k:sum(d['first_hits']['audit'][k] for d in f1 if 'first_hits' in d) for k in ('n','changed')})
    output['vit']['visual']=read(RUN/'vit-seal.json')
    output['critic']['diagnostics']={}
    for scope in ('all','development','drone_quad','mono'):
        keys=list(profiles) if scope=='all' else [u for u in profiles if earlier[u]['target'] in
                ({'drone','quad'} if scope=='drone_quad' else {'mono'} if scope=='mono' else set(TARGETS)-{'mono'})]
        applied=[profiles[u] for u in keys]
        output['critic']['diagnostics'][scope]=dict(n=len(applied), **{
            key:float(np.mean([a[key] for a in applied if a[key] is not None])) for key in next(iter(profiles.values()))})
    output['fovea']['decode']=read(RUN/'fovea-seal.json')
    xx=[r['details'] for r in scored['fovea']]
    output['fovea']['decoded_cases']=dict(n=len(xx),new_particles_cases=sum(d['particles']>0 for d in xx),
                            particles=sum(d['particles'] for d in xx),view_drops=sum(d['view_drops'] for d in xx))
    private=RUN/'scores-private.json'
    write(private,scored)
    write(RUN/'results.json',dict(outputs=output,sealed_inputs=read(RUN/'input-seal.json'),
         source_training=read(RUN/'training-seal.json'),baseline_max_iou_delta=baseline_delta,
         waymo_train_recording_path_overlap_frames=len(overlap),
         labels_after_all_seals=True,offline_score_sha=digest(private),score_code=digest(__file__),
         protocol=digest(HERE/'PROTOCOL_ZH.md')))


if __name__=='__main__':
    main()
