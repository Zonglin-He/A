"""Score the content-disjoint controls only after all single-utterance predictions are sealed."""
import numpy as np
import pandas as pd
from local_paths import *
from fresh_inference import FRESH, pins
from method_local import ARMS, GEOMS
from score_roles import score_boxes, sensor_frame_box
from endpoint_score import arrays_to_owners

def score():
    sealed=read(FRESH/'SEALED.json');main=read(WORK/'SEALED.json')
    assert not sealed['labels_read'] and sealed['pins']==pins() and not main['labels_used']
    assert digest(FRESH/'score_only.json')==read(OUT/'fresh_panel.json')['scoring']
    labels=read(FRESH/'score_only.json');rows=[];identities=[];worst=0.
    for s in SOURCES:
        f=FRESH/s/'SEALED.json';assert digest(f)==sealed['sources'][s]
        for uid,pin in read(f)['files'].items():
            case=labels[uid];gt=np.array(case['truth']);ts=case['subject'];tr=case['reference']
            path=FRESH/s/(uid+'.npz');assert digest(path)==pin
            with np.load(path) as z:
                sub=[z[g] for g in GEOMS];refs=[z[k] for k in ('ref','ref_anchored','ref_fit','ref_direct_fit')]
                own=arrays_to_owners(sub+refs,gt,z['seeds'])
                box,inv=np.unique(np.concatenate(sub),axis=0,return_inverse=True)
                q,e=score_boxes(box,gt[ts]);view=int(z['view'])
                vals,f=score_boxes(np.array([sensor_frame_box(b,'waymo',view) for b in box]),sensor_frame_box(gt[ts],'waymo',view))
                worst=max(worst,e,f,float(abs(vals-q).max()));assert worst<1e-8
                mat=q[inv].reshape(len(GEOMS),len(ARMS))
                for i,g in enumerate(GEOMS):
                    for j,arm in enumerate(ARMS):
                        base=dict(source=s,target='waymo',case=uid,frame=case['frame'],recording=case['recording'],
                            pair=case['pair'],half=case['half'],geometry=g,arm=arm)
                        rows.append(dict(**base,iou=float(mat[i,j]),nominated=len(z['nominated'])))
                        for rule,o in own.items():
                            subject=int(o[i][j]==ts);reference=int(o[len(GEOMS)+i][j]==tr)
                            identities.append(dict(**base,rule=rule,subject=subject,reference=reference,both=subject*reference))
    p=pd.DataFrame(rows);p['acc25']=100*(p.iou>.25);p['acc50']=100*(p.iou>.5);p['miou']=100*p.iou
    p.to_csv(OUT/'fresh_task_rows.csv.gz',index=False)
    d=pd.DataFrame(identities);d.to_csv(OUT/'fresh_identity_rows.csv.gz',index=False)
    g=p.groupby(['source','geometry','arm'])
    met=g[['acc25','acc50','miou']].mean();met['records']=g.size()
    met.to_csv(OUT/'fresh_metrics.csv')
    pair=d.groupby(['source','pair','geometry','arm','rule'])
    assert pair.size().eq(2).all()
    grouped=pair[['subject','reference','both']].min().reset_index().groupby(['geometry','arm','rule'])
    counts=grouped[['subject','reference','both']].sum();counts['pairs']=grouped.size()
    counts.to_csv(OUT/'fresh_pair_counts.csv')
    save(OUT/'fresh_check.json',dict(utterances=p[['source','case']].drop_duplicates().shape[0],
        physical_frames=p.frame.nunique(),recordings=p.recording.nunique(),worst_iou_error=worst,
        sealed=digest(FRESH/'SEALED.json')))
    print('new-frame evaluation',len(p),len(d),worst)
if __name__=='__main__':score()
