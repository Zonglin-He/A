"""Sealed native-oriented sensor-frame evaluation of the ordinary terrain-only readout."""
from collections import defaultdict
import numpy as np

from shared_paths import OUT,OLD,NEW,HERE,read,save,digest,anonymous
from terrain_followup import BASE,DST,FOLLOW,ARMS,geometry_predictions,boxes_in_model
from common_lex import sample
from scope183 import ordinary,frame
from score_roles import score_boxes,sensor_truth,sensor_frame_box,inventory,normalize,csv_file,measure,record_intervals
from endpoint_score import arrays_to_owners,write_gz
from batch_yaw_iou import many_iou


def grounded(source,attr,uid,others):
    before,after=boxes_in_model(others,source['platform'],source['suffix'])
    labeled,caption=inventory(source['metapath'],source['platform'],source['suffix'],source['object_index'])
    assert normalize(source['caption'])==caption
    points=np.unique(sample(frame(source)).astype(float),axis=0)
    owners=arrays_to_owners([np.stack((before,after))],labeled,points)
    fields={}
    for mode,label in [('unique_iou','iou'),('seed_ownership','return')]:
        so,fo=owners[mode][0]
        fields[label+'_before']=int(so==0);fields[label+'_after']=int(fo==0)
    return fields


def score():
    roster=ordinary();ma={};recs={};rows=[];summary=[];cert=[]
    diffs=0.;total=0;native_count=0
    metric=[];comparisons=[]
    def stats(which,mat,vals,group):
        n=len(mat)
        for i,a in enumerate(ARMS):
            metric.append(dict(direction=which,arm=a,n=n,acc25=vals[i,0],acc50=vals[i,1],miou=vals[i,2],
                edited=sum(r['edited'] for r in group if r['arm']==a)))
            for j,b in enumerate(ARMS):
                if i==j or (i!=1 and j!=0):continue
                s=dict(direction=which,arm=a,control=b,n=n,delta25=vals[i,0]-vals[j,0],
                     delta50=vals[i,1]-vals[j,1],delta_miou=vals[i,2]-vals[j,2])
                for t in (25,50):
                    old,actual=mat[:,j]>t/100,mat[:,i]>t/100
                    s['rescue'+str(t)]=int((actual&~old).sum());s['harm'+str(t)]=int((~actual&old).sum())
                comparisons.append(s)
    for direc,boxes,picks in geometry_predictions():
        src,target=direc.split('_to_')
        path=BASE/('scored_'+direc+'.npz')
        seal=read(path.with_suffix('.json'))
        assert seal['prediction_seal']==digest(BASE/'SEALED.json') and seal['file_sha256']==digest(path)
        with np.load(path) as z:all_iou=z['pool_iou']
        raw_iou=all_iou[np.arange(len(picks)),picks];mat=np.zeros((len(picks),len(ARMS)))
        before_rows=len(rows)
        # Open the precise archived A256 only for potential geometry opportunities.
        box_blocks=read(BASE/direc/'SEALED.json')['files'];shard=None;bounds=None
        for i,item in enumerate(roster[target]):
            gt=sensor_truth(target,item['metapath'],item['caption'],item['suffix'])
            scores,err=score_boxes(boxes[i],gt);diffs=max(diffs,err,abs(scores[0]-raw_iou[i]))
            assert diffs<1.e-7 and all((raw_iou[i]>c)==(scores[0]>c) for c in (.25,.5))
            mat[i]=scores;total+=1
            table=np.c_[boxes[i,:,2],boxes[i,:,5]]
            candidate_scores=all_iou[i]
            oriented_best=candidate_scores.max()
            union=oriented_best
            if max(scores)>.5>=oriented_best:
                f=f'{i//128*128:05d}.npz'
                if f!=shard:
                    assert digest(BASE/direc/f)==box_blocks[f]
                    with np.load(BASE/direc/f) as p:
                        bounds=p['pool'];ordinals=p['ordinals']
                    shard=f
                p=int(i%128);assert ordinals[p]==i
                flat=boxes_in_model(bounds[p],target,item['suffix'])
                flat[:,6]=0.
                sensor=np.stack([sensor_frame_box(b,target,item['suffix']) for b in flat])
                union=max(union,many_iou(sensor,gt).max())
                native_count+=1
            uid=anonymous((target,i))
            for a,score in enumerate(scores):
                cross=(scores[0]>.5)!=(score>.5)
                r=dict(case=uid,source=src,target=target,recording=anonymous((target,item['recording'])),
                    frame=anonymous(item['pointpath']),arm=ARMS[a],iou=float(score),v1_iou=float(scores[0]),
                    edited=int(np.max(abs(table[a]-table[0]))>1.e-8),cross50=int(cross),
                    beyond_a256=int(score>.5>=union),iou_before=-1,iou_after=-1,return_before=-1,return_after=-1)
                if cross:r.update(grounded(item,ARMS[a],uid,boxes[i,[0,a]]))
                rows.append(r)
                if cross:cert.append(r)
        ma[src,target]=mat
        recs[src,target]=dict(recording=np.array([r['recording'] for r in roster[target]]))
        vals=measure(mat);summary.append(vals)
        stats(direc,mat,vals,rows[before_rows:]);print('scored terrain',direc,len(picks),flush=True)
    assert total==18620 and len(rows)==18620*len(ARMS)
    stats('macro',np.vstack(list(ma.values())),np.mean(summary,axis=0),rows)
    cis=[]
    for j in (0,2,3):
        lo,hi=record_intervals(recs,ma,1,j)
        cis.append(dict(arm=ARMS[1],control=ARMS[j],delta50_ci_low=lo,delta50_ci_high=hi))
    write_gz(OUT/'terrain_task_rows.csv.gz',rows)
    if cert:write_gz(OUT/'terrain_crossing_rows.csv.gz',cert)
    csv_file(OUT/'terrain_metrics.csv',metric);csv_file(OUT/'terrain_paired.csv',comparisons);csv_file(OUT/'terrain_intervals.csv',cis)
    save(OUT/'terrain_audit.json',dict(records=total,arms=len(ARMS),raw_and_pca_rescans=native_count,
        v1_verified=total,max_scorer_difference=diffs,prediction=digest(DST/'SEALED.json'),
        protocol=digest(FOLLOW),negative_control=read(DST/'SEALED.json')['negative_control'],
        source_training=0,target_updates=0,FINAL_access=False))


if __name__=='__main__':score()
