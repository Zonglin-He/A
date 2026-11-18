"""Evaluate the already-sealed additional A-only predictions with two IoU implementations."""
from collections import Counter,defaultdict
import numpy as np
from transfer_inputs158 import HERE,ROOT,WORK,read,digest
import sys
sys.path.insert(0,str(ROOT/'research/round156'))
from score156 import ground_truth, many_iou, oriented_iou, rotation, table, save

ARMS=('old_top1','overlap','old_axis','overlap_axis')


def main():
    dest=HERE/'transfer_results';dest.mkdir(exist_ok=True)
    reg=WORK/'EXTRA_REGISTRY_PRIVATE.json';pred=WORK/'EXTRA_PRIVATE.npz';seal=read(WORK/'EXTRA_SEAL.json')
    if digest(reg)!=seal['registry'] or digest(pred)!=seal['sha256'] or seal['target_GT_read'] or seal['FINAL_access']:
        raise ValueError('prediction seal')
    rr=read(reg)
    with np.load(pred,allow_pickle=False) as f:p=dict(f)
    if not np.array_equal(p['ordinal'],np.arange(1,len(rr)+1)) or len(rr)!=seal['n']:raise ValueError('manifest join')
    group={k:j for j,k in enumerate(sorted(set(r['group'] for r in rr)),1)}
    scenes={k:j for j,k in enumerate(sorted(set((r['target'],r['row']['scene']) for r in rr)),1)}
    rows=[];maxerr=0.;calls=0
    for i,e in enumerate(rr):
        f=e['row'];gt=ground_truth(e['target'],f['metapath'],f['object_index'],f['caption'])
        b=p['boxes'][i].astype(float);b[:,3:6]=np.maximum(b[:,3:6],1e-6);axis=b.copy();axis[:,-1]=0
        uniq,ind=np.unique(np.concatenate([b,axis]),return_inverse=True,axis=0)
        y=many_iou(uniq,gt)
        y2=np.array([oriented_iou(bb[:3],bb[3:6],rotation(bb[6]),gt[:3],gt[3:6],rotation(gt[6])) for bb in uniq])
        maxerr=max(float(abs(y-y2).max()),maxerr);calls+=len(uniq)
        if not (np.array_equal(y>.25,y2>.25) and np.array_equal(y>.5,y2>.5)):raise ValueError('threshold disagreement')
        rows.append(dict(ordinal=i+1,source=e['source'],target=e['target'],scene_cluster=scenes[e['target'],f['scene']],
             point_content_cluster=group[e['group']],changed=bool(p['indices'][i,0]!=p['indices'][i,1]),**dict(zip(ARMS,map(float,y[ind])))))
    if maxerr>1e-7: raise ValueError('3D IoU audit')
    table(dest/'all_rows.csv',rows)
    summary=[];dirs=[];scene_rows=[]
    for weighting in ('ordinary','content_equal'):
        for s,t in sorted(set((r['source'],r['target']) for r in rows)):
            records=[r for r in rows if (r['source'],r['target'])==(s,t)]
            mult=Counter(r['point_content_cluster'] for r in records)
            w=np.array([1 if weighting=='ordinary' else 1/mult[r['point_content_cluster']] for r in records])
            def stats(values,weights):
                return {k:float(100*np.average(x,weights=weights)) for k,x in [('mIoU',values),('Acc25',values>.25),('Acc50',values>.5)]}
            for arm in ARMS:
                y=np.array([r[arm] for r in records])
                dirs.append(dict(weighting=weighting,source=s,target=t,n=len(records),arm=arm,**stats(y,w)))
                for scene in sorted({r['scene_cluster'] for r in records}):
                    mask=np.array([r['scene_cluster']==scene for r in records])
                    scene_rows.append(dict(weighting=weighting,source=s,target=t,scene_cluster=scene,n=int(mask.sum()),arm=arm,**stats(y[mask],w[mask])))
        for arm in ARMS:
            ss=[r for r in dirs if (r['weighting'],r['arm'])==(weighting,arm)]
            if len(ss)!=6:raise ValueError('missing direction')
            summary.append(dict(arm=arm,weighting=weighting,n=len(rr),**{k:float(np.mean([r[k] for r in ss])) for k in ('mIoU','Acc25','Acc50')}))
    table(dest/'direction.csv',dirs);table(dest/'macro.csv',summary);table(dest/'scene.csv',scene_rows)
    delta=np.array([r['overlap']-r['old_top1'] for r in rows])
    audit=dict(status='PASS',n=len(rr),physical_groups=len(group),target_scene_clusters=len(scenes),
               better=int((delta>1e-12).sum()),worse=int((delta< -1e-12).sum()),same=int((abs(delta)<=1e-12).sum()),
               changed=int(sum(r['changed'] for r in rows)),independent_OBB_calls=calls,max_error=maxerr,
               no_target_outcome_used_for_choice=True,previous_full_VAL_exposed=True,FINAL_access=False)
    save(dest/'AUDIT.json',audit);print(audit)

if __name__=='__main__':main()
