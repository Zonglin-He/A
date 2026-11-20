"""Post-seal attribution. Expose the true reference only here and measure the fixed joint table."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from shared_paths import SEG,PRED,OLD,NEW,OUT,ARMS,inputs,name,read,digest,HERE,save
from scene_roles import anchor_returns,world,role,infer
from physical_evidence import Returns
from ordered_choice import prepare
from endpoint_score import arrays_to_owners,write_gz
from score_roles import csv_file
from panel_witness import inventory,projection,nearest
from scope183 import VIEWS
from tables_scene import grouped


def work(item):
    (row,zpath,_,_,_,fid,words,_,_),entry=item
    assert 'pair' in entry
    f=row['frame'];source=row['source'];n=name(row)
    st=entry['subject'];rr=entry['reference'];truth=np.asarray(entry['truth'])
    view=VIEWS.index(f['waymo_view'])
    with np.load(zpath) as z,np.load(SEG/(fid+'.npz')) as s,np.load(PRED/n) as p:
        T,_,mt,_=prepare(z,'t');R,_,mr,_=prepare(z,'r')
        owner=arrays_to_owners([T,R],truth,z['seeds'])
        tables=[];points=Returns(s['xyz']);tr=anchor_returns(T,points);rs=anchor_returns(R,points)
        for k in range(8):
            scene=world(s,k,int(fid[:8],16))
            tables.append((role(tr,mt,words[0],scene),role(rs,mr,words[1],scene)))
        J,_=infer(tables,s['labels'],int(z['side']),int(z['depth']),bool(z['immediate']),'joint_raw',len(T),len(R))
        np.testing.assert_allclose(J.sum(1),p['subject_scores'][ARMS.index('joint_raw')],rtol=1.e-10,atol=1.e-14)
        baseline=arrays_to_owners([p['raw'][[ARMS.index('joint_raw')]]],truth,z['seeds'])
        outcomes=[]
        for rule,(sid,rid) in owner.items():
            correct=J[:,rid==rr].sum(1)
            wrong=J[:,(rid>=0)&(rid!=rr)].sum(1)
            in_table=bool((J[np.ix_(sid==st,rid==rr)]>0).any())
            def win(v):return bool(v.sum()>0 and sid[v.argmax()]==st)
            outcomes.append(dict(panel='new_side' if entry['group']=='new_side' else 'old_switch',
                  case=row['uid'],source=source,pair=entry['pair'],half=entry['half'],rule=rule,
                  subject_in_pool=int(st in sid),ref_in_pool=int(rr in rid),both_in_pool=int(st in sid and rr in rid),
                  positive_correct_pair=int(in_table),original_correct=int(baseline[rule][0][0]==st),
                  true_ref_correct=int(win(correct)),wrong_refs_correct=int(win(wrong)),
                  no_true_ref_mass=int(correct.sum()==0)))
    central=None
    if entry['group']=='new_side':
        cls,boxes,bad=inventory(dict(metapath=f['calibration'],platform=f['target'],suffix=view))
        assert not bad and np.array_equal(boxes,truth)
        px=projection(boxes,f['target'],view,f['calibration'],f['rgb'])
        # Matched annotated alternatives: exclude the same known reference in both comparisons.
        # As input to this offline diagnostic, the eligible object set is deliberately an oracle.
        noun=cls[st];sign='left' if entry['half']==0 else 'right'
        assert nearest(noun,rr,sign,cls,px)==st
        synthetic=px.copy();synthetic[rr,[0,2,3]]=.5
        center=nearest(noun,rr,sign,cls,synthetic)
        central=dict(pair=entry['pair'],half=entry['half'],same_side=float(px[st,0]-.5),
                     correct_at_center=int(center==st),abstained_at_center=int(center is None))
    return outcomes,central


def main():
    status=read(OUT/'audit.json')
    assert status['prediction_seal']==digest(PRED/'SEALED.json')
    label={**read(OLD/'labels.json'),**read(NEW/'labels.json')}
    jobs=[(r,label[r[0]['uid']]) for r in inputs() if 'pair' in label[r[0]['uid']]]
    items=[];centers={}
    with ProcessPoolExecutor(max_workers=7) as ex:
        for result,central in ex.map(work,jobs,chunksize=2):
            items.extend(result)
            if central:
                key=central['pair'],central['half']
                assert central==centers.setdefault(key,central)
    stats=[];keys=('subject_in_pool','ref_in_pool','both_in_pool','positive_correct_pair','original_correct','true_ref_correct','wrong_refs_correct','no_true_ref_mass')
    for (g,rule),v in grouped(items,['panel','rule']).items():
        pp=grouped(v,['pair','source'])
        values={k:sum(r[k] for r in v) for k in keys}
        pair_values={'pair_'+k:sum(all(r[k] for r in p) for p in pp.values()) for k in keys}
        stats.append(dict(panel=g,rule=rule,n=len(v),pairs=len(pp),**values,**pair_values))
    out=[]
    for (identity,half),a in centers.items():
        if half!=0:continue
        b=centers[identity,1]
        out.append(dict(pair=identity,both_targets_on_one_side_of_image=int(a['same_side']*b['same_side']>0),
                both_correct_at_center=int(a['correct_at_center'] and b['correct_at_center']),
                any_unknown_at_center=int(a['abstained_at_center'] or b['abstained_at_center'])))
    write_gz(OUT/'oracle_rows.csv.gz',items);csv_file(OUT/'reference_upper.csv',stats)
    save(OUT/'center_diagnostic.json',dict(physical_frames=len(out),**{k:sum(r[k] for r in out) for k in out[0] if k!='pair'},
       introduced_after_scoring=True,deployable=False))


if __name__=='__main__':main()
