"""Only this post-seal entry reads target annotations. Known boxes are a separate diagnostic."""
from collections import defaultdict

import numpy as np
import pandas as pd

from token_paths import ARMS, CLASSES, OUT, SOURCES, WORK, digest, prior, read, roster, row_id, save, signatures
from decide_token import observations, posteriors
from endpoint_score import arrays_to_owners
from oracle_local import DIAG
from panel_witness import inventory
from physical_evidence import Returns
from role_cut import layout, medoid, separate
from score_roles import score_boxes,sensor_frame_box
from score_swap import best


def solve(p,words,view,side,depth,near,mode):
    n=len(p)
    single=np.ones(n)
    identity=np.eye(n)
    valid=~identity.astype(bool)
    if mode!='wrong_reference':
        joint,landmark=separate(valid,single,p,words,view,side,depth,near)
    else:
        v,r,vis,gap=layout(view,side,'wrong_reference_binding')
        earlier=(depth*(r[:,3,None]-r[:,3])<0)&vis[:,None]
        sub=p[:,words[0]]*vis
        ref=p[:,words[1]]*vis
        survival=1.-earlier*ref[:,None]
        landmark=ref*survival.prod(0)
        candidate=np.divide(sub[:,None]*(~earlier if words[0]==words[1] else 1),survival,
                            out=np.zeros_like(survival),where=survival>0)
        candidate*=valid & (gap>0) & vis[None,:]
        joint=candidate*landmark
        if near:
            order=np.argsort(side*v[:,1],kind='stable')
            free=np.vstack((np.ones(n),np.cumprod(1.-candidate[order][:-1],axis=0)))
            start=np.maximum.accumulate(np.where(np.r_[True,np.diff(v[order,1])!=0],np.arange(n),0))
            joint[order]*=free[start]
    sub,ref,_=medoid(joint,identity,single,single,valid)
    if mode=='independent':
        unary=p[:,words[0]]*np.isfinite(view[:,0])
        sub=int(unary.argmax()) if unary.max(initial=0)>0 else -1
        ref=int(landmark.argmax()) if landmark.max(initial=0)>0 else -1
    return sub,ref


def disclose_class(p,truth,word):
    # Conditioning on the truth value of one physical object's class test preserves a
    # proper categorical distribution among the other classes on a negative test.
    out=p.copy()
    yes=truth[:,word].astype(bool)
    out[yes]=0.
    out[yes,word]=1.
    out[~yes,word]=0.
    missing=(~yes)&(out.sum(1)==0)
    out[missing]=1.
    out[missing,word]=0.
    return out/out.sum(1,keepdims=True)


def error_type(sel,wanted,names):
    if sel<0:
        return 'no_choice'
    if sel==wanted:
        return 'correct'
    return 'same_class_other_object' if names[sel]==names[wanted] else 'wrong_class'


def aggregates(table,keys, output):
    first=table.groupby([*keys,'source','pair'],dropna=False)
    assert first.size().eq(2).all()
    halves=first[['subject','reference','both']].min().reset_index()
    g=halves.groupby(keys,dropna=False)
    counts=g[['subject','reference','both']].sum()
    counts['pairs']=g.size()
    counts.to_csv(OUT/output)
    return counts


def run():
    seal=read(WORK/'SEALED.json')
    assert seal['pins']==signatures() and not seal['labels'] and not seal['FINAL']
    rows,frames=roster()
    labels_path=prior.previous.EXTRA/'labels.json'
    assert digest(labels_path)==read(prior.old.OUT/'noun_panel.json')['labels']
    labels=read(labels_path)
    reg_seal=read(DIAG/'REGIONS.json')['files']
    previous_seal={s:read(DIAG/s/'SEALED.json')['files'] for s in SOURCES}
    new_seal={s:read(WORK/s/'SEALED.json')['files'] for s in SOURCES}
    cached={};pairs=defaultdict(list);tasks=[];endpoints=[];oracle=[];shapes=[];assignment=[]
    max_error=0.
    for row in rows:
        src=row['source'];fid=row['frame'];info=frames[fid]
        rid=row_id(row);e=labels[row['uid']];words=row['words'];ts=e['subject'];tr=e['reference']
        truth=np.asarray(e['truth'])
        path=WORK/'predictions'/(rid+'.npz')
        assert digest(path)==seal['files'][rid]
        head=dict(source=src,target=row['target'],frame=e['frame'],recording=e['recording'],pair=e['pair'],half=e['half'])
        with np.load(row['prior']) as predecessor:
            view_id=int(predecessor['view']);seeds=predecessor['seeds'];pool=predecessor['native_a']
        with np.load(path) as pred:
            sub=[pred['raw'],pred['anchored']];ref=[pred['ref'],pred['ref_anchored']]
            owners=arrays_to_owners(sub+ref,truth,seeds)
            allbox=np.vstack(sub)
            values,_=score_boxes(allbox,truth[ts])
            alt,check=score_boxes(np.stack([sensor_frame_box(b,row['target'],view_id) for b in allbox]),
                                   sensor_frame_box(truth[ts],row['target'],view_id))
            max_error=max(max_error,float(np.abs(alt-values).max()),float(check))
            np.testing.assert_equal(values[:,None]>[.25,.5],alt[:,None]>[.25,.5])
            ious=values.reshape(2,len(ARMS))
            for j,a in enumerate(ARMS):
                for k,g in enumerate(('raw','completed')):
                    tasks.append(dict(**head,arm=a,geometry=g,iou=ious[k,j],fallback=int(pred['nodes'][j,0]<0)))
                    for rule,own in owners.items():
                        s=int(own[k][j]==ts);r=int(own[k+2][j]==tr)
                        endpoints.append(dict(**head,arm=a,geometry=g,rule=rule,subject=s,reference=r,both=s*r))
                was,now=ious[:,j]>.5
                if was!=now and pred['nodes'][j,0]>=0:
                    stable=all(own[0][j]==ts and own[1][j]==ts for own in owners.values())
                    pool_best=best(pool,truth[ts])
                    shapes.append(dict(**head,arm=a,before=ious[0,j],after=ious[1,j],pool_best=pool_best,
                                       rescue=int(now),harm=int(was),stable=int(stable),beyond=int(now and stable and pool_best<=.5)))
        if fid not in cached:
            reg=DIAG/(fid+'_regions.npz');old=DIAG/src/(fid+'.npz');obs=WORK/src/(fid+'.npz')
            assert digest(reg)==reg_seal[fid] and digest(old)==previous_seal[src][fid] and digest(obs)==new_seal[src][fid]
            with np.load(reg) as z:
                np.testing.assert_array_equal(z['boxes'],truth);view=z['view']
            names,boxes,conflict=inventory(dict(platform=row['target'],metapath=info['f']['calibration'],suffix=view_id))
            assert not conflict
            np.testing.assert_array_equal(boxes,truth)
            with np.load(info['lex']) as l,np.load(obs) as v,np.load(old) as z:
                cloud=Returns(l['sample']);claims=[cloud.local(b)[0] for b in truth]
                mult=np.zeros(len(cloud.xyz),int)
                for ids in claims:mult[ids]+=1
                unique=[ids[mult[ids]==1] for ids in claims]
                deltas=observations(v['xyz'],v['values'],cloud.xyz,unique)
                prob=posteriors(z['combined'],deltas)
            exact=np.array([[float(c==n) for n in CLASSES[:-1]] for c in names])
            cached[fid]=(prob,view,names,exact)
        p,view,names,exact=cached[fid]
        assert names[ts]==CLASSES[words[0]] and names[tr]==CLASSES[words[1]]
        with np.load(row['role']) as z:
            relation=(int(z['side']),int(z['depth']),bool(z['immediate']))
        pairs[(src,e['pair'])].append((e,words,p))
        # Negative instance-binding control for the offline class disclosures. Keep the
        # visible category histogram exactly and move the full vector jointly by object.
        wrong=exact.copy()
        ix=np.flatnonzero(np.isfinite(view[:,0]))
        order=ix[np.argsort(view[ix,1],kind='stable')]
        if len(order)>1:
            wrong[order]=exact[np.roll(order,max(1,len(order)//2))]
        np.testing.assert_equal(exact[order].sum(0),wrong[order].sum(0))
        extra=dict(reveal_subject_word=disclose_class(p['tree'],exact,words[0]),
                   reveal_reference_word=disclose_class(p['tree'],exact,words[1]),
                   wrong_subject_word=disclose_class(p['tree'],wrong,words[0]),
                   wrong_reference_word=disclose_class(p['tree'],wrong,words[1]),true_classes=exact)
        for a in (*ARMS,*extra):
            field=p.get(a,p['token_interaction']) if a not in extra else extra[a]
            si,ri=solve(field,words,view,*relation,a)
            s=int(si==ts);r=int(ri==tr)
            oracle.append(dict(**head,arm=a,subject=s,reference=r,both=s*r,same_noun=int(words[0]==words[1]),
                        subject_error=error_type(si,ts,names),reference_error=error_type(ri,tr,names),
                        subject_chose_true_reference=int(si==tr),reference_chose_true_subject=int(ri==ts)))
    for (src,pair),halves in pairs.items():
        (e,w,fields),(other,v,other_fields)=sorted(halves,key=lambda h:h[0]['half'])
        assert fields is other_fields and len({e['subject'],other['subject'],e['reference'],other['reference']})==4
        for a,p in fields.items():
            i,j=e['reference'],other['reference'];u,t=w[1],v[1]
            gap=p[i,u]*p[j,t]-p[i,t]*p[j,u]
            assignment.append(dict(source=src,pair=pair,frame=e['frame'],recording=e['recording'],arm=a,
                                   correct=int(gap>1e-12),tie=int(abs(gap)<=1e-12),wrong=int(gap< -1e-12)))
    t=pd.DataFrame(tasks);d=pd.DataFrame(endpoints);o=pd.DataFrame(oracle);a=pd.DataFrame(assignment)
    for tab,name in [(t,'task'),(d,'physical_identity'),(o,'inventory'),(a,'name_assignment'),(pd.DataFrame(shapes),'geometry')]:
        tab.to_csv(OUT/(name+'_rows.csv.gz'),index=False)
    overall=aggregates(o,['arm'],'inventory_pairs.csv')
    assert overall.loc['tree','both']==7 and overall.loc['true_classes','both']==64
    actual=aggregates(d,['arm','geometry','rule'],'physical_pairs.csv')
    a.groupby('arm')[['correct','tie','wrong']].sum().to_csv(OUT/'name_assignment.csv')
    error_summary=[]
    for arm,part in o.groupby('arm'):
        for role in ('subject','reference'):
            error_summary.append(dict(arm=arm,role=role,**part[role+'_error'].value_counts().to_dict()))
    pd.DataFrame(error_summary).fillna(0).to_csv(OUT/'competitors.csv',index=False)
    t['acc25']=(t.iou>.25)*100.;t['acc50']=(t.iou>.5)*100.;t['miou']=100*t.iou
    t.groupby(['source','target','arm','geometry'])[['acc25','acc50','miou']].mean().to_csv(OUT/'task_directions.csv')
    assert max_error<1e-8 and len(t)==128*len(ARMS)*2 and len(d)==2*len(t)
    save(OUT/'checks.json',dict(prediction_seal=digest(WORK/'SEALED.json'),old_tree_box_error=seal['tree_max_error'],
         samples=128,physical_frames=t.frame.nunique(),recordings=t.recording.nunique(),direction_pairs=len(pairs),
         native_query_check={s:read(WORK/s/'SEALED.json')['checks'] for s in SOURCES},max_iou_error=max_error,
         class_oracle_four=int(overall.loc['true_classes','both']),tree_oracle_four=int(overall.loc['tree','both']),
         labels_during_prediction=False,source_training=0,target_updates=0,FINAL_access=False))
    print('physical\n',actual.loc[('token_interaction','completed')].to_string(),'\nknown boxes\n',overall.to_string(),flush=True)

if __name__=='__main__':
    run()
