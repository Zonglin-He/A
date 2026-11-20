"""Retrospective scoring. This module is never imported by inference."""
import argparse
from pathlib import Path
import numpy as np
from partition_paths import ARMS, GEOMS, OUT, ROOT, SOURCES, WORK, archive, catalog, digest, read, row_id, save, signed
from score_delivery import previous_rows
from evaluate_witness import metadata, PANELS, COHORTS, historical_records
from score_tree import labels
from score_roles import score_boxes, sensor_frame_box, sensor_truth, inventory, normalize
from shared_paths import anonymous
from endpoint_score import arrays_to_owners, write_gz
from score_swap import best
from scope183 import ordinary


def measure(z,truth,target,tgt,view,sensor_gt):
    stack=np.stack([z[g] for g in GEOMS]);boxes,inv=np.unique(stack.reshape(-1,7),axis=0,return_inverse=True)
    raw,a=score_boxes(boxes,truth[target])
    sensor,b=score_boxes(np.array([sensor_frame_box(x,tgt,view) for x in boxes]),
                         sensor_frame_box(truth[target],tgt,view) if sensor_gt is None else sensor_gt)
    err=max(a,b,float(np.abs(raw-sensor).max()))
    assert err<1e-8
    np.testing.assert_equal(raw[:,None]>[.25,.5],sensor[:,None]>[.25,.5])
    return raw[inv].reshape(len(GEOMS),len(ARMS)),err


def identify(z,truth,seeds,ts,tr,hdr,pair,half):
    allboxes=[z[pre+g] for pre in ('','ref_') for g in GEOMS]
    own=arrays_to_owners(allboxes,truth,seeds)
    assert 'seed_ownership' in own
    data=[]
    if ts is not None and tr is not None:
        for d,g in enumerate(GEOMS):
            for j,arm in enumerate(ARMS):
                s=np.array([int(arr[d][j]) for arr in own.values()])
                r=np.array([int(arr[d+len(GEOMS)][j]) for arr in own.values()])
                for rule,yes_s,yes_r in [(k,int(o[d][j]==ts),int(o[len(GEOMS)+d][j]==tr)) for k,o in own.items()]+[
                      ('both_tests',int((s==ts).all()),int((r==tr).all()))]:
                    data.append(dict(**hdr,arm=arm,geometry=g,rule=rule,pair=pair,half=half,
                                     subject=yes_s,reference=yes_r,both=yes_s*yes_r))
    return data,own


def geo_diagnostics(hdr,z,mat,own,pool,target,tr):
    lines=[]
    for j,arm in enumerate(ARMS[2:],2):
        if z['selected'][j,0]<0:continue
        for i,k in [(1,0),(2,0),(3,0),(3,1),(3,2)]:
            old,new=mat[k,j],mat[i,j]
            if (old>.5)==(new>.5):continue
            stable=all(arr[i][j]==target and arr[k][j]==target for arr in own.values())
            line=dict(**hdr,arm=arm,geometry=GEOMS[i],control=GEOMS[k],before=old,after=new,a256=pool,
                rescue=int(new>.5),harm=int(old>.5),stable=int(stable),outside_a256=int(new>.5 and stable and pool<=.5),
                ref_verified=int(tr is not None and all(arr[i+len(GEOMS)][j]==tr for arr in own.values())))
            for peer in ('independent','wrong_reference','wrong_marks','same_count','union','previous_tree','v1'):
                arr=[o[i][ARMS.index(peer)] for o in own.values()]
                line[peer+'_wrong_instance']=int(len(set(arr))==1 and 0<=arr[0]!=target)
            lines.append(line)
    return lines


def coverage(z,plan,truth,seeds,words,ts,tr,hdr,side):
    lines=[]
    for g in ('measured','envelope'):
        b=[plan['boxes'] if g=='measured' else plan[g][:,c] for c in words]
        if not len(b[0]):continue
        o=arrays_to_owners(b,truth,seeds)
        visible=np.isfinite(plan['view'][:,0]);par=plan['parent']
        ancestor=np.eye(len(par),dtype=bool)
        for v in range(len(par)-1,-1,-1):
            if par[v]>=0:ancestor[v]|=ancestor[par[v]]
        disjoint=~(ancestor|ancestor.T)
        v=plan['view'];lr=v[:,2,None]<v[:,0] if side<0 else v[:,0,None]>v[:,2]
        for name,mask in [('union',np.ones_like(plan['mark'][:,:6])),('word_cut',plan['mark'][:,:6])]:
            pair=[]
            for role,true in enumerate((ts,tr)):
                hits=np.logical_and.reduce([x[role]==true for x in o.values()])
                pair.append(hits & visible & mask[:,words[role]])
            s,r=pair
            lines.append(dict(**hdr,geometry=g,arm=name,subject=int(s.any()),reference=int(r.any()),
                  both=int(s.any() and r.any()),admissible=int(np.any(s[:,None]&r&lr&disjoint))))
    return lines


def evaluate(src,repo):
    seal=read(WORK/'SEALED.json');assert seal['code']==signed() and not seal['labels_used']
    roster=catalog();table=ordinary();fallback=previous_rows();past=archive.old
    prior={(r['source'],r['uid']):r for r,*_ in past.observations()}
    labs=labels();panels={g:read(PANELS[g]) for g in COHORTS};hist=historical_records(repo);cache={}
    nf=archive.previous.EXTRA/'labels.json';assert digest(nf)==read(past.OUT/'noun_panel.json')['labels']
    nouns=read(nf);tasks=[];identities=[];geometry=[];covers=[];solved=set();error=0.;legacy_error=0.
    for row in roster['rows']:
        if row['source']!=src:continue
        rid=row_id(row);f=row['frame'];uid=row['uid'];tgt=row['target'];scope=row['scope']
        p=WORK/'decisions'/(rid+'.npz');assert digest(p)==seal['files'][rid]
        pair='';half=-1;case=uid
        with np.load(p) as z,np.load(row['prior']) as old,np.load(row['role']) as native:
            if scope=='ordinary':
                case=anonymous((tgt,row['ordinal']));rr=fallback[src,case]['v1'];solved.add(case)
                inp=table[tgt][row['ordinal']];target=ts=0;tr=None;seeds=None;view=inp['suffix']
                truth,cap=inventory(inp['metapath'],tgt,view,inp['object_index']);assert normalize(cap)==normalize(inp['caption'])
                sg=sensor_truth(tgt,str(inp['metapath']),inp['caption'],view)
                panel='full';fr,rec=rr['frame'],rr['recording']
            elif scope=='noun':
                e=nouns[uid];truth=e['truth'];target=ts=e['subject'];tr=e['reference'];panel='noun'
                fr,rec=e['frame'],e['recording'];pair,half=e['pair'],e['half'];view=int(old['view']);sg=None;seeds=old['seeds']
            else:
                e=labs[uid];group=e['group'];panel='natural' if group in COHORTS else ('change_landmark' if group in ('name','depth') else group)
                truth,ts,tr,fr,rec,view,sg=metadata(prior[src,uid],e,panels,hist,cache)
                target=e.get('subject',0);seeds=native['seeds'];pair=e.get('pair','');half=e.get('half',-1)
                if group in COHORTS:
                    rel=f"{COHORTS[group]}/{panels[group][e['case']]['case']:02d}_{src}.npz"
                    dep=ROOT/'crossplatform/round171/predictions'/rel;assert digest(dep)==hist[0][rel]
                    with np.load(dep) as saved:seeds=saved['seeds']
            truth=np.asarray(truth);hdr=dict(panel=panel,source=src,target=tgt,case=case,frame=fr,recording=rec)
            mat,err=measure(z,truth,target,tgt,view,sg);error=max(error,err)
            if scope=='ordinary':
                for i,oldarm in enumerate(('v1','shared_cut')):
                    raw=fallback[src,case][oldarm]
                    legacy_error=max(legacy_error,abs(mat[0,i]-float(raw['measured'])),abs(mat[1,i]-float(raw['completed'])))
                assert legacy_error<1e-7
            for j,arm in enumerate(ARMS):
                tasks.append(dict(**hdr,arm=arm,eligible=int(row['active']),emitted=int(z['selected'][j,0]>=0),
                             **{g:mat[i,j] for i,g in enumerate(GEOMS)}))
            known=ts is not None and tr is not None
            crossed=row['active'] and np.any((mat[1:,2:]>.5)!=(mat[:1,2:]>.5))
            if not (known or crossed):continue
            if seeds is None:
                with np.load(roster['frames'][f]['lex']) as raw:seeds=np.unique(raw['sample'].astype(float),axis=0)
            ids,own=identify(z,truth,seeds,ts,tr,hdr,pair,half);identities.extend(ids)
            if scope=='noun' and row['active']:
                plan=WORK/'plans'/(f+'.npz');record=WORK/'frame_seals'/(f+'.json')
                assert digest(record)==seal['frame_seals'][f] and digest(plan)==read(record)['plan']
                with np.load(plan) as s:covers.extend(coverage(z,s,truth,seeds,row['words'],ts,tr,dict(**hdr,pair=pair,half=half),int(native['side'])))
            if crossed:
                pool=best(native['a'] if scope=='panel' else old['native_a'],truth[target])
                geometry.extend(geo_diagnostics(hdr,z,mat,own,pool,target,tr))
    for (source,case),group in fallback.items():
        if source!=src or case in solved:continue
        x=group['v1'];y=group['shared_cut']
        assert x['parsed']=='0' and x['measured']==y['measured'] and x['completed']==y['completed']
        tasks.append(dict(panel='full',source=src,target=x['target'],case=case,frame=x['frame'],recording=x['recording'],
              arm='all_fallback',eligible=0,emitted=0,**{g:float(x['measured' if g=='measured' else 'completed']) for g in GEOMS}))
    for name,rows in [('task',tasks),('identity',identities),('geometry',geometry),('coverage',covers)]:
        if rows:write_gz(OUT/(src+'_'+name+'_rows.csv.gz'),rows)
    save(OUT/(src+'_score.json'),dict(source=src,registered_full=len(solved),iou_chain_error=error,
        baseline_replay_error=legacy_error,predictions=digest(WORK/'SEALED.json'),FINAL=False))
    print('scored',src,len(tasks),len(identities),len(geometry),len(covers),flush=True)


def fresh_score():
    root=WORK/'fresh';s=read(root/'SEALED.json');assert s['code']==signed() and not s['labels_read']
    prev=archive.WORK/'fresh'/'score_only.json';assert digest(prev)==read(archive.OUT/'fresh_panel.json')['scoring']
    docs=read(prev);tasks=[];identities=[];geometry=[];error=0.
    for key,h in s['files'].items():
        src,uid=key.split('_',1);e=docs[uid];truth=np.array(e['truth']);target=e['subject'];tr=e['reference']
        path=root/(key+'.npz');assert digest(path)==h
        hdr=dict(panel='fresh',source=src,target='waymo',case=uid,frame=e['frame'],recording=e['recording'])
        with np.load(path) as z:
            mat,err=measure(z,truth,target,'waymo',int(z['view']),None);error=max(error,err)
            for j,arm in enumerate(ARMS):tasks.append(dict(**hdr,arm=arm,eligible=1,emitted=int(z['selected'][j,0]>=0),
                    **{g:mat[i,j] for i,g in enumerate(GEOMS)}))
            ids,own=identify(z,truth,z['seeds'],target,tr,hdr,e['pair'],e['half']);identities.extend(ids)
            geometry.extend(geo_diagnostics(hdr,z,mat,own,best(z['native_a'],truth[target]),target,tr))
    for name,rows in [('task',tasks),('identity',identities),('geometry',geometry)]:
        if rows:write_gz(OUT/('fresh_'+name+'_rows.csv.gz'),rows)
    save(OUT/'fresh_score.json',dict(records=len(s['files']),iou_chain_error=error,predictions=digest(root/'SEALED.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',choices=[*SOURCES,'fresh']);p.add_argument('--archive',type=Path)
    args=p.parse_args()
    if args.source=='fresh':fresh_score()
    else:evaluate(args.source,args.archive)
