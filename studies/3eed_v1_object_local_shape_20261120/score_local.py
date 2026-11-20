"""Paired task and physical endpoint evaluation strictly downstream of the prediction seal."""
import argparse
from pathlib import Path
import numpy as np
from local_paths import *
from method_local import ARMS,GEOMS,evidence
from predict_local import code
from score_delivery import previous_rows
from evaluate_witness import metadata,PANELS,COHORTS,historical_records
from score_tree import labels
from score_roles import score_boxes,sensor_frame_box,sensor_truth,inventory,normalize,many_iou
from shared_paths import anonymous
from endpoint_score import arrays_to_owners,write_gz
from score_swap import best
from scope183 import ordinary

def evaluate(src,repo):
    sealed=read(WORK/'SEALED.json');assert sealed['code']==code() and not sealed['labels_used'] and sealed['upstream']==base()
    roster=catalog();table=ordinary();fallback=previous_rows()
    prior={(r['source'],r['uid']):r for r,*_ in old.observations()}
    labs=labels();panels={g:read(PANELS[g]) for g in COHORTS};hist=historical_records(repo);cache={}
    nf=previous.EXTRA/'labels.json';assert digest(nf)==read(old.OUT/'noun_panel.json')['labels'];nouns=read(nf)
    tasks=[];endpoints=[];geometry=[];ranking=[];solved=set();err=0.;repro=0.;main=ARMS.index('shape_difference')
    for row in roster['rows']:
        if row['source']!=src:continue
        rid=row_id(row);fid=row['frame'];uid=row['uid'];tgt=row['target'];scope=row['scope']
        filename=WORK/'predictions'/(rid+'.npz')
        assert digest(filename)==sealed['files'][rid] and digest(row['prior'])==row['ph']
        pair='';half=-1;oldpanel=0;case=uid
        with np.load(filename) as pred,np.load(row['prior']) as oldpred,np.load(row['role']) as native:
            if scope=='ordinary':
                case=anonymous((tgt,row['ordinal']));original=fallback[src,case]['v1'];solved.add(case)
                f=table[tgt][row['ordinal']];ts=0;tr=None;target=0;seeds=None;view=f['suffix']
                truth,cap=inventory(f['metapath'],tgt,view,f['object_index']);assert normalize(cap)==normalize(f['caption'])
                sg=sensor_truth(tgt,str(f['metapath']),f['caption'],view);panel='full'
                fr,rec=original['frame'],original['recording'];oldpanel=int(original['in_small_panel'])
            elif scope=='noun':
                e=nouns[uid];truth=np.array(e['truth']);ts=e['subject'];tr=e['reference'];target=ts
                fr,rec=e['frame'],e['recording'];view=int(oldpred['view']);sg=None;seeds=oldpred['seeds']
                panel='noun';pair=e['pair'];half=e['half']
            else:
                e=labs[uid];g=e['group'];panel='natural' if g in COHORTS else ('change_landmark' if g in ('name','depth') else g)
                truth,ts,tr,fr,rec,view,sg=metadata(prior[src,uid],e,panels,hist,cache)
                target=e.get('subject',0);seeds=native['seeds'];pair=e.get('pair','');half=e.get('half',-1)
                if g in COHORTS:
                    case_ix=panels[g][e['case']]['case'];rel=f'{COHORTS[g]}/{case_ix:02d}_{src}.npz'
                    dep=ROOT/'crossplatform/round171/predictions'/rel;assert digest(dep)==hist[0][rel]
                    with np.load(dep) as ss:seeds=ss['seeds']
            truth=np.asarray(truth);gt=truth[target]
            header=dict(panel=panel,source=src,target=tgt,case=case,frame=fr,recording=rec)
            stack=np.stack([pred[g] for g in GEOMS]);boxes,idx=np.unique(stack.reshape(-1,7),return_inverse=True,axis=0)
            score,e1=score_boxes(boxes,gt)
            alt,e2=score_boxes(np.array([sensor_frame_box(b,tgt,view) for b in boxes]),sg if sg is not None else sensor_frame_box(gt,tgt,view))
            err=max(err,e1,e2,float(abs(score-alt).max()));assert err<1e-8
            np.testing.assert_equal(score[:,None]>[.25,.5],alt[:,None]>[.25,.5])
            mat=score[idx].reshape(len(GEOMS),len(ARMS))
            if scope=='ordinary':
                for arm,a in [('v1','v1'),('tree','shared_cut')]:
                    saved=fallback[src,case][a];k=ARMS.index(arm)
                    repro=max(repro,abs(mat[0,k]-float(saved['measured'])),abs(mat[1,k]-float(saved['completed'])))
                assert repro<1e-7
            for j,a in enumerate(ARMS):
                tasks.append(dict(**header,arm=a,eligible=int(row['active']),old_panel=oldpanel,nominated=len(pred['nominated']),
                      observed_s=int(pred['seen'][j,0]),observed_r=int(pred['seen'][j,1]),**{g:mat[i,j] for i,g in enumerate(GEOMS)}))
            known=ts is not None and tr is not None
            crosses=(mat>.5)!=(mat[0:1]>.5)
            if not row['active'] and not known:continue
            if not known and not crosses.any():continue
            if seeds is None:
                info=roster['frames'][fid]
                with np.load(info['lex']) as l:seeds=np.unique(l['sample'].astype(float),axis=0)
            refs=['ref','ref_anchored','ref_fit','ref_direct_fit']
            owners=arrays_to_owners([pred[g] for g in (*GEOMS,*refs)],truth,seeds)
            if known:
                for j,a in enumerate(ARMS):
                    for rule,own in owners.items():
                        for i,g in enumerate(GEOMS):
                            s=int(own[i][j]);r=int(own[4+i][j])
                            endpoints.append(dict(**header,arm=a,rule=rule,geometry=g,pair=pair,half=half,subject=int(s==ts),
                                      reference=int(r==tr),both=int(s==ts and r==tr),s_owner=s))
                if row['active'] and len(pred['nominated']):
                    with np.load(WORK/'plans'/(fid+'.npz')) as pp,np.load(WORK/'observed'/src/(fid+'.npz')) as local:
                        loc={int(v):i for i,v in enumerate(local['ids'])};ids=pred['nominated'];kk=np.array([loc[int(k)] for k in ids])
                        post=evidence(pp['p'],ids,pp['boxes'],local['logit'][kk])
                        data_row=np.flatnonzero(pp['row']==rid).item()
                        from endpoint_score import unique_iou
                        for role,word,identity in [(0,row['words'][0],ts),(1,row['words'][1],tr)]:
                            box=pp['completed'][:,word];visible=np.isfinite(pp['view'][:,0]);ow=unique_iou(box,truth)
                            good=(ow==identity)&visible;different=(ow>=0)&(ow!=identity)&visible
                            rivals=np.unique(ow[different]);visited=pp['nom'][data_row,role]
                            for a in post:
                                p=post[a][:,word]
                                available=bool(good.any() and len(rivals))
                                margin=float(p[good].max()-max(p[(ow==k)&visible].max() for k in rivals)) if available else ''
                                ranking.append(dict(**header,arm=a,role=('subject','reference')[role],cover=int(good.any()),
                                    role_nomination=int((good&visited).any()),either_nomination=int(good[ids].any()),rivals=len(rivals),
                                    comparable=int(available),margin=margin))
            if not crosses.any() or not row['active']:continue
            if scope=='panel':pool=best(native['a'],gt)
            else:pool=best(oldpred['native_a'],gt)
            for j,a in enumerate(ARMS):
                if pred['selected'][j,0]<0:continue
                for i,g in enumerate(GEOMS[1:],1):
                    for k in ([0] if i==1 else [0,1]):
                        if (mat[i,j]>.5)==(mat[k,j]>.5):continue
                        stable=all(o[i][j]==target and o[k][j]==target for o in owners.values())
                        win=mat[i,j]>.5
                        line=dict(**header,arm=a,geometry=g,control=GEOMS[k],before=mat[k,j],after=mat[i,j],a256=pool,
                                  rescue=int(win),harm=int(not win),stable=int(stable),
                                  outside_a256=int(win and stable and pool<=.5),both_known=int(known and stable and all(o[4+i][j]==tr for o in owners.values())))
                        for peer in ('v1','tree','misbound_reference','independent','misbound_crop'):
                            z=ARMS.index(peer);vv=[int(o[0][z]) for o in owners.values()]
                            line[peer+'_wrong_instance']=int(len(set(vv))==1 and vv[0]>=0 and vv[0]!=target)
                            line[peer+'_same_node']=int(pred['selected'][j,0]==pred['selected'][z,0])
                        geometry.append(line)
    for (source,case),rec in fallback.items():
        if source!=src or case in solved:continue
        x=rec['v1'];y=rec['shared_cut'];assert x['parsed']=='0' and x['measured']==y['measured'] and x['completed']==y['completed']
        tasks.append(dict(panel='full',source=src,target=x['target'],case=case,frame=x['frame'],recording=x['recording'],
                  arm='all_fallback',eligible=0,old_panel=int(x['in_small_panel']),nominated=0,observed_s=0,observed_r=0,
                  raw=float(x['measured']),anchored=float(x['completed']),fit=float(x['completed']),direct_fit=float(x['completed'])))
    for name,rows in [('task',tasks),('identity',endpoints),('geometry',geometry),('rank',ranking)]:
        if rows:write_gz(OUT/(src+'_'+name+'_rows.csv.gz'),rows)
    save(OUT/(src+'_score.json'),dict(source=src,records=len(tasks),max_iou_crosscheck=err,predecessor_error=repro,
         registered_full=len(solved),sealed=digest(WORK/'SEALED.json'),source_steps=0,target_steps=0,FINAL=False))
    print('scored',src,len(tasks),len(endpoints),len(geometry),len(ranking),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',choices=SOURCES);p.add_argument('--archive',required=True,type=Path)
    a=p.parse_args();evaluate(a.source,a.archive)
