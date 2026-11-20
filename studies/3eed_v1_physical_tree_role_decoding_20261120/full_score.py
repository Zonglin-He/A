"""Ordinary scoring after six predictions are sealed, including all untouched cases and losses."""
from collections import defaultdict
import csv
import gzip
from pathlib import Path
import numpy as np
from paths_tree import *
from full_roster import FULL,CFG_FULL,full_pins
from scope183 import ordinary,frame
from full_observe import key
from terrain_followup import boxes_in_model
from shared_paths import anonymous
from common_lex import sample
from score_roles import score_boxes,sensor_truth,inventory,normalize,record_intervals,csv_file
from endpoint_score import arrays_to_owners,write_gz
from score_swap import best
from score_lex import lexical_has_pass
from batch_yaw_iou import many_iou
from tables_scene import paired_counts

def score():
    out=FULL/'predictions';seal=read(out/'SEALED.json')
    assert not seal['labels_used'] and seal['pins']==full_pins() and seal['records']==18620
    assert seal['code']==digest(HERE/'full_decide.py') and seal['protocol']==digest(CFG_FULL)
    table=ordinary();small=set()
    for row,*_ in observations():small.add((row['source'],key(row['frame']),row['frame']['text'].strip()))
    oldpath=parent.previous.OUT/'terrain_task_rows.csv.gz'
    old={}
    with gzip.open(oldpath,'rt') as h:
        for y in csv.DictReader(h):
            if y['arm']=='local_two_halves':old[y['case'],y['source']]=(float(y['v1_iou']),float(y['iou']))
    tasks=[];crosses=[];stats=[];deltas=[];specimens={}
    mats={};rosters={};err=0.;checked=0;overlap=0
    j=ARMS.index('shared_cut')
    for d,sh in sorted(seal['dirs'].items()):
        src,tgt=d.split('_to_');folder=out/d
        assert digest(folder/'SEALED.json')==sh
        sealed=read(folder/'SEALED.json');assert digest(folder/'all.npz')==sealed['all']
        captured=read(FULL/'observations'/(src+'.json'))
        with np.load(folder/'all.npz') as dec:
            raw=dec['measured'];after=dec['completed'];active=dec['applied'];reg=dec['registered']
        n=len(raw);mat=np.empty((n,2,len(ARMS)));oldpanel=np.zeros(n,bool)
        for i,v in enumerate(table[tgt]):
            f=frame(v);caseid=anonymous((tgt,i));rowrec=anonymous((tgt,v['recording']));frameid=anonymous(v['pointpath'])
            oldpanel[i]=(src,key(f),f['text'].strip()) in small
            uid=caseid,src;gt=sensor_truth(tgt,v['metapath'],v['caption'],v['suffix'])
            poses,inv=np.unique(np.vstack((raw[i],after[i])),axis=0,return_inverse=True)
            scores,de=score_boxes(poses,gt);mat[i]=scores[inv].reshape(2,-1)
            err=max(err,de,float(np.max(abs(mat[i,:,0]-old[uid]))))
            assert err<1.e-7
            before,done=mat[i]
            if reg[i]:
                path=folder/(str(i)+'.npz');assert digest(path)==sealed['files'][str(i)]
                with np.load(path) as z:
                    np.testing.assert_equal(z['raw'],raw[i]);np.testing.assert_equal(z['completed'],after[i])
                    np.testing.assert_equal(active[i],z['applied'])
            else:
                assert not active[i].any()
                np.testing.assert_equal(mat[i],np.broadcast_to(mat[i,:,0,None],mat[i].shape))
            keyed={}
            if reg[i] and np.any((done>.5)!=(before>.5)):
                # Post-seal assessment of physical identity. Compare both box-overlap owner and raw-return owner.
                truth,cap=inventory(v['metapath'],tgt,v['suffix'],v['object_index'])
                assert normalize(cap)==normalize(v['caption'])
                with np.load(path) as z:
                    check=boxes_in_model(np.vstack([raw[i],after[i]]),tgt,v['suffix'])
                    pts=np.unique(sample(f).astype(float),axis=0)
                    owner=arrays_to_owners([check],truth,pts)
                    pool=best(z['native_a'],truth[0])
                    upper=max([pool]+[best(z[k],truth[0]) for k in ('native_t','native_r') if len(z[k])])
                    rawupper=many_iou(z['visible_physical'],truth[0]).max(initial=0.)
                    possible=bool(upper<=.5 and rawupper<=.5 and np.any((done>.5)&(before<=.5)&active[i]))
                    lexpass=False
                    if possible:
                        fid=captured['records'][tgt+'_'+str(i)]['frame'];word=captured['frames'][fid]
                        assert digest(word['lex'])==word['lex_sha']
                        with np.load(word['lex']) as l:lexpass=lexical_has_pass(l['native_boxes'],l['sample'],truth[0])
                        checked+=1
                    for k,arm in enumerate(ARMS):
                        if (done[k]>.5)==(before[k]>.5):continue
                        own={m:int(o[0][k]==0 and o[0][len(ARMS)+k]==0) for m,o in owner.items()}
                        g=dict(case=caseid,source=src,target=tgt,recording=rowrec,frame=frameid,arm=arm,
                            original=before[k],completed=done[k],raw_a256_and_pca=pool,phrase_pools=upper,
                            best_measured_physical=float(rawupper),from_physical_node=int(k>=2 and active[i,k]),**own)
                        stable=bool(min(own.values()))
                        g['verified_a256_rescue']=int(g['from_physical_node'] and done[k]>.5 and pool<=.5 and stable)
                        g['verified_all_native_rescue']=int(g['verified_a256_rescue'] and possible and not lexpass)
                        crosses.append(g);keyed[k]=g
            for k,arm in enumerate(ARMS):
                tasks.append(dict(case=caseid,source=src,target=tgt,recording=rowrec,frame=frameid,arm=arm,
                    in_small_panel=int(oldpanel[i]),parsed=int(reg[i]),applied=int(active[i,k]),
                    measured=before[k],completed=done[k],v1=before[0],v1_ground=done[0]))
        mats[src,tgt]=mat;rosters[src,tgt]=dict(recording=np.array([y['recording'] for y in table[tgt]]),
              registered=reg,old=oldpanel,applied=active)
        overlap+=int(oldpanel.sum());print('scored ordinary',d,n,int(reg.sum()),flush=True)
    for subset in ('all','outside_small','in_small','parsed','fallback'):
        for phase,setting in enumerate(('measured','completed')):
            data={};recs={};values=[]
            for (s,t),m in mats.items():
                r=rosters[s,t];mask={'all':np.ones(len(m),bool),'outside_small':~r['old'],'in_small':r['old'],
                     'parsed':r['registered'],'fallback':~r['registered']}[subset]
                if not mask.any():continue
                v=m[mask,phase]
                # Original strict v1 is an additional baseline even for the completed geometry condition.
                v=np.column_stack([v,m[mask,0,0]])
                data[s,t]=v;recs[s,t]=dict(recording=r['recording'][mask])
                numbers=100*np.array([(v>.25).mean(0),(v>.5).mean(0),v.mean(0)]);values.append(numbers)
                for k,arm in enumerate(ARMS+('v1_raw',)):
                    stats.append(dict(subset=subset,geometry=setting,direction=s+'_to_'+t,n=len(v),arm=arm,
                                    acc25=numbers[0,k],acc50=numbers[1,k],miou=numbers[2,k]))
            if not data:continue
            macro=np.mean(values,axis=0);merged=np.vstack(list(data.values()))
            for k,arm in enumerate(ARMS+('v1_raw',)):
                stats.append(dict(subset=subset,geometry=setting,direction='macro',n=len(merged),arm=arm,
                                acc25=macro[0,k],acc50=macro[1,k],miou=macro[2,k]))
                ci=record_intervals(recs,data,j,k)
                for d,v in [('all',merged)]+[(s+'_to_'+t,m) for (s,t),m in data.items()]:
                    deltas.append(dict(subset=subset,geometry=setting,direction=d,n=len(v),control=arm,
                        **paired_counts(v,j,k),improved=int((v[:,j]>v[:,k]+1e-10).sum()),
                        worsened=int((v[:,j]<v[:,k]-1e-10).sum()),macro_delta50_low=ci[0] if d=='all' else '',
                        macro_delta50_high=ci[1] if d=='all' else ''))
    assert len(tasks)==18620*len(ARMS) and len(old)==18620
    csv_file(OUT/'full_metrics.csv',stats);csv_file(OUT/'full_paired.csv',deltas)
    write_gz(OUT/'full_task_rows.csv.gz',tasks)
    if crosses:write_gz(OUT/'full_geometry_rows.csv.gz',crosses)
    save(OUT/'full_audit.json',dict(records=18620,arms=len(ARMS),scored_predictions=len(tasks),max_error=err,
        small_overlap=overlap,outside_small=18620-overlap,parsed=sum(int(x['registered'].sum()) for x in rosters.values()),
        verified_earlier_terrain_and_original=len(old),explicit_full_lexical_ceiling_checks=checked,
        full_geometry_crossings_logged_on_eligible_records=len(crosses),predictions=digest(out/'SEALED.json'),
        geometry_code=digest(HERE/'full_score.py'),scored_after_sealing=True,FINAL_access=False))
if __name__=='__main__':score()
