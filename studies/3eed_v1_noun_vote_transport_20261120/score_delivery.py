"""Post-seal task, endpoint, noun-ranking and fixed-identity geometry scoring."""
import argparse
import csv
import gzip
from pathlib import Path
import numpy as np
from vote_paths import *
from predict_delivery import codepins
from evaluate_witness import metadata, PANELS, COHORTS, historical_records
from score_tree import labels
from score_roles import score_boxes, sensor_frame_box, sensor_truth, inventory, normalize, many_iou
from score_swap import best
from score_lex import lexical_has_pass
from shared_paths import anonymous
from endpoint_score import arrays_to_owners, write_gz

GEOMS=('raw','anchored','fit','null_fit')
REFS=('ref','ref_anchored','ref_fit','ref_null_fit')
RULES=('unique_iou','seed_ownership')

def previous_rows():
    path=old.OUT/'full_task_rows.csv.gz'
    source={}
    with gzip.open(path,'rt') as f:
        for r in csv.DictReader(f):
            if r['arm'] in ('v1','shared_cut'):
                key=(r['source'],r['case'])
                source.setdefault(key,{})[r['arm']]=r
    assert len(source)==18620 and all(len(v)==2 for v in source.values())
    return source

def match_ranks(field, truth, refid, words, hdr, results):
    # One maximum per independently annotated physical instance, not one vote per duplicate box.
    word=words[1]
    visible=field['visible']
    for name,boxes in [('raw',field['boxes']),('anchored',field['completed'][:,word])]:
        if not visible.any():continue
        from endpoint_score import unique_iou
        owner=unique_iou(boxes[visible],truth)
        good=owner==refid
        others=np.unique(owner[(owner>=0)&(~good)])
        if not good.any() or not len(others):continue
        for arm in ('tree','association','direct_votes','direct_null','inverse_association','centers_only','sizes_only'):
            probability=field[arm][visible,word]
            target=float(probability[good].max())
            distractors=np.array([probability[owner==k].max() for k in others])
            results.append(dict(**hdr,arm=arm,ownership=name,rivals=len(others),
                  beats_strongest=int(target>distractors.max()),margin=target-distractors.max(),
                  pair_accuracy=np.mean((target>distractors)+.5*(target==distractors))))

def beyond_ceiling(row,info,subject_truth,measured_ceiling,original,phrase):
    if max(original,phrase,measured_ceiling)>.5:return 0
    assert digest(info['lex'])==info['lh'] and digest(info['word'])==info['wh']
    with np.load(info['lex']) as l,np.load(info['word']) as w, np.load(WORK/'votes'/row['source']/(row['frame']+'.npz')) as v:
        # Both zero yaw and registered PCA yaw are checked for every feasible raw candidate.
        if lexical_has_pass(l['native_boxes'],l['sample'],subject_truth):return 0
        isolated=np.concatenate([v['center'],w['extent']],axis=-1)
        return int(not lexical_has_pass(isolated,l['sample'],subject_truth))

def rows_for_source(src,repo):
    cat=catalog();seal=read(WORK/'SEALED.json')
    assert not seal['labels_used'] and seal['code']==codepins() and seal['catalog']==digest(WORK/'CATALOG.json')
    assert seal['protocol']==digest(SPEC) and seal['original_full_predictions']==pins()['ordinary_seal']
    olds=previous_rows();tab=ordinary()
    original_rows={(r['source'],r['uid']):r for r,*_ in old.observations()}
    entry=labels();panels={g:read(PANELS[g]) for g in COHORTS};history=historical_records(repo);cache={}
    nounfile=EXTRA/'labels.json';assert digest(nounfile)==read(old.OUT/'noun_panel.json')['labels']
    nouns=read(nounfile)
    scores=[];identity=[];ranks=[];geometry=[];covered=set();maxerr=0.;priorerr=0.;audited=0
    jtree=ARMS.index('tree'); jmain=ARMS.index('association')
    for row in cat['rows']:
        if row['source']!=src:continue
        scope=row['scope'];tgt=row['target'];uid=row['uid'];rid=row_id(row)
        pred=WORK/'predictions'/(rid+'.npz')
        assert digest(pred)==seal['files'][rid] and digest(row['prior'])==row['ph'] and digest(row['role'])==row['rh']
        info=cat['frames'].get(row['frame'])
        case=uid;small=0;pair='';half=-1
        with np.load(pred) as z,np.load(row['role']) as native,np.load(row['prior']) as past:
            if scope=='ordinary':
                i=row['ordinal'];v=tab[tgt][i];case=anonymous((tgt,i));gi=olds[src,case]['v1'];covered.add(case)
                frameid,rec=gi['frame'],gi['recording'];small=int(gi['in_small_panel']);panel='full';view=v['suffix']
                truth,cap=inventory(v['metapath'],tgt,v['suffix'],v['object_index'])
                assert normalize(cap)==normalize(v['caption'])
                sg=sensor_truth(tgt,str(v['metapath']),v['caption'],v['suffix'])
                ts=0;tr=None;subject=0;seeds=None
            elif scope=='noun':
                lab=nouns[uid];panel='noun';truth=np.asarray(lab['truth']);ts, tr=lab['subject'],lab['reference']
                frameid,rec=lab['frame'],lab['recording'];pair=lab['pair'];half=int(lab['half'])
                subject=ts;view=int(past['view']);sg=None;seeds=past['seeds']
            else:
                lab=entry[uid];g=lab['group'];panel='natural' if g in COHORTS else ('change_landmark' if g in ('name','depth') else g)
                truth,ts,tr,frameid,rec,view,sg=metadata(original_rows[src,uid],lab,panels,history,cache)
                subject=lab.get('subject',0);seeds=native['seeds']
                if 'pair' in lab:pair=lab['pair'];half=int(lab['half'])
                if g in COHORTS:
                    n=panels[g][lab['case']]['case'];key=f'{COHORTS[g]}/{n:02d}_{src}.npz'
                    file=ROOT/'crossplatform/round171/predictions'/key;assert digest(file)==history[0][key]
                    with np.load(file) as archived:seeds=archived['seeds']
            truth=np.asarray(truth);ground=truth[subject]
            hdr=dict(panel=panel,source=src,target=tgt,case=case,frame=frameid,recording=rec)
            boxes,idx=np.unique(np.vstack([z[g] for g in GEOMS]),axis=0,return_inverse=True)
            a,ae=score_boxes(boxes,ground)
            sensor=np.array([sensor_frame_box(b,tgt,view) for b in boxes])
            b,be=score_boxes(sensor,sg if sg is not None else sensor_frame_box(ground,tgt,view))
            maxerr=max(maxerr,ae,be,float(abs(a-b).max()));assert maxerr<1.e-8
            np.testing.assert_equal(a[:,None]>[.25,.5],b[:,None]>[.25,.5])
            mat=a[idx].reshape(4,len(ARMS))
            if scope=='ordinary':
                for arm,k in [('v1',0),('shared_cut',jtree)]:
                    g0=olds[src,case][arm]
                    priorerr=max(priorerr,abs(mat[0,k]-float(g0['measured'])),abs(mat[1,k]-float(g0['completed'])))
                assert priorerr<1e-7
            for j,arm in enumerate(ARMS):
                scores.append(dict(**hdr,arm=arm,eligible=int(row['active']),selected=int(z['selected'][j,0]>=0),
                       old_panel=small,**{k:mat[i,j] for i,k in enumerate(GEOMS)}))
            known=ts is not None and tr is not None
            crossing=(mat>.5)!=(mat[0:1]>.5)
            if not known and not crossing.any():continue
            if seeds is None:
                if not info:continue
                with np.load(info['lex']) as l:seeds=np.unique(l['sample'].astype(float),axis=0)
            owner=arrays_to_owners([z[g] for g in (*GEOMS,*REFS)],truth,seeds)
            field=None
            if row['active']:
                path=WORK/'fields'/(row['frame']+'.npz');assert digest(path)==seal['observed'][row['frame']]
                with np.load(path) as fp:field=dict(fp)
            if known:
                for j,arm in enumerate(ARMS):
                    for rule,own in owner.items():
                        for i,geom in enumerate(GEOMS):
                            identity.append(dict(**hdr,arm=arm,geometry=geom,rule=rule,pair=pair,half=half,
                                subject=int(own[i][j]==ts),reference=int(own[4+i][j]==tr),
                                both=int(own[i][j]==ts and own[4+i][j]==tr),subject_owner=int(own[i][j])))
                if field is not None:match_ranks(field,truth,tr,row['words'],hdr,ranks)
            if not crossing.any() or field is None:continue
            if scope=='panel':
                orig=best(native['a'],ground);phr=max([orig]+[best(native[k],ground) for k in ('t_box','r_box') if k in native and len(native[k])])
            else:
                orig=best(past['native_a'],ground);phr=max([orig]+[best(past[k],ground) for k in ('native_t','native_r') if len(past[k])])
            cover=many_iou(field['boxes'][field['visible']],ground).max(initial=0)
            strict=None
            for j,arm in enumerate(ARMS):
                if z['selected'][j,0]<0:continue
                for i,geom in enumerate(GEOMS[1:],1):
                    for k in ([0] if i==1 else [0,1]):
                        if (mat[i,j]>.5)==(mat[k,j]>.5):continue
                        st=all(own[k][j]==subject and own[i][j]==subject for own in owner.values())
                        rescue=int(mat[i,j]>.5)
                        if st and rescue and strict is None and max(orig,phr,cover)<=.5:
                            strict=beyond_ceiling(row,info,ground,cover,orig,phr);audited+=1
                        all_native=bool(st and rescue and strict==1)
                        geometry.append(dict(**hdr,arm=arm,geometry=geom,control=GEOMS[k],
                              before=mat[k,j],after=mat[i,j],rescue=rescue,harm=1-rescue,
                              subject_kept=int(st),reference_known=int(known),
                              both_endpoints=int(known and st and all(own[4+i][j]==tr for own in owner.values())),
                              a256=orig,phrase=phr,measured_physical=cover,
                              outside_a256=int(st and rescue and orig<=.5),outside_all_native=int(all_native)))
    for (source,case),prev in olds.items():
        if source!=src or case in covered:continue
        x=prev['v1'];y=prev['shared_cut'];assert x['parsed']=='0'
        assert x['measured']==y['measured'] and x['completed']==y['completed']
        scores.append(dict(panel='full',source=src,target=x['target'],case=case,frame=x['frame'],recording=x['recording'],
                  arm='all_fallback',eligible=0,selected=0,old_panel=int(x['in_small_panel']),raw=float(x['measured']),
                  anchored=float(x['completed']),fit=float(x['completed']),null_fit=float(x['completed'])))
    for name,rows in [('task',scores),('identity',identity),('ranking',ranks),('geometry',geometry)]:
        if rows:write_gz(OUT/(src+'_'+name+'_rows.csv.gz'),rows)
    save(OUT/(src+'_scoring.json'),dict(ground_error=maxerr,prior_error=priorerr,explicit_word_ceiling_checks=audited,
        ordinary_eligible=len(covered),identity_rows=len(identity),files=digest(WORK/'SEALED.json'),
        same_information=codepins(),private_targets_read_after_sealing=True))
    print(src,'complete',len(scores),len(identity),len(ranks),len(geometry),'error',maxerr,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',choices=SOURCES);p.add_argument('--archive',required=True,type=Path)
    args=p.parse_args();rows_for_source(args.source,args.archive)

