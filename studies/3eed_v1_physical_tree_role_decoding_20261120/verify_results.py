"""Recompute public decision counts and independently audit full-cohort threshold flips."""
import csv
import gzip
from collections import Counter,defaultdict
import hashlib
import numpy as np
from paths_tree import *
from full_roster import FULL
from shared_paths import anonymous
from score_roles import inventory,csv_file
from endpoint_score import arrays_to_owners,write_gz
from scope183 import ordinary,frame
from common_lex import sample
from terrain_followup import boxes_in_model
from verify_kernel import run as check_kernel

def readrows(name):
    with (gzip.open(OUT/name,'rt') if name.endswith('.gz') else (OUT/name).open()) as f:return list(csv.DictReader(f))

def compare_sealed_boxes():
    root=FULL/'predictions'
    manifest=read(root/'SEALED.json')
    main=ARMS.index('shared_cut')
    totals=defaultdict(Counter)
    for direction,pin in manifest['dirs'].items():
        folder=root/direction
        assert digest(folder/'SEALED.json')==pin
        record=read(folder/'SEALED.json')
        assert digest(folder/'all.npz')==record['all']
        with np.load(folder/'all.npz') as all_boxes:
            for j,control in enumerate(ARMS):
                counter=totals[control]
                counter['records']+=record['n']
                for setting in ('measured','completed'):
                    boxes=all_boxes[setting]
                    counter[setting+'_subject_different']+=int(np.any(boxes[:,main]!=boxes[:,j],axis=-1).sum())
                counter['applied_main']+=int(all_boxes['applied'][:,main].sum())
        for index,pin in record['files'].items():
            f=folder/(index+'.npz')
            assert digest(f)==pin
            with np.load(f) as case:
                for field in ('reference','reference_completed'):
                    refs=case[field]
                    for j,control in enumerate(ARMS):
                        totals[control][field+'_different_on_eligible']+=int(not np.array_equal(refs[main],refs[j],equal_nan=True))
    assert all(v['records']==18620 for v in totals.values())
    assert all(totals['pair_relation'][f]==0 for f in ('measured_subject_different','completed_subject_different',
           'reference_different_on_eligible','reference_completed_different_on_eligible'))
    csv_file(OUT/'full_prediction_comparison.csv',[dict(control=k,**v) for k,v in totals.items()])

def validate():
    fields=readrows('full_task_rows.csv.gz');summ=readrows('full_metrics.csv');deltas=readrows('full_paired.csv')
    n=len(fields); by=defaultdict(dict)
    for r in fields:
        k=r['case'],r['source'];assert r['arm'] not in by[k]
        by[k][r['arm']]=r
    assert n==18620*len(ARMS) and len(by)==18620 and all(set(x)==set(ARMS) for x in by.values())
    catalog=ordinary()
    paths={(anonymous((t,i)),s):(i,v) for t,lst in catalog.items() for i,v in enumerate(lst) for s in SOURCES if s!=t}
    pairchecks=0;allchanges=[];types=defaultdict(Counter)
    keycols=('source','target','direction','reference')
    scores=defaultdict(list);hits=defaultdict(Counter)
    for key,aa in by.items():
        primary=aa['shared_cut'];sf=primary['source'];tt=primary['target']
        for arm,r in aa.items():
            for k in ('in_small_panel','parsed','v1_ground','v1'):assert r[k]==primary[k]
            for f in ('measured','completed'):
                x=float(primary[f]);y=float(r[f]);scores[f,sf+'_to_'+tt,arm].append(y)
                for cut in (25,50):
                    if (x>cut/100)!=(y>cut/100):hits[f,arm]['rescue' + str(cut) if x>cut/100 else 'harm'+str(cut)]+=1
        for control in ('v1','independent','wrong_reference_binding','separate_occupancy'):
            xx=float(primary['completed']);yy=float(aa[control]['completed'])
            if (xx>.5)==(yy>.5):continue
            i,row=paths[key];direc=sf+'_to_'+tt
            fn=FULL/'predictions'/direc/(str(i)+'.npz')
            cert=read(FULL/'predictions'/direc/'SEALED.json');assert cert['files'][str(i)]==digest(fn)
            with np.load(fn) as v:
                ids=[ARMS.index(control),ARMS.index('shared_cut')]
                measured=boxes_in_model(v['raw'][ids],tt,row['suffix']);after=boxes_in_model(v['completed'][ids],tt,row['suffix'])
            truth,_=inventory(row['metapath'],tt,row['suffix'],row['object_index'])
            cloud=np.unique(sample(frame(row)).astype(float),axis=0)
            owners=arrays_to_owners([after,measured],truth,cloud)
            old,new=(tuple(int(owners[k][0][ii]) for k in ('unique_iou','seed_ownership')) for ii in (0,1))
            # Unknown labels and conflicting ownership definitions receive their own bin.
            if old==(0,0) and new==(0,0):kind='same_verified_target'
            elif min(old+new)>=0 and old[0]==old[1] and new[0]==new[1] and old[0]!=new[0]:
                kind='distinct_annotated_instances'
            else:kind='ambiguous_or_other'
            transition='rescue' if xx>.5 else 'harm'
            types[control,transition][kind]+=1
            allchanges.append(dict(case=key[0],source=sf,target=tt,recording=primary['recording'],frame=primary['frame'],
                control=control,transition=transition,attribution=kind,control_iou=yy,main_iou=xx,
                control_target_both_rules=int(old==(0,0)),main_target_both_rules=int(new==(0,0))))
        pairchecks+=1
    for r in summ:
        if r['subset']!='all' or r['direction']=='macro' or r['arm']=='v1_raw':continue
        v=np.array(scores[r['geometry'],r['direction'],r['arm']])
        np.testing.assert_allclose([float(r['acc25']),float(r['acc50']),float(r['miou'])],
                                   100*np.array([(v>.25).mean(),(v>.5).mean(),v.mean()]),atol=1e-10)
    for r in deltas:
        if r['subset']!='all' or r['direction']!='all' or r['control']=='v1_raw':continue
        assert all(int(r[k])==hits[r['geometry'],r['control']][k] for k in ('rescue25','harm25','rescue50','harm50'))
    write_gz(OUT/'full_change_attribution_rows.csv.gz',allchanges)
    csv_file(OUT/'full_change_attribution.csv',[dict(control=c,transition=tr,n=sum(b.values()),
        **{k:b.get(k,0) for k in ('same_verified_target','distinct_annotated_instances','ambiguous_or_other')})
        for (c,tr),b in sorted(types.items())])
    compare_sealed_boxes()
    tests,err=check_kernel()
    save(OUT/'verification.json',dict(ordinary_records=pairchecks,scope_rows=n,full_threshold_counts_and_all_direction_metrics=True,
           full_output=digest(OUT/'full_task_rows.csv.gz'),annotated_crossings=len(allchanges),enumerated_tests=tests,
           maximum_probability_error=err,FINAL_access=False))
if __name__=='__main__':validate()
