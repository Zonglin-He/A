"""Offline diagnostic on constructed pairs. Known categories and instance IDs never enter task predictions."""
import re
from collections import defaultdict
import numpy as np
from paths_witness import WORK, OBS, OUT, read, name, save
from observe_witness import PATTERN
from panel_witness import inventory
from projection_witness import projection
from ordered_choice import prepare,reference_mass,relations,joint,choices
from roles_joint import affinity
from endpoint_score import arrays_to_owners,write_gz
from score_roles import csv_file
from scope183 import VIEWS
from evaluate_witness import check_seals

def annotated_choice(classes,pr,subject,reference,depth,side):
    st=np.flatnonzero((classes==subject) & pr[:,7].astype(bool))
    rs=np.flatnonzero((classes==reference) & pr[:,7].astype(bool))
    if depth:
        j=rs[np.argmin(depth*pr[rs,1])]
    elif len(rs)==1:
        j=rs[0]
    else:
        return -1,-1
    st=st[side*(pr[st,0]-pr[j,0])>0]
    if not len(st):return -1,int(j)
    return int(st[np.argmin(abs(pr[st,0]-pr[j,0]))]),int(j)

def noun_ceilings(T,R,pt,pr,tp,rp,ts,rs,classes,sword,rword,depth,side,grouping):
    ids_t=np.array([i for i,o in enumerate(ts) if o>=0 and classes[o]==sword],int)
    ids_r=np.array([i for i,o in enumerate(rs) if o>=0 and classes[o]==rword],int)
    if not len(ids_t) or not len(ids_r):return -1,-1
    if grouping:
        def one_per(ids,own,conf):
            return np.array([subset[conf[subset].argmax()] for o in np.unique(own[ids]) for subset in [ids[own[ids]==o]]])
        ids_t=one_per(ids_t,ts,pt);ids_r=one_per(ids_r,rs,pr)
    # Restrict the noun only; the location of the named answer is still unknown to the decision.
    ref=reference_mass(np.ones(len(ids_r)),np.ones(len(ids_r))/len(ids_r),rp[ids_r],depth,False,None)
    J=joint(np.ones(len(ids_t)),ref,relations(tp[ids_t],rp[ids_r],side),True)
    pick=choices(J,affinity(T[ids_t],T[ids_t]),affinity(R[ids_r],R[ids_r]))
    if pick is None:return -1,-1
    i,j=pick
    return int(ts[ids_t[i]]),int(rs[ids_r[j]])

def run():
    check_seals(); truth=read(WORK/'labels.json')
    rows=[]
    for case in read(WORK/'roster.json'):
        entry=truth[case['uid']]
        if 'pair' not in entry:continue
        frame=case['frame'];target=frame['target'];view=VIEWS.index(frame['waymo_view']) if target=='waymo' else 0
        classes,boxes,conflict=inventory(dict(metapath=frame['calibration'],platform=target,suffix=view))
        assert not conflict
        np.testing.assert_array_equal(boxes,np.asarray(entry['truth']))
        gtproj=projection(boxes,target,view,frame['calibration'],frame['rgb'])
        st,re=entry['subject'],entry['reference']
        phrase=PATTERN.fullmatch(frame['text']);sw,_,rw,_=phrase.groups()
        with np.load(OBS/name(case)) as z:
            dirn=int(z['side']);dep=int(z['depth']); seeds=z['seeds']
            assert np.array_equal(annotated_choice(classes,gtproj,sw,rw,dep,dirn),[st,re])
            T,pt,_,tp=prepare(z,'t');R,pr,_,rp=prepare(z,'r')
            pools=arrays_to_owners([z['t_box'],z['r_box'],T,R],boxes,seeds)
            for rule,(full_t,full_r,ts,rs) in pools.items():
                start=dict(group=entry['group'],case=case['uid'],pair=entry['pair'],source=case['source'],target=target,
                        rule=rule,native_subject=int(st in full_t),native_reference=int(re in full_r),
                        available_subject=int(st in ts),available_reference=int(re in rs),
                        native_both=int(st in full_t and re in full_r),available_both=int(st in ts and re in rs))
                for arm in ('perfect_nouns','perfect_nouns_and_grouping'):
                    i,j=noun_ceilings(T,R,pt,pr,tp,rp,ts,rs,classes,sw,rw,dep,dirn,arm.endswith('grouping'))
                    rows.append(dict(**start,arm=arm,selected_subject=int(i==st),selected_ref=int(j==re),selected_both=int(i==st and j==re)))
    write_gz(OUT/'oracle_rows.csv.gz',rows);summary=[]
    for rule in ('unique_iou','seed_ownership'):
        for arm in ('perfect_nouns','perfect_nouns_and_grouping'):
            for group in ('name','depth','combined'):
                sub=[r for r in rows if r['rule']==rule and r['arm']==arm and (r['group']==group or group=='combined')]
                res=dict(rule=rule,arm=arm,group=group,records=len(sub),gt_geometry_program_correct=len(sub))
                for k in ('native_subject','native_reference','native_both','available_subject','available_reference',
                          'available_both','selected_subject','selected_ref','selected_both'):
                    res[k]=sum(r[k] for r in sub)
                bounds=defaultdict(list)
                for r in sub:bounds[r['pair'],r['source']].append(r)
                assert all(len(v)==2 for v in bounds.values())
                res['pair_directions']=len(bounds)
                for k in ('available_both','selected_subject','selected_both'):
                    res['pairs_'+k]=sum(all(vv[k] for vv in v) for v in bounds.values())
                summary.append(res)
    csv_file(OUT/'semantic_positive_controls.csv',summary)

if __name__=='__main__':run()
