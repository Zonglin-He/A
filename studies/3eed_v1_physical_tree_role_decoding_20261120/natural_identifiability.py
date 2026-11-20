"""Post-seal natural-language control with the dataset's available annotated object inventory."""
from collections import Counter,defaultdict
import numpy as np
from paths_tree import CLASSES, OUT, digest, read
from full_roster import FULL
from shared_paths import anonymous
from scope183 import ordinary,frame
from panel_witness import inventory,projection
from score_roles import csv_file,inventory as truth_first,many_iou
from endpoint_score import write_gz


def run():
    assert digest(FULL/'predictions/SEALED.json')==read(OUT/'full_audit.json')['predictions']
    parsed=read(FULL/'parsed.json')
    assert digest(FULL/'parsed.json')==read(FULL/'ROSTER.json')['parsed']
    counts=defaultdict(Counter)
    rows=[]
    for target,roster in ordinary().items():
        cache={}
        for i,(row,words) in enumerate(zip(roster,parsed[target])):
            if words is None:
                continue
            sub,ref,side,depth,near=words
            assert not (depth or near)
            view=int(row['suffix'] or 0)
            key=row['metapath'],view
            if key not in cache:
                names,boxes,conflict=inventory(dict(platform=target,metapath=row['metapath'],suffix=view))
                f=frame(row)
                projected=projection(boxes,target,view,row['metapath'],f['rgb']) if len(boxes) else np.empty((0,8))
                cache[key]=names,boxes,projected,conflict
            names,boxes,projected,conflict=cache[key]
            truth,_=truth_first(row['metapath'],target,view,row['object_index'])
            overlap=many_iou(boxes,truth[0]) if len(boxes) else np.empty(0)
            match=np.flatnonzero(overlap>.96)
            invalid=bool(conflict or len(match)!=1)
            subject=int(match[0]) if not invalid else -1
            visible=projected[:,7]>0
            subject_word=(names==CLASSES[sub]) & visible
            ref_word=(names==CLASSES[ref]) & visible
            for mode in ('projected_centers','disjoint_projected_extents'):
                if mode=='projected_centers':
                    spatial=side*(projected[:,0,None]-projected[:,0])>0
                elif side<0:
                    spatial=projected[:,3,None]<projected[:,2]
                else:
                    spatial=projected[:,2,None]>projected[:,3]
                pairs=(subject_word[:,None] & ref_word[None,:] & spatial) & ~np.eye(len(names),dtype=bool)
                totals=pairs.any(1)
                ever=int(totals.sum())
                if invalid:result='unresolved_inventory'
                elif names[subject]!=CLASSES[sub]:result='subject_noun_disagrees'
                elif not visible[subject]:result='subject_not_visible_by_projection'
                elif not np.any(ref_word & (np.arange(len(names))!=subject)):result='no_visible_reference_of_requested_class'
                elif not totals[subject]:result='target_excluded_by_direction'
                elif ever==1:result='target_only_feasible_subject'
                else:result='target_and_other_feasible_subjects'
                nref=int(pairs[subject].sum()) if subject>=0 else -1
                rows.append(dict(case=anonymous((target,i)),target=target,geometry=mode,verdict=result,
                     eligible_subjects=ever,visible_subject_class=int(subject_word.sum()),
                     eligible_references_for_target=nref))
                for group in (target,'all'):
                    stats=counts[group,mode];stats[result]+=1
                    stats['descriptions']+=1
                    stats['unique_subject_and_one_reference']+=int(result=='target_only_feasible_subject' and nref==1)
                    stats['direction_alone_resolved_competing_subjects']+=int(result=='target_only_feasible_subject' and subject_word.sum()>1)
    assert len(rows)==2*sum(sum(x is not None for x in parsed[t]) for t in parsed)
    kinds=sorted({k for stats in counts.values() for k in stats})
    summary=[dict(target=target,geometry=mode,**{k:stats[k] for k in kinds})
             for (target,mode),stats in sorted(counts.items())]
    csv_file(OUT/'natural_identifiability.csv',summary)
    write_gz(OUT/'natural_identifiability_rows.csv.gz',rows)


if __name__=='__main__':
    run()
