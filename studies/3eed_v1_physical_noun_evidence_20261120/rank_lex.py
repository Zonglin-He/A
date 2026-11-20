"""Post-seal diagnosis of physical ranking hidden by the frame-wide noun preference."""
from collections import defaultdict
import numpy as np
from common_lex import *
from panel_witness import inventory
from ordered_choice import prepare
from physical_evidence import lexicon
from endpoint_score import arrays_to_owners,write_gz
from score_roles import csv_file
from scope183 import VIEWS


def rank(rows):
    out=[]
    for a in ('spatial','misplaced'):
        for side in ('t','r'):
            for metric in ('true_vs_same','true_vs_other','class_vs_other'):
                for rule in ('unique_iou','seed_ownership'):
                    cells=defaultdict(list)
                    for r in rows:
                        if (r['arm'],r['role'],r['rule'],r['test'])==(a,side,rule,metric):
                            cells[r['frame'],r['source']].append(r['auc'])
                    if not cells:continue
                    per_frame=defaultdict(list)
                    for (frame,source),values in cells.items():per_frame[frame].append(np.mean(values))
                    # Both source checkpoints and both captions from the same physical scan form one trial.
                    out.append(dict(arm=a,role=side,test=metric,rule=rule,frames=len(per_frame),frame_sources=len(cells),
                        valid_records=sum(map(len,cells.values())),auc=float(np.mean([np.mean(v) for v in per_frame.values()]))))
    return out


def run():
    assert read(OUT/'audit.json')['sealed']==digest(PRED/'SEALED.json')
    labels=read(OLD/'labels.json');obs=read(OBS/'SEALED.json')
    records=[]
    for row in panel():
        case=row['uid'];entry=labels[case]
        if 'pair' not in entry:continue
        f=row['frame'];idx=VIEWS.index(f['waymo_view'])
        nouns,truth,conflict=inventory(dict(metapath=f['calibration'],platform=f['target'],suffix=idx))
        assert not conflict and np.array_equal(truth,np.asarray(entry['truth']))
        n=name(row)
        with np.load(OBS/(obs['files'][n]+'.npz')) as h,np.load(OLD/'observed'/n) as z:
            seeds,scores,_=lexicon(h['xyz'],h['logits'])
            T=prepare(z,'t')[0];R=prepare(z,'r')[0]
            owners=arrays_to_owners([T,R],truth,z['seeds'])
            for l,(boxes,side,goal,word) in enumerate(zip((T,R),('t','r'),(entry['subject'],entry['reference']),obs['nouns'][n])):
                ids=[seeds.local(b)[0] for b in boxes]
                wordscores=scores[:,word];wrong=np.roll(wordscores,len(wordscores)//2)
                for arm,signal in [('spatial',wordscores),('misplaced',wrong)]:
                    boxscore=np.array([signal[x].mean() if len(x) else np.nan for x in ids])
                    for rule,own in owners.items():
                        indices=own[l]
                        obj={i:np.median(boxscore[(indices==i)&np.isfinite(boxscore)])
                             for i in np.unique(indices[np.isfinite(boxscore)]) if i>=0}
                        siblings=[i for i in obj if i!=goal and nouns[i]==CLASSES[word]]
                        others=[i for i in obj if nouns[i]!=CLASSES[word]]
                        right=[i for i in obj if nouns[i]==CLASSES[word]]
                        for test,left,neg in [('true_vs_same',[goal] if goal in obj else [],siblings),
                                              ('true_vs_other',[goal] if goal in obj else [],others),
                                              ('class_vs_other',right,others)]:
                            if not left or not neg:continue
                            d=np.array([obj[i]-obj[j] for i in left for j in neg])
                            records.append(dict(pair=entry['pair'],frame=entry['frame'],source=row['source'],case=case,
                                role=side,rule=rule,arm=arm,test=test,comparisons=len(d),
                                auc=float(np.mean((d>0)+.5*(d==0)))))
    write_gz(OUT/'within_scene_rank_rows.csv.gz',records)
    csv_file(OUT/'within_scene_rank.csv',rank(records))


if __name__=='__main__':run()
