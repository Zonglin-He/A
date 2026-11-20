"""For every verified new crossing, test whether a stronger candidate inventory already passed."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from partition_paths import ARMS, OUT, WORK, SOURCES, archive, catalog, digest, read, row_id
from evaluate_witness import PANELS, COHORTS, historical_records, metadata
from score_tree import labels
from score_roles import inventory
from shared_paths import anonymous
from batch_yaw_iou import many_iou
from score_swap import best
from score_lex import lexical_has_pass
from scope183 import ordinary


def run(repo):
    data=pd.concat([pd.read_csv(OUT/(src+'_geometry_rows.csv.gz')) for src in SOURCES])
    wins=data.query("arm=='word_cut' and control=='measured' and geometry=='envelope' and outside_a256==1")
    assert not wins.duplicated(['panel','source','case']).any()
    roster=catalog();sealed=read(WORK/'SEALED.json');table=ordinary()
    prior={(r['source'],r['uid']):r for r,*_ in archive.old.observations()}
    labs=labels();panels={g:read(PANELS[g]) for g in COHORTS};hist=historical_records(repo);cache={}
    noun=read(archive.previous.EXTRA/'labels.json');out=[]
    for row in roster['rows']:
        case=anonymous((row['target'],row['ordinal'])) if row['scope']=='ordinary' else row['uid']
        selected=wins[(wins.source==row['source'])&(wins.case==case)]
        if selected.empty:continue
        assert len(selected)==1
        old=selected.iloc[0];uid=row['uid'];fid=row['frame'];src=row['source'];tgt=row['target']
        if row['scope']=='ordinary':
            item=table[tgt][row['ordinal']];gt=inventory(item['metapath'],tgt,item['suffix'],item['object_index'])[0][0]
        elif row['scope']=='noun':
            gt=noun[uid]['truth'][noun[uid]['subject']]
        else:
            e=labs[uid];gt=metadata(prior[src,uid],e,panels,hist,cache)[0][e.get('subject',0)]
        gt=np.asarray(gt);frameinfo=roster['frames'][fid]
        p=WORK/'decisions'/(row_id(row)+'.npz')
        receipt=WORK/'frame_seals'/(fid+'.json');plan=WORK/'plans'/(fid+'.npz')
        assert digest(p)==sealed['files'][row_id(row)] and digest(receipt)==sealed['frame_seals'][fid]
        assert digest(plan)==read(receipt)['plan'] and digest(row['role'])==row['rh'] and digest(row['prior'])==row['ph']
        with np.load(p) as z,np.load(row['role']) as native,np.load(row['prior']) as past,np.load(plan) as q:
            original=best(native['a'],gt);assert abs(original-old.a256)<1e-7
            phrase=max([original]+[best(native[k],gt) for k in ('t_box','r_box') if len(native[k])])
            previous_raw=past['physical_raw' if row['scope']=='panel' else 'visible_physical']
            before=float(max(many_iou(q['boxes'],gt).max(initial=0.),many_iou(previous_raw,gt).max(initial=0.)))
            assert digest(frameinfo['lex'])==frameinfo['lh']
            with np.load(frameinfo['lex']) as lex:
                noun_pass=lexical_has_pass(lex['native_boxes'],lex['sample'],gt)
            main=ARMS.index('word_cut')
            out.append(dict(panel=old.panel,source=src,target=tgt,case=case,frame=old.frame,recording=old.recording,
                before=old.before,after=old.after,original=original,phrase=phrase,all_measured=before,
                six_word_pool_pass=int(noun_pass),beyond_all_native=int(phrase<=.5 and not noun_pass),
                beyond_all_native_and_measured=int(phrase<=.5 and not noun_pass and before<=.5),
                known_ref=int(old.ref_verified),independent_wrong=int(old.independent_wrong_instance),
                wrong_reference_wrong=int(old.wrong_reference_wrong_instance),
                same_node_as_union=int(z['selected'][main,0]==z['selected'][ARMS.index('union'),0]),
                same_node_as_same_count=int(z['selected'][main,0]==z['selected'][ARMS.index('same_count'),0])))
    assert len(out)==len(wins)
    pd.DataFrame(out).to_csv(OUT/'beyond_pool_rows.csv.gz',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path);run(p.parse_args().archive)
