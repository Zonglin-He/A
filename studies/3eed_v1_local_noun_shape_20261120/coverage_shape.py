"""Identify the earliest lost physical endpoint in each two-utterance diagnostic after sealing."""
import numpy as np
from paths_shape import *
from predict_shape import observations
from ordered_choice import prepare,relations
from endpoint_score import arrays_to_owners,write_gz
from score_roles import csv_file
from collections import defaultdict
from verify_shape import grouped_local


def run():
    labels={}
    for folder in (previous.OLD,previous.NEW,WORK):
        labels.update(read(folder/'labels.json'))
    seal=read(PRED/'SEALED.json')
    groups=defaultdict(dict)
    for row,path,sha,_,_,_,words,*_ in observations():
        entry=labels[row['uid']]
        if 'pair' not in entry:continue
        name_=name(row);assert digest(path)==sha and digest(PRED/name_)==seal['files'][name_]
        with np.load(path) as native,np.load(PRED/name_) as out:
            T,_,_,tv=prepare(native,'t');R,_,_,rv=prepare(native,'r')
            np.testing.assert_equal(T,out['candidates'])
            np.testing.assert_equal(R,out['ref_candidates'])
            within=np.isfinite(relations(tv,rv,int(native['side'])))
            owners=arrays_to_owners([native['a'],native['t_box'],native['r_box'],T,R,
                                     np.vstack((out['raw'],out['ref']))],np.asarray(entry['truth']),native['seeds'])
            for rule,(a,st,sr,t,r,choices) in owners.items():
                s0,r0=entry['subject'],entry['reference']
                it,ir=t==s0,r==r0
                real=it.any() and ir.any() and np.any(within[it][:,ir])
                k=ARMS.index('local_shape');correct=(choices[k]==s0 and choices[len(ARMS)+k]==r0)
                panel=entry['group'] if entry['group'] not in ('name','depth') else 'change_landmark'
                key=(panel,entry['pair'],row['source'],row['frame']['target'],entry['frame'],entry['recording'],rule)
                groups[key][entry['half']]=dict(original_subject=bool(s0 in a),original_either_role_both=bool(s0 in a and r0 in a),
                    native_subject=bool(s0 in st),native_ref=bool(r0 in sr),native_both=bool(s0 in st and r0 in sr),
                    visible_subject=bool(it.any()),visible_ref=bool(ir.any()),visible_both=bool(it.any() and ir.any()),
                    positive_geometric_pair=bool(real),both_chosen=bool(correct))
    detailed=[]
    for (panel,pair,src,tar,frame,rec,rule),v in groups.items():
        x,y=v[0],v[1]
        detailed.append(dict(panel=panel,pair=pair,source=src,target=tar,frame=frame,recording=rec,rule=rule,
            **{k:int(x[k] and y[k]) for k in x}))
    write_gz(OUT/'candidate_pair_rows.csv.gz',detailed)
    tab=[]
    for (panel,rule),all_ in grouped_local(detailed,['panel','rule']).items():
        tab.append(dict(panel=panel,rule=rule,pairs=len(all_),frames=len({r['frame'] for r in all_}),
            **{k:sum(r[k] for r in all_) for k in x}))
    csv_file(OUT/'candidate_ladder.csv',tab)

if __name__=='__main__':run()
