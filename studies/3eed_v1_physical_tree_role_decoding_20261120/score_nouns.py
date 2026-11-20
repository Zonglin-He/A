"""Do paired adjudication for the preselected single-word challenge after all predictions exist."""
from collections import defaultdict
import numpy as np
from paths_tree import *
from panel_nouns import EXTRA,PIN
from predict_nouns import inputs
from terrain_followup import boxes_in_model
from score_roles import csv_file,score_boxes,sensor_frame_box,many_iou
from endpoint_score import arrays_to_owners,write_gz
from tables_scene import grouped
from score_swap import best
from score_lex import lexical_has_pass

def score():
    seal=read(EXTRA/'SEALED.json')
    assert seal['small']==digest(PRED/'SEALED.json') and not seal['labels_used'] and seal['protocol']==digest(PIN)
    assert seal['implementation']==digest(HERE/'full_decide.py')
    efile=EXTRA/'labels.json';assert digest(efile)==read(OUT/'noun_panel.json')['labels'];labels=read(efile)
    srcseals={}
    for src,sha in seal['sources'].items():
        f=EXTRA/src/'SEALED.json';assert digest(f)==sha;srcseals[src]=read(f)['files']
    rows=[];half=defaultdict(dict);sig={};maxerr=0.;dots={}
    for row in inputs():
        uid=row['uid'];src=row['source'];tg=row['frame']['target'];name_=name(row);folder=EXTRA/src
        zfile=folder/name_;file=srcseals[src][name_]
        assert file['pred']==digest(zfile)
        e=labels[uid];truth=np.array(e['truth']);ts=e['subject'];tr=e['reference']
        with np.load(zfile) as z:
            view=int(z['view']);raw=boxes_in_model(z['raw'],tg,view)
            final=boxes_in_model(z['completed'],tg,view)
            ref=boxes_in_model(z['reference'],tg,view);rfinal=boxes_in_model(z['reference_completed'],tg,view)
            rawpool=z['visible_physical'];tpool=z['subject_pool'];rpool=z['reference_pool']
            import hashlib
            s=hashlib.sha256(rawpool.tobytes()+tpool.tobytes()).hexdigest()
            assert sig.setdefault((src,e['pair']),s)==s
            owns=arrays_to_owners([raw,final,ref,rfinal,rawpool,tpool,rpool],truth,z['seeds'])
            a,ea=score_boxes(np.vstack([raw,final]),truth[ts])
            b,eb=score_boxes(np.vstack([z['raw'],z['completed']]),sensor_frame_box(truth[ts],tg,view))
            maxerr=max(maxerr,ea,eb,float(np.max(abs(a-b))));assert maxerr<1.e-8
            before,done=np.split(a,2)
            ceiling=best(z['native_a'],truth[ts])
            native=max(ceiling,best(z['native_t'],truth[ts]),best(z['native_r'],truth[ts]))
            empirical=many_iou(rawpool,truth[ts]).max(initial=0.)
            beyond=False
            if min(native,empirical)<=.5 and max(native,empirical)<=.5 and np.any((done>.5)&(before<=.5)&z['applied']):
                l=folder/(file['fid']+'_lex.npz');assert digest(l)==file['lex']
                with np.load(l) as lex:beyond=not lexical_has_pass(lex['native_boxes'],lex['sample'],truth[ts])
            for j,arm in enumerate(ARMS):
                result=dict(case=uid,pair=e['pair'],frame=e['frame'],recording=e['recording'],source=src,
                            target=tg,arm=arm,same_half=int(e['same_half']),applied=int(z['applied'][j]),
                            raw_iou=before[j],final_iou=done[j],a256=ceiling,native_phrase=native,raw_boxes=empirical)
                for rule,o in owns.items():
                    so,fo,ro,rf,cp,tp,rp=o
                    values=dict(raw_subject=int(so[j]==ts),raw_ref=int(ro[j]==tr),
                                final_subject=int(fo[j]==ts),final_ref=int(rf[j]==tr),
                                raw_cover=int(ts in cp and tr in cp),final_cover=int(ts in tp and tr in rp))
                    half[src,e['pair'],arm,rule][e['half']]=values
                    for k,v in values.items():result[rule+'_'+k]=v
                identified=all(result[rule+'_'+f] for rule in owns for f in ('raw_subject','final_subject'))
                result['verified_a256_rescue']=int(identified and done[j]>.5>=max(ceiling,before[j]))
                result['verified_all_native_rescue']=int(result['verified_a256_rescue'] and beyond and j>=2)
                rows.append(result)
    pairrows=[]
    for (src,pair,arm,rule),two in half.items():
        assert set(two)=={0,1}
        a,b=two[0],two[1]
        q=dict(source=src,pair=pair,arm=arm,rule=rule)
        for k in a:q[k]=int(a[k] and b[k])
        for st in ('raw','final'):q[st+'_four']=int(q[st+'_subject'] and q[st+'_ref'])
        pairrows.append(q)
    summary=[];metrics=[]
    for (arm,rule),sub in grouped(pairrows,['arm','rule']).items():
        summary.append(dict(arm=arm,rule=rule,pairs=len(sub),**{k:sum(x[k] for x in sub) for k in
            ('raw_subject','raw_ref','raw_four','final_subject','final_ref','final_four','raw_cover','final_cover')}))
    for (arm,),sub in grouped(rows,['arm']).items():
        scores=np.array([[r['raw_iou'],r['final_iou']] for r in sub])
        metrics.append(dict(arm=arm,n=len(sub),raw_acc25=100*np.mean(scores[:,0]>.25),raw_acc50=100*np.mean(scores[:,0]>.5),
                      final_acc25=100*np.mean(scores[:,1]>.25),final_acc50=100*np.mean(scores[:,1]>.5),
                      geometry_rescue=int(np.sum((scores[:,1]>.5)&(scores[:,0]<=.5))),
                      geometry_harm=int(np.sum((scores[:,0]>.5)&(scores[:,1]<=.5))),
                      verified_a256_rescue=sum(r['verified_a256_rescue'] for r in sub),
                      verified_all_native_rescue=sum(r['verified_all_native_rescue'] for r in sub)))
    csv_file(OUT/'noun_pairs.csv',summary);csv_file(OUT/'noun_metrics.csv',metrics)
    write_gz(OUT/'noun_task_rows.csv.gz',rows);write_gz(OUT/'noun_endpoint_rows.csv.gz',pairrows)
    save(OUT/'noun_audit.json',dict(n=len(inputs()),pair_directions=len(sig),
        frames=len({r['frame'] for r in rows}),recordings=len({r['recording'] for r in rows}),
        same_half_physical_frames=len({r['frame'] for r in rows if r['same_half']}),max_error=maxerr,
        predictions=digest(EXTRA/'SEALED.json'),FINAL_access=False))
if __name__=='__main__':score()
