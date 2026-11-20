"""Post-seal attribution of the incremental effect of shared split probabilities."""
import numpy as np
import pandas as pd
from score_scene import owners
from io_scene import OUT,PRIVATE,read,write,digest
from belief_scene import ARMS
def run():
    seal=read(OUT/'prediction_seal.json');panel=read(OUT/'panel_seal.json')
    assert digest(PRIVATE/'answer_key.json')==panel['truth']
    qs={q['uid']:q for q in read(PRIVATE/'questions.json')};gt=read(PRIVATE/'answer_key.json')
    jm=ARMS.index('tree_joint'); im=ARMS.index('masked_product')
    details=[]
    for rel,h in seal['files'].items():
        src,uid=rel.split('/'); path=PRIVATE/'predictions'/src/(uid+'.npz'); assert digest(path)==h
        q=qs[uid]; e=gt[uid]
        with np.load(path) as d:
            p=d['p'][jm];b=d['p'][im]
            tv=.5*abs(p-b).sum()
            margin=abs(d['answers'][im,:2]-.5)
            lower=margin>tv+1.e-12
            diff=d['answers'][jm,:2]-d['answers'][im,:2]
            assert (abs(diff)<=tv+1.e-12).all()
            truths=np.array(e['boxes'])
            s=owners(d['sb'],truths)==e['s'];r=owners(d['rb'],truths)==e['r'];good=s[:,None]&r
            details.append(dict(frame=uid,recording=q['recording'],direction=src+'_to_'+q['frame']['target'],tv=tv,
                guaranteed=int(lower.sum()),covered=int(good.any()),compatible_covered=int((good&(p>0)).any()),
                highest_correct=int(bool(p.size and p.sum()>0 and good.flat[p.argmax()])),
                proxy_visibility=float(np.median(d['visible_count']/d['size'])) if len(s) else 0.,
                max_answer_change=abs(diff).max(),right_change=diff[0],forward_change=diff[1]))
    df=pd.DataFrame(details)
    df.to_csv(OUT/'shared_split_rows.csv.gz',index=False,compression={'method':'gzip','mtime':0},float_format='%.16g')
    frame=pd.read_csv(OUT/'static_all_rows.csv.gz')
    idx=['frame','direction','task','truth','cohort']
    a=frame.query('arm=="tree_joint"').set_index(idx);b=frame.query('arm=="masked_product"').set_index(idx)
    j=a.join(b,lsuffix='_j',rsuffix='_m')
    changed=j.query('hit_j!=hit_m').reset_index()
    changed[['frame','direction','task','truth','cohort','prob_j','prob_m','hit_j','hit_m']].to_csv(OUT/'changed_answers.csv',index=False)
    res=dict(postseal=True,source_frames=len(df),tv_median=float(df.tv.median()),tv_p95=float(df.tv.quantile(.95)),
             tv_max=float(df.tv.max()),structurally_invariant=int(df.guaranteed.sum()),questions=2*len(df),
             unchanged=len(j)-len(changed),separate_cover=int(df.covered.sum()),compatible_cover=int(df.compatible_covered.sum()),
             peak_correct=int(df.highest_correct.sum()),visibility_median_strictly_below_one=int((df.proxy_visibility<1).sum()),
             rows_sha256=digest(OUT/'shared_split_rows.csv.gz'))
    write(OUT/'shared_split_diagnostic.json',res)
    print(res)
if __name__=='__main__':run()
