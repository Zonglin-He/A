"""Offline availability of verified endpoints and physical support for the registered geometry."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax

from observe import RESULTS, WORK, read
from predict_profiles import DECISIONS
from profiles import ARMS
from roles_joint import affinity
from score_profiles import COHORTS, PANELS, SOURCES, anon, historical_endpoints, historical_records, inventory, many_iou, sealed_inputs


def associations(boxes, inventory):
    boxes=np.asarray(boxes).reshape(-1,7)
    hits=np.stack([many_iou(boxes,b)>.25 for b in inventory],axis=1)
    return np.where(hits.sum(1)==1,hits.argmax(1),-1)


def audit(archive):
    seal=sealed_inputs()
    history=historical_records(archive)
    covers,neighbours=[],[]
    for panel,path in PANELS.items():
        for case,item in enumerate(read(path)):
            row=item['row']
            target=row['platform']
            physical,_=inventory(row['metapath'],target,int(row['suffix'] or 0),int(row['object_index']))
            ends=(0,int(item['ref_id'])) if panel=='source_proxy' else historical_endpoints(COHORTS[panel],item,physical,history)[0]
            for src in SOURCES:
                if panel!='source_proxy' and src==target:
                    continue
                tag=f'{panel}_{case:03d}_{src}.npz'
                meta=dict(panel=panel,group=panel if panel!='source_proxy' else ('own_proxy' if src==target else 'cross_proxy'),
                          case=anon(f'{panel}|{case}'),source=src,target=target)
                with np.load(WORK/tag,allow_pickle=False) as z,np.load(DECISIONS/tag,allow_pickle=False) as d:
                    active=bool(z['active'])
                    if None not in ends:
                        subject,ref=z['subject'],z['reference']
                        pca=z['pool']
                        def role_seen(ps,pr):
                            s=associations(ps,physical)
                            rr=s if pr is ps else associations(pr,physical)
                            return int(ends[0] in s),int((ends[1] in rr) and bool(z['original_ref_valid']))
                        for category,ps,pr in [('original_roles',subject[0],ref[0]),('coherent_roles',subject[:2],ref[:2]),
                                               ('original_pca_pool',pca[0],None),('coherent_pca_pools',pca[:2],None),
                                               ('all_pca_pools',pca,None)]:
                            # The full scene inventory may contain a reference even if parsing disabled its use.
                            if pr is None:
                                a=associations(ps,physical)
                                s,r=(int(e in a) for e in ends)
                            else:
                                s,r=role_seen(ps,pr)
                            covers.append(dict(**meta,active=int(active),pool=category,has_subject=s,has_reference=r,
                                               both=s*r,only_subject=s*(1-r),only_reference=(1-s)*r,neither=(1-r)*(1-s)))
                    if not active:
                        continue
                    i,j=d['choice'][-1]
                    subj=z['subject'];ref=z['reference']
                    anchor_s=subj[:2].reshape(8,7)[i:i+1];anchor_r=ref[:2].reshape(8,7)[j:j+1]
                    p=softmax(z['joint'].reshape(4,16),axis=1).reshape(4,4,4)
                    mi=np.sum(p*np.log(p/(p.sum(2,keepdims=True)*p.sum(1,keepdims=True))),axis=(1,2))
                    supports=[]
                    for s,r,prob in zip(subj,ref,p):
                        ks=affinity(anchor_s,s)[0]
                        kr=affinity(anchor_r,r)[0]
                        supports.append(ks*(prob@kr))
                    weight=np.stack(supports)
                    different=np.array([np.any(abs(s-anchor_s[0])>1e-7,axis=1) for s in subj])
                    alternate=(weight>0)&different
                    den=weight[:2].sum()
                    assert den>0
                    factor=ARMS.index('independent_roles')
                    neighbours.append(dict(**meta,alt_coherent=int(np.any(alternate[:2])),
                      alt_any=int(np.any(alternate)),alternative_positive_fraction=float(weight[:2][different[:2]].sum()/den),
                      native_mutual_information=float(np.mean(mi)),same_factorized_subject=int(np.array_equal(d['raw'][-1],d['raw'][factor])),
                      same_factorized_reference=int(np.array_equal(d['refs'][-1],d['refs'][factor])),
                      same_factorized_box=int(np.array_equal(d['boxes'][-1],d['boxes'][factor])),
                      modified=int(np.any(abs(d['raw'][-1]-d['boxes'][-1])>1e-8))))
    cov=pd.DataFrame(covers)
    diag=pd.DataFrame(neighbours)
    assert len(diag)==seal['active'] and len(cov)==(333+42)*5
    cov.to_csv(RESULTS/'coverage_rows.csv.gz',index=False)
    diag.to_csv(RESULTS/'support_rows.csv.gz',index=False)
    fields=['both','only_subject','only_reference','neither','has_subject','has_reference']
    ds=[]
    for group in ('natural_all','natural170','natural171','own_proxy','cross_proxy'):
        c=cov[cov.group.str.startswith('natural')] if group=='natural_all' else cov[cov.group==group]
        for (pool,),part in c.groupby(['pool']):
            for label,sub in [('all',part),('active',part[part.active==1])]:
                ds.append(dict(group=group,pool=pool,stage=label,n=len(sub),**{f:int(sub[f].sum()) for f in fields}))
    pd.DataFrame(ds).to_csv(RESULTS/'coverage.csv',index=False)
    print('Validated coverage',len(cov),'candidate-pool evaluations and',len(diag),'active relation cases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path)
    audit(p.parse_args().archive)

