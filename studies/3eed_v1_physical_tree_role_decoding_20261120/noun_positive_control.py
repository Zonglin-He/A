"""Score-only class-identity interventions on the new category-word panel."""
from collections import defaultdict
import numpy as np
from paths_tree import *
from panel_nouns import EXTRA,PIN
from predict_nouns import inputs
from panel_witness import inventory,projection
from scene_partition import build
from physical_nodes import assemble,overlap_prior
from physical_evidence import Returns
from role_cut import structure,shared,separate,medoid
from endpoint_score import arrays_to_owners,write_gz
from tables_scene import grouped
from score_roles import csv_file

def evaluate():
    rows=inputs();seal=read(EXTRA/'SEALED.json')
    assert seal['n']==len(rows) and not seal['labels_used']
    assert digest(EXTRA/'labels.json')==read(OUT/'noun_panel.json')['labels']
    entries=read(EXTRA/'labels.json');cert={s:read(EXTRA/s/'SEALED.json')['files'] for s in SOURCES}
    store={};halves=defaultdict(dict);records=[]
    for row in rows:
        src=row['source'];n=name(row);f=row['frame'];e=entries[row['uid']]
        folder=EXTRA/src;pin=cert[src][n];fid=pin['fid']
        p=folder/n;rp=folder/(n[:-4]+'_native.npz')
        assert digest(p)==pin['pred'] and digest(rp)==pin['native']
        with np.load(p) as result,np.load(rp) as z:
            if (src,fid) not in store:
                lp=folder/(fid+'_lex.npz');wp=folder/(fid+'_word.npz')
                assert digest(lp)==pin['lex'] and digest(wp)==pin['word']
                with np.load(lp) as l,np.load(wp) as w:
                    data=assemble(build(l['sample'],l['logits'],l['xyz'],f),w)
                cls,b,conflict=inventory(dict(platform=f['target'],metapath=f['calibration'],suffix=int(result['view'])))
                np.testing.assert_equal(b,np.array(e['truth']));assert not conflict
                owner=arrays_to_owners([data['boxes']],b,None)['unique_iou'][0]
                q=np.zeros_like(data['p'])
                for v,i in enumerate(owner):
                    if i>=0 and cls[i] in CLASSES[:-1]:q[v,CLASSES.index(cls[i])]=1.
                _,K=structure(data['parent'],data['stop'],data['size'])
                proj=projection(b,f['target'],int(result['view']),f['calibration'],f['rgb'])
                gtview=proj[:,[2,0,3,1,7]].copy();gtview[proj[:,7]<=0]=np.nan
                scene=np.zeros((len(b),len(CLASSES)-1))
                for i,c in enumerate(cls):
                    if c in CLASSES[:-1]:scene[i,CLASSES.index(c)]=1.
                store[src,fid]=data,q,owner,K,Returns(data['xyz']),cls,gtview,scene
            data,q,owners,K,points,cls,gtview,scene=store[src,fid]
            from observe_witness import interpret
            a,b,side,depth,near=interpret(f['text'],{});words=[CLASSES.index(a),CLASSES.index(b)]
            s,r=e['subject'],e['reference']
            assert cls[s]==a and cls[r]==b
            us=overlap_prior(data,z,'t',points);ur=overlap_prior(data,z,'r',points)
            for arm in ('actual','true_landmark','true_classes','true_classes_and_landmark','known_inventory_and_classes'):
                if arm=='known_inventory_and_classes':
                    allowed=~np.eye(len(scene),dtype=bool)
                    weights=separate(allowed,np.ones(len(scene)),scene,words,gtview,side,depth,near)[0]
                    i,j,_=medoid(weights,np.eye(len(scene)),np.ones(len(scene)),np.ones(len(scene)),allowed)
                    so,ro=i,j
                else:
                    prob=q if arm in ('true_classes','true_classes_and_landmark') else data['p']
                    mass=shared(data['parent'],K==0,data['stop'],prob,words,data['view'],side,depth,near,'physical')[0]
                    if arm in ('true_landmark','true_classes_and_landmark'):mass*=owners[None,:]==r
                    i,j,_=medoid(mass,K,us,ur,K==0)
                    so,ro=owners[i] if i>=0 else -1,owners[j] if j>=0 else -1
                halves[src,e['pair'],arm][e['half']]=(so==s,ro==r)
                records.append(dict(source=src,pair=e['pair'],arm=arm,half=e['half'],subject=int(so==s),reference=int(ro==r)))
    pairrows=[]
    for (src,pair,arm),v in halves.items():
        x,y=v[0],v[1]
        pairrows.append(dict(source=src,pair=pair,arm=arm,two_subjects=int(x[0] and y[0]),two_refs=int(x[1] and y[1]),
                        four=int(all(x) and all(y))))
    result=[]
    for (arm,),v in grouped(pairrows,['arm']).items():
        result.append(dict(arm=arm,pairs=len(v),**{k:sum(y[k] for y in v) for k in ('two_subjects','two_refs','four')}))
    csv_file(OUT/'noun_offline.csv',result);write_gz(OUT/'noun_offline_rows.csv.gz',records)
if __name__=='__main__':evaluate()
