"""After-seal ceiling: supply annotated boxes; withhold their class names from the tested readouts."""
import numpy as np
import pandas as pd
from scipy.special import softmax
from vote_paths import *
from physical_evidence import Returns,lexicon
from panel_witness import inventory,projection
from role_cut import separate,medoid
from delivery import seed_words,distribute,counts,unit,update

def build_diagnostic(info,src,fid,truth,classes):
    with np.load(info['lex']) as l,np.load(info['word']) as z,np.load(WORK/'votes'/src/(fid+'.npz')) as captured:
        seed,centered,_=lexicon(l['xyz'],l['logits']);centered-=centered.mean(0)
        cloud=Returns(l['sample']);nearest=seed.tree.query(cloud.xyz)[1]
        seed_ids=np.unique(z['xyz'],axis=0,return_inverse=True)[1];dim=np.maximum(z['extent'],1e-6)
        mean=np.zeros((len(seed.xyz),CATS,3));np.add.at(mean,seed_ids,dim);mean/=np.bincount(seed_ids)[:,None,None]
        all_hits=[cloud.local(b)[0] for b in truth];multiplicity=np.zeros(len(cloud.xyz),int)
        for ids in all_hits:multiplicity[ids]+=1
        assignment=np.full(len(cloud.xyz),-1,int)
        lexical=np.full((len(truth),CATS),1./CATS);shapes=lexical.copy();supported=np.zeros(len(truth),int)
        for i,(b,ids) in enumerate(zip(truth,all_hits)):
            # A return claimed by multiple annotated boxes remains unassigned.
            ids=ids[multiplicity[ids]==1];assignment[ids]=i;supported[i]=len(ids)
            if not len(ids):continue
            lexical[i]=softmax(centered[nearest[ids]].mean(0)[:-1])
            proxy=np.median(mean[np.unique(nearest[ids])],axis=0)
            proxy[:,:2]=np.sort(proxy[:,:2],axis=1)[:,::-1]
            obs=b[3:6].copy();obs[:2]=np.sort(obs[:2])[::-1]
            shapes[i]=np.prod(np.minimum(proxy,obs)/np.maximum(proxy,obs),axis=1)
        p=unit(lexical*shapes)
        gb=np.concatenate([captured['center'],dim],axis=-1)
        q,w=seed_words(l['logits'],l['xyz'])
        g,_=distribute(dict(xyz=cloud.xyz,assign=assignment[None]),gb,np.ones(len(truth)))
        observed,uncoupled=counts(g,q,w)
    oracle=np.array([[int(c==k) for k in CLASSES[:-1]] for c in classes],float)
    return dict(uniform=np.full_like(p,1/CATS),lexical=lexical,size=unit(shapes),combined=p,
          association=update(p,observed,uncoupled),direct_votes=unit(p+observed),direct_null=unit(p+uncoupled),
          true_reference_id=update(p,observed,uncoupled),true_classes=oracle),supported

def run():
    catalog_=catalog();sealed=read(WORK/'SEALED.json');entry=read(EXTRA/'labels.json')
    assert not sealed['labels_used'] and digest(EXTRA/'labels.json')==read(old.OUT/'noun_panel.json')['labels']
    store={};rows=[]
    for r in catalog_['rows']:
        if r['scope']!='noun':continue
        e=entry[r['uid']];src=r['source'];f=r['frame'];info=catalog_['frames'][f]
        with np.load(r['prior']) as prev:iv=int(prev['view'])
        if f not in store:
            b=np.array(e['truth'])
            cls,official,conflict=inventory(dict(platform=r['target'],metapath=info['f']['calibration'],suffix=iv))
            np.testing.assert_equal(b,official);assert not conflict
            pp,counts_=build_diagnostic(info,src,f,b,cls)
            proj=projection(b,r['target'],iv,info['f']['calibration'],info['f']['rgb'])
            view=proj[:,[2,0,3,1,7]].copy();view[proj[:,7]<=0]=np.nan
            store[f]=(pp,view,counts_)
        fields,view,count=store[f]
        with np.load(r['role']) as z:side,dep,near=int(z['side']),int(z['depth']),bool(z['immediate'])
        n=len(view);identity=np.eye(n);valid=~identity.astype(bool);ones=np.ones(n)
        for arm,p in fields.items():
            joint,_=separate(valid,ones,p,r['words'],view,side,dep,near)
            if arm=='true_reference_id':joint*=np.arange(n)==e['reference']
            s,ref,_=medoid(joint,identity,ones,ones,valid)
            rows.append(dict(arm=arm,source=src,frame=e['frame'],recording=e['recording'],pair=e['pair'],
                half=int(e['half']),subject=int(s==e['subject']),reference=int(ref==e['reference']),
                both=int(s==e['subject'] and ref==e['reference']),subject_has_returns=int(count[e['subject']]>0),
                reference_has_returns=int(count[e['reference']]>0)))
    x=pd.DataFrame(rows);x.to_csv(OUT/'inventory_diagnostic_rows.csv.gz',index=False)
    g=x.groupby(['arm','source','pair']);assert g.size().eq(2).all()
    halves=g[['subject','reference','both','subject_has_returns','reference_has_returns']].min().reset_index()
    a=halves.groupby('arm');out=a.sum(numeric_only=True);out['pairs']=a.size();out.to_csv(OUT/'inventory_diagnostic.csv')
    assert out.loc['true_classes','both']==64
if __name__=='__main__':run()

