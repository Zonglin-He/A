"""Diagnostic-only reader on supplied physical regions. Genuinely label-free runs are sealed first."""
import argparse
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from local_paths import *
from local_reader import LocalReader,observe_objects
from method_local import evidence
from inventory_diagnostic import build_diagnostic
from physical_evidence import Returns
from panel_witness import projection,inventory
from collect import Grounder
from role_cut import separate,medoid,layout
from observe_witness import interpret

DIAG=WORK/'oracle'
def rows():
    return [r for r in catalog()['rows'] if r['scope']=='noun']

def prepare():
    # This is the single interface which passes annotation geometry to the diagnostic observer.
    assert not read(WORK/'SEALED.json')['labels_used']
    p=previous.EXTRA/'labels.json';assert digest(p)==read(old.OUT/'noun_panel.json')['labels']
    labels=read(p);frames={};cat=catalog()
    DIAG.mkdir(exist_ok=True,mode=0o700)
    for r in rows():
        fid=r['frame'];b=np.array(labels[r['uid']]['truth'],float)
        if fid in frames:
            with np.load(DIAG/(fid+'_regions.npz')) as x:np.testing.assert_equal(x['boxes'],b)
            continue
        f=cat['frames'][fid]['f']
        with np.load(r['prior']) as stored:view=int(stored['view'])
        px=projection(b,r['target'],view,f['calibration'],f['rgb'])
        viewcoords=px[:,[2,0,3,1,7]].copy();viewcoords[px[:,7]<=0]=np.nan
        path=DIAG/(fid+'_regions.npz');archive_npz(path,boxes=b,view=viewcoords)
        frames[fid]=digest(path)
    save(DIAG/'REGIONS.json',dict(files=frames,main_seal=digest(WORK/'SEALED.json'),scope='diagnostic known boxes; object names and target identities removed'))

@torch.inference_mode()
def capture(source):
    rec=read(DIAG/'REGIONS.json');assert rec['main_seal']==digest(WORK/'SEALED.json')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    cat=catalog();fids={r['frame'] for r in cat['rows'] if r['scope']=='noun' and r['source']==source}
    reader=LocalReader(Grounder(source)._frontend.base) if fids else None
    hashes={};out=DIAG/source;out.mkdir(exist_ok=True,mode=0o700)
    for fid in sorted(fids):
        info=cat['frames'][fid];path=DIAG/(fid+'_regions.npz')
        assert digest(path)==rec['files'][fid]
        with np.load(path) as z,np.load(info['lex']) as l:
            boxes=z['boxes'];visible=np.isfinite(z['view'][:,0]);cloud=Returns(l['sample'])
            sets=[cloud.local(b)[0] for b in boxes];multi=np.zeros(len(cloud.xyz),int)
            for ids in sets:multi[ids]+=1
            supported=[ix[multi[ix]==1] for ix in sets]
            ids=np.array([i for i,v in enumerate(supported) if visible[i] and len(v)],int)
            examples=[]
            for i in ids:
                points=cloud.xyz[supported[i]];xy=points[:,:2]-points[:,:2].mean(0)
                _,axes_=np.linalg.eigh(xy.T@xy);v=axes_[:,-1]
                examples.append((points,float(np.arctan2(v[1],v[0])%np.pi)))
            found=observe_objects(reader,examples)
            original,_=build_diagnostic(info,source,fid,boxes,[None]*len(boxes))
            probs=evidence(original['combined'],ids,boxes,found['logit'])
            probs['combined']=original['combined'];probs['old_lexical']=original['lexical'];probs['uniform']=original['uniform']
            for j,name in enumerate(('real_alone','scrambled_alone')):
                p=original['uniform'].copy();p[ids]=softmax(found['logit'][:,j,:-1],axis=1);probs[name]=p
            file=out/(fid+'.npz');archive_npz(file,**probs,ids=ids,support=found['support'])
            hashes[fid]=digest(file)
    save(out/'SEALED.json',dict(regions=digest(DIAG/'REGIONS.json'),files=hashes,reader=digest(HERE/'local_reader.py'),
        code=digest(Path(__file__)),classes_read_at_observation=False,known_geometry=True))

def score():
    cat=catalog();labels=read(previous.EXTRA/'labels.json');cache={};pred=[];instances=[];assign=[]
    files={}
    for src in SOURCES:
        seal=read(DIAG/src/'SEALED.json')
        assert seal['regions']==digest(DIAG/'REGIONS.json') and seal['code']==digest(Path(__file__)) and not seal['classes_read_at_observation']
        files.update(seal['files'])
    pairs={}
    texts={(r['source'],r['uid']):r['frame']['text'] for r in previous.noun_inputs()}
    for r in rows():
        fid=r['frame'];e=labels[r['uid']];src=r['source'];f=cat['frames'][fid]['f'];key=(src,e['pair'])
        if fid not in cache:
            with np.load(r['prior']) as stored:iv=int(stored['view'])
            names,boxes,bad=inventory(dict(platform=r['target'],metapath=f['calibration'],suffix=iv))
            assert not bad;np.testing.assert_equal(boxes,np.array(e['truth']))
            path=DIAG/src/(fid+'.npz');assert digest(path)==files[fid]
            with np.load(path) as z:info=dict(z)
            with np.load(DIAG/(fid+'_regions.npz')) as z:view=z['view']
            ids=info.pop('ids');counts=info.pop('support')
            exact=np.array([[float(c==n) for n in CLASSES[:-1]] for c in names]);info['true_classes']=exact
            cache[fid]=info,names,view
            eligible=np.array([i for i in ids if names[i] in CLASSES[:-1]],int)
            for a,p in info.items():
                for i in eligible:
                    correct=CLASSES.index(names[i]);scores=p[i];leader=scores.max();tied=scores==leader
                    instances.append(dict(source=src,recording=e['recording'],frame=e['frame'],object=int(i),category=CLASSES[correct],
                        arm=a,top1=int(scores.argmax()==correct),tie_split_credit=float(tied[correct])/tied.sum(),
                        probability=float(scores[correct])))
        info,names,view=cache[fid]
        a,b,direction,depth,near=interpret(texts[src,r['uid']],{})
        words=r['words'];assert a==names[e['subject']] and b==names[e['reference']]
        n=len(view);unit=np.eye(n);ones=np.ones(n);allowed=~unit.astype(bool)
        for arm in (*info,'known_reference','independent','misbound_reference'):
            p=info.get(arm,info['shape_difference'])
            v=view
            if arm=='misbound_reference':
                # Rotate only the reference spatial coordinates as in the deployed role control.
                V,R,vis,gap=layout(v,direction,'wrong_reference_binding')
                earlier=(depth*(R[:,3,None]-R[:,3])<0)&vis[:,None]
                sub=p[:,words[0]]*vis;ref=p[:,words[1]]*vis;survival=1.-earlier*ref[:,None]
                rf=ref*survival.prod(0)
                right=np.divide(sub[:,None]*(~earlier if words[0]==words[1] else 1.),survival,
                              out=np.zeros_like(survival),where=survival>0)*allowed*(gap>0)*vis[None,:]
                joint=right*rf
                if near:
                    order=np.argsort(direction*V[:,1],kind='stable')
                    free=np.vstack([np.ones(n),np.cumprod(1.-right[order][:-1],axis=0)])
                    start=np.maximum.accumulate(np.where(np.r_[True,np.diff(V[order,1])!=0],np.arange(n),0))
                    joint[order]*=free[start]
            else:
                joint,rf=separate(allowed,ones,p,words,v,direction,depth,near)
            if arm=='known_reference':joint*=np.arange(n)==e['reference']
            s,j,_=medoid(joint,unit,ones,ones,allowed)
            if arm=='independent':
                vis=np.isfinite(v[:,0]);u=p[:,words[0]]*vis
                s=int(u.argmax()) if u.max(initial=0)>0 else -1;j=int(rf.argmax()) if rf.max(initial=0)>0 else -1
            pred.append(dict(source=src,pair=e['pair'],frame=e['frame'],recording=e['recording'],half=e['half'],arm=arm,
                      subject=int(s==e['subject']),reference=int(j==e['reference']),both=int(s==e['subject'] and j==e['reference'])))
        pairs.setdefault(key,{})[e['half']]=(e,words,info)
    for (src,pair),v in pairs.items():
        (left,wl,fields),(right,wr,_)=v[0],v[1]
        i,j=left['reference'],right['reference'];a,b=wl[1],wr[1]
        assert i!=j and a!=b
        for arm,p in fields.items():
            x,y=p[i,a]*p[j,b],p[i,b]*p[j,a]
            assign.append(dict(source=src,frame=left['frame'],recording=left['recording'],pair=pair,arm=arm,
                 correct=int(x>y+1e-12),tie=int(abs(x-y)<=1e-12),incorrect=int(y>x+1e-12),difference=float(x-y)))
    pp=pd.DataFrame(pred);ii=pd.DataFrame(instances);aa=pd.DataFrame(assign)
    pp.to_csv(OUT/'oracle_grounding_rows.csv.gz',index=False);ii.to_csv(OUT/'oracle_class_rows.csv.gz',index=False)
    aa.to_csv(OUT/'oracle_assignment_rows.csv.gz',index=False)
    g=pp.groupby(['source','pair','arm']);assert g.size().eq(2).all()
    both=g[['subject','reference','both']].min().reset_index().groupby('arm');counts=both[['subject','reference','both']].sum()
    counts['pairs']=both.size();counts.to_csv(OUT/'oracle_grounding.csv')
    assert counts.loc['combined','both']==7 and counts.loc['true_classes','both']==64
    res=ii.groupby(['arm','category']).tie_split_credit.mean().groupby('arm').mean().to_frame('class_macro')
    res['micro']=ii.groupby('arm').tie_split_credit.mean();res['objects']=ii.groupby('arm').size();res.to_csv(OUT/'oracle_classes.csv')
    vals=aa.groupby('arm')[['correct','tie','incorrect']].sum();vals.to_csv(OUT/'oracle_assignment.csv')
    print('oracle verified',len(cache),len(pairs),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','score',*SOURCES]);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='score':score()
    else:capture(a.stage)
