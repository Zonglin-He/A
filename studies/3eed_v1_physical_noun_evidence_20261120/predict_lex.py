"""Sealed subject, independent and misplaced-role decisions, and single-axis shape controls."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from common_lex import *
from physical_evidence import lexicon, box_classes, Returns, choose, complete
from ordered_choice import prepare,relations


def wrong(scores):
    # Seed rows are sorted by physical XYZ. A half-cycle is a derangement for N>1.
    return np.roll(scores,len(scores)//2,axis=0)


def predict(z,prior,h,categories):
    raw=np.tile(z['raw'],(len(ARMS),1));refs=np.tile(z['ref'],(len(ARMS),1));out=raw.copy()
    applied=np.zeros(len(ARMS),bool);extended=np.zeros((len(ARMS),2))
    witnesses=np.zeros_like(extended,dtype=int)
    target_word,ref_word=categories
    seeds,balanced,single=lexicon(h['xyz'],h['logits'])
    shuffled=wrong(balanced)
    raw[1],refs[1],out[1]=prior['raw'][5],prior['ref'][5],prior['boxes'][5]
    applied[1]=prior['applied'][5]
    diagnostic={}
    if z['active']:
        T,pt,mt,tp=prepare(z,'t');R,pr,mr,rp=prepare(z,'r')
        dist=relations(tp,rp,int(z['side']))
        for boxes,prefix in ((T,'t'),(R,'r')):
            main,ids=box_classes(boxes,seeds,balanced)
            diagnostic[prefix+'_main']=main
            diagnostic[prefix+'_one']=box_classes(boxes,seeds,single,ids)[0]
            diagnostic[prefix+'_wrong']=box_classes(boxes,seeds,shuffled,ids)[0]
            diagnostic[prefix+'_count']=np.array([len(x) for x in ids])
        for k,arm in enumerate(ARMS[2:8],2):
            tk='one' if arm=='single_order' else ('wrong' if arm=='wrong_placement' else 'main')
            rk='wrong' if arm=='wrong_reference' else tk
            eligible_t=(diagnostic['t_'+tk]==target_word)&(target_word>=0)
            eligible_r=(diagnostic['r_'+rk]==ref_word)&(ref_word>=0)
            if arm=='no_noun':
                eligible_t=np.ones(len(T),bool);eligible_r=np.ones(len(R),bool)
            result=choose(T,R,tp,rp,mt,mr,dist,eligible_t,eligible_r,int(z['depth']),bool(z['immediate']),
                          z['raw'],independent=(arm=='independent'))
            if result is not None:
                raw[k],refs[k]=result
                out[k]=raw[k];applied[k]=True
    else:
        for prefix in ('t','r'):
            for suffix in ('main','one','wrong','count'):
                diagnostic[prefix+'_'+suffix]=np.empty(0,int)
    idx=ARMS.index('joint_raw')
    distinct,ii=np.unique(z['a'],axis=0,return_index=True)
    # Use summed candidate weights when exact proposal copies exist.
    _,inverse=np.unique(z['a'],axis=0,return_inverse=True)
    masses=np.bincount(inverse,weights=z['ma'],minlength=len(distinct))
    candidates=distinct[np.argsort(-masses,kind='stable')]
    points=Returns(h['sample'])
    points.nearest_seed=seeds.tree.query(points.xyz)[1]
    owners=balanced.argmax(1)[points.nearest_seed]
    if target_word>=0:
        geom={}
        for mode,q in [('physical',balanced),('blind',balanced),('wrong',shuffled)]:
            geom[mode]=complete(raw[idx],candidates,target_word,seeds,q,points,owners,donor_mode=mode)
    for k in range(idx+1,len(ARMS)):
        raw[k]=raw[idx];refs[k]=refs[idx];out[k]=raw[idx];applied[k]=applied[idx]
        if target_word<0:continue
        mode={'blind_donors':'blind','wrong_donors':'wrong'}.get(ARMS[k],'physical')
        moved,mid,sides,delta,used=geom[mode]
        out[k]=mid if ARMS[k]=='symmetric' else moved
        extended[k]=delta;witnesses[k]=used
    assert np.isfinite(out).all()
    return dict(raw=raw,ref=refs,boxes=out,applied=applied,extended=extended,witnesses=witnesses,**diagnostic)


def job(task):
    n,fid,ref0,ref1,h_sha,cats=task
    old=OLD/'observed'/n;old_p=OLD/'predictions'/n;current=OBS/(fid+'.npz')
    assert digest(old)==ref0 and digest(old_p)==ref1 and digest(current)==h_sha
    with np.load(old) as z,np.load(old_p) as prior,np.load(current) as h:
        res=predict(z,prior,h,cats)
    archive_npz(PRED/n,**res)
    return n,digest(PRED/n)


def run():
    panel();s=read(OBS/'SEALED.json')
    assert s['protocol']==digest(SPEC) and s['code']==digest(HERE/'observe_lex.py')
    parents=[read(OLD/v/'SEALED.json')['files'] for v in ('observed','predictions')]
    tasks=[(n,fid,parents[0][n],parents[1][n],s['hashes'][fid],s['nouns'][n]) for n,fid in s['files'].items()]
    PRED.mkdir(exist_ok=True,mode=0o700)
    checks={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,(n,h) in enumerate(pool.map(job,tasks,chunksize=3),1):
            checks[n]=h
            if i%100==0:print(i,len(tasks),flush=True)
    assert len(checks)==844 and all(digest(PRED/n)==h for n,h in checks.items())
    save(PRED/'SEALED.json',dict(files=checks,observations=digest(OBS/'SEALED.json'),protocol=digest(SPEC),
        code={n:digest(HERE/n) for n in ('predict_lex.py','physical_evidence.py')},labels_used=False,FINAL_access=False))


if __name__=='__main__':run()
