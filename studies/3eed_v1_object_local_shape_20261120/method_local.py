"""Frozen joint instance selection and observed-face geometry with local shape interventions."""
import numpy as np
from scipy.special import softmax
from local_paths import *
from physical_nodes import complete
from role_cut import shared,medoid
ARMS=ARMS+(read(HERE/'ROLE_CONTROL.json')['arm'],)
GEOMS=('raw','anchored','fit','direct_fit')

def tilt(p,logits):
    if not len(p):return p.copy()
    with np.errstate(divide='ignore'):
        old=np.log(p)
    alive=p.sum(1)>0
    post=p.copy()
    post[alive]=softmax(old[alive]+logits[alive],axis=1)
    return post

def evidence(p,ids,measured,logits):
    # Every arm starts from identical full-scene probabilities, including all unexamined objects.
    z={a:p.copy() for a in ('tree','local_direct','scrambled_direct','inverse','shape_difference','misbound_crop')}
    if len(ids)==0:return z
    actual,shuffled=logits[:,0,:-1],logits[:,1,:-1]
    delta=actual-shuffled
    ix=np.lexsort((ids,measured[ids,2],measured[ids,1],measured[ids,0]))
    wrong=delta.copy()
    if len(ids)>1:wrong[ix]=delta[np.roll(ix,max(1,len(ix)//2))]
    for arm,s in [('local_direct',actual),('scrambled_direct',shuffled),('inverse',-delta),
                  ('shape_difference',delta),('misbound_crop',wrong)]:
        z[arm][ids]=tilt(p[ids],s)
    return z

def fit_on_node(measured,base,real,scrambled):
    if not np.isfinite(real).all():return base,base
    direct=complete(measured,real)
    if not np.isfinite(scrambled).all() or (scrambled<=0).any() or np.array_equal(real,scrambled):
        return base,direct
    old=base[3:6].copy();old[:2]=np.sort(old[:2])[::-1]
    predicted=old*(real/scrambled)
    assert np.isfinite(predicted).all() and (predicted>0).all()
    return complete(measured,predicted),direct

def decide(plan,fields,words,us,ur,side,depth,near,m,K):
    ok=K==0;view=plan['view'];vis=np.isfinite(view[:,0]);choices={}
    for arm in ('tree','local_direct','scrambled_direct','inverse','shape_difference','misbound_reference','misbound_crop'):
        p=fields['shape_difference'] if arm=='misbound_reference' else fields[arm]
        mode='wrong_reference_binding' if arm=='misbound_reference' else 'physical'
        joint,ref=shared(plan['parent'],ok,plan['stop'],p,words,view,side,depth,near,mode)
        i,j,_=medoid(joint,K,us,ur,ok)
        choices[arm]=i,j
        if arm=='shape_difference':
            su=K@(m*p[:,words[0]]*us*vis)
            ru=K@(ur*ref)
            choices['independent']=(int(np.argmax(np.where(vis,su,-1))) if su.max(initial=0)>0 else -1,
                                    int(np.argmax(np.where(vis,ru,-1))) if ru.max(initial=0)>0 else -1)
    return choices
