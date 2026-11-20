"""Greedy four-by-four retention of an ordered physical pair, followed by conditional geometry."""
import numpy as np

from collect import HERE, CONFIG, PARENT, read
from roles_joint import affinity, direction_tensor

ARMS=('v1','original_slots','independent_nomination','independent_roles','no_bearing','wrong_reference',
      'reverse_bearing','whole_pool_map','joint_nomination_raw','subject_only_shape','joint_nomination')
SLOTS=4


def bearings(left_hist,right_hist,sign):
    a,b=np.asarray(left_hist,float),np.asarray(right_hist,float)
    if a.shape[1]!=b.shape[1] or sign not in (1,-1):
        raise ValueError('Expected camera-column measurements for an oriented relation')
    sa,sb=a.sum(1),b.sum(1)
    b_survival=sb[:,None]-np.cumsum(b,axis=1)+.5*b
    total=sa[:,None]*sb[None,:]
    prob=np.divide(a@b_survival.T,total,out=np.full(total.shape,.5),where=total>0)
    return np.clip(prob if sign==1 else 1-prob,0.,1.)


def normalize_pair(a,d,separation,bearing):
    joint=a[:,None]*d[None,:]*separation*bearing
    total=joint.sum()
    return joint/total if total>0 else None


def covered(kernel):
    return 1-np.prod(1-kernel,axis=0)


def nominate_joint(ks,kr,p,count=SLOTS):
    """Exact one-round gain in expected simultaneous coverage, without an optimality claim for all rounds."""
    u=np.ones(p.shape[0]);v=np.ones(p.shape[1])
    ss=[];rr=[]
    for _ in range(count):
        a,b=1-u,1-v
        left=ks@(u*(p@b))
        right=kr@(v*(p.T@a))
        cross=(ks@((u[:,None]*p)*v[None,:]))@kr.T
        gain=left[:,None]+right[None,:]+cross
        gain[ss,:]=-np.inf;gain[:,rr]=-np.inf
        i,j=np.unravel_index(gain.argmax(),gain.shape)
        ss.append(i);rr.append(j)
        u*=1-ks[i];v*=1-kr[j]
    return np.array(ss),np.array(rr)


def nominate_single(k,p,count=SLOTS):
    unexplained=p.copy();selected=[]
    for _ in range(count):
        gain=k@unexplained
        gain[selected]=-np.inf
        i=int(np.argmax(gain));selected.append(i)
        unexplained*=1-k[i]
    return np.array(selected)


def assignment(kernel):
    retained=covered(kernel)
    total=kernel.sum(0)
    return kernel*np.divide(retained,total,out=np.zeros_like(total),where=total>0)


def estimate_shape(boxes,weights,anchor):
    total=weights.sum()
    if total<=0:
        return anchor.copy()
    w=weights/total
    if (w>0).sum()==1:
        return boxes[w.argmax()].copy()
    center=w@boxes[:,:3]
    xx,xy,yy=w@direction_tensor(boxes)
    eig,basis=np.linalg.eigh([[xx,xy],[xy,yy]])
    if eig.min()<=0:
        raise ArithmeticError('Degenerate shape covariance')
    yaw=np.arctan2(basis[1,-1],basis[0,-1]) if np.ptp(eig)>1.e-12*eig[-1] else anchor[6]
    box=np.r_[center,np.sqrt(eig[::-1]),np.exp(w@np.log(boxes[:,5])),(yaw+np.pi/2)%np.pi-np.pi/2]
    if np.allclose(box[:3],anchor[:3],atol=1.e-10,rtol=0.) and abs(box[5]-anchor[5])<1.e-10:
        if np.allclose(direction_tensor(box[None]),direction_tensor(anchor[None]),atol=1.e-10,rtol=1.e-12):
            return anchor.copy()
    return box


def decode(slate_s,slate_r,sk,rk,p,native_boxes):
    sc,rc=assignment(sk),assignment(rk)
    joint=sc@p@rc.T
    coverage=covered(sk)@p@covered(rk)
    np.testing.assert_allclose(joint.sum(),coverage,atol=2.e-13)
    i,j=np.unravel_index(joint.argmax(),joint.shape)
    weight=sc[i]*(p@rc[j])
    np.testing.assert_allclose(weight.sum(),joint[i,j],atol=2.e-13)
    anchor=slate_s[i]
    box=estimate_shape(native_boxes,weight,anchor)
    single=estimate_shape(native_boxes,sc[i]*p.sum(1),anchor)
    return anchor,slate_r[j],box,single,float(coverage),float(joint[i,j])


def wrong_mass(boxes,mass):
    order=np.lexsort(tuple(boxes[:,j] for j in range(6,-1,-1)))
    out=mass.copy()
    out[order]=mass[np.roll(order,len(order)//2)]
    return out


def predict(z):
    s,r=z['native_s'],z['native_r']
    i,j=z['before']
    ref_orig=r if bool(z['native_ref_valid']) else np.full_like(r,np.nan)
    count=len(ARMS)
    raw=np.tile(s[i],(count,1))
    boxes=raw.copy();refs=np.tile(ref_orig[j],(count,1))
    subjects=np.tile(s[None],(count,1,1))
    references=np.tile(ref_orig[None],(count,1,1))
    coverage=np.full((count,2),np.nan)
    out=dict(raw=raw,boxes=boxes,refs=refs,subjects=subjects,references=references,coverage=coverage,
             full_pool=np.zeros(count,bool),applied=False,mutual_information=0.,visible_fraction=np.zeros(2))
    if not bool(z['active']):
        return out
    a,d,ma,md=[z[k] for k in ('a','d','mass_a','mass_d')]
    ks,kr=affinity(a,a),affinity(d,d)
    # Area integration is symmetric; enforce exact diagonal and symmetry for deterministic ties.
    ks=(ks+ks.T)/2; kr=(kr+kr.T)/2
    np.fill_diagonal(ks,1.);np.fill_diagonal(kr,1.)
    sep=1-affinity(a,d)
    bearing=bearings(z['ha'],z['hd'],int(z['sign']))
    original=normalize_pair(ma,md,sep,bearing)
    if original is None:
        return out
    out['applied']=True
    out['visible_fraction']=np.array([(z[k].sum(1)>0).mean() for k in ('ha','hd')])
    ms,mr=original.sum(1),original.sum(0)
    independent=ms[:,None]*mr[None,:]
    valid=(original>0)&(independent>0)
    out['mutual_information']=float(np.sum(original[valid]*np.log(original[valid]/independent[valid])))
    si,ri=nominate_joint(ks,kr,original)
    left,right=nominate_single(ks,ms),nominate_single(kr,mr)
    for k,arm in enumerate(ARMS[1:],1):
        p=original
        ids,ids_r=si,ri
        if arm in ('independent_nomination','independent_roles'):
            ids,ids_r=left,right
            if arm=='independent_roles':
                p=independent
        elif arm in ('no_bearing','wrong_reference','reverse_bearing'):
            field=.5 if arm=='no_bearing' else (1-bearing if arm=='reverse_bearing' else bearing)
            score=wrong_mass(d,md) if arm=='wrong_reference' else md
            p=normalize_pair(ma,score,sep,field)
            if p is None:
                continue
            ids,ids_r=nominate_joint(ks,kr,p)
        if arm=='whole_pool_map':
            ii,jj=np.unravel_index(p.argmax(),p.shape)
            raw[k],refs[k]=a[ii],d[jj]
            w=ks[ii]*(p@kr[jj])
            boxes[k]=estimate_shape(a,w,a[ii])
            out['full_pool'][k]=True
            subjects[k],references[k]=np.nan,np.nan
            coverage[k]=[1.,p[ii,jj]]
            continue
        if arm=='original_slots':
            ss,rr=s,r
            sk,rk=affinity(s,a),affinity(r,d)
        else:
            ss,rr=a[ids],d[ids_r]
            sk,rk=ks[ids],kr[ids_r]
        anchor,ref,box,single,total,mass=decode(ss,rr,sk,rk,p,a)
        # Store all nominated slates' coverage under the SAME factual posterior for comparison.
        coverage[k]=[covered(sk)@original@covered(rk),mass]
        raw[k],refs[k],subjects[k],references[k]=anchor,ref,ss,rr
        boxes[k]=anchor if arm=='joint_nomination_raw' else (single if arm=='subject_only_shape' else box)
    np.testing.assert_array_equal(raw[-1],raw[ARMS.index('joint_nomination_raw')])
    np.testing.assert_array_equal(raw[-1],raw[ARMS.index('subject_only_shape')])
    return out

