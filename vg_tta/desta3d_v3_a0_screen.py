"""A0 cached direction screen. No model, labels, decoder or data loader here."""
import hashlib
import torch
from torch.nn import functional as F

def rank(tag, value):
    return hashlib.sha256(('DESTA-A0-v1|'+tag+'|'+str(value)).encode()).hexdigest()

def select_rows(train, dev):
    groups={p:[r for r in train if r['source']==p] for p in sorted({r['source'] for r in train})}
    first=[min(rows,key=lambda r:rank('train-query',r['key'])) for rows in groups.values()]
    keys={r['key'] for r in first}
    rest=sorted([r for r in train if r['key'] not in keys],key=lambda r:rank('train-query',r['key']))
    chosen=sorted(first+rest[:128-len(first)],key=lambda r:rank('train-order',r['key']))
    parents=sorted([p for p in {r['source'] for r in dev} if sum(r['source']==p for r in dev)>=4],key=lambda p:rank('dev-parent',p))[:16]
    picked=[]
    for p in parents:
        picked.extend(sorted([r for r in dev if r['source']==p],key=lambda r:rank('dev-query',r['key']))[:4])
    return chosen,picked

def coefficients(model, cache):
    """Exact coefficient path of StateAwareDirectionMixer, avoids unused 2560D projection."""
    z,qT,qS,ev,st=[cache[k].detach() for k in ('z','qT','qS','evidence8','state33')]
    shape=z.shape[:-1]
    if st.shape!=(shape[0],shape[1],33) or ev.shape!=(*shape,8):raise ValueError('A0 cache support')
    qT=qT[:,None,None,None].expand(*shape,-1);qS=qS[:,None,None,None].expand(*shape,-1)
    st=st[:,:,None,None].expand(*shape,-1)
    h=F.silu(model.input(torch.cat((z,qT,qS,ev,st),-1)))
    h=h+F.silu(model.local(h.movedim(-1,1)).movedim(1,-1))
    return model.output(F.silu(model.mix(h)))

def fixed_field(model, coeff, stock_norm):
    field=F.linear(coeff,model.basis.detach())
    norm=field.flatten().norm()
    return field*(model.radius*stock_norm/norm.clamp_min(1e-12))

def terminal_decision(train_median,dev_median):
    if dev_median is None:return 'undefined_direction_stop'
    if dev_median>=.1:return 'native_dev64'
    if train_median is None or train_median<.3:return 'capacity_or_optimization'
    return 'conditioning_or_generalization'
