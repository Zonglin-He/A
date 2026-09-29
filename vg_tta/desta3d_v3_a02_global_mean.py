"""One parameter-free THW mean residual; original A0 initialization and scope."""
from torch.nn import functional as F
import torch
from vg_tta.desta3d_v3_a01_triage import FitabilityDirectionMixer
from vg_tta.desta3d_v3_gap_candidates import direction_field


def coefficients(model, cache, *, global_mean=True):
    z,qT,qS,ev,st=[cache[k].detach() for k in ('z','qT','qS','evidence8','state33')]
    shape=z.shape[:-1]
    if st.shape!=(shape[0],shape[1],33) or ev.shape!=(*shape,8):
        raise ValueError('A0.2 cache support')
    qT=qT[:,None,None,None].expand(*shape,-1)
    qS=qS[:,None,None,None].expand(*shape,-1)
    st=st[:,:,None,None].expand(*shape,-1)
    h0=F.silu(model.input(torch.cat((z,qT,qS,ev,st),-1)))
    h=h0+F.silu(model.local(h0.movedim(-1,1)).movedim(1,-1))
    if global_mean:
        h=h+h0.mean(dim=(1,2,3),keepdim=True)
    return model.output(F.silu(model.mix(h)))


class GlobalMeanDirectionMixer(FitabilityDirectionMixer):
    def forward(self,z,qT,qS,evidence,state,stock):
        a=coefficients(self,dict(z=z,qT=qT,qS=qS,evidence8=evidence,state33=state))
        return direction_field(a,self.basis,stock,self.radius),a


def route(short,long):
    for name,r in [('GMean-S200',short),('GMean-S2000',long)]:
        if r['train']['median']>=.3 and r['dev']['median']>=.1:
            return 'native_dev64:'+name
    return 'stop_architecture_tuning_cached_structure_audit'
