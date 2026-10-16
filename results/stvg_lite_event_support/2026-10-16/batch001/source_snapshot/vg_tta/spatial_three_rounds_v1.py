"""Fixed temporal weighting and bounded spatial diagnostic helpers."""
import numpy as np
import torch
from methods.decota_final_simplified_v1.objectives import generalized_iou,project,native_view_indices

def normalize(w):
    w=torch.as_tensor(w,dtype=torch.float64)
    if not torch.isfinite(w).all() or (w<0).any():raise ValueError('invalid weights')
    return w/w.sum() if float(w.sum())>0 else w

def temporal_weights(x,config):
    n=len(x['frame_ids']);a=torch.ones(n,dtype=torch.float64)
    def mask(indices):
        w=torch.zeros(n,dtype=torch.float64);s,e=indices;w[s:e+1]=1;return w
    ev=project(x['native_logits'],[a['raw_logits'] for a in x['teacher']],x['records'],config)
    aa=[torch.sigmoid(v['standardized_logits']).clamp_min(config.epsilon) for v in ev['offsets']]
    d=torch.stack([aa[i%2][i//2] for i in range(n)])
    weights=dict(A_uniform=a,B_native=mask(x['predictions']['Frozen']['indices']),
       C_adapted=mask(x['predictions']['Full_DeCoTA']['indices']),D_soft=d)
    weights={k:normalize(v) for k,v in weights.items()};weights['E_shifted']=torch.roll(weights['D_soft'],n//2)
    strength=[]
    for j,z in enumerate(ev['offsets']):
        pairs={tuple(k):i for i,k in enumerate(z['ij'].T.tolist())}
        ni=native_view_indices(x['native_logits'][j]);ai=native_view_indices(x['predictions']['Full_DeCoTA']['logits'][j])
        strength.append(float(z['score'][pairs[ai]]-z['score'][pairs[ni]]))
    return weights,dict(offset_strength=strength,strength=float(np.mean(strength)),shift_positions=n//2,
                       support='normalized locked sigmoid standardized dense actionness',epsilon=config.epsilon)

class WeightedROI:
    def __init__(self,target,weights,uniform=False):
        self.target=target.detach().clone();self.weights=normalize(weights).to(target)
        self.uniform=uniform
    def __call__(self,boxes):
        terms=5*(boxes-self.target).abs().sum(-1)+2*(1-generalized_iou(boxes,self.target))
        if self.uniform:return terms.sum()/len(terms)
        return (terms*self.weights).sum()

def temporal_state(x,eta):
    return {k:v+eta*(x['temporal']['state'][k]-v) for k,v in x['temporal']['initial_state'].items()}

def summary(values,seed=20260917):
    v=np.array([x for x in values if x is not None],dtype=float)
    if not len(v):return dict(n=0,mean=None,median=None,ci95=None,negative_rate=None,q10=None)
    rng=np.random.default_rng(seed);b=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),ci95=np.quantile(b,[.025,.975]).tolist(),
       negative_rate=float((v<0).mean()),practical_negative_rate=float((v<-.001).mean()),
       practical_positive_rate=float((v>.001).mean()),q10=float(np.quantile(v,.1)),harm_gt5pp=int((v<-.05).sum()))

def continuation(stats):
    """Predetermined exploratory P0 gate; no per-source GT selection."""
    eps=.001
    def avg(arm,metric,ref):
        return sum(stats[c]['contrasts'][f'{arm}-{ref}'][metric]['mean'] for c in stats)/len(stats)
    oracle=avg('F_GT_step5','sIoU','A_uniform_step5')
    candidates={}
    for arm,control in [('C_adapted','B_native'),('D_soft','E_shifted')]:
        ds=avg(arm+'_step5','sIoU','A_uniform_step5');dv=avg(arm+'_step5','vIoU_corrected','A_uniform_step5')
        dc=avg(arm+'_step5','sIoU',control+'_step5')
        candidates[arm]=dict(delta_s=ds,delta_v=dv,matched_control_delta_s=dc,
                            pass_gate=oracle>eps and ds>eps and dv>=-eps and dc>eps)
    eligible=[k for k,v in candidates.items() if v['pass_gate']]
    selected=max(eligible,key=lambda k:(candidates[k]['delta_s'],candidates[k]['delta_v'],k=='C_adapted')) if eligible else None
    return dict(oracle_F_minus_A_s=oracle,candidates=candidates,selected_weighting=selected,
                P1_run=bool(selected),criterion='predeclared exploratory point-effect gate, not significance or promotion')
