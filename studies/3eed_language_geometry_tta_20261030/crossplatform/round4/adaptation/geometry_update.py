"""Shape-prior preserving visibleface interval; centerlatent only, fixedsize/yaw."""
import numpy as np,torch
from scipy.spatial import cKDTree
from torch import nn

def rotation(yaw):
    c,s=np.cos(float(yaw)),np.sin(float(yaw));return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float64)

def interval_teacher(points,box7,camera_origin):
    p=np.unique(np.asarray(points,dtype=np.float64),axis=0);source=np.asarray(box7,dtype=np.float64);R=rotation(source[6]);dims=source[3:6];local=(p-source[:3])@R;mask=np.all(abs(local)<=1.25*dims/2+1e-7,axis=1);crop=p[mask];loc=local[mask];lower=np.zeros(3);upper=np.zeros(3);observed=np.zeros(3,dtype=bool);diag={'crop_count':len(crop),'normals_from_fullcloud_graph':True,'axes':[]}
    if np.any(dims<=0) or len(crop)<8 or len(p)<16:return lower,upper,observed,dict(diag,reason='invalid_size_or_insufficient_observation')
    ids=cKDTree(p).query(crop,k=16)[1];neighbors=p[ids];neighbors-=neighbors.mean(1,keepdims=True);cov=np.einsum('nki,nkj->nij',neighbors,neighbors)/15;eig,vec=np.linalg.eigh(cov);normal=vec[:,:,0]@R;planarity=(eig[:,1]-eig[:,0])/np.maximum(eig[:,2],1e-12);view=(np.asarray(camera_origin)-source[:3])@R
    for j in range(3):
        outside=bool(abs(view[j])>dims[j]/2);sign=float(np.sign(view[j])) if outside else 0.;tangent=[k for k in range(3) if k!=j];take=(abs(normal[:,j])>=.9)&(planarity>=.3)&(sign*loc[:,j]>0) if sign else np.zeros(len(loc),bool);count=int(take.sum());record={'axis':j,'camera_near_face_sign':sign,'camera_outside_axis_slab':outside,'support':count,'accepted':False}
        if count>=8:
            xyz=loc[take];b=float(np.median(xyz[:,j]));mad=float(np.median(abs(xyz[:,j]-b)));coverage=np.ptp(xyz[:,tangent],axis=0)/dims[tangent];record.update(plane_offset=b,MAD_m=mad,tangent_coverage=coverage.tolist())
            if np.all(coverage>=.2) and mad<=.05:
                center=b-sign*dims[j]/2;uncertainty=max(.03,1.4826*mad);lower[j]=center-uncertainty;upper[j]=center+uncertainty;observed[j]=True;record.update(accepted=True,center_interval_offset_m=[lower[j],upper[j]])
        diag['axes'].append(record)
    diag.update(reason='measured_singleaxis_visibleface_constraints',observable_axes=int(observed.sum()),violated_observable_axes=int(((lower>0)|(upper<0))[observed].sum()),visibility_is_camera_nearface_heuristic_not_true_instance_or_occlusion=True)
    return lower,upper,observed,diag

def direct_center(lower,upper,observed):
    boundary=np.clip(np.zeros(3),lower,upper)*observed;delta=boundary/1.25;norm=np.linalg.norm(delta)
    if norm>.15:delta*=.15/norm
    return delta

def box_from_delta(source,delta):
    b=np.asarray(source).copy();b[:3]=(np.asarray(source[:3],dtype=np.float64)+np.asarray(delta)@rotation(source[6]).T).astype(b.dtype);assert np.array_equal(b[3:7],source[3:7]);return b

class CenterLatent(nn.Module):
    def __init__(self,observed):
        super().__init__();self.offset=nn.Parameter(torch.zeros(3,dtype=torch.float64));self.register_buffer('observed',torch.as_tensor(observed,dtype=torch.bool))
    def forward(self):return torch.where(self.observed,self.offset,torch.zeros_like(self.offset))
    @torch.no_grad()
    def project(self):
        self.offset[~self.observed]=0;norm=self.offset.norm()
        if norm>.15:self.offset.mul_(.15/norm)

def run_center(source,lower,upper,observed,control=False):
    observed=np.zeros(3,dtype=bool) if control else np.asarray(observed,bool);model=CenterLatent(observed);opt=torch.optim.Adam(model.parameters(),lr=.05);lo=torch.as_tensor(lower);hi=torch.as_tensor(upper);mask=torch.as_tensor(observed);outputs={};trace=[]
    def forwardloss():
        delta=model();distance=torch.maximum(torch.maximum(lo-delta,delta-hi),torch.zeros(3,dtype=torch.float64))*mask;constraint=distance.square().sum();anchor=.25*delta.square().sum();return constraint+anchor,constraint,anchor
    for step in range(4):
        loss,constraint,anchor=forwardloss();delta=model()
        if step in [0,1,3]:outputs[step]=box_from_delta(source,delta.detach().numpy())
        if step==3:break
        before=model.offset.detach().clone();beforeloss=float(loss.detach());opt.zero_grad(set_to_none=True);loss.backward();grad=float(model.offset.grad.norm());opt.step();model.project();proposal=model.offset.detach()-before;accepted=False;alpha=1.
        for halving in range(11):
            with torch.no_grad():
                model.offset.copy_(before+alpha*proposal);model.project();after=forwardloss()[0]
            if float(after)<=beforeloss+1e-12:accepted=True;break
            alpha*=.5
        if not accepted:
            with torch.no_grad():model.offset.copy_(before)
            alpha=0.
        with torch.no_grad():after=forwardloss()[0]
        trace.append({'step':step+1,'constraint_loss_before':float(constraint.detach()),'anchor_loss_before':float(anchor.detach()),'total_loss_before':beforeloss,'total_loss_after':float(after),'gradient_norm':grad,'latent_delta_L2_m':float(model().detach().norm()),'center_offset_local_m':model().detach().numpy().tolist(),'proposal_scale':alpha,'backtrack_halvings':halving,'nonincreasing_step_accepted':accepted})
    assert np.array_equal(outputs[0],source)
    if control or not observed.any() or np.all(((lower<=0)&(upper>=0))|~observed):assert all(np.array_equal(b,source) for b in outputs.values())
    with torch.no_grad():model.offset.zero_();model.offset.grad=None
    assert np.array_equal(box_from_delta(source,model().detach().numpy()),source)
    return outputs,{'trace':trace,'parameter_count':3,'latent_not_neuralfeatureTTA':True,'sizes_PCAyaw_exact_frozen':True,'unobserved_axes_exact_zero':True,'episodic_reset_source_exact':True,'control':control,'backtracking':'label-free lossnonincrease, max10halvings, originalAdam.05proposal retained'}
