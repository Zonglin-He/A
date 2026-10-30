"""GT-free partial-face latent teacher; no network/model imports."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

CONFIG={'candidates':8,'crop_min_points':16,'normal_knn':8,'outlier_probability':.2,
        'normal_coefficient':.2,'likelihood_temperature':.02,'center_prior':.5,
        'center_bound_fraction':.15,'max_iterations':20,'posterior_min':.65,
        'normal_planarity_min':.1,'vertical_normal_xy_min':.8,
        'face_support_min':3,'tangent_coverage_min':.25,'mode_merge_radians':np.pi/18,
        'mode_merge_center_fraction':.05,'accepted_fit_check_angle_radians':np.pi/12}

def wrap(theta):return (theta+np.pi/2)%np.pi-np.pi/2
def anglediff(a,b):return abs(wrap(a-b))

def observations(points):
    """Build this half's own normal graph; fit/check never share kNN graphs."""
    p=np.asarray(points,dtype=np.float64);k=min(CONFIG['normal_knn'],len(p))
    nn=cKDTree(p).query(p,k=k)[1];local=p[nn];local-=local.mean(1,keepdims=True)
    eig,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',local,local)/max(k-1,1))
    normal=vec[:,:,0];planarity=(eig[:,1]-eig[:,0])/np.maximum(eig[:,2],1e-12)
    return {'points':p,'normal':normal,'planarity':np.clip(planarity,0,1)}

def face_cost(obs,box):
    center,extent,theta=box[:3],box[3:6],box[6]
    co,si=np.cos(theta),np.sin(theta);rot=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
    local=(obs['points']-center)@rot;normals=obs['normal']@rot;scale=min(extent[:2])
    costs=[]
    for axis in range(3):
        tangent=[a for a in range(3) if a!=axis]
        outside=np.maximum(np.abs(local[:,tangent])-extent[tangent]/2,0)
        normal_error=CONFIG['normal_coefficient']*obs['planarity']*(1-normals[:,axis]**2)
        for sign in [-1,1]:
            distance=(local[:,axis]-sign*extent[axis]/2)/scale
            costs.append(distance**2+np.sum((outside/scale)**2,axis=1)+normal_error)
    return np.stack(costs,1),local,normals

def energy(obs,box):
    cost,_,_=face_cost(obs,box);temp=CONFIG['likelihood_temperature'];out=CONFIG['outlier_probability']
    density=(1-out)*np.exp(-np.minimum(cost,100)/temp).mean(1)+out*np.exp(-1.)
    # Normalize to the likelihood of one exactly matched visible face.
    reference=(1-out)/6+out*np.exp(-1.)
    return float((-temp*np.log(np.maximum(density,1e-30))+temp*np.log(reference)).mean())

def coverage(obs,box):
    costs,local,normals=face_cost(obs,box);assignment=costs.argmin(1);supported=[]
    for face in range(4): # only the two horizontal-axis face families, not roof/floor
        axis=face//2;tangent=1-axis
        mask=(assignment==face)&(obs['planarity']>=CONFIG['normal_planarity_min'])&(np.linalg.norm(normals[:,:2],axis=1)>=CONFIG['vertical_normal_xy_min'])
        count=int(mask.sum());span=float(np.ptp(local[mask,tangent])/box[3+tangent]) if count>1 else 0.
        if count>=CONFIG['face_support_min'] and span>=CONFIG['tangent_coverage_min']:supported.append(axis)
    families=sorted(set(supported));return {'horizontal_face_families':families,'two_direction_support':len(families)>=2}

def fit_modes(fit,check,source):
    cov=np.cov(fit['points'][:,:2],rowvar=False);_,v=np.linalg.eigh(cov);pca=np.arctan2(v[1,-1],v[0,-1])
    def box(x):return np.r_[source[:3]+x[:3]*source[3:6],source[3:6],wrap(x[3])]
    def objective(x):return energy(fit,box(x))+CONFIG['center_prior']*np.sum(x[:3]**2)
    modes=[]
    for theta in [0.,wrap(pca),wrap(pca+np.pi/2)]:
        x0=np.r_[np.zeros(3),theta]
        res=minimize(objective,x0,method='L-BFGS-B',bounds=[(-CONFIG['center_bound_fraction'],CONFIG['center_bound_fraction'])]*3+[(-np.pi/2,np.pi/2)],options={'maxiter':CONFIG['max_iterations'],'ftol':1e-9,'maxls':15})
        b=box(res.x);record={'box':b,'fit_energy':float(objective(res.x)),'check_energy':energy(check,b)+CONFIG['center_prior']*np.sum(res.x[:3]**2),'optimizer_success':bool(res.success),'iterations':int(res.nit)}
        # Merge converged modes without multiplying their posterior by number of starts.
        matches=[i for i,m in enumerate(modes) if anglediff(b[6],m['box'][6])<CONFIG['mode_merge_radians'] and np.max(abs((b[:3]-m['box'][:3])/source[3:6]))<CONFIG['mode_merge_center_fraction']]
        if matches:
            i=matches[0]
            if record['fit_energy']<modes[i]['fit_energy']:modes[i]=record
        else:modes.append(record)
    energies=np.array([(x['fit_energy']+x['check_energy'])/2 for x in modes]);p=np.exp(-(energies-energies.min())/CONFIG['likelihood_temperature']);p/=p.sum();j=int(energies.argmin());best=modes[j]
    fitbest=min(modes,key=lambda m:m['fit_energy']);checkbest=min(modes,key=lambda m:m['check_energy']);fitcheck_agreement=anglediff(fitbest['box'][6],checkbest['box'][6])<=CONFIG['accepted_fit_check_angle_radians']
    cov_fit=coverage(fit,best['box']);cov_check=coverage(check,best['box'])
    accept=bool(p[j]>=CONFIG['posterior_min'] and fitcheck_agreement and cov_fit['two_direction_support'] and cov_check['two_direction_support'])
    diag={'retained_modes':len(modes),'best_mode_probability':float(p[j]),'mode_entropy':float(-np.sum(p*np.log(p+1e-30))),'fit_check_mode_agreement':fitcheck_agreement,'fit_coverage':cov_fit,'check_coverage':cov_check,'accepted':accept,'modes':[{'yaw':float(x['box'][6]),'fit_energy':x['fit_energy'],'check_energy':x['check_energy'],'posterior':float(pp),'iterations':x['iterations'],'optimizer_success':x['optimizer_success']} for x,pp in zip(modes,p)]}
    return best['box'],float(energies[j]),diag

def build_teacher(points,boxes,semantic_scores,seed=0):
    points=np.unique(np.asarray(points,dtype=np.float64),axis=0);boxes=np.asarray(boxes,dtype=np.float64);s=np.asarray(semantic_scores,dtype=np.float64)
    assert np.isfinite(boxes).all() and np.isfinite(s).all() and np.all(s>=0)
    ids=np.argsort(-s,kind='stable')[:CONFIG['candidates']];prior=s[ids]/max(s[ids].sum(),1e-30);target=np.c_[boxes[ids],np.zeros(len(ids))];energies=np.zeros(len(ids));accepted=np.zeros(len(ids),dtype=bool);diags=[]
    for i,j in enumerate(ids):
        src=boxes[j]
        if np.any(src[3:6]<=0):diags.append({'query':int(j),'reason':'invalid_source_size_not_abs_or_clipped','accepted':False});continue
        crop=points[np.all(abs(points-src[:3])<=src[3:6]/2+1e-7,axis=1)]
        if len(crop)<CONFIG['crop_min_points']:diags.append({'query':int(j),'reason':'fewer_than16_unique_crop_points','n':len(crop),'accepted':False});continue
        # A seeded shuffle is a partition of observed points, not a second independent scan.
        perm=np.random.default_rng(seed+int(j)).permutation(len(crop));fit=observations(crop[perm[::2]]);check=observations(crop[perm[1::2]])
        b,e,d=fit_modes(fit,check,src);d.update(query=int(j),n=len(crop));diags.append(d)
        if d['accepted']:target[i]=b;accepted[i]=True;energies[i]=e
    # Evidence changes prior only when geometry passes the fixed support/ambiguity rule.
    # Non-accepted candidates have zero bonus, not an arbitrary favorable zero-energy fit.
    bonus=np.zeros(len(ids));bonus[accepted]=np.clip((.02-energies[accepted])/.02,-1.,1.)
    q=prior*np.exp(bonus);q/=max(q.sum(),1e-30);winner=int(q.argmax())
    return {'query_indices':ids,'prior':prior,'posterior':q,'logscore_bonus':bonus,'target_boxes7':target,'accepted':accepted,'direct_box7':target[winner].copy(),'direct_query':int(ids[winner]),'diagnostics':diags}
