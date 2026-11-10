"""Independent raw-state formula, source prefix, assignment and readout audit.

Does not import the detector/runner or any task labels. Independently recomputes
both gates, source-derived fixed gate and donor set, random edit norm/count,
every real normalized prediction and complete source restoration.
"""
import argparse,time,torch
from common60 import *

def independently_detect(f,a,W,b):
    c=[];energies=[];roles=[];allx=[]
    for view in 'ABC':
        X=f[view+'_X'].astype(np.float64);T=f[view+'_tokens'].astype(np.float64);mask=f[view+'_mask'].astype(bool)
        y=X.dot(W.T)+b;norm=np.clip(np.linalg.norm(y,axis=-1),1e-12,None)
        v=T[mask].sum(0)/mask.sum()-T[~mask].sum(0)/(~mask).sum();v=v/max(np.linalg.norm(v),1e-12)
        c.append(X/norm[:,None]*np.dot(W.T,v)[None,:]);energies.append((X/norm[:,None])**2*(W*W).sum(0)[None]);roles.append((np.dot(W.T,v)**2)/np.clip((W*W).sum(0),1e-20,None));allx.append(X)
    aligned=[c[0]];rho=[]
    for k,view in enumerate('BC',1):
        transport=a[view+'_W'].astype(np.float64);mass=transport.sum(1);tmp=np.zeros_like(c[k]);ok=mass>0;tmp[ok]=transport[ok].dot(c[k])/mass[ok,None];aligned.append(tmp);rho.append(a[view+'_affinity'].max(1)*ok)
    reliability=np.minimum(*rho);total=float(reliability.sum());noise=np.zeros(288);signal=np.zeros(288)
    if total>1e-12:
        weights=reliability/total;stack=np.array(aligned);mu=stack.mean(0);noise=(weights[:,None]*((stack-mu[None])**2).mean(0)).sum(0);cent=(weights[:,None]*mu).sum(0);signal=(weights[:,None]*(mu-cent)**2).sum(0)
    excess=np.clip(noise-signal,0,None);rank=excess;strength=excess/(noise+signal+1e-12)
    def choose(rank,strength):
        indices=sorted(range(288),key=lambda j:(-rank[j],j))[:8];out=np.zeros(288)
        for j in indices:
            if rank[j]>1e-12:out[j]=.25*np.clip(strength[j],0,1)
        return out
    ng=choose(rank,strength);X=np.vstack(allx);fraction=np.square(X.mean(0))/np.clip(np.square(X).mean(0),1e-20,None);relevance=np.array(roles).mean(0);energy=np.vstack(energies).mean(0);strength=np.clip(2*fraction-1,0,1)*(1-relevance);cg=choose(energy*strength,strength)
    return {'noise':ng,'common':cg},{'support_mass':total,'supported_anchor_count':int((reliability>0).sum()),'noise_energy':noise,'between_candidate_signal':signal,'common_fraction':fraction,'entity_direction_fraction':relevance,'normalized_channel_energy':energy}

def independent_score(f,g,W,b,gate,activation=False):
    parts=[]
    for view in 'ABC':
        x=torch.from_numpy(f[view+'_X']);weight=torch.tensor(W,dtype=torch.float32);bias=torch.tensor(b,dtype=torch.float32);factor=torch.tensor(1-gate,dtype=torch.float32)
        if activation:y=torch.nn.functional.linear(x*factor[None],weight,bias)
        else:y=torch.nn.functional.linear(x,weight*factor[None],bias)
        q=torch.nn.functional.normalize(y,dim=1);t=torch.from_numpy(f[view+'_tokens']);mask=torch.from_numpy(f[view+'_mask']).bool();l=q.matmul(t.transpose(0,1))/.07;z=torch.logsumexp(l[:,mask],1)-torch.logsumexp(l[:,~mask],1);prob=torch.softmax(z.to(torch.float64),dim=0).numpy()
        if view!='A':
            prob=g[view+'_W'].dot(prob);mass=float(g[view+'_fixed_mass']);prob=mass*prob/max(prob.sum(),1e-300)+(1-mass)*g['base']
        parts.append(prob)
    return np.array(parts).sum(0)/3

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.perf_counter();rows=read(ROOT/p['manifest']);D=ROOT/p['detection_root'];P=ROOT/p['prediction_root'];F=ROOT/p['feature_root'];A=ROOT/p['association_root'];ds=read(D/'SEALED.json');ps=read(P/'SEALED.json');fs=read(F/'SEALED.json');ss=read(A/'SEALED.json');assert ps['arm_names']==ARMS and ps['n']==ds['n']==336
    for root,sealdata in [(D,ds),(P,ps)]:
        for rel,h in sealdata['files'].items():assert sha(root/rel)==h
    orig=load(ROOT/p['source_parameters']);W=orig['4.weight'].astype(float);b=orig['4.bias'].astype(float);allg=[];maxgate=0.;maxscore=0.;changed={arm:0 for arm in ARMS[6:]};effects=[]
    AB=ROOT/'crossplatform/round50/episodic336_capture_attempt001';C=ROOT/'crossplatform/round57/capture336_attempt001';abseal=read(AB/'SEALED.json');cs=read(C/'SEALED.json')
    # Independent two-hidden-layer replay, no import of prior producer.
    def prefix(h):
        h=torch.tensor(h)[None]
        for layer in [0,2]:h=torch.relu(torch.nn.functional.linear(h,torch.tensor(orig[str(layer)+'.weight']),torch.tensor(orig[str(layer)+'.bias'])))
        return h[0].numpy()
    for i,row in enumerate(rows):
        rel='targets_PRIVATE/'+row['id']+'.npz';assert sha(F/rel)==fs['files'][rel];f=load(F/rel,[v+'_'+k for v in 'ABC' for k in ['X','tokens','mask']]);rel='samples_PRIVATE/'+row['id']+'.npz';assert sha(A/rel)==ss['files'][rel];g=load(A/rel)
        assert sha(AB/rel)==abseal['files'][rel] and sha(C/rel)==cs['files'][rel];ab=load(AB/rel,['A_h','B_h']);c=load(C/rel,['h'])
        for view,h in [('A',ab['A_h']),('B',ab['B_h']),('C',c['h'])]:assert np.array_equal(prefix(h),f[view+'_X'])
        gates,diag=independently_detect(f,g,W,b);saved=load(D/rel)
        for k,v in diag.items():assert np.allclose(v,saved[k],rtol=1e-12,atol=1e-12),k
        for z in MECHANISMS:
            err=float(np.max(abs(gates[z]-saved[z+'_gate'])));maxgate=max(maxgate,err);assert err<1e-12
        allg.append(gates)
    expectedfixed={}
    for z in MECHANISMS:
        avg=np.mean([x[z] for x in allg],0);ind=sorted(range(288),key=lambda j:(-avg[j],j))[:8];out=np.zeros(288);out[ind]=avg[ind];expectedfixed[z]=out
    fixed=load(D/'FIXED_PATCHES_PRIVATE.npz')
    for z in MECHANISMS:assert np.array_equal(fixed[z],expectedfixed[z])
    donors=read(D/'PATCH_DONORS_PRIVATE.json');expected=np.arange(336)
    for pl,seed in [('drone',60301),('quad',60302)]:
        ids=np.array([i for i,r in enumerate(rows) if r['platform']==pl]);rng=np.random.default_rng(seed);order=rng.permutation(ids);expected[order]=np.roll(order,1)
    assert np.array_equal(donors,expected) and set(donors)==set(range(336)) and all(i!=j for i,j in enumerate(donors))
    for i,row in enumerate(rows):
        rel='samples_PRIVATE/'+row['id']+'.npz';f=load(F/('targets_PRIVATE/'+row['id']+'.npz'),[v+'_'+k for v in 'ABC' for k in ['X','tokens','mask']]);g=load(A/rel);old=load(F/rel);new=load(P/rel);patch=load(P/('patches_PRIVATE/'+row['id']+'.npz'));case=read(P/('cases_PRIVATE/'+row['id']+'.json'));assert case['optimizer_calls']==0 and case['no_pseudo_labels_fitted'] and case['source_restored_exact']
        assert np.array_equal(new['source_boxes'],old['source_boxes']) and np.array_equal(new['source_scores'],old['source_scores']);assert np.array_equal(new['selected_original_boxes'],old['source_boxes'][new['queries']])
        for j,arm in enumerate(ARMS[:6]):assert np.array_equal(new['scores'][j],old['scores'][list(old['arm_names']).index('direct_all' if arm=='r59_direct_all' else arm)])
        for j,arm in enumerate(ARMS[6:],6):
            z,variant=arm.split('_');gate=patch[arm];assert np.count_nonzero(gate)<=8 and gate.min()>=0 and gate.max()<=.5+1e-12
            if variant in ['dynamic','activation']:assert np.max(abs(gate-allg[i][z]))<1e-12
            elif variant=='fixed':assert np.array_equal(gate,fixed[z])
            elif variant=='shuffled':assert np.max(abs(gate-allg[donors[i]][z]))<1e-12
            else:
                dyn=allg[i][z];assert not (set(np.flatnonzero(dyn)) & set(np.flatnonzero(gate)))
                e=np.sort(np.linalg.norm(W,axis=0)[dyn>0]*dyn[dyn>0]);r=np.sort(np.linalg.norm(W,axis=0)[gate>0]*gate[gate>0]);assert np.allclose(e,r,rtol=1e-12,atol=1e-12)
            scores=independent_score(f,g,W,b,gate,variant=='activation');error=float(np.max(abs(scores-new['scores'][j])));maxscore=max(maxscore,error);assert error==0 and int(scores.argmax())==int(new['queries'][j]);changed[arm]+=case['arms'][arm]['actual_weight_changed'];assert case['arms'][arm]['actual_weight_changed']==(variant!='activation' and np.count_nonzero(gate)>0)
        assert np.array_equal(new['queries'],new['scores'].argmax(1))
    out=ROOT/p['audit_root'];assert not out.exists();out.mkdir(parents=True);result={'state':'PASS_R60_INDEPENDENT_DETECTOR_COLUMN_ASSIGNMENT_CONTROL_AND_NATIVE_READOUT_AUDIT','n':336,'independent_noninherited_readouts':3360,'prefix_features_checked':1008,'max_detector_error':maxgate,'max_native_score_error':maxscore,'source_boxes_and_parameters_restoration_verified':True,'all_reference_readouts_exact':True,'fixed_patch_label_free_target_calibration_verified':True,'donor_patch_multisets_exact':True,'random_count_and_edit_norm_matched':True,'no_autograd_solver_or_pseudo_label_fit':True,'actual_weight_changed_cases':changed,'prediction_seal_sha256':sha(P/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result,flush=True)
if __name__=='__main__':main()
