"""A-only, no-optimizer gradient diagnostic. Must be launched under night_guard."""
import sys,time,json,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'crossplatform/round29'),str(ROOT/'crossplatform/round24')]
import common30
from common30 import read,dump,sha,seal,verify,load_npz
import torch
import fit24
from fit29 import logodds_scores  # Legacy import can overwrite fit24.feature.
fit24.feature=common30.feature   # Restore only the already verified A source cache.
from fit24 import module,batch
F=torch.nn.functional
BASE=Path(__file__).resolve().parent


def fingerprint(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def flat_grad(loss,parameters):
    gg=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=False)
    return torch.cat([g.reshape(-1) for g in gg]).detach().cpu().numpy().astype(np.float64)


def cosine(a,b):
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return None if na==0. or nb==0. else float(np.dot(a,b)/(na*nb))


def pair_statistics(G,norms,mask):
    ix=np.flatnonzero(mask);pairs=[]
    for ii,i in enumerate(ix):
        for j in ix[ii+1:]:
            dot=float(G[i]@G[j]);mass=float(norms[i]*norms[j]);pairs.append((dot,dot/mass,mass))
    if not pairs:return {'observations':len(ix),'pairs':0,'negative_pairs':0,'raw_negative_fraction':None,'norm_weighted_cosine':None,'negative_dot_energy_fraction':None,'minimum_cosine':None}
    dots=np.array([x[0] for x in pairs]);cs=np.array([x[1] for x in pairs]);mass=np.array([x[2] for x in pairs]);energy=float(abs(dots).sum())
    return {'observations':len(ix),'pairs':len(pairs),'negative_pairs':int((dots<0).sum()),'raw_negative_fraction':float((dots<0).mean()),'norm_weighted_cosine':float(dots.sum()/mass.sum()),'negative_dot_energy_fraction':float(np.maximum(-dots,0).sum()/energy) if energy>0 else None,'minimum_cosine':float(cs.min()),'maximum_cosine':float(cs.max()),'sum_absolute_dot':energy}


def main():
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=BASE/'attempt001';out.mkdir(exist_ok=False);(out/'gradients_PRIVATE').mkdir();start=time.perf_counter()
    verify(ROOT/'crossplatform/round29/direct208_attempt001');verify(ROOT/'crossplatform/round30/screen_attempt001')
    feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');assert len(feedback)==22
    A=common30.rows('A208');assert len(A)==208;assert all(r['stage'] not in ['R','S','T'] for r in A)
    spec={'scope':'A208 only; source and R30b1 query projection states; autograd only','optimizer_calls':0,'training_steps':0,'new_model_states_saved':0,'feedback_sha256':sha(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json'),'original_platform_denominators':{'drone':10,'quad':12},'U_counts':{'drone':96,'quad':93},'tau':.1,'beta':1,'sourceq0':'fixed official source score argmax','relative_active_norm_threshold':1e-6,'absolute_active_norm_threshold':1e-12,'thresholds_are_diagnostic_not_label_filter':True,'program_sha256':sha(__file__),'no_R_S_T_labels_or_inputs':True}
    dump(out/'SPEC_LOCK.json',spec);all_states=[];all_observations=[];all_pairs=[]
    for pl in ['drone','quad']:
        rr=[r for r in A if r['platform']==pl];ff=[f for f in feedback if f['platform']==pl];N=len(ff);assert N==spec['original_platform_denominators'][pl]
        lookup={r['id']:i for i,r in enumerate(rr)};by_id=collections.OrderedDict()
        for j,f in enumerate(ff):by_id.setdefault(f['id_PRIVATE'],[]).append(j)
        U=[i for i,r in enumerate(rr) if r['id'] not in by_id];assert len(U)==spec['U_counts'][pl]
        a=batch(rr);assert all(not v.requires_grad for v in a.values() if isinstance(v,torch.Tensor));q0=a['scores'].argmax(1)
        source=module('query_mlp')
        with torch.no_grad():z0=logodds_scores(source,a);d0=z0-z0.gather(1,q0[:,None])
        for state in ['source','R30_b1']:
            m=module('query_mlp')
            if state=='R30_b1':
                ss=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));m.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in ss.items()})
            before=fingerprint(m);parameters=list(m.parameters());assert sum(p.numel() for p in parameters)==184960
            z=logodds_scores(m,a);gaps=torch.stack([z[lookup[f['id_PRIVATE']],f['positive_group_PRIVATE']].max()-z[lookup[f['id_PRIVATE']],f['negative_group_PRIVATE']].max() for f in ff]);loss_terms=F.softplus(-gaps/.1);pref=loss_terms.sum()/N;dist=z-z.gather(1,q0[:,None]);keep=(dist[U]-d0[U]).square().mean();gp=flat_grad(pref,parameters);gk=flat_grad(keep,parameters);gt=flat_grad(pref+keep,parameters)
            obsids=list(by_id);G=np.stack([flat_grad(loss_terms[js].sum()/N,parameters) for js in by_id.values()]);norms=np.linalg.norm(G,axis=1);sumG=G.sum(0);sum_error=float(np.linalg.norm(sumG-gp));sum_relative=sum_error/max(float(np.linalg.norm(gp)),1e-30);assert sum_error<2e-5 and sum_relative<2e-5,(pl,state,sum_error,sum_relative)
            total_error=float(np.linalg.norm(gt-gp-gk));assert total_error<2e-5,(pl,state,total_error)
            if state=='source':assert float(np.linalg.norm(gk))==0.,'source keep loss gradient must be zero'
            nonzero=norms>0.;maxnorm=float(norms.max());active_rel=nonzero&(norms>=maxnorm*spec['relative_active_norm_threshold']);active_abs=nonzero&(norms>=spec['absolute_active_norm_threshold']);cosmat=np.full((len(G),len(G)),np.nan);dotmat=G@G.T
            for i in range(len(G)):
                for j in range(len(G)):
                    if norms[i]>0 and norms[j]>0:cosmat[i,j]=dotmat[i,j]/(norms[i]*norms[j])
            aliases=[rr[lookup[i]]['alias'] for i in obsids]
            for i,(obid,js) in enumerate(by_id.items()):
                row={'platform':pl,'state':state,'feedback_alias':aliases[i],'preference_count':len(js),'loss_contribution_original_denominator':float((loss_terms[js].sum()/N).detach()),'loss_unscaled_sum':float(loss_terms[js].sum().detach()),'gradient_norm':float(norms[i]),'gradient_norm_share':float(norms[i]/norms.sum()) if norms.sum()>0 else None,'relative_norm_to_max':float(norms[i]/maxnorm) if maxnorm>0 else None,'active_relative_1e6':bool(active_rel[i]),'active_absolute_1e12':bool(active_abs[i]),'cosine_to_total_pref':cosine(G[i],gp),'cosine_to_keep':cosine(G[i],gk),'dot_to_keep':float(G[i]@gk),'original_feedback_labels':[ff[j]['label'] for j in js],'no_label_removed_or_reweighted':True};all_observations.append(row)
            for i in range(len(G)):
                for j in range(i+1,len(G)):
                    all_pairs.append({'platform':pl,'state':state,'a':aliases[i],'b':aliases[j],'dot':float(dotmat[i,j]),'cosine':float(cosmat[i,j]) if np.isfinite(cosmat[i,j]) else None,'norm_product':float(norms[i]*norms[j]),'both_relative_active':bool(active_rel[i] and active_rel[j]),'both_absolute_active':bool(active_abs[i] and active_abs[j])})
            after=fingerprint(m);assert before==after;assert all(p.grad is None for p in parameters)
            np.savez_compressed(out/'gradients_PRIVATE'/(pl+'_'+state+'.npz'),gradient_by_observation=G.astype(np.float32),gradient_pref=gp.astype(np.float32),gradient_keep=gk.astype(np.float32),aliases=np.array(aliases),cosine_matrix=cosmat,dot_matrix=dotmat,norms=norms)
            record={'platform':pl,'state':state,'feedback_observations':len(G),'feedback_pairs':N,'keep_observations':len(U),'preference_loss':float(pref.detach()),'keep_loss':float(keep.detach()),'pref_gradient_norm':float(np.linalg.norm(gp)),'keep_gradient_norm':float(np.linalg.norm(gk)),'total_gradient_norm':float(np.linalg.norm(gt)),'pref_keep_cosine':cosine(gp,gk),'pref_keep_dot':float(gp@gk),'zero_keep_is_not_compatibility_evidence':state=='source','perobs_sum_vs_original_pref_L2_error':sum_error,'perobs_sum_vs_original_pref_relative_error':sum_relative,'total_vs_pref_plus_keep_L2_error':total_error,'gradient_sum_norm_over_sum_norms':float(np.linalg.norm(gp)/norms.sum()) if norms.sum()>0 else None,'all_exact_nonzero':pair_statistics(G,norms,nonzero),'relative_active_1e6':pair_statistics(G,norms,active_rel),'absolute_active_1e12':pair_statistics(G,norms,active_abs),'zero_gradient_observations':int((~nonzero).sum()),'model_state_sha256_before':before,'model_state_sha256_after':after,'parameter_grad_buffers_still_None':True,'parameters_unchanged':True,'optimizer_calls':0};all_states.append(record);print('GRAD33',pl,state,'norm',record['pref_gradient_norm'],'keep',record['keep_gradient_norm'],'pref-keep',record['pref_keep_cosine'],'active',int(active_rel.sum()),flush=True)
            del z,gaps,loss_terms,pref,keep,m
        del a,source,z0,d0
    dump(out/'STATE_SUMMARY.json',all_states);dump(out/'OBSERVATION_ROWS.json',all_observations);dump(out/'PAIRWISE_ROWS.json',all_pairs);seal(out,'R33_A_only_source_and_R30_gradient_diagnosis_no_optimizer',states=4,gradient_observation_rows=len(all_observations),pairwise_rows=len(all_pairs),seconds=time.perf_counter()-start,optimizer_calls=0,training_steps=0,model_parameters_unchanged=True,peak_bytes=torch.cuda.max_memory_allocated());print('GRADIENT_DIAGNOSTIC33_SEALED',flush=True)

if __name__=='__main__':main()
