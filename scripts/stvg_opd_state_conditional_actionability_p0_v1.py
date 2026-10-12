"""P9-P0 full cache-only paired state/evidence/randomness mechanism audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import collections,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts.stvg_opd_evidence_actionability_audit_v1 import BASE as P9,OUT as P9OUT,OLD,B,rows,put_rows,verify as p9_verify
BASE=ROOT/'artifacts/stvg_opd_state_conditional_actionability_v1'
OUT=ROOT/'results/stvg_opd_state_conditional_actionability/2026-10-13'
BOOT=10000;SEED=20261013;EPS=1e-12
def canonical(x):
    import torch
    if isinstance(x,torch.Tensor):return dict(dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest())
    if isinstance(x,np.ndarray):return dict(dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(x.tobytes()).hexdigest())
    if isinstance(x,dict):return {str(k):canonical(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [canonical(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x
def digest(x):return hashlib.sha256(json.dumps(canonical(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ln_vector(state):return np.concatenate([state[k].numpy().reshape(-1).astype(np.float64) for k in sorted(state) if 'query_residual' not in k])
def random_trace(f):
    import torch
    out=[];checks=0
    for k,r in enumerate(f['rounds']):
        a=r['rollout'];assert r['round']==k and a['mean'].dtype==a['samples'].dtype==torch.float32
        g=torch.Generator(device='cpu').manual_seed(a['seed']);n=torch.randn(len(a['mean']),16,4,generator=g,dtype=torch.float32);n=torch.cat((n,-n),1)
        z=a['mean'][:,None,:]+f['config']['sigma']*n
        assert torch.equal(z,a['samples']);checks+=z.numel()
        out.append(dict(round=k,seed=a['seed'],noise_shape=list(n.shape),noise_sha256=digest(n),samples_replay_exact=True))
    assert len(out)==(0 if f['empty'] else f['config']['steps'])
    return out,checks
def origins(records):
    out={}
    for ds in ['hc2','vidstg']:
        first=[r for r in records if r['dataset']==ds and r['arrival']==0];assert len(first)==2
        fits=[load(ROOT/r['original_Full']['path'])['fit'] for r in first]
        assert digest(fits[0]['initial'])==digest(fits[1]['initial'])
        out[ds]=fits[0]['initial']
    return out
def extract_pair(group,source):
    import torch
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from scripts.run_decota_paper_main_v1 import read_row
    group=sorted(group,key=lambda r:r['order']);assert [r['order'] for r in group]==['order1','order2']
    zs=[];fits=[];inputs=[];bindings={};traces=[];coords=0
    for r in group:
        p=P9/'readouts'/f"{r['public_cell']:05}.pt";assert sha(p)==read(P9/'GLOBAL_READOUT_BARRIER.json')['files'][str(p.relative_to(ROOT))]
        z=load(p);zs.append(z);bindings[str(p.relative_to(ROOT))]=sha(p)
        for rel,h in z['source_bindings'].items():assert sha(ROOT/rel)==h;bindings[rel]=h
        inp=normalized(load(ROOT/r['original_Full']['input_path']));inputs.append(inp)
        a=load(ROOT/r['original_Full']['path'])['fit'];c=load(B/'formal'/f"{r['public_cell']:05}"/'C_temporal_guided_OPD_repeat0.pt')['fit'];fits.append([a,c])
        assert digest(a['initial'])==digest(c['initial']) and a['config']==c['config']==r['config']
        assert len(a['initial'])==7 and all(v.numel()==256 and v.dtype==torch.float32 for v in a['initial'].values())
        q=next(v for k,v in a['initial'].items() if 'query_residual' in k);assert torch.count_nonzero(q).item()==0
        row=read_row(r['dataset'],r['query_ordinal']);key=row['key'];ordertrace=[]
        for j,f in enumerate([a,c]):
            t,n=random_trace(f);coords+=n
            for v in t:assert v['seed']==int(hashlib.sha256(f"spatial-opd-v1|{key}|{v['round']}".encode()).hexdigest()[:15],16)
            assert np.array_equal(z['before'],f['before'].numpy()) and np.array_equal(z['after'][j],f['final'].numpy())
            assert f['positions']==[int(v['position']) for v in z['anchors'][j]]
            ordertrace.append(t)
        traces.append(ordertrace)
    for field in ['dataset','source','parent_id','query_ordinal','capture_index','previous_P6','config']:assert group[0][field]==group[1][field]
    for field in ['source_model_sha256','pixel_sha256','frame_ids','anchors','visited']:assert digest(zs[0][field])==digest(zs[1][field]),field
    assert inputs[0]['interval']==inputs[1]['interval'] and inputs[0]['indices']==inputs[1]['indices']
    assert traces[0]==traces[1] and digest(inputs[0]['expert'])==digest(inputs[1]['expert'])
    ep=B/'expert_I1'/f"{group[0]['capture_index']:05}.pt";assert str(ep.relative_to(ROOT)) in bindings
    evidence=[digest(inputs[0]['expert']),sha(ep)]
    initial=[v[0]['initial'] for v in fits];x=[ln_vector(v) for v in initial];s=ln_vector(source);assert len(s)==1536
    delta=x[1]-x[0];distance=float(np.linalg.norm(delta));rms=float(np.sqrt(np.mean(delta**2)))
    stats=dict(LN_difference_L2=distance,LN_difference_RMS=rms,LN_difference_relative_source_norm=distance/float(np.linalg.norm(s)),LN_order1_source_L2=float(np.linalg.norm(x[0]-s)),LN_order2_source_L2=float(np.linalg.norm(x[1]-s)),query_residuals_both_exact_zero=True,initial_states_identical=digest(initial[0])==digest(initial[1]),before_box_displacement=float(np.linalg.norm(zs[1]['before'].astype(float)-zs[0]['before'].astype(float),axis=1).mean()))
    if stats['initial_states_identical']:
        assert np.array_equal(zs[0]['before'],zs[1]['before']) and all(np.array_equal(a,b) for a,b in zip(zs[0]['after'],zs[1]['after']))
    for rel,h in bindings.items():assert sha(ROOT/rel)==h
    return dict(dataset=group[0]['dataset'],parent_id=group[0]['parent_id'],previous_P6=group[0]['previous_P6'],public_cells=[r['public_cell'] for r in group],query_identity_sha256=digest(dict(dataset=group[0]['dataset'],query_ordinal=group[0]['query_ordinal'])),pixel_sha256=zs[0]['pixel_sha256'],source_model_sha256=zs[0]['source_model_sha256'],frame_ids_sha256=digest(zs[0]['frame_ids']),evidence_sha256=evidence,visited=zs[0]['visited'],initial_states=initial,source_initial_state=source,initial_sha256=[digest(v) for v in initial],random_traces=traces[0],random_sample_coordinate_checks=coords,stats=stats,source_bindings=bindings,CPU_only=True,GT_read=False)
def ci(x):
    x=np.asarray(x,float);v=x[np.isfinite(x)];return dict(ci95=np.quantile(v,[.025,.975]).tolist() if len(v) else None,valid_resamples=len(v),undefined_resamples=len(x)-len(v))
def correlation(x,y,method):
    from scipy.stats import rankdata
    x=np.asarray(x,float);y=np.asarray(y,float)
    if method=='Spearman':x=rankdata(x,axis=-1);y=rankdata(y,axis=-1)
    x=x-x.mean(axis=-1,keepdims=True);y=y-y.mean(axis=-1,keepdims=True);den=np.sqrt((x*x).sum(-1)*(y*y).sum(-1))
    return np.divide((x*y).sum(-1),den,out=np.full_like(den,np.nan,dtype=float),where=den>0)
def association(rr,xfield,yfield):
    values=[(r[xfield],r[yfield]) for r in rr if r.get(xfield) is not None and r.get(yfield) is not None];n=len(values)
    if n<3:return dict(parents=n,Pearson=None,Spearman=None)
    x,y=np.array(values).T;idx=np.random.default_rng(SEED).integers(n,size=(BOOT,n));out=dict(parents=n)
    for method in ['Pearson','Spearman']:
        v=float(correlation(x,y,method));out[method]=dict(value=v if math.isfinite(v) else None,**ci(correlation(x[idx],y[idx],method)))
    return out
def effect_stats(rr,outcome):
    def sign(v):return 'unknown' if v is None else 'positive' if v>EPS else 'negative' if v< -EPS else 'neutral'
    matrix=collections.Counter(sign(r['order1_'+outcome])+'_'+sign(r['order2_'+outcome]) for r in rr)
    pairs=[r for r in rr if r['order1_'+outcome] is not None and r['order2_'+outcome] is not None];n=len(pairs)
    nonzero=[r for r in pairs if abs(r['order1_'+outcome])>EPS and abs(r['order2_'+outcome])>EPS]
    flips=[r for r in nonzero if r['order1_'+outcome]*r['order2_'+outcome]<0]
    z=dict(total_parents=len(rr),defined_pairs=n,unknown_pairs=len(rr)-n,both_nonneutral_pairs=len(nonzero),transition_matrix=dict(matrix),sign_flip_parents=len(flips),sign_flip_all_parent_rate=len(flips)/len(rr),sign_flip_nonneutral_rate=len(flips)/len(nonzero) if nonzero else None,flip_both_over1pp=sum(min(abs(r['order1_'+outcome]),abs(r['order2_'+outcome]))>=.01 for r in flips),flip_both_over5pp=sum(min(abs(r['order1_'+outcome]),abs(r['order2_'+outcome]))>=.05 for r in flips),flip_effect_difference_below1pp=sum(abs(r['order2_'+outcome]-r['order1_'+outcome])<.01 for r in flips),difference_over5pp=sum(abs(r['order2_'+outcome]-r['order1_'+outcome])>=.05 for r in pairs),difference_over20pp=sum(abs(r['order2_'+outcome]-r['order1_'+outcome])>=.2 for r in pairs),any_order_harm5pp=sum(min(r['order1_'+outcome],r['order2_'+outcome])<-.05 for r in pairs),any_order_harm20pp=sum(min(r['order1_'+outcome],r['order2_'+outcome])<-.2 for r in pairs))
    if not n:return z
    a=np.array([r['order1_'+outcome] for r in pairs]);b=np.array([r['order2_'+outcome] for r in pairs]);idx=np.random.default_rng(SEED).integers(n,size=(BOOT,n))
    for k,v in [('order1',a),('order2',b),('order2_minus_order1',b-a),('absolute_order_difference',abs(b-a))]:z[k]=dict(mean=float(v.mean()),**ci(v[idx].mean(1)))
    z['flip_magnitudes']=dict(mean=float(np.mean([abs(r['order2_'+outcome]-r['order1_'+outcome]) for r in flips])) if flips else None,maximum=max(abs(r['order2_'+outcome]-r['order1_'+outcome]) for r in flips) if flips else None)
    indicators=((a>EPS)&(b< -EPS))|((a< -EPS)&(b>EPS));z['sign_flip_defined_pair_rate']=dict(value=float(indicators.mean()),**ci(indicators[idx].mean(1)))
    return z
def summarize(rr):
    out={}
    for ds in ['hc2','vidstg']:
        out[ds]={}
        for scope in ['Omega0','Omega1']:
            out[ds][scope]={}
            for subset in ['all128','previous_P6_32','added96']:
                q=[r for r in rr if r['dataset']==ds and r['scope']==scope and (subset=='all128' or r['previous_P6']==(subset=='previous_P6_32'))];assert len(q)=={'all128':128,'previous_P6_32':32,'added96':96}[subset]
                out[ds][scope][subset]=dict(parents=len(q),same_initial_state_pairs=sum(r['initial_states_identical'] for r in q),outcomes={o:effect_stats(q,o) for o in ['common_U_delta','own_U_delta']},associations={o:dict(LN_RMS_vs_absolute_effect_difference=association(q,'LN_difference_RMS','absolute_'+o+'_difference'),Before_teacher_agreement_difference_vs_effect_difference=association(q,'before_training_agreement_difference',o+'_difference')) for o in ['common_U_delta','own_U_delta']})
    return out
def contracts():
    from scipy.stats import pearsonr,spearmanr
    x=np.array([0,0,1,3,7.]);y=np.array([9,5,1,2,4.])
    assert abs(float(correlation(x,y,'Pearson'))-pearsonr(x,y).statistic)<1e-14 and abs(float(correlation(x,y,'Spearman'))-spearmanr(x,y).statistic)<1e-14
    assert not math.isfinite(float(correlation(x*0,y,'Pearson')))
    test=[dict(order1_y=.1,order2_y=-.2),dict(order1_y=None,order2_y=-1),dict(order1_y=0,order2_y=0)];s=effect_stats(test,'y');assert s['sign_flip_parents']==1 and s['unknown_pairs']==1 and s['flip_both_over5pp']==1
    return dict(status='pass',scope='synthetic tied Pearson/Spearman versus scipy, constant undefined correlation, full unknown transition and meaningful flip magnitudes; no GT/model/GPU',contracts=6,CPU_only=True,GT_read=False)
def prepare():
    p9_verify();assert read(P9/'FINAL_COMPLETION.json')['status']=='complete';assert not BASE.exists();BASE.mkdir();OUT.mkdir(parents=True,exist_ok=False)
    records=read(OLD/'DESIGN_LOCK.json')['records'];assert len(records)==512
    write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_new_paired_state_seal_and_reused_GT_labels',records=records,cells=512,parent_scope_pairs=512,parents=256,bootstrap=BOOT,seed=SEED,tiers_pp=[1,5,20],same_common_random_numbers_required=True,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,new_GT_scoring_calls=0,causal_state_intervention=False,production_promoted=False,time=time.time()))
    write(BASE/'CPU_CONTRACTS.json',contracts());names=['protocols/stvg_opd_state_conditional_actionability_p0_v1.md','scripts/stvg_opd_state_conditional_actionability_p0_v1.py','scripts/audit_stvg_opd_state_conditional_actionability_p0_v1.py']
    original=[P9/'RUNTIME.json',P9/'ROOT_READBACK.json',P9/'GLOBAL_READOUT_BARRIER.json',P9/'FINAL_COMPLETION.json',P9OUT/'ROWS.jsonl.gz',ROOT/'vg_tta/decota_spatial_opd_tunable_v1.py',ROOT/'scripts/run_stvg_opd_temporal_guided_p7b_v1.py']
    write(BASE/'RUNTIME.json',dict(status='locked_cache_only_same_evidence_state_and_randomness',pins={n:sha(ROOT/n) for n in names},original_bindings={str(p.relative_to(ROOT)):sha(p) for p in original},design_sha256=sha(BASE/'DESIGN_LOCK.json'),contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),CPU_only=True,new_GPU_fits=0,new_DINO_calls=0,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_actual_all_pair_seal',CPU_only=True,GT_read=False,time=time.time()))
def verify():
    p9_verify();r=read(BASE/'RUNTIME.json');assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256'] and sha(BASE/'CPU_CONTRACTS.json')==r['contracts_sha256']
    for p,h in {**r['pins'],**r['original_bindings']}.items():assert sha(ROOT/p)==h,p
    return r
def seal():
    import torch
    torch.set_num_threads(2);verify();assert not torch.cuda.is_initialized();records=read(BASE/'DESIGN_LOCK.json')['records'];source=origins(records);groups=collections.defaultdict(list)
    for r in records:groups[r['parent_id']].append(r)
    bindings={};noise=0;rounds_checked=0
    for parent,group in groups.items():
        z=extract_pair(group,source[group[0]['dataset']]);p=BASE/'pairs'/f'{parent:05}.pt';save(p,z);assert digest(load(p))==digest(z);noise+=z['random_sample_coordinate_checks'];rounds_checked+=sum(len(v) for v in z['random_traces'])*2;bindings[str(p.relative_to(ROOT))]=sha(p)
        status(BASE/'STATUS.json',dict(status='sealing_same_evidence_states_and_exact_random_replay',done=len(bindings),total=256,CPU_only=True,GT_read=False,time=time.time()))
    verify();assert len(bindings)==256 and not torch.cuda.is_initialized()
    write(BASE/'GLOBAL_PAIRED_READOUT_BARRIER.json',dict(status='sealed',files=bindings,parents=256,parent_scope_pairs=512,saved_updates=1024,initial_state_coordinates=256*2*1792,random_rounds=rounds_checked,random_sample_coordinate_checks=noise,all_source_evidence_config_noise_matched=True,all_saved_sample_replay_bitwise=True,runtime_sha256=sha(BASE/'RUNTIME.json'),CPU_only=True,GT_read=False,time=time.time()))
def assemble():
    verify();bar=read(BASE/'GLOBAL_PAIRED_READOUT_BARRIER.json');assert bar['status']=='sealed' and bar['runtime_sha256']==sha(BASE/'RUNTIME.json')
    old=rows(P9OUT/'ROWS.jsonl.gz');lookup={(r['parent_id'],r['scope'],r['order']):r for r in old};out=[]
    for rel,h in bar['files'].items():
        assert sha(ROOT/rel)==h;z=load(ROOT/rel)
        for scope in ['Omega0','Omega1']:
            a=lookup[(z['parent_id'],scope,'order1')];b=lookup[(z['parent_id'],scope,'order2')]
            for k in ['dataset','parent_id','previous_P6','training_anchors','teacher_quality','teacher_labeled_anchors','teacher_unknown_anchors','teacher_fully_labeled_correct','own_U_frames','common_U_frames','GT_event_real_frames']:assert a[k]==b[k],(k,a[k],b[k])
            r=dict(dataset=z['dataset'],parent_id=z['parent_id'],scope=scope,previous_P6=z['previous_P6'],same_evidence=True,common_random_numbers_exact=True,teacher_quality=a['teacher_quality'],teacher_fully_labeled_correct=a['teacher_fully_labeled_correct'],own_U_frames=a['own_U_frames'],common_U_frames=a['common_U_frames'],**z['stats'])
            r['before_training_agreement_difference']=None if a['before_training_agreement'] is None else b['before_training_agreement']-a['before_training_agreement']
            for o in ['common_U_delta','own_U_delta']:
                r['order1_'+o]=a[o];r['order2_'+o]=b[o];d=None if a[o] is None or b[o] is None else b[o]-a[o];r[o+'_difference']=d;r['absolute_'+o+'_difference']=None if d is None else abs(d)
            out.append(r)
    assert len(out)==512;return out
def analyze():
    write(BASE/'REUSED_GT_LABEL_EXPOSURE.json',dict(status='existing_verified_P9_GT_derived_labels_only_after_all_pair_seal',raw_GT_read=False,new_GT_scoring_calls=0,barrier_sha256=sha(BASE/'GLOBAL_PAIRED_READOUT_BARRIER.json'),source_P9_labels_sha256=sha(P9OUT/'ROWS.jsonl.gz'),CPU_only=True,GT_read=True,time=time.time()))
    rr=assemble();put_rows(OUT/'ROWS.jsonl.gz',rr);write(OUT/'SUMMARY.json',summarize(rr))
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_independent_root_PDF_view_publication_archive',parents=256,parent_scope_pairs=512,saved_updates=1024,CPU_only=True,raw_GT_read=False,new_GT_scoring_calls=0,GT_read=True,time=time.time()))
def root():
    import torch
    from scripts.audit_stvg_opd_state_conditional_actionability_p0_v1 import independent_states,independent_summary
    verify();records=read(BASE/'DESIGN_LOCK.json')['records'];sources=origins(records);groups=collections.defaultdict(list)
    for r in records:groups[r['parent_id']].append(r)
    bar=read(BASE/'GLOBAL_PAIRED_READOUT_BARRIER.json');checks=0;source_hashes=0;noise=0
    for parent,group in groups.items():
        p=BASE/'pairs'/f'{parent:05}.pt';assert sha(p)==bar['files'][str(p.relative_to(ROOT))];z=load(p);fresh=extract_pair(group,sources[group[0]['dataset']]);assert digest(z)==digest(fresh)
        checks+=independent_states(z);source_hashes+=len(z['source_bindings']);noise+=z['random_sample_coordinate_checks']
        status(BASE/'ROOT_STATUS.json',dict(status='independent_full_source_state_random_replay_readback',done=parent+1,total=256,CPU_only=True,GT_read=False,time=time.time()))
    rr=assemble();assert rr==rows(OUT/'ROWS.jsonl.gz');checks+=independent_summary(rr,read(OUT/'SUMMARY.json'));assert not torch.cuda.is_initialized();verify()
    write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='all256 original paired parents/all512 scope pairs/1024 saved fits; exact evidence/source/state/common random numbers and actual saved sample replay; complete reused labels and independent paired statistics',parents=256,parent_scope_pairs=512,saved_updates=1024,independent_scalar_checks=checks,original_source_second_hash_checks=source_hashes,random_sample_coordinate_checks=noise,readout_second_hash_checks=256,actual_raw_GT_read=False,new_GT_scoring_calls=0,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,causal_LN_attribution=False,CPU_only=True,GT_read=True,time=time.time()))
    write(BASE/'MECHANISM_SCOPE_DECISION.json',dict(status='descriptive_state_conditional_results_require_complete_C_before_any_causal_intervention',same_evidence_and_randomness_verified=True,causal_LN_attribution=False,new_gate_created=False,new_GPU_intervention_launched=False,C_GPU_interrupted=False,production_promoted=False,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':globals()[sys.argv[1]]()
