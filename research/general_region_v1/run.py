"""Independent CPU-only class-free feedback experiment; frozen K16 stays paused."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,hashlib,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
OLD=R.parent/'stvg-mainline';PROVIDER=R.parent/'stvg-opd'
A=OLD/'outputs/target_semantic128_collect_v3';B=OLD/'outputs/target_semantic128_shrink_v1'
D=R/'outputs/general_region_v1';P=R/'results/general_region_v1'
from research.general_region_v1.core import PRIMARY, candidates, weights, pool
ARMS=['Source','NativeG6','K16',PRIMARY,'FullGrid','Reflected','PositionOnly','WrongTime','K16Matched','DirectSameEvidence','Sharpen','ConstantNoOp']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
    p=read(D/'plan.json');assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    for f,h in p['code'].items():assert sha(R/f)==h,f
    for f,h in p['bound_inputs'].items():assert sha(f)==h,f
    full=read(R/'outputs/k16_full_v1/plan.json')
    for base,key in [(R,'code'),(OLD,'old_paper_code'),(PROVIDER,'provider_sources')]:
        for f,h in full[key].items():assert sha(base/f)==h,(key,f)
    assert read(R/'outputs/k16_full_v1/state.json')['status']=='paused_by_user'
    return p
def prepare():
    assert not(D/'plan.json').exists(),'Never replace a frozen experiment'
    acquisition=read(A/'plan.json');s=read(A/'sealed.json');base=read(B/'plan.json');bs=read(B/'sealed.json')
    assert s['count']==bs['count']==256 and not s['GT_read']
    code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
    p=dict(schema='general_region_v1',primary=PRIMARY,arms=ARMS,items=acquisition['items'],count=256,labels=base['labels'],code=code,
        authorization='User explicitly authorizes class-free target-content prototype validation, switch full ONLY after successful validation. Latest user explicitly permits stopping current full; K16 paused with HC320 receipts preserved.',
        mechanism='Original full-caption crossmodal time-decoder memory visual suffix BEFORE temporal decoder; fractional-area mean within same complete-box fulltime context1.5. No ASA layer/query/head/word probability is used for primary. Fit opposite-offset mean; subtract fitting mean and L2-normalize both fitting/evaluation vectors. K=(1+cos)/2. Positive/negative q0-membership weighted K means shrunk by fixed16 to unconditional fitting-fold K mean; log ratio summed on inclusive interval nodes, original q0 prior ONCE; original physical MBR/spatial DP.',
        meaning='Class-free target-state compatibility, NOT calibrated event probability. Main evidence is original trained Source multimodal memory. Opposite offsets are correlated. Spatial pool discards withinROI layout and may retain background/identity rather than event. Linear cosine kernel has no bandwidth/temperature parameter. No covariance inverse/learned projection/wordhead training/query optimization.',
        controls='FullGrid samecontentinterface without targetselection; Reflected uses exact rotated area weights preserving total support but may overlap target; WrongTime halfcycle of primary per parity preserves all evidence values/norms; PositionOnly same estimator on physical time; K16Matched oldK16 L scaled to EXACT primary L2 norm; DirectSameEvidence same scores uniformlegalprior; Sharpen q0 squared; ConstantNoOp exactnative.',
        fixed=dict(context=1.5,kappa=16,rounds=1,seed=20261023,numeric_configurations=1,kernel='(1+cos)/2',observed=.5,guard=.1,link=.25,feature='actual native temporal encoded_feature final H*W tokens, textprefix excluded; original fullcaption already fused',centering='otherparity unconditional mean; no allquery fit',epsilon=1e-12),
        exposure='Eachdirection128parent/onequery, previous expanded panel FULLY EXPOSED DEVELOPMENT. Prior64confirm is now exposed; no selection or tuning on this round. Not independent confirmation or cross-backbone validation.',
        qualification='EACHdirection: actual t and completev positive vsSource AND G6; primary t/v at leastK16 (replacement must not reduce eithermean); Source short<=5s/long>5s t positive; timewins>harms vsG6; >5pp t/v harm counts no greater thanK16. Primary mean t beyond FullGrid/Reflected/WrongTime/PositionOnly/K16Matched/Sharpen each; pairedCIs reported, do not call independent mechanism established if CIs includezero. On failure preserve all controls and STOP this fixed candidate; no controlpromotion/weight/kernel/layer/kappa scan and do NOT restart any full.',
        conditional_confirmation='OnlyifALLdevelopmentchecks pass, freeze unchanged and test32additionalparents per direction excluded from expanded128 and oldHC64/HC32/Vid32 once, samecriteria; originalfull exposure disclosed. ActualSource reconstruction only ifmissing strictbound memory, no newexpert. Allnewpredictions sealbeforeindependentCPUlabels. Onlyafterpassing fixeddual-domainconfirmation create NEWfullplan/fingerprint, preserve oldK16 receipts, never resume oldfull asnewmethod.',
        budget=dict(max_CPU_seconds=900,max_threads=1,new_model=0,new_GPU=0,new_RGB=0,new_encoder=0,new_text=0,new_fusion=0,new_wordhead=0,new_expert=0,new_observations=0,new_training=0,new_optimizer=0),
        bound_inputs={str(f):sha(f)for f in [A/'plan.json',A/'sealed.json',B/'plan.json',B/'sealed.json',R/'outputs/k16_full_v1/plan.json',R/'outputs/maintenance/k16_pause_general_20261023/pause_receipt.json']},
        bootstrap_seed=20261023,GT_in_inference=False,all256_seal_before_GT=True)
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p)
    put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['items','labels','bound_inputs']})
    for f in code:
        q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
    put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));verify();print(p['fingerprint'],flush=True)

def extract(cap,aux):
    import numpy as np
    ids=list(map(int,cap['frame_ids']));N=len(ids);out={k:np.empty((N,256))for k in [PRIMARY,'FullGrid','Reflected']};stats=[]
    for off in cap['temporal_cache']['offsets']:
        h,w=map(int,off['grid_size']);mem=off['inputs']['encoded_feature'];mask=off['inputs']['encoded_mask']
        assert mem.ndim==3 and mem.shape[0]>h*w and mem.shape[2]==256
        visual=mem[-h*w:].permute(1,0,2).numpy().reshape(-1,h,w,256)
        valid=(~mask[:,-h*w:]).numpy().reshape(-1,h,w)
        assert off['frame_ids']==ids[off['offset']::2] and not mask.any(),'Actual archived native grids have zero padding; do not assume a different token order'
        for j in range(len(visual)):
            t=off['offset']+2*j;support,fallback=weights(aux[t],h,w,valid[j]);reflected=support[::-1,::-1].copy()
            assert abs(reflected.sum()-support.sum())<1e-10 and not(reflected[~valid[j]]>0).any()
            out[PRIMARY][t]=pool(visual[j],support);out['Reflected'][t]=pool(visual[j],reflected);out['FullGrid'][t]=pool(visual[j],valid[j])
            stats.append(dict(frame=t,fallback=bool(fallback),area=float(support.sum()),grid=h*w,
                target_reflected_soft_overlap=float(np.minimum(support,reflected).sum()/support.sum())))
    assert all(np.isfinite(x).all()for x in out.values());return out,stats

def predict():
    import torch,numpy as np
    from stvg.runtime import setup
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.temporal import final_mbr
    from stvg.global6 import posterior
    setup();torch.set_num_threads(1);p=verify();assert not(D/'sealed.json').exists()and not(D/'predictions').exists()
    put(D/'state.json',dict(status='predicting_CPU',fingerprint=p['fingerprint'],**who()));s=read(A/'sealed.json');bs=read(B/'sealed.json');files={};start=time.perf_counter()
    for item in p['items']:
        assert time.perf_counter()-start<p['budget']['max_CPU_seconds'];key=item['alias'];a=s['files'][key];b=bs['files'][key];tick=time.perf_counter()
        for root,e,pairs in [(A,a,[('capture_path','capture_sha256'),('path','sha256')]),(B,b,[('trace_path','trace_sha256'),('path','sha256')])]:
            for name,h in pairs:assert sha(root/e[name])==e[h],key
        cap=torch.load(A/a['capture_path'],weights_only=False,map_location='cpu');old=torch.load(B/b['trace_path'],weights_only=False,map_location='cpu');sem=torch.load(A/a['path'],weights_only=False,map_location='cpu')
        assert cap['GT_read'] is False;ids=cap['frame_ids'];q0=posterior(cap['records'],cap['logits'],ids)
        assert np.array_equal(q0,old['q']['NativeG6'])and np.array_equal(q0,sem['q0'])
        geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);aux=evidence_dp(geom.for_interval([ids[0],ids[-1]+1]))['final'].numpy()
        features,pooling=extract(cap,aux);qs,feedback=candidates(features,q0,old['steps']['ShrinkK16']['L'],ids)
        qs.update(NativeG6=q0,K16=old['q']['ShrinkK16']);iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()}
        arr={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in qs}
        with np.load(B/b['path'],allow_pickle=False)as z:
            for name,ref in [('NativeG6','NativeG6'),('K16','ShrinkK16')]:
                assert np.array_equal(arr[name],z[ref])and iv[name]==z[ref+'_interval'].tolist(),(key,name)
            assert np.array_equal(cap['Source'],z['Source']);arr['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
        f=D/'predictions'/f'{key}.npz';f.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(f,**arr,**{a+'_interval':v for a,v in iv.items()},frame_ids=ids,positions8=cap['positions8'])
        tr=D/'traces'/f'{key}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,feedback=feedback,features=features,pooling=pooling,GT_read=False),tr)
        files[key]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,input_capture_sha256=a['capture_sha256'],Source_G6_K16_bitwise=True)
        put(D/'progress.json',dict(completed=len(files),total=256,seconds=time.perf_counter()-start,GT_read=False,**who()))
        if len(files)%16==0:print(len(files),round(time.perf_counter()-start,2),flush=True)
    verify();assert not torch.cuda.is_initialized();put(D/'sealed.json',dict(count=256,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,new_model=0,new_expert=0,new_GPU=0,all256_Source_G6_K16_noop_bitwise=True,seconds=time.perf_counter()-start))
    put(D/'state.json',dict(status='sealed_before_new_GT',fingerprint=p['fingerprint']));print('SEALED256',flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));raise
