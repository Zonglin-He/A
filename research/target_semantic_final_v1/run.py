"""Final user-authorized temporal repair. Frozen single candidate, no parameter scan."""
import os,sys,json,time,hashlib,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put
from research.target_semantic_final_v1.core import candidates
A=R/'outputs/target_semantic128_collect_v3';B=R/'outputs/target_semantic128_shrink_v1'
D=R/'outputs/target_semantic_final_v1';P=R/'results/target_semantic_final_v1'
PROVIDER=R.parent/'stvg-opd'
PRIMARY='JackknifePrototype'
BASE=['NativeG6','Loop2','Loop1','ShrinkK16','WeakB075']
NEW=['JackknifePrototype','UniformMatched','PermutedMatched','WrongTime','RandomSemantic','ReflectedSemantic','DirectSameEvidence']
ARMS=BASE+NEW+['Source']

def verify():
    p=read(D/'plan.json')
    assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
    for root,key in [(R,'code'),(R,'paper_code'),(PROVIDER,'provider_sources')]:
        for f,h in p[key].items(): assert sha(root/f)==h,f
    for path,h in p['bound_inputs'].items(): assert sha(R/path)==h,path
    return p

def prepare():
    assert not(D/'plan.json').exists()
    a=read(A/'plan.json');s=read(A/'sealed.json');bs=read(B/'sealed.json')
    assert s['count']==bs['count']==256 and not s['GT_read'] and not bs['GT_read']
    code={str(f.relative_to(R)):sha(f) for f in Path(__file__).parent.glob('*.py')}
    for f in ['research/native_event_semantic_prototype_v1/core.py','research/target_semantic128_shrink_v1/core.py','research/native_event_target_semantic_v1/score.py','research/target_semantic128_shrink_v1/score.py','research/target_semantic128_rank_audit.py','research/joint_event_v1/run.py']:
        code[f]=sha(R/f)
    parent=read(B/'plan.json')
    p=dict(schema='target_semantic_final_v1',primary=PRIMARY,arms=ARMS,items=a['items'],count=256,
       authorization='User selects continue repair, meet criteria before full; THIS IS THE LAST repair round. No additional candidate, tuning range, control promotion or automatic fallback after failure.',
       mechanism='One-round q0 softmembership. In each opposite-parity fitting fold split chronologically into exactly4 groups; delete each group and recompute Bernoulli logodds prototype contrast eta. Sensitivity V=.75*sum((eta_delete-mean_delete)^2). Coordinate reliability r=eta^2/(eta^2+V), 0/0=0. Shrink BOTH class means toward same fitted mu0 by coordinate r. This can change direction unlike common scalar kappa. No optimized state or actionness. Stability need not mean event correctness.',
       prior_evidence='Source pooled BCE word classification is not per-frame event supervision. Shared-mean scalar kappa preserves prototype contrast direction; confirmed to1.39e-16. Candidate ranking HC fails. Last-iteration mechanistic hypothesis: locally concentrated semantic contrast is unstable under removal of temporal blocks. Stable wrong semantics can survive this mechanism.',
       source_conditions='Source pipeline.py123/139 reads SpatialActivation over selected frames; classifier.py pools head(query).mean(0), criterion.py65-75 uses multi-hot BCE. We retain original data/word-head; no foreground/actionness scores used as added feedback. No private checkpoint training-history claim.',
       variance_scope='Delete-group replication is a sensitivity proxy. Four dependent chronological blocks, soft q0 labels and overlapping offsets do NOT establish unbiased/calibrated uncertainty. r formula is a new fixed heuristic, not a reproduced trained method or statistically proved Bayesian estimate.',
       reference='https://data.nass.usda.gov/Education_and_Outreach/Reports,_Presentations_and_Conferences/allreports/Using_the_Delete-a-Group_Jackknife_Estimator_in_NASS_Surveys.pdf',
       controls='UniformMatched multiplies raw one-round evidence by its L2 norm ratio to primary; PermutedMatched applies exactly same reliability multiset to fixed shuffled axes then matches L2. Direct uses same evidence with uniform legal prior. WrongTime reverses semantic sequence. Random is existing exact-per-axis histogram matched projection; Reflected uses same-size reflected ROI and is not a known-negative identity. No control promoted.',
       exposure='All256 previous128-per-direction rows are exposed DEVELOPMENT ONLY, including prior confirmation. No split relabelled unseen and NO selection in this round; primary/blockcount/formula fixed before new metrics.',
       qualification='For EACH direction across all128 development parents: actual t and complete-v means beyond BOTH Source/Global6 strictlypositive, Source short<=5s AND long>5s t strictlypositive, timewins>harms, time/full-v severe counts no greater than originalLoop2 and total time-severe strictlyfewer across domains. Primary mean t must exceed UniformMatched/PermutedMatched/WrongTime/RandomSemantic/ReflectedSemantic in both domains. Report paired CIs and candidate rankings; positive means alone not enough to claim mechanism. On ANY failed condition STOP last temporal repair, preserve/publish all outcomes, full remains paused, no further candidate or parameter scan.',
       conditional_confirmation='ONLY if every development condition passes: freeze unchanged candidate, input-SHA seed20261023 select32 new parents per direction excluding expanded128 and oldHC64/HC32/Vid32 parents (remaining53HC/578Vid), onequeryeach. These are prospective-for-this-candidate; original full historically exposed. Reconstruct missing Source224 features only with strict cached old evidence, original Source/Full exact, no new teacher. All64seal before independent CPU fixed labels, SAME criteria; no retuning. If pass, freeze new full plan then run required new temporal outputs; never mix oldpaper fingerprint. If fail stop. No automatic original full resume.',
       budget=dict(blocks=4,rounds=1,seed=20261023,numeric_configurations=1,max_CPU_seconds=600,new_model=0,new_RGB=0,new_expert=0,new_observations=0,new_optimizer=0),
       unchanged='Same originalq0, target1.5context, allwordcoordinates, inclusiveSUM, physicalMBR, complete-boxDP, originalSource/E/eightobservations. No kappa/beta stacking or query veto.',
       code=code,paper_code=a['paper_code'],provider_sources=a['provider_sources'],labels=parent['labels'],bootstrap_seed=20261023,
       bound_inputs={str(f.relative_to(R)):sha(f) for f in [A/'plan.json',A/'sealed.json',B/'plan.json',B/'sealed.json',P/'PREDECLARED_DESIGN.json']},GT_in_inference=False,all256_seal_before_GT=True,no_automatic_full=True)
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    put(D/'plan.json',p)
    put(P/'protocol.json',{k:v for k,v in p.items() if k not in ['items','paper_code','provider_sources','labels']})
    for f in code:
        path=D/'source_snapshot'/f;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((R/f).read_bytes())
    put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']))
    print(p['fingerprint'])

def predict():
    import torch,numpy as np
    from stvg.temporal import final_mbr
    from stvg.spatial import SpatialGeometry,evidence_dp
    torch.set_num_threads(1);p=verify();assert not(D/'sealed.json').exists()
    assert not(D/'predictions').exists(),'No duplicate/mixed-code inference'
    acq=read(A/'sealed.json');oldseal=read(B/'sealed.json');start=time.perf_counter();files={};reliabilities=[]
    startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    put(D/'state.json',dict(status='predicting_CPU',pid=os.getpid(),startticks=startticks,fingerprint=p['fingerprint']))
    for item in p['items']:
        assert time.perf_counter()-start<p['budget']['max_CPU_seconds']
        key=item['alias'];ae=acq['files'][key];be=oldseal['files'][key]
        for root,e in [(A,ae),(B,be)]:
            for path,checksum in [('path','sha256'),('trace_path','trace_sha256')] if root==B else [('path','sha256'),('capture_path','capture_sha256')]:
                assert sha(root/e[path])==e[checksum]
        x=torch.load(A/ae['path'],weights_only=False,map_location='cpu');cap=torch.load(A/ae['capture_path'],weights_only=False,map_location='cpu')
        prior=torch.load(B/be['trace_path'],weights_only=False,map_location='cpu');q0=x['q0'];sem=x['semantic_p']
        seed=int(hashlib.sha256((str(p['budget']['seed'])+key).encode()).hexdigest()[:16],16)
        qs,trace=candidates(sem['TargetContext'],q0,x['random_p'],sem['ReflectedContext'],seed)
        qs.update({a:prior['q'][a] for a in BASE});assert np.array_equal(qs['NativeG6'],q0)
        geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);ids=cap['frame_ids']
        intervals={a:final_mbr(ids,q)['interval'] for a,q in qs.items()}
        arr={a:evidence_dp(geom.for_interval(intervals[a]))['final'].numpy() for a in qs}
        with np.load(B/be['path'],allow_pickle=False) as z:
            for a in BASE:
                assert np.array_equal(arr[a],z[a]) and intervals[a]==z[a+'_interval'].tolist(),(key,a)
            arr['Source']=z['Source'].copy();intervals['Source']=z['Source_interval'].tolist()
        f=D/'predictions'/f'{key}.npz';f.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(f,**arr,**{a+'_interval':v for a,v in intervals.items()},frame_ids=ids,positions8=x['positions8'])
        tf=D/'traces'/f'{key}.pt';tf.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(q=qs,feedback=trace,GT_read=False),tf)
        files[key]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tf.relative_to(D)),trace_sha256=sha(tf))
        reliabilities.extend(float(v['reliability'].mean()) for v in trace['prototypes'])
        put(D/'progress.json',dict(completed=len(files),total=256,pid=os.getpid(),seconds=time.perf_counter()-start,GT_read=False))
    verify();put(D/'sealed.json',dict(count=256,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,seconds=time.perf_counter()-start,
         original_base_outputs_bitwise=256,reliability_mean=float(np.mean(reliabilities)),new_model=0,new_expert=0,new_observations=0))
    put(D/'state.json',dict(status='sealed_before_new_GT',fingerprint=p['fingerprint']))
    print('sealed256',flush=True)

if __name__=='__main__':
    try: globals()[sys.argv[1]]()
    except BaseException:
        put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True));raise
