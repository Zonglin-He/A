"""Bounded cached simplification study; current full remains immutable/running."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from full import read,sha,atomic,identity,check,PROVIDER,RUN
from fastmath import positive_hull,linear_step,linear_evidence
from lightspatial import known_grid,box_links,build
from stvg.evidence import SourceEvidence,extract_visual_grids
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.temporal import probabilities,mass_table,final_mbr
from stvg.prototype import one_step,evidence
from stvg.prior import from_capture
spec=importlib.util.spec_from_file_location('cycle_study',ROOT/'studies/cycle_reliability_ablation/run.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
D=ROOT/'outputs/algorithm_simplification_v1';P=ROOT/'results/algorithm_simplification_v1';PARENT=ROOT/'outputs/cycle_reliability_ablation_v1';OLD=ROOT.parent/'stvg-native2d/outputs/native_prior_v1'
ARMS=('Source','Current','NoCycle','BoxLink','NearestBoxLink')
PAIRS=[(a,b)for a,b in [('NoCycle','Current'),('BoxLink','Current'),('BoxLink','NoCycle'),('NearestBoxLink','Current'),('NearestBoxLink','BoxLink'),('NearestBoxLink','Source'),('Current','Source')]]

def verify():
    p=read(D/'plan.json');assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    for f,h in p['code'].items():assert sha(ROOT/f)==h,f
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert not torch.cuda.is_initialized();return p

def prepare():
    assert not(D/'plan.json').exists();prod=check();parent=read(PARENT/'plan.json');assert read(PARENT/'state.json')['status']=='complete_scored';torch.set_num_threads(1)
    rng=np.random.default_rng(20260920);tests=[]
    for n in (6,7,16):
        grids=[]
        for off in (0,1):
            ids=list(range(off,n,2));ij=np.asarray(np.triu_indices(len(ids),1));v=rng.random(ij.shape[1]);v/=v.sum();grids.append(dict(ids=ids,ij=ij,p=v))
        ids,a=mass_table(grids);bids,b=positive_hull(grids);assert ids==bids and np.max(abs(a-b))<1e-13;tests.append(dict(n=n,max_hull_error=float(np.max(abs(a-b)))))
    q=np.triu(np.ones((9,9)),1);q/=q.sum();assert np.array_equal(linear_step(np.ones((9,5)),q)[0],q)
    z=rng.normal(size=(9,5));a,_=evidence(z,q);b,_=linear_evidence(z,q);assert np.max(abs(a-b))<1e-12
    code={f:sha(ROOT/f)for f in prod['code']}
    for folder in ('studies/algorithm_simplification','studies/cycle_reliability_ablation'):
        for f in (ROOT/folder).glob('*.py'):code[str(f.relative_to(ROOT))]=sha(f)
    inputs={str(f):sha(f)for f in [PARENT/'plan.json',PARENT/'sealed.json',PARENT/'score_receipt.json',D/'profile_current.json',D/'natural_runtime_cost.json',OLD/'plan.json',OLD/'sealed.json',RUN/'hc2/sealed.json',RUN/'plan.json']}
    for i in parent['items']:
        for k in ('raw','mask','capture','prediction'):inputs[i[k+'_path']]=i[k+'_sha256']
    p=dict(schema='algorithm_simplification_v1',authorization='User requests actual study of simpler formulas/less costly spatial propagation and other components; retain ongoing full unchanged.',
        primary='NearestBoxLink-Current',arms=ARMS,pairs=PAIRS,
        exact='Positive O(T^2) minstart/maxend hull mass by unique endpoint-owner cases, no cancellation/pruning; vector-mean RegionKernel instead of explicit pairwise kernel. Also direct source-grid extraction using explicit original encoder H,W rather than position factor search; exact tensor checks.',
        structural='One fixed nearest physical observation per unobserved node; tie earlier observation. Original bidirectional affinities and exact affine fit+validity; no cycle or fallback-to-farther-anchor selection. Uniform valid support, original extent/appearance/guard/unary coefficients and exact DP. Plain adjacent complete-box IoU link (.25 original coefficient), no adjacent feature/mask transport. Invalid fit -> E. No learned state, threshold/weight/anchor-count scan.',
        controls='Current original; NoCycle exact sealed parent alias; BoxLink keeps BOTH past/future anchors with only no-cycle and boxIoU link; NearestBoxLink is the ONLY predeclared structural candidate. Do not promote a control posthoc.',
        fixed_time='Same original kappa64 Full interval and Native2D region features/prior for every spatial arm. This estimates spatial replacement, not end-to-end rerun of changed ROI-to-time feedback.',
        scope='Same67 availability-selected exposed official-HC-compatible cached inputs. Exact math audited on officialHC1902 plus existingVid128 (2030 unique queries); no model/RGB/teacher/GT in exact phase. Spatial labels only after all67 predictions seal.',
        timing='One common source-grid extraction/seed input per query, one warmup before measured three alternating-repeat direct constructors. Current includes adjacent-affinity construction+geometry; BoxLink/Nearest implementations never build and discard original geometry. Known-shape grid extraction paired exact and timed separately. Cached CPU component costs only, no uncached deployment/GPU speed claim.',
        budget=dict(spatial_queries=67,exact_math_queries=2030,new_model=0,new_RGB=0,new_expert=0,new_training=0,GPU=0),no_production_change=True,no_selection=True,code=code,inputs=inputs,items=parent['items'],job=parent['job'],seed=20260920,production_fingerprint=prod['fingerprint'])
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(D/'plan.json',p)
    for f in code:
        dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    atomic(P/'protocol.json',{k:v for k,v in p.items()if k not in('items','job','inputs')});atomic(D/'tests.json',dict(positive_hull_small_grid_reference=tests,constant_representation_exact_noop=True,vector_kernel_small_grid_error=float(np.max(abs(a-b))),GT_read=False));atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print('prepared',p['fingerprint'],flush=True)

def exact():
    p=verify();assert not(D/'exact_receipt.json').exists();torch.set_num_threads(1);atomic(D/'exact_worker.json',identity());started=time.perf_counter();seal=read(RUN/'hc2/sealed.json');oldp=read(OLD/'plan.json');olds=read(OLD/'sealed.json');cases=[]
    for key,e in seal['files'].items():cases.append(dict(ds='hc2',path=str(RUN/'hc2'/e['capture_path']),sha=e['capture_sha256'],pred=str(RUN/'hc2'/e['path']),pred_sha=e['sha256'],arm='Full'))
    for i in oldp['items']:
        if i['dataset']=='vidstg':
            e=olds['files']['vidstg_'+i['alias']];cases.append(dict(ds='vidstg',path=i['path'],sha=i['sha256'],features=i['feature_path'],features_sha=i['feature_sha256'],pred=str(OLD/e['path']),pred_sha=e['sha256'],arm='NativeFeedback'))
    assert len(cases)==2030;max_hull=max_post=max_L=0.;hist={};reps={};intervals_exact=0
    for n,c in enumerate(cases,1):
        assert sha(c['path'])==c['sha'] and sha(c['pred'])==c['pred_sha'];cap=torch.load(c['path'],map_location='cpu',weights_only=False)
        if'features'in c:
            assert sha(c['features'])==c['features_sha'];tr=torch.load(c['features'],map_location='cpu',weights_only=False);z=tr['features']['Native2D']if isinstance(tr['features'],dict)else tr['features']
        else:z=cap['content_features']
        logs=cap.get('native_logits');logs=logs if logs is not None else[x[-1]for x in cap['logits']];grids=probabilities(cap['records'],logs);ids,q0=mass_table(grids);fids,fq=positive_hull(grids);assert ids==fids
        old,st=one_step(z,q0);new,ns=linear_step(z,fq);err=float(np.max(abs(q0-fq)));pe=float(np.max(abs(old-new)));le=float(np.max(abs(st['L']-ns['L'])));max_hull=max(max_hull,err);max_post=max(max_post,pe);max_L=max(max_L,le);assert err<1e-12 and pe<1e-11 and le<1e-11
        iv=final_mbr(ids,old)['interval'];nv=final_mbr(ids,new)['interval'];assert iv==nv
        with np.load(c['pred'],allow_pickle=False)as a:assert np.array_equal(iv,a[c['arm']+'_interval'])
        intervals_exact+=1;hist[len(ids)]=hist.get(len(ids),0)+1;reps.setdefault(len(ids),(grids,z.copy(),q0.copy()))
        if n%200==0:print('exact',n,flush=True)
    sizes=sorted(reps);chosen=sorted({sizes[0],min(sizes,key=lambda x:abs(x-65)),min(sizes,key=lambda x:abs(x-128)),sizes[-1]});timing={}
    for n in chosen:
        grids,z,q0=reps[n];mass_table(grids);positive_hull(grids);one_step(z,q0);linear_step(z,q0);vals={k:[]for k in('original_hull','positive_hull','original_prototype','vector_prototype')}
        for rep in range(3):
            for key,fn in [('original_hull',lambda:mass_table(grids)),('positive_hull',lambda:positive_hull(grids)),('original_prototype',lambda:one_step(z,q0)),('vector_prototype',lambda:linear_step(z,q0))]:
                s=time.perf_counter();fn();vals[key].append(time.perf_counter()-s)
        timing[n]={k:float(np.median(v))for k,v in vals.items()};timing[n]['original_interval_pair_product']=len(grids[0]['p'])*len(grids[1]['p']);timing[n]['physical_matrix_cells']=n*n
    verify();assert intervals_exact==2030 and not torch.cuda.is_initialized();out=dict(status='passed',queries=2030,HC1902=1902,Vid128=128,source_bindings_sha256=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),all_intervals_exact=2030,unchanged_spatial_outputs_alias='Same interval and immutable same geometry/DP deterministically return same existing boxes; not new label scoring or network inference.',posterior_bitwise_claim=False,FP64_operation_order_differs=True,max_native_hull_error=max_hull,max_updated_posterior_error=max_post,max_frame_evidence_error=max_L,frame_count_histogram=hist,paired_median_CPU_seconds=timing,seconds=time.perf_counter()-started,new_models=0,GT_read=False,production_unchanged=True)
    atomic(P/'EXACT_MATH.json',out);atomic(D/'exact_receipt.json',dict(status='passed',public_sha256=sha(P/'EXACT_MATH.json'),**identity()));print(json.dumps(out),flush=True)

def predict():
    p=verify();assert not(D/'sealed.json').exists();torch.set_num_threads(1);atomic(D/'worker.json',identity());atomic(D/'state.json',dict(status='predicting',**identity()))
    parent=read(PARENT/'sealed.json');files={};timings=[];stats=[];start=time.perf_counter()
    for n,i in enumerate(p['items'],1):
        cap=torch.load(i['capture_path'],map_location='cpu',weights_only=False);raw=torch.load(i['raw_path'],map_location='cpu',weights_only=False);pack=torch.load(i['mask_path'],map_location='cpu',weights_only=False);j=int(pack['selected_query']);ids=cap['frame_ids'];pos=cap['positions8'];g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry'])
        features,valid,_,shape,_=extract_visual_grids(raw['spatial_inputs'],ids);ff,vv=known_grid(raw['spatial_inputs'],ids,g.height,g.width);assert torch.equal(features,ff)and torch.equal(valid,vv)
        ev=C.load_evidence(i,cap);h=C.reweight(g,ev,use_cycle=False);two=build(features,valid,ids,pos,ev.seeds,cap['E'],g.native_boxes,nearest=False);near=build(features,valid,ids,pos,ev.seeds,cap['E'],g.native_boxes,nearest=True)
        assert torch.equal(two.boxes,h.boxes)and torch.equal(two.valid,h.valid)and torch.equal(two.projection,h.projection)and torch.equal(two.guard,h.guard)
        control=copy.deepcopy(h);control.link=box_links(control.boxes);e=parent['files'][i['source_key']]
        with np.load(PARENT/e['path'],allow_pickle=False)as old:
            span=old['Current_interval'].copy();arr={a:old[a].copy()for a in('Source','Current','NoCycle')};arr.update({a+'_interval':old[a+'_interval'].copy()for a in('Source','Current','NoCycle')});arr.update(frame_ids=old['frame_ids'].copy(),positions8=old['positions8'].copy())
            oc=evidence_dp(control.for_interval(span));tc=evidence_dp(two.for_interval(span));nc=evidence_dp(near.for_interval(span));assert torch.equal(oc['final'],tc['final'])and torch.equal(oc['actions'],tc['actions'])
            arr.update(BoxLink=tc['final'].numpy(),BoxLink_interval=span,NearestBoxLink=nc['final'].numpy(),NearestBoxLink_interval=span)
        times={k:[]for k in('Current_core','BoxLink_core','Nearest_core','Original_grid_adapter','Known_grid_adapter')}
        # Constructors above are the warmups; alternate measurement order.
        for rep in range(3):
            def current():
                ee=SourceEvidence.from_arrays(features,valid,ids,pos,ev.seeds);return SpatialGeometry(ee,cap['E'],g.native_boxes)
            calls=[('Current_core',current),('BoxLink_core',lambda:build(features,valid,ids,pos,ev.seeds,cap['E'],g.native_boxes,nearest=False)),('Nearest_core',lambda:build(features,valid,ids,pos,ev.seeds,cap['E'],g.native_boxes,nearest=True)),('Original_grid_adapter',lambda:extract_visual_grids(raw['spatial_inputs'],ids)),('Known_grid_adapter',lambda:known_grid(raw['spatial_inputs'],ids,g.height,g.width))]
            for key,fn in calls if rep%2==0 else reversed(calls):
                t=time.perf_counter();fn();times[key].append(time.perf_counter()-t)
        timings.append(dict(query=f'H{n:03d}',median_seconds={k:float(np.median(v))for k,v in times.items()}));stats.append(dict(query=f'H{n:03d}',frames=len(ids),original_adjacent_affinity=len(ids)-1,BoxLink_affinity=two.metadata['affinity_calls'],Nearest_affinity=near.metadata['affinity_calls'],Current_NoCycle_candidate_geometry_exact=True,BoxLink_direct_constructor_matches_cached_control=True))
        key=f'H{n:03d}';f=D/'predictions'/(key+'.npz');f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arr);files[i['source_key']]=dict(alias=key,path=str(f.relative_to(D)),sha256=sha(f));atomic(D/'progress.json',dict(completed=n,total=67,GT_read=False))
        if n%16==0:print('predicted',n,flush=True)
    verify();assert not torch.cuda.is_initialized();atomic(D/'sealed.json',dict(status='sealed',queries=67,fingerprint=p['fingerprint'],files=files,GT_read=False,timings=timings,checks=stats,seconds=time.perf_counter()-start));atomic(D/'state.json',dict(status='sealed_waiting_independent_CPU_score',fingerprint=p['fingerprint']));print('sealed',flush=True)

def summarize(rows):
    from full_score import cluster
    fs=rows[0]['values']['Source'];means={a:{f:float(np.mean([x['values'][a][f]for x in rows if x['values'][a][f]is not None]))if any(x['values'][a][f]is not None for x in rows)else None for f in fs}for a in ARMS};risk={}
    for a,b in PAIRS:
        v=[x['values'][a]['v']-x['values'][b]['v']<-.05 for x in rows];s=[x['values'][a]['unobserved_actual_sIoU']-x['values'][b]['unobserved_actual_sIoU']<-.05 for x in rows];risk[a+'-'+b]=dict(v_count=sum(v),unobserved_s_count=sum(s),union_count=sum(i or j for i,j in zip(v,s)))
    return dict(queries=len(rows),parents=len({x['parent']for x in rows}),means=means,paired={a+'-'+b:cluster(rows,a,b)for a,b in PAIRS},severe=risk)

def score():
    from full_score import metric
    p=verify();s=read(D/'sealed.json');assert s['queries']==67 and not s['GT_read'] and s['fingerprint']==p['fingerprint'] and not(D/'score_receipt.json').exists();torch.set_num_threads(1);atomic(D/'score_worker.json',identity())
    for e in s['files'].values():assert sha(D/e['path'])==e['sha256']
    lab=PROVIDER/p['job']['labels'];assert sha(lab)==p['job']['labels_sha256'];truth={x['id']:x for x in map(json.loads,lab.read_text().splitlines())if x['dataset']=='hc2'};old={x['query']:x for x in read(ROOT/'results/cycle_reliability_ablation_v1/report.json')['rows']};rows=[];checks=0;maxerr=0.;start=time.perf_counter();signed=['query\tparent\tmethod\tbaseline\tdelta_v_pp\tdelta_unobserved_s_pp'];severe=[signed[0]]
    for i in p['items']:
        e=s['files'][i['source_key']];r=i['row'];gt=truth[i['source_key']];boxes={int(k):v for k,v in gt['boxes'].items()};values={}
        with np.load(D/e['path'],allow_pickle=False)as z:
            for a in ARMS:values[a],err=metric(z[a],r,boxes,gt['interval'],z[a+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err)
        for a in('Source','Current','NoCycle'):assert values[a]==old[e['alias']]['values'][a]
        assert len({values[a]['t']for a in('Current','NoCycle','BoxLink','NearestBoxLink')})==1
        x=dict(query=e['alias'],parent=old[e['alias']]['parent'],duration_group=old[e['alias']]['duration_group'],values=values);rows.append(x)
        for a,b in PAIRS:
            dv=100*(values[a]['v']-values[b]['v']);ds=100*(values[a]['unobserved_actual_sIoU']-values[b]['unobserved_actual_sIoU']);line=f"{e['alias']}\t{x['parent']}\t{a}\t{b}\t{dv:.12g}\t{ds:.12g}";signed.append(line)
            if min(dv,ds)<-5:severe.append(line)
    groups=dict(all67=summarize(rows));groups['all67']['duration']={tag:summarize([x for x in rows if x['duration_group']==tag])for tag in('short_le5s','long_gt5s')}
    assert checks==335 and maxerr<2e-10 and not torch.cuda.is_initialized();timing={k:dict(mean_seconds=float(np.mean([r['median_seconds'][k]for r in s['timings']])),median_seconds=float(np.median([r['median_seconds'][k]for r in s['timings']])))for k in s['timings'][0]['median_seconds']}
    out=dict(status='complete_scored',groups=groups,rows=rows,timing=timing,timing_scope=p['timing'],all67_grid_features_validity_exact=True,all67_full_time_equal=True,all67_direct_BoxLink_constructor_matches_cached_control=True,checks=checks,max_metric_error=maxerr,seconds=time.perf_counter()-start,scope=p['scope'],no_selection=True,production_unchanged=True,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'))
    atomic(P/'report.json',out);(P/'signed.tsv').write_text('\n'.join(signed)+'\n');(P/'severe.tsv').write_text('\n'.join(severe)+'\n');verify();atomic(D/'score_receipt.json',dict(status='scored',report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),**identity()));atomic(D/'state.json',dict(status='complete_scored',fingerprint=p['fingerprint']));print(json.dumps(dict(means=groups['all67']['means'],paired=groups['all67']['paired']['NearestBoxLink-Current'],timing=timing)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','exact','predict','score'));a=ap.parse_args();globals()[a.command]()
