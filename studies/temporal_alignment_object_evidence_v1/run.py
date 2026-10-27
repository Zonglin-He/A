"""Frozen cached temporal-vector permutation; no new acquisition or model."""
from pathlib import Path
import os,sys,json,hashlib,time,shutil,importlib.util,argparse,traceback,csv
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
D=R/'outputs/temporal_alignment_object_evidence_v1';P=R/'results/temporal_alignment_object_evidence_v1'
spec=importlib.util.spec_from_file_location('_alignment_parent',R/'studies/temporal_feedback_risk_v1/common.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
selected=parent.selected;read=parent.read;sha=parent.sha;atomic=parent.atomic;identity=parent.identity;digest=parent.digest
ARMS=['CurrentAligned','FixedTemporalPermutation'];SEED=20261027

def array_hash(x):
    import numpy as np
    if hasattr(x,'detach'):x=x.detach().cpu().numpy()
    x=np.asarray(x);assert x.dtype.kind!='O'
    h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes());return h.hexdigest()

def state_hash(x):
    import numpy as np
    if isinstance(x,np.generic):return digest(x.item())
    if isinstance(x,dict):return digest({str(k):state_hash(v) for k,v in sorted(x.items(),key=lambda a:str(a[0]))})
    if isinstance(x,(list,tuple)):return digest([state_hash(v) for v in x])
    if isinstance(x,np.ndarray) or hasattr(x,'detach'):return array_hash(x)
    return digest(x)

def rows(ds):return read(D/'selection.json')[ds]

def validate():
    p=read(D/'plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for path,h in p[field].items():assert sha(R/path)==h,path
    assert read(R/p['predecessor_path'])['status']=='complete_published_matched_pipeline_timing_v1'
    selected.validate();return p

def resources(start):
    assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>=80*2**30
    assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
    assert sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30

def prepare():
    import numpy as np
    assert not(D/'plan_v1.json').exists()
    predecessor=R/'outputs/matched_pipeline_timing_v1/complete_published_v1.json'
    assert read(predecessor)['status']=='complete_published_matched_pipeline_timing_v1'
    completion=read(predecessor)['completion_path'];assert sha(completion)==read(predecessor)['completion_sha256']
    assert read(completion)['fresh_independent_bare_all_blob_bytes_verified']
    selection=read(R/'outputs/matched_pipeline_timing_v1/selection.json')
    atomic(D/'selection.json',selection);permutations={};inventory={};start=time.perf_counter()
    for ds,rr in selection.items():
        assert len(rr)==128 and len({row['id'] for row in rr})==128
        permutations[ds]={};inventory[ds]={}
        for row in rr:
            resources(start);key=row['id'];c,proof=parent.source(ds,row);z=np.asarray(c['features']['AuxE_ROI']);n=len(z)
            assert n==len(c['ids']) and z.ndim==2 and np.isfinite(z).all() and not c['GT_read']
            perm=sorted(range(n),key=lambda i:(hashlib.sha256(f'{SEED}|{ds}|{key}|{i}'.encode('utf-8')).digest(),i))
            rotated=False
            if n>1 and perm==list(range(n)):perm=perm[1:]+perm[:1];rotated=True
            permutations[ds][key]=dict(source_indices=perm,identity_fallback_left_rotation=rotated,nonintervened=n<2)
            current=read(selected.D/f'components_{ds}_sealed.json')['files'][key]
            assert sha(selected.D/current['path'])==current['sha256'] and sha(selected.D/current['receipt_path'])==current['receipt_sha256']
            inventory[ds][key]=dict(feature_sha256=array_hash(z),nodes=n,channels=z.shape[1],dtype=str(z.dtype),q0_sha256=array_hash(c['q0']),
                                    fixed_spatial_input_sha256=state_hash(dict(Source=c['Source'],positions8=c['pos'],ids=c['ids'],geometry=c['geometries']['main'])),
                                    compact_sha256=proof['compact_sha256'],parent_prediction_sha256=current['sha256'],parent_receipt_sha256=current['receipt_sha256'])
    atomic(D/'permutation_manifest_v1.json',permutations);atomic(D/'cached_input_inventory_v1.json',inventory)
    inputs=[predecessor,Path(completion),D/'selection.json',D/'permutation_manifest_v1.json',D/'cached_input_inventory_v1.json',
            R/'outputs/matched_pipeline_timing_v1/plan_v2.json',selected.D/'plan.json']
    for ds in selection:inputs += [selected.PREV/'spatial'/f'{ds}_sealed.json',selected.D/f'components_{ds}_sealed.json',selected.D/f'components_{ds}_reporting_receipt.json']
    code=[Path(__file__),R/'studies/temporal_feedback_risk_v1/common.py',R/'studies/shared_corrected_roi_v1/shared_common.py',R/'stvg/prototype.py',R/'stvg/spatial.py',R/'stvg/current_shared_roi.py',R/'tools/full_score.py']
    p=dict(schema='cached_temporal_vector_permutation_v1',arms=ARMS,queries_per_domain=128,seed=SEED,
           intervention='Reassign complete cached E-ROI vectors to fixed temporal positions by SHA256 order. Only temporal input vectors change; spatial geometry/candidates/unary/link features remain fixed. Refit prototypes, evidence, posterior, legal jointMAP and final own-interval DP.',
           permutation_rule='UTF8 20261027|dataset|original_query_id|source_index SHA256 ascending; source_index breaks ties; identity N>1 leftrotate1; N<2 explicit nonintervened, never dropped',
           params=dict(kappa=0,beta=1,rounds=1,no_first_aux_DP=True),primary='CurrentAligned minus FixedTemporalPermutation t/v pp;128parent paired10000 bootstrap seed20260920 pointwise95%',
           limitations='Previously exposed development panel. Supports temporal correspondence conditional on this intervention; not action semantics or self-reinforcement causality. Temporal weighted prototypes may change by design.',
           predecessor_path=str(predecessor.relative_to(R)),code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs},
           new_Source=0,new_expert=0,new_model=0,new_optimizer=0,new_acquisition_authorized=False)
    p['fingerprint']=digest(p);atomic(D/'plan_v1.json',p)
    atomic(P/'protocol_v1.json',{k:v for k,v in p.items()if k!='input_sha256'})
    atomic(D/'active_execution_v1.json',dict(status='prepared_frozen_pending_four_case_preflight',plan_fingerprint=p['fingerprint'],worker=identity()))
    print(json.dumps(dict(plan=p['fingerprint'],cached_queries=256,new_acquisition=0)))

def derive(ds,row):
    import numpy as np,torch
    from stvg.prototype import one_step
    from stvg.current_shared_roi import map_interval
    from stvg.spatial import SpatialGeometry,evidence_dp
    c,proof=parent.source(ds,row);key=row['id'];inv=read(D/'cached_input_inventory_v1.json')[ds][key];z=np.asarray(c['features']['AuxE_ROI']);q0=np.asarray(c['q0']);ids=c['ids']
    fixed=dict(Source=c['Source'],positions8=c['pos'],ids=ids,geometry=c['geometries']['main']);before=state_hash(fixed)
    assert before==inv['fixed_spatial_input_sha256'] and array_hash(z)==inv['feature_sha256'] and array_hash(q0)==inv['q0_sha256'] and proof['compact_sha256']==inv['compact_sha256']
    perm=np.array(read(D/'permutation_manifest_v1.json')[ds][key]['source_indices'],dtype=np.int64)
    assert np.array_equal(np.sort(perm),np.arange(len(z)));shuffled=z[perm].copy();assert np.array_equal(shuffled[np.argsort(perm)],z)
    g=SpatialGeometry.from_cached_geometry(c['geometries']['main']);a=dict(frame_ids=np.asarray(ids),positions8=np.asarray(c['pos']));posterior_hash={}
    for name,x in zip(ARMS,[z,shuffled]):
        q,_=one_step(x,q0,kappa=0,beta=1);iv=np.asarray(map_interval(ids,q));a[name]=evidence_dp(g.for_interval(iv))['final'].numpy();a[name+'_interval']=iv;posterior_hash[name]=array_hash(q)
    assert state_hash(fixed)==before and array_hash(z)==inv['feature_sha256'] and array_hash(q0)==inv['q0_sha256']
    e=read(selected.D/f'components_{ds}_sealed.json')['files'][key]
    assert sha(selected.D/e['path'])==inv['parent_prediction_sha256'] and sha(selected.D/e['receipt_path'])==inv['parent_receipt_sha256']
    with np.load(selected.D/e['path'])as old:assert np.array_equal(a['CurrentAligned'],old['FullMAP'])and np.array_equal(a['CurrentAligned_interval'],old['FullMAP_interval'])
    displacement=np.abs(perm-np.arange(len(perm)));physical=np.abs(np.asarray(ids)[perm]-np.asarray(ids))/row['metadata']['fps']
    diag=dict(nodes=len(z),fixed_points=int((displacement==0).sum()),mean_absolute_index_displacement=float(displacement.mean()),mean_absolute_seconds_displacement=float(physical.mean()),max_absolute_seconds_displacement=float(physical.max()),nonintervened=len(z)<2,vector_multiset_inverse_exact=True,fixed_spatial_inputs_exact=True,posterior_sha256=posterior_hash,aligned_current_arrays_exact=True)
    proof.update(current_prediction_sha256=inv['parent_prediction_sha256'],current_receipt_sha256=inv['parent_receipt_sha256'],fixed_spatial_input_sha256=before,feature_sha256=inv['feature_sha256'],permutation_sha256=array_hash(perm),new_Source=0,new_expert=0,new_optimizer=0,GT_read=False)
    assert all(np.isfinite(v).all()for v in a.values());return a,proof,diag

def preflight():
    import torch,numpy as np
    torch.set_num_threads(1);validate();out=[];start=time.perf_counter()
    for ds in ['hc2','vidstg']:
        for row in rows(ds)[:2]:
            a,p,m=derive(ds,row);b,pp,mm=derive(ds,row);assert p==pp and m==mm and set(a)==set(b) and all(np.array_equal(v,b[k])for k,v in a.items());out.append(dict(dataset=ds,aligned_identity_exact=True,multiset_inverse_exact=True,spatial_inputs_unchanged=True,repeat_all_arrays_exact=True,input_compact_sha256=p['compact_sha256']))
    rec=dict(status='passed_four_actual_cache_cases_identity_multiset_spatial_input_repeat_exact',cases=out,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_model=0,new_GT=0)
    atomic(D/'preflight_receipt_v1.json',rec);atomic(P/'preflight_receipt_v1.json',rec);print(rec['status'])

def predict(ds):
    import torch,numpy as np
    from stvg.runtime import guard
    sys.addaudithook(guard);torch.set_num_threads(1);plan=validate();assert read(D/'preflight_receipt_v1.json')['status'].startswith('passed_four_actual')
    folder=D/ds;assert not(folder/'sealed.json').exists();start=time.perf_counter();who=identity();atomic(folder/'predict_worker.json',who);files={}
    for i,row in enumerate(rows(ds),1):
        resources(start);t=time.perf_counter();a,proof,diag=derive(ds,row);key=row['id'];out=folder/'predictions'/(key+'.npz');rp=folder/'receipts'/(key+'.json');out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
        with out.with_suffix('.tmp').open('wb')as h:np.savez_compressed(h,**a)
        os.replace(out.with_suffix('.tmp'),out)
        with np.load(out)as z:assert set(z.files)==set(a) and all(np.array_equal(v,z[k])for k,v in a.items())
        e=dict(path=str(out.relative_to(D)),sha256=sha(out),provenance=proof,diagnostics=diag,worker=who,conditional_elapsed_seconds=time.perf_counter()-t,GT_read=False,plan_fingerprint=plan['fingerprint']);atomic(rp,e);files[key]=dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));atomic(folder/'progress.json',dict(completed=i,total=128,stage='predict',worker=who,GT_read=False))
    atomic(folder/'sealed.json',dict(queries=128,files=files,worker=who,GT_read=False,elapsed_wall_seconds=time.perf_counter()-start,plan_fingerprint=plan['fingerprint']))

def score(ds):
    import torch,numpy as np
    from full_score import metric
    torch.set_num_threads(1);validate();folder=D/ds;seal=read(folder/'sealed.json');assert seal['queries']==128 and not seal['GT_read'];truth=selected.labels(ds);out=[];checks=0;maxerr=0.;start=time.perf_counter()
    for i,row in enumerate(rows(ds),1):
        resources(start);e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256'] and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];values={}
        with np.load(D/e['path'])as z:
            inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
            for arm in ARMS:
                values[arm],err=metric(z[arm],inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err)
        out.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values,diagnostics=e['diagnostics']))
    assert len(out)==128 and len({v['parent']for v in out})==128
    atomic(folder/'metrics.json',dict(private_rows=out));rec=dict(status='scored_whole128_after_prediction_seal',queries=128,parents=128,official_checks=checks,max_error=maxerr,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,seal_sha256=sha(folder/'sealed.json'),metrics_sha256=sha(folder/'metrics.json'));atomic(folder/'score_receipt.json',rec);atomic(P/ds/'score_receipt.json',rec)

def audit(ds):
    import torch,numpy as np
    torch.set_num_threads(1);validate();folder=D/ds;seal=read(folder/'sealed.json');s=read(folder/'score_receipt.json');assert s['seal_sha256']==sha(folder/'sealed.json') and s['metrics_sha256']==sha(folder/'metrics.json');start=time.perf_counter();rr=read(folder/'metrics.json')['private_rows'];assert len(rr)==128
    for row in rows(ds):
        resources(start);e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256'] and sha(D/e['receipt_path'])==e['receipt_sha256'];a,p,m=derive(ds,row);assert p==e['provenance'] and m==e['diagnostics']
        with np.load(D/e['path'])as z:assert set(z.files)==set(a) and all(np.array_equal(v,z[k])for k,v in a.items())
    means={arm:{k:float(np.mean(vv))if(vv:=[x['values'][arm][k]for x in rr if x['values'][arm][k]is not None])else None for k in rr[0]['values'][arm]}for arm in ARMS}
    rec=dict(status='verified_whole128_all_arrays_inputs_parent_SHA_means',queries=128,parents=128,prediction_receipt_SHA_checks=256,all_arrays_cold_rederived_exact=True,means=means,score_sha256=sha(folder/'score_receipt.json'),metrics_sha256=sha(folder/'metrics.json'),seal_sha256=sha(folder/'sealed.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_GT=0)
    atomic(folder/'reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec);print(json.dumps(dict(dataset=ds,status=rec['status'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','predict','score','audit']);p.add_argument('--dataset',choices=['hc2','vidstg']);a=p.parse_args()
    try:globals()[a.command]()if a.command in ['prepare','preflight']else globals()[a.command](a.dataset)
    except BaseException:
        atomic(D/'failures'/(a.command+'_'+str(a.dataset)+'.json'),dict(error=traceback.format_exc(),worker=identity()));raise
