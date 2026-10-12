"""Finite CPU geometry readback of sealed fits; no GPU/model or raw GT loader."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load
BASE=ROOT/'artifacts/stvg_opd_optimization_geometry_v1'
PUB=ROOT/'results/stvg_opd_optimization_geometry/2026-10-12'
OLD=ROOT/'artifacts/stvg_opd_mechanism_review_v1'
ROWS=ROOT/'results/stvg_opd_mechanism_review/2026-10-11'
ATTACHMENT=Path('<LOCAL_AUTHORIZATION_ATTACHMENT>')

def rows(name):
    with gzip.open(ROWS/name,'rt') as f:return [json.loads(s) for s in f]

def cosine(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1)
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return float(np.dot(a,b)/(na*nb)) if na>0 and nb>0 else None

def direct_gradient(boxes,evidence):
    import torch
    from methods.decota_final_simplified_v1.objectives import generalized_iou
    p=torch.as_tensor(np.array(boxes),dtype=torch.float64).requires_grad_(True)
    e=torch.as_tensor(np.array(evidence),dtype=torch.float64)
    loss=(5*(p-e).abs().sum(-1)+2*(1-generalized_iou(p,e))).mean()
    g=torch.autograd.grad(loss,p)[0]
    assert torch.isfinite(g).all()
    return g.detach().numpy()*(p.detach().numpy()*(1-p.detach().numpy()))

def contracts():
    import torch
    torch.set_num_threads(1)
    # All enclosure/intersection edges distinct: finite differences do not
    # identify a unique derivative at GIoU min/max contact ties.
    b=np.array([[.40,.40,.21,.18],[.68,.60,.14,.22]])
    e=np.array([[.43,.45,.22,.20],[.64,.625,.16,.235]])
    z=np.log(b/(1-b));analytic=direct_gradient(b,e)
    from methods.decota_final_simplified_v1.objectives import generalized_iou
    def value(x):
        p=torch.sigmoid(torch.tensor(x,dtype=torch.float64));q=torch.tensor(e,dtype=torch.float64)
        return float((5*(p-q).abs().sum(-1)+2*(1-generalized_iou(p,q))).mean())
    for i in np.ndindex(z.shape):
        a=z.copy();c=z.copy();a[i]+=1e-6;c[i]-=1e-6
        assert abs((value(a)-value(c))/2e-6-analytic[i])<2e-7
    assert cosine([0,0],[1,2]) is None
    assert abs(cosine([1,2],[2,4])-1)<1e-15
    assert abs(cosine([1,2],[-2,-4])+1)<1e-15
    assert abs(cosine([1,0],[0,1]))<1e-15
    w=np.full(32,1/32);assert 1/np.dot(w,w)==32
    assert float(np.var(np.zeros(32)))==0
    return dict(status='pass',valid_contracts=14,finite_difference_coordinates=8,
                scope='same-output Direct chart gradient and undefined cosine arithmetic; not GPU qualification',GT_read=False)

def prepare():
    if (BASE/'CPU_RUNTIME.json').exists():return verify()
    assert read(OLD/'ROOT_DIAGNOSTIC_CLOSING_RECEIPT.json')['status']=='complete'
    bindings=read(OLD/'PRIVATE_INPUT_BINDINGS.json')['records'];assert len(bindings)==3584
    write(BASE/'CPU_CONTRACTS.json',contracts())
    write(BASE/'DESIGN_LOCK.json',dict(status='fixed_before_new_diagnostic_and_common_state_GPU',
      authorization_attachment_sha256=sha(ATTACHMENT),old_matched_cells=1792,old_complete_fits=3584,
      common_state_logical_fits=5376,new_formal_Direct_fits=3584,exact_original_Full_references=1792,
      common_initial_history='exact_original_Full_initial_per_cell',new_models=0,main_method_changed=False,
      source_checkpoints={'vidstg':'5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83',
                         'hc2':'47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036'},
      bootstrap=10000,bootstrap_seed=20261012,GT_before_new_global=False,historical_exposure=True,time=time.time()))
    files=['protocols/stvg_opd_optimization_geometry_v1.md','scripts/analyze_stvg_opd_optimization_geometry_v1.py',
      'methods/decota_final_simplified_v1/objectives.py','methods/CURRENT_METHOD.json',
      'methods/decota_spatial_opd_v1/configs.json',str((OLD/'PRIVATE_INPUT_BINDINGS.json').relative_to(ROOT)),
      str((OLD/'ROOT_DIAGNOSTIC_CLOSING_RECEIPT.json').relative_to(ROOT))]
    files += [str((ROWS/f).relative_to(ROOT)) for f in ['CELL_ROWS.jsonl.gz','QUALITY_ROWS.jsonl.gz','FRAME_ROWS.jsonl.gz']]
    write(BASE/'CPU_RUNTIME.json',dict(status='locked_before_CPU_readback',pins={p:sha(ROOT/p) for p in files},
      design_sha256=sha(BASE/'DESIGN_LOCK.json'),contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),GT_read=False,time=time.time()))
    return verify()

def verify():
    r=read(BASE/'CPU_RUNTIME.json');assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
    assert sha(BASE/'CPU_CONTRACTS.json')==r['contracts_sha256']
    for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
    return r

def summarize(cell_rows):
    groups=collections.defaultdict(list)
    fields=['ESS_mean','reward_variance_mean','weighted_target_displacement_mean','output_gradient_cosine_mean',
      'Direct_output_gradient_norm_mean','OPD_output_gradient_norm_mean','Adam_parameter_movement_mean',
      'observed_output_movement_mean','unobserved_output_movement_mean']
    for c in cell_rows:groups[(c['dataset'],c['setting'])].append(c)
    result={}
    for (ds,setting),rr in groups.items():
        for population in ['all_historical','excluding_development']:
            part=[c for c in rr if population=='all_historical' or not c['development']]
            parents=collections.defaultdict(list)
            for c in part:parents[c['source_id']].append(c)
            out={}
            for field in fields:
                vals=[]
                for pp in parents.values():
                    v=[p[field] for p in pp if p[field] is not None]
                    if v:vals.append(math.fsum(v)/len(v))
                if not vals:out[field]=dict(parents=0,mean=None,ci95=None);continue
                a=np.asarray(vals);rng=np.random.default_rng(20261012);boot=[]
                for _ in range(100):boot.extend(a[rng.integers(0,len(a),(100,len(a)))].mean(1).tolist())
                out[field]=dict(parents=len(a),mean=math.fsum(vals)/len(a),median=float(np.median(a)),ci95=np.quantile(boot,[.025,.975]).tolist())
            result[ds+'/'+setting+'/'+population]=dict(cells=len(part),parent_sources=len(parents),metrics=out)
    return dict(status='CPU_diagnostic_complete_not_common_state_GPU_result',statistics=result,
      output_chart_gradients=True,parameter_Jacobian_independently_replayed=False,
      old_independent_histories_not_causal=True,new_model_calls=0,new_GPU_fits=0,
      bootstrap=10000,bootstrap_seed=20261012,historical_exposure=True,unadjusted_CI=True)

def run():
    verify();import torch;torch.set_num_threads(1)
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    bindings=read(OLD/'PRIVATE_INPUT_BINDINGS.json')['records'];cells=rows('CELL_ROWS.jsonl.gz');quality=rows('QUALITY_ROWS.jsonl.gz')
    assert len(cells)==1792 and len(bindings)==3584
    key=lambda c:(c['dataset'],c['source_id'],c['condition'],c['order'],c['arrival'])
    qmap=collections.defaultdict(dict)
    for q in quality:qmap[key(q)][q['observation_index']]=q
    PUB.mkdir(parents=True,exist_ok=True);out_cells=[];rounds=observations=updates=undefined=bytes_read=0;start=time.time()
    with gzip.open(PUB/'ROUND_GEOMETRY.jsonl.gz','wt') as stream:
        for i,cell in enumerate(cells):
            bs=bindings[2*i:2*i+2];zs=[]
            for binding in bs:
                p=ROOT/binding['path'];ip=ROOT/binding['input_path']
                assert sha(p)==binding['sha256'] and sha(ip)==binding['input_sha256']
                bytes_read+=p.stat().st_size+ip.stat().st_size;zs.append(load(p))
            z=zs[0];fit=z['fit'];assert z['arm']=='on_policy' and not z['GT_read'] and not fit['GT_used']
            assert z['query_ordinal']==cell['query_ordinal'] and len(fit['rounds'])==fit['config']['steps']
            inp=load(ROOT/bs[0]['input_path']);expert=unpack_expert(inp['expert']);support=top1_support(expert)
            pos=[p for p,e in support];assert pos==fit['positions']
            evidence=np.asarray([e for p,e in support],dtype=np.float64)
            values=collections.defaultdict(list)
            for k,r in enumerate(fit['rounds']):
                rounds+=1
                if not pos:assert not r['updated'];continue
                rr=r['rollout'];w=rr['weights'].numpy().astype(float);reward=rr['rewards'].numpy().astype(float)
                samples=rr['samples'].numpy().astype(float);mu=r['mean_before'].numpy().astype(float)
                gopd=r['autograd_mean_gradient'].numpy().astype(float)
                boxes=fit['path'][k]['boxes'].numpy();gd=direct_gradient(boxes[pos],evidence)
                post=fit['path'][k+1]['boxes'].numpy();mask=np.ones(len(boxes),dtype=bool);mask[pos]=False
                u=fit['path'][k].get('update');pn=float(np.linalg.norm(u['raw'].numpy().astype(float))) if u else 0.
                if u:updates+=1
                targets=np.einsum('jm,jmd->jd',w,samples)
                for j in range(len(pos)):
                    observations+=1;q=qmap[key(cell)].get(j)
                    co=cosine(gopd[j],gd[j]);undefined+=co is None
                    rec={n:cell[n] for n in ['dataset','source_id','condition','order','arrival','setting','development']}
                    rec.update(round_index=k,observation_index=j,ESS=1/math.fsum(float(a*a) for a in w[j]),
                      reward_variance=float(np.var(reward[j])),weighted_target_displacement=float(np.linalg.norm(targets[j]-mu[j])),
                      output_gradient_cosine=co,OPD_output_gradient_norm=float(np.linalg.norm(gopd[j])),
                      Direct_output_gradient_norm=float(np.linalg.norm(gd[j])),Adam_parameter_movement=pn,
                      observed_output_movement=float(np.linalg.norm(post[pos]-boxes[pos])),
                      unobserved_output_movement=float(np.linalg.norm(post[mask]-boxes[mask])) if mask.any() else None,
                      boundary_coordinates=int(np.sum((boxes[pos[j]]<=0)|(boxes[pos[j]]>=1))),
                      quality=q['quality'] if q else 'GT_unavailable',first_reachability=q['reachability'] if q else 'unavailable',
                      historical_Full_minus_Direct_current=q['Full_minus_Direct_current'] if q else None,
                      historical_Before_difference=q['Before_difference'] if q else None,
                      different_online_histories=True,parameter_cosine_not_claimed=True)
                    stream.write(json.dumps(rec,allow_nan=False)+'\n')
                    for n,v in rec.items():
                        if n in ['ESS','reward_variance','weighted_target_displacement','output_gradient_cosine','Direct_output_gradient_norm','OPD_output_gradient_norm','Adam_parameter_movement','observed_output_movement','unobserved_output_movement'] and v is not None:values[n].append(v)
            c={n:cell[n] for n in ['dataset','source_id','condition','order','arrival','query_ordinal','setting','development']}
            c.update({n+'_mean':math.fsum(values[n])/len(values[n]) if values[n] else None for n in ['ESS','reward_variance','weighted_target_displacement','output_gradient_cosine','Direct_output_gradient_norm','OPD_output_gradient_norm','Adam_parameter_movement','observed_output_movement','unobserved_output_movement']})
            c['no_admission']=not pos;c['complete_original_Full_sha256']=bs[0]['sha256'];c['complete_original_Direct_sha256']=bs[1]['sha256'];out_cells.append(c)
            if i%32==0:
                status(BASE/'CPU_STATUS.json',dict(status='running_sealed_fit_CPU_diagnostic',done=i+1,total=1792,rounds=rounds,CPU_only=True,new_model_calls=0,GT_read='existing_postseal_anonymous_labels_only',pid=os.getpid(),time=time.time()))
                print('GEOMETRY_CPU_PROGRESS',i+1,1792,rounds,flush=True)
    with gzip.open(PUB/'CELL_GEOMETRY.jsonl.gz','wt') as f:
        for c in out_cells:f.write(json.dumps(c,allow_nan=False)+'\n')
    write(PUB/'SUMMARY.json',summarize(out_cells));verify()
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_independent_root_readback_plot_view_publication',scope='sealed_cache_geometry_diagnostic_only',
      cells=len(cells),complete_fit_byte_readbacks=len(bindings),complete_rounds=rounds,round_observation_records=observations,
      actual_original_Adam_updates=updates,undefined_output_cosines=undefined,bytes_read=bytes_read,CPU_only=True,new_model_calls=0,new_GPU_fits=0,
      common_state_experiment_complete=False,GT_read='existing_postseal_anonymous_labels_only',elapsed_seconds=time.time()-start,
      files={str(p.relative_to(ROOT)):sha(p) for p in PUB.iterdir() if p.is_file()},time=time.time()))
    status(BASE/'CPU_STATUS.json',dict(status='pending_independent_root_readback_plot_view_publication',done=1792,total=1792,CPU_only=True,new_GPU_fits=0,time=time.time()))
    print('GEOMETRY_CPU_COMPLETE',len(cells),rounds,observations,flush=True)

if __name__=='__main__':
    try:prepare() if len(sys.argv)>1 and sys.argv[1]=='prepare' else run()
    except BaseException:
        p=BASE/'CPU_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc());raise
