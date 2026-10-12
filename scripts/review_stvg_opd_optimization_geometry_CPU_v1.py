"""Independent NumPy Direct derivative and all recorded scalar aggregation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,load
BASE=ROOT/'artifacts/stvg_opd_optimization_geometry_v1'
PUB=ROOT/'results/stvg_opd_optimization_geometry/2026-10-12'

def derivative(p,e):
    p=np.asarray(p,float);e=np.asarray(e,float);n=len(p)
    pc=np.concatenate((p[:,:2]-p[:,2:]/2,p[:,:2]+p[:,2:]/2),axis=1)
    ec=np.concatenate((e[:,:2]-e[:,2:]/2,e[:,:2]+e[:,2:]/2),axis=1)
    jac=np.array([[1,0,-.5,0],[0,1,0,-.5],[1,0,.5,0],[0,1,0,.5]])
    ge=lambda a,b:np.where(a>b,1.,np.where(a<b,0.,.5))
    le=lambda a,b:ge(b,a)
    dw=le(pc[:,2],ec[:,2])[:,None]*jac[2]-ge(pc[:,0],ec[:,0])[:,None]*jac[0]
    dh=le(pc[:,3],ec[:,3])[:,None]*jac[3]-ge(pc[:,1],ec[:,1])[:,None]*jac[1]
    wr=np.minimum(pc[:,2],ec[:,2])-np.maximum(pc[:,0],ec[:,0]);hr=np.minimum(pc[:,3],ec[:,3])-np.maximum(pc[:,1],ec[:,1])
    w=np.maximum(wr,0);h=np.maximum(hr,0);dw*= (wr>=0)[:,None];dh*= (hr>=0)[:,None]
    inter=w*h;di=dw*h[:,None]+dh*w[:,None]
    da=np.zeros((n,4));da[:,2]=p[:,3];da[:,3]=p[:,2]
    ur=p[:,2]*p[:,3]+e[:,2]*e[:,3]-inter;du=(da-di)*(ur>=1e-7)[:,None];u=np.maximum(ur,1e-7)
    cw=np.maximum(pc[:,2],ec[:,2])-np.minimum(pc[:,0],ec[:,0]);ch=np.maximum(pc[:,3],ec[:,3])-np.minimum(pc[:,1],ec[:,1])
    dcw=ge(pc[:,2],ec[:,2])[:,None]*jac[2]-le(pc[:,0],ec[:,0])[:,None]*jac[0]
    dch=ge(pc[:,3],ec[:,3])[:,None]*jac[3]-le(pc[:,1],ec[:,1])[:,None]*jac[1]
    cr=cw*ch;dc=(dcw*ch[:,None]+dch*cw[:,None])*(cr>=1e-7)[:,None];c=np.maximum(cr,1e-7)
    dg=di/u[:,None]+du*(1/c-inter/u**2)[:,None]-dc*(u/c**2)[:,None]
    return ((5*np.sign(p-e)-2*dg)/n)*(p*(1-p))

def cosine(a,b):
    aa=float(np.linalg.norm(a));bb=float(np.linalg.norm(b))
    return float(np.dot(a,b)/(aa*bb)) if aa>0 and bb>0 else None

def run():
    import torch;torch.set_num_threads(1)
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    completion=read(BASE/'CPU_COMPLETION.json');assert completion['cells']==1792
    for p,h in completion['files'].items():assert sha(ROOT/p)==h
    bindings=read(ROOT/'artifacts/stvg_opd_mechanism_review_v1/PRIVATE_INPUT_BINDINGS.json')['records']
    count=checks=undefined=0;maxgrad=0.;cells=[];time0=time.time()
    with gzip.open(PUB/'ROUND_GEOMETRY.jsonl.gz','rt') as stream:
        for i in range(1792):
            b=bindings[2*i];assert sha(ROOT/b['path'])==b['sha256'] and sha(ROOT/b['input_path'])==b['input_sha256']
            fit=load(ROOT/b['path'])['fit'];inp=load(ROOT/b['input_path']);support=top1_support(unpack_expert(inp['expert']))
            pos=[p for p,e in support];e=np.asarray([e for p,e in support],float);values=collections.defaultdict(list)
            for k,r in enumerate(fit['rounds']):
                if not pos:continue
                p=fit['path'][k]['boxes'].numpy()[pos];direct=derivative(p,e);g=r['autograd_mean_gradient'].numpy().astype(float)
                rr=r['rollout'];w=rr['weights'].numpy().astype(float);rw=rr['rewards'].numpy().astype(float)
                samples=rr['samples'].numpy().astype(float);mu=r['mean_before'].numpy().astype(float)
                for j in range(len(pos)):
                    row=json.loads(next(stream));assert row['round_index']==k and row['observation_index']==j
                    count+=1;co=cosine(g[j],direct[j]);undefined+=co is None
                    expect=dict(ESS=1/math.fsum(v*v for v in w[j]),reward_variance=math.fsum((v-math.fsum(rw[j])/32)**2 for v in rw[j])/32,
                      weighted_target_displacement=float(np.linalg.norm(np.array([math.fsum(w[j,m]*samples[j,m,d] for m in range(32)) for d in range(4)])-mu[j])),
                      Direct_output_gradient_norm=float(np.linalg.norm(direct[j])),OPD_output_gradient_norm=float(np.linalg.norm(g[j])),output_gradient_cosine=co)
                    for name,v in expect.items():
                        a=row[name]
                        if v is None:assert a is None
                        else:
                            err=abs(a-v);assert err<2e-10*(1+abs(v)),(i,k,j,name,err)
                            if name=='Direct_output_gradient_norm':maxgrad=max(maxgrad,err)
                            values[name].append(a)
                        checks+=1
            cells.append(values)
            if i%256==0:print('GEOMETRY_ROOT_PROGRESS',i,count,checks,flush=True)
        assert next(stream,None) is None
    assert count==completion['round_observation_records'] and undefined==completion['undefined_output_cosines']
    # Recompute every cell mean from the continuous round stream independently.
    with gzip.open(PUB/'CELL_GEOMETRY.jsonl.gz','rt') as f:published=[json.loads(s) for s in f]
    for c,v in zip(published,cells):
        for name,values in v.items():
            expected=math.fsum(values)/len(values) if values else None
            assert c[name+'_mean']==expected;checks+=1
    summary=read(PUB/'SUMMARY.json')
    fields=['ESS_mean','reward_variance_mean','weighted_target_displacement_mean','output_gradient_cosine_mean',
       'Direct_output_gradient_norm_mean','OPD_output_gradient_norm_mean','Adam_parameter_movement_mean',
       'observed_output_movement_mean','unobserved_output_movement_mean']
    for key,z in summary['statistics'].items():
        ds,rest=key.split('/',1);setting,pop=rest.rsplit('/',1)
        rr=[c for c in published if c['dataset']==ds and c['setting']==setting and (pop=='all_historical' or not c['development'])]
        pp=collections.defaultdict(list)
        for c in rr:pp[c['source_id']].append(c)
        assert len(rr)==z['cells'] and len(pp)==z['parent_sources']
        for field in fields:
            vals=[math.fsum(a[field] for a in v if a[field] is not None)/sum(a[field] is not None for a in v) for v in pp.values() if any(a[field] is not None for a in v)]
            r=z['metrics'][field];assert r['parents']==len(vals)
            if not vals:assert r['mean'] is None;continue
            assert r['mean']==math.fsum(vals)/len(vals)
            a=np.asarray(vals);rng=np.random.default_rng(20261012);boot=[]
            for _ in range(100):boot.extend(np.sum(a[rng.integers(len(a),size=(100,len(a)))],axis=1)/len(a))
            assert np.max(np.abs(np.quantile(boot,[.025,.975])-r['ci95']))<2e-12;checks+=5
    write(BASE/'ROOT_CPU_DIAGNOSTIC_READBACK.json',dict(status='pass',scope='independent piecewise NumPy GIoU/Direct output gradient, every continuous round/cell and parent/bootstrap scalar',
      cells=1792,round_observation_records=count,independent_scalar_checks=checks,undefined_cosines=undefined,
      maximum_Direct_gradient_norm_difference=maxgrad,CPU_only=True,new_model_calls=0,new_GPU_fits=0,
      common_state_GPU_result=False,elapsed_seconds=time.time()-time0,time=time.time()))
    print('GEOMETRY_ROOT_PASS',count,checks,flush=True)

if __name__=='__main__':run()
