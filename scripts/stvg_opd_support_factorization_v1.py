"""P8: exact cached support factorization, no model calls or GPU fits."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import collections,gzip,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts.stvg_opd_temporal_guided_p7_common_v1 import BASE as OLD,A as OLD_A,B as OLD_B,OUT as OLD_OUT,verify as verify_old
BASE=ROOT/'artifacts/stvg_opd_support_factorization_v1'
OUT=ROOT/'results/stvg_opd_support_factorization/2026-10-12'
ARMS={'A':('Omega0','I0'),'B':('Omega0','I1'),'E':('Omega1','I0'),'C':('Omega1','I1')}
META=['public_cell','parent_id','dataset','order','previous_P6']
def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(v) for v in f]
def rows_write(path,values):
    assert not path.exists()
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:
            for v in values:f.write((json.dumps(v,allow_nan=False,separators=(',',':'))+'\n').encode())
def near(a,b):
    if isinstance(a,dict):assert set(a)==set(b);return sum(near(v,b[k]) for k,v in a.items())
    if isinstance(a,list):assert len(a)==len(b);return sum(near(x,y) for x,y in zip(a,b))
    if a is None or isinstance(a,(str,bool,int)):assert a==b,(a,b)
    else:assert abs(a-b)<2e-10*(1+abs(a)),(a,b)
    return 1
def contracts():
    old=np.array([.2,.7,.1]);new=np.array([.4,.3,.2]);w0=np.array([0,.5,.5]);w1=np.array([.25,.25,.25])
    a=float(w0@old);b=float(w1@old);e=float(w0@new);c=float(w1@new)
    assert abs((c-e)-(b-a)-float((w1-w0)@(new-old)))<1e-15
    assert abs((c-e)-(b-a)-((c-b)-(e-a)))<1e-15
    assert np.array_equal(old,old.copy()) and np.array_equal(new,new.copy())
    assert (c-c)-(a-a)==0
    assert np.mean([1.,3.])==2
    valid=[v for v in [None,0.,1.] if v is not None]
    assert len(valid)==2 and np.mean(valid)==.5
    return dict(status='pass',scope='synthetic exact interaction/alternative factorization/fixed path/no temporal spatial-IoU intervention/parent averaging/undefined partition',contracts=6,CPU_only=True,GT_read=False,new_GPU_fits=0)
def prepare():
    verify_old();assert read(OLD/'FINAL_COMPLETION.json')['status']=='complete'
    assert all(read(p)['status']=='pass' for p in [OLD_A/'ROOT_READBACK.json',OLD_B/'ROOT_READBACK.json'])
    assert not BASE.exists();BASE.mkdir(parents=True);OUT.mkdir(parents=True,exist_ok=False)
    d=read(OLD/'DESIGN_LOCK.json');records=d['records'];assert len(records)==512
    write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_new_E_readout_and_GT',records=records,cells=512,parents=256,
        targets=['hc2','vidstg'],orders=['order1','order2'],previous_P6_each=32,added_each=96,history_exposed=True,
        arms=ARMS,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,new_optimizer_steps=0,GT_read=False,time=time.time()))
    write(BASE/'CPU_CONTRACTS.json',contracts())
    names=['protocols/stvg_opd_support_factorization_v1.md','scripts/stvg_opd_support_factorization_v1.py',
        'scripts/audit_stvg_opd_support_factorization_public_v1.py',
        'scripts/review_stvg_opd_revised_p1_v2.py','vg_tta/tastvg_oracle_event5_v1.py',
        'vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py']
    original=[OLD/'DESIGN_LOCK.json',OLD/'RUNTIME.json',OLD/'FINAL_COMPLETION.json',OLD_A/'GLOBAL_READOUT_BARRIER.json',
        OLD_B/'GLOBAL_PREDICTION_BARRIER.json',OLD_A/'ROOT_READBACK.json',OLD_B/'ROOT_READBACK.json',
        OLD_OUT/'P7_A/ROWS.jsonl.gz',OLD_OUT/'P7_B/ROWS.jsonl.gz',OLD_OUT/'P7_B/DIAGNOSTIC_EXTRA_ROWS.jsonl.gz']
    write(BASE/'RUNTIME.json',dict(status='locked_cache_only',pins={n:sha(ROOT/n) for n in names},
        original_bindings={str(p.relative_to(ROOT)):sha(p) for p in original},design_sha256=sha(BASE/'DESIGN_LOCK.json'),
        CPU_contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),bootstrap=10000,seed=20261012,new_GPU_fits=0,
        new_model_calls=0,new_DINO_calls=0,GT_before_new_global=False,production_promoted=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_actual_cache_readout_seal',CPU_only=True,GT_read=False,time=time.time()))
def verify():
    verify_old();r=read(BASE/'RUNTIME.json')
    assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256'] and sha(BASE/'CPU_CONTRACTS.json')==r['CPU_contracts_sha256']
    for name,h in {**r['pins'],**r['original_bindings']}.items():assert sha(ROOT/name)==h,name
    return r
def seal():
    import torch
    torch.set_num_threads(2);verify();assert not torch.cuda.is_initialized()
    b=read(OLD_B/'GLOBAL_PREDICTION_BARRIER.json');a=read(OLD_A/'GLOBAL_READOUT_BARRIER.json');bindings={};coordinates=0
    for r in read(BASE/'DESIGN_LOCK.json')['records']:
        ap=OLD_A/'readouts'/f"{r['public_cell']:05}.pt";cp=OLD_B/'formal'/f"{r['public_cell']:05}/C_temporal_guided_OPD_repeat0.pt"
        assert sha(ap)==a['files'][str(ap.relative_to(ROOT))] and sha(cp)==b['files'][str(cp.relative_to(ROOT))]
        az=load(ap);cz=load(cp);assert cz['interval']==az['intervals']['T1'] and cz['Native_interval']==az['intervals']['T0']
        target=BASE/'readouts'/f"{r['public_cell']:05}.pt"
        z={**{k:r[k] for k in META},'space':{'Omega0':az['boxes']['OPD'],'Omega1':cz['fit']['final']},
            'intervals':{'I0':az['intervals']['T0'],'I1':az['intervals']['T1']},
            'source_bindings':{str(ap.relative_to(ROOT)):sha(ap),str(cp.relative_to(ROOT)):sha(cp)},
            'source_model_sha256':cz['source_model_sha256'],'pixel_sha256':cz['pixel_sha256'],'frame_ids':cz['frame_ids'],
            'actual_NEW_GPU_fits':0,'GT_read':False}
        save(target,z);rz=load(target)
        assert all(torch.equal(v,rz['space'][k]) for k,v in z['space'].items());coordinates+=sum(v.numel() for v in z['space'].values())
        assert rz['intervals']==z['intervals'];bindings[str(target.relative_to(ROOT))]=sha(target)
        status(BASE/'STATUS.json',dict(status='sealing_all_cached_2x2_readouts',done=len(bindings),total=512,CPU_only=True,GT_read=False,time=time.time()))
    assert len(bindings)==512 and not torch.cuda.is_initialized();verify()
    write(BASE/'GLOBAL_READOUT_BARRIER.json',dict(status='sealed',cells=512,logical_factorial_readouts=2048,new_E_readouts=512,
        files=bindings,coordinate_checks=coordinates,runtime_sha256=sha(BASE/'RUNTIME.json'),new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,CPU_only=True,GT_read=False,time=time.time()))
def require_seal():
    verify();b=read(BASE/'GLOBAL_READOUT_BARRIER.json');assert b['status']=='sealed' and b['cells']==512 and b['logical_factorial_readouts']==2048 and not b['GT_read']
    assert b['runtime_sha256']==sha(BASE/'RUNTIME.json');return b
def assemble(independent=False):
    import torch
    torch.set_num_threads(2);barrier=require_seal();assert not torch.cuda.is_initialized()
    from scripts.score_stvg_opd_temporal_capture_p7a_v1 import context_truth,array
    from scripts.run_decota_paper_main_v1 import read_row
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    gt,spans=context_truth();old={r['public_cell']:r for r in rows(OLD_OUT/'P7_B/ROWS.jsonl.gz')}
    aa={r['public_cell']:r for r in rows(OLD_OUT/'P7_A/ROWS.jsonl.gz')};extra={r['public_cell']:r for r in rows(OLD_OUT/'P7_B/DIAGNOSTIC_EXTRA_ROWS.jsonl.gz')}
    values=[];checks=0;hashes=0;coordinates=0;old_barrier=read(OLD_B/'EXPERT_EVIDENCE_BARRIER.json')
    for r in read(BASE/'DESIGN_LOCK.json')['records']:
        p=BASE/'readouts'/f"{r['public_cell']:05}.pt";h=sha(p);assert h==barrier['files'][str(p.relative_to(ROOT))];z=load(p)
        if independent:
            ap=OLD_A/'readouts'/f"{r['public_cell']:05}.pt";cp=OLD_B/'formal'/f"{r['public_cell']:05}/C_temporal_guided_OPD_repeat0.pt";az=load(ap);cz=load(cp)
            for source,hh in z['source_bindings'].items():assert sha(ROOT/source)==hh;hashes+=1
            assert torch.equal(az['boxes']['OPD'],z['space']['Omega0']) and torch.equal(cz['fit']['final'],z['space']['Omega1'])
            assert az['intervals']['T0']==z['intervals']['I0'] and az['intervals']['T1']==z['intervals']['I1']
            assert cz['source_model_sha256']==z['source_model_sha256'] and cz['pixel_sha256']==z['pixel_sha256'] and cz['frame_ids']==z['frame_ids']
            coordinates+=sum(v.numel() for v in z['space'].values())
        ds=r['dataset'];q=r['query_ordinal'];raw=read_row(ds,q);truth=gt[ds][q];span=spans[ds][q]
        assert raw['frame_ids']==z['frame_ids'];row={k:r[k] for k in META};curves={}
        for arm,(space,interval) in ARMS.items():
            bb=array(z['space'][space]);iv=z['intervals'][interval]
            if independent:m,fids,curve=dense_metric(bb,raw,truth,span,iv,ds)
            else:
                dense=DenseTube(bb,raw,truth,span,clip=ds=='hc2');m=dense.score(iv);fids=dense.fids;curve=dense.iou;checks+=near(m,official(bb,raw,truth,span,iv,ds))
            for k,v in m.items():row[arm+'_'+k]=v
            for th in [.3,.5]:row[arm+'_R'+str(int(th*100))]=float(m['v']>th)
            curves[arm]=curve
            if arm in ['A','B','C']:
                prefix={'A':'A_original','B':'B_posthoc','C':'C_temporal_guided_OPD'}[arm]
                for k in ['v','t','s','R30','R50']:checks+=near(row[arm+'_'+k],old[r['public_cell']][prefix+'_'+k])
        assert np.array_equal(curves['A'],curves['B']) and np.array_equal(curves['E'],curves['C'])
        for k in ['v','t','s','R30','R50']:
            for name,x,y in [('temporal_Omega0','B','A'),('temporal_Omega1','C','E'),('observation_I0','E','A'),('observation_I1','C','B')]:row[name+'_'+k]=row[x+'_'+k]-row[y+'_'+k]
            row['interaction_'+k]=row['temporal_Omega1_'+k]-row['temporal_Omega0_'+k]
            checks+=near(row['interaction_'+k],row['observation_I1_'+k]-row['observation_I0_'+k])
            row['D_reference_'+k]=old[r['public_cell']]['D_temporal_guided_Direct_'+k]
        assert row['interaction_t']==row['interaction_s']==0 and row['A_s']==row['B_s'] and row['E_s']==row['C_s']
        ids=np.array(fids);event=(ids>=span[0])&(ids<span[1]);real=np.isin(ids,raw['frame_ids']);supported=(ids>=raw['frame_ids'][0])&(ids<=raw['frame_ids'][-1])
        inp=normalized(load(ROOT/r['original_Full']['input_path']));ep=OLD_B/'expert_I1'/f"{r['capture_index']:05}.pt"
        assert sha(ep)==old_barrier['files'][str(ep.relative_to(ROOT))];ex=load(ep)
        visited0=[raw['frame_ids'][v] for v in inp['expert']['actual_observation_positions']];visited1=[raw['frame_ids'][v] for v in ex['expert']['actual_observation_positions']]
        visited=np.isin(ids,visited0+visited1);i0=(ids>=z['intervals']['I0'][0])&(ids<z['intervals']['I0'][1]);i1=(ids>=z['intervals']['I1'][0])&(ids<z['intervals']['I1'][1])
        masks={'common_unobserved_real':event&real&~visited,'visited_union_real':event&real&visited,'interpolation_only':event&supported&~real,
            'gained_event':event&i1&~i0,'lost_event':event&i0&~i1,'common_event':event&i0&i1,
            'gained_unobserved_real':event&i1&~i0&real&~visited,'lost_unobserved_real':event&i0&~i1&real&~visited,
            'unsupported_event':event&~supported}
        delta=curves['C']-curves['B']
        n0=max(max(z['intervals']['I0'][1],span[1])-min(z['intervals']['I0'][0],span[0]),1)
        n1=max(max(z['intervals']['I1'][1],span[1])-min(z['intervals']['I1'][0],span[0]),1)
        weights=event*i1/n1-event*i0/n0;contributions=weights*delta
        checks+=near(float(contributions.sum()),row['interaction_v'])
        for name,mask in masks.items():
            row[name+'_frames']=int(mask.sum());row[name+'_C_minus_B_spatial_IoU']=float(delta[mask].mean()) if mask.any() else None
            row[name+'_interaction_contribution']=float(contributions[mask].sum())
        checks+=near(sum(row[n+'_interaction_contribution'] for n in ['gained_event','lost_event','common_event']),row['interaction_v'])
        for name in ['gained_event_frames','lost_event_frames','T0_outside_event_duration','T1_outside_event_duration','T0_event_coverage','T1_event_coverage','delta_event_coverage']:
            row['original_'+name]=aa[r['public_cell']][name]
        for name in ['I0_offline_admitted_expert_quality','I1_minus_I0_offline_expert_quality','I0_admitted_positions','I1_minus_I0_admitted_positions']:
            row[name]=extra[r['public_cell']][name]
        checks+=near(row['common_unobserved_real_C_minus_B_spatial_IoU'],old[r['public_cell']]['C_minus_B_common_unobserved_real_IoU'])
        assert sha(p)==h;hashes+=2;values.append(row)
        status(BASE/('ROOT_STATUS.json' if independent else 'STATUS.json'),dict(status='independent_all512_support_root' if independent else 'actual_official_2x2_cached_scoring',done=len(values),total=512,CPU_only=True,GT_read=True,time=time.time()))
    assert len(values)==512 and not torch.cuda.is_initialized();verify()
    return values,dict(scalar_checks=checks,second_hash_checks=hashes,coordinate_checks=coordinates,cells=512,logical_factorial_readouts=2048,new_E_readouts=512,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0)
def aggregate(rr):
    from scripts.score_stvg_opd_temporal_guided_p7b_v1 import aggregate as original_parent_summary
    return original_parent_summary(rr)
def score():
    require_seal();write(BASE/'GT_EXPOSURE.json',dict(status='offline_after_all512_new_E_and_all2048_factorial_readouts_sealed',barrier_sha256=sha(BASE/'GLOBAL_READOUT_BARRIER.json'),GT_for_choice=False,history_exposed=True,CPU_only=True,time=time.time()))
    rr,counts=assemble();rows_write(OUT/'ROWS.jsonl.gz',rr);write(OUT/'SUMMARY.json',aggregate(rr))
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_independent_root_PDF_view_publication_archive',counts=counts,cells=512,parents=256,CPU_only=True,GT_read=True,time=time.time()))
def root():
    assert read(BASE/'CPU_COMPLETION.json')['GT_read'];rr,counts=assemble(True);checks=near(rr,rows(OUT/'ROWS.jsonl.gz'))
    from scripts.audit_stvg_opd_support_factorization_public_v1 import independent_summary
    expected,n=independent_summary(rr,read(OUT/'SUMMARY.json'));checks+=n
    write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='all512 exact sealed E coordinates/immutable C and original source/2048 official-independent dense outputs/temporal-observation interaction/GT support and true-unobserved partitions/full parent bootstrap',counts=counts,independent_scalar_checks=checks,cells=512,parents=256,original_complete_math_state_root_reused=True,new_model_or_Jacobian_replay=False,CPU_only=True,GT_read=True,time=time.time()))
    status(BASE/'STATUS.json',dict(status='pending_actual_PDF_view_publication_archive',done=512,total=512,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':{'prepare':prepare,'seal':seal,'score':score,'root':root}[sys.argv[1]]()
