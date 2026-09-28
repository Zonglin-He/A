"""Register and run a write-once, CPU-only attribution of saved Joint residuals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import argparse,json,sys,time,shutil,traceback,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.external_qualification_io import read,write,sha,check_pins,tensor_sha
OLD=ROOT/'artifacts/desta3d_v3/latent_oracle_v1/decomposition001'
DEST=ROOT/'artifacts/desta3d_v3/latent_oracle_v1/joint_components_v1'
PROTOCOL=ROOT/'protocols/desta3d_v3_joint_component_attribution_v1.md'


def register():
    import unittest
    from tests.test_desta3d_v3_joint_attribution import Contracts
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    assert result.wasSuccessful() and result.testsRun==4
    assert not DEST.exists();assert shutil.disk_usage(ROOT).free>8*2**30+16*2**20
    paths=[PROTOCOL,Path(__file__),ROOT/'vg_tta/desta3d_v3_joint_attribution.py',
           ROOT/'tests/test_desta3d_v3_joint_attribution.py',ROOT/'scripts/crosscheck_desta3d_v3_joint_components.py',
           ROOT/'vg_tta/external_qualification_io.py',OLD/'BASIS.pt',OLD/'ROOT_BASIS_SEAL.json',
           OLD/'COMPLETE.json',OLD/'PREDICTIONS_SEAL.json',OLD/'INPUTS.json',OLD/'CONFIG.json',
           OLD/'independent_readback_v1/REPORT.json',OLD/'independent_readback_v1/COMPLETE.json']
    sealed=read(OLD/'PREDICTIONS_SEAL.json')['files'];consumed={}
    for i in range(16):
        for f in ['GEOMETRY.pt','INPUT.json','COMPLETE.json']:
            p=OLD/'episodes'/f'{i:02}'/f;h=sha(p);assert h==sealed[str(p.relative_to(OLD))]
            paths.append(p);consumed[str(p.relative_to(OLD))]=h
    assert sha(OLD/'BASIS.pt')==read(OLD/'ROOT_BASIS_SEAL.json')['basis_sha']
    assert sha(OLD/'PREDICTIONS_SEAL.json')==read(OLD/'COMPLETE.json')['seal_sha']
    report=OLD/'independent_readback_v1/REPORT.json';assert sha(report)==read(report.parent/'COMPLETE.json')['report_sha']
    write(DEST/'CPU_PREFLIGHT.json',dict(status='passed',tests=4,no_real_attribution_yet=True))
    write(DEST/'CONFIG.json',dict(queries=16,parents=16,cpu_seconds_limit=900,blas_threads=4,
        storage_bytes_limit=16*2**20,disk_floor=8*2**30,rank_rtol=1e-10,
        reduction_atol=1e-8,reduction_rtol=2e-7,reconstruction_relative=2e-6,
        source_GT_already_in_gradients=True,new_GT_read=False,target_read=False,
        model_forward=0,backward=0,optimizer_steps=0,GPU_seconds=0,
        cumulative_GPU_seconds=43107.875212573104,cap=None))
    paths += [DEST/'CPU_PREFLIGHT.json',DEST/'CONFIG.json']
    write(DEST/'LOCK.json',dict(pins={str(p):sha(p) for p in paths},consumed_original_seal_files=consumed))
    write(DEST/'REGISTRATION.json',dict(time=time.time(),status='registered_before_CPU_attribution',
        old_result_preserved=True,new_correction_or_architecture=False,old_report_sha=sha(report)))
    print(json.dumps(dict(status='registered',consumed_sealed_files=len(consumed),tests=4)))


def run():
    import numpy as np,torch
    from vg_tta.desta3d_v3_joint_attribution import bases,attribution,gradient_capacity,dot
    torch.set_num_threads(4);cfg=read(DEST/'CONFIG.json');check_pins(read(DEST/'LOCK.json')['pins'])
    start=time.monotonic();status='running';assert not (DEST/'STARTED.json').exists()
    write(DEST/'STARTED.json',dict(time=time.time(),pid=os.getpid(),CPU_only=True))
    def guard():
        assert time.monotonic()-start<cfg['cpu_seconds_limit'];assert shutil.disk_usage(ROOT).free>=cfg['disk_floor']
    def load(p):return torch.load(p,map_location='cpu',weights_only=False)
    def arr(x):return x.detach().numpy().astype(np.float64).reshape(-1,x.shape[-1])
    def compress(x,u,measure_residual=False):
        x=arr(x);out=x@u;en=dot(x,x)
        re=None
        if measure_residual:
            re=math.fsum(dot(x[k:k+256]-out[k:k+256]@u.T,x[k:k+256]-out[k:k+256]@u.T) for k in range(0,len(x),256))
        return out,en,re
    try:
        guard();old_basis=load(OLD/'BASIS.pt');b=bases(old_basis['T'].numpy(),old_basis['S'].numpy(),cfg['rank_rtol'])
        assert b['report']['T_rank']==b['report']['S_rank']==128 and b['report']['union_rank']==256
        u=b['U'];qj=old_basis['J'].double().numpy();b['report']['saved_union_relative_residual']=float(np.linalg.norm(qj-u@(u.T@qj))/np.linalg.norm(qj))
        assert b['report']['saved_union_relative_residual']<cfg['reconstruction_relative']
        write(DEST/'BASIS_GEOMETRY.json',b['report']);torch.save({k:torch.from_numpy(b[k]) for k in ['U','T','S','inverse']},DEST/'AUDIT_BASIS.pt')
        old=read(OLD/'independent_readback_v1/REPORT.json');inputs=read(OLD/'INPUTS.json');assert len(inputs)==16 and len({x['source'] for x in inputs})==16
        rows=[]
        for i,row in enumerate(inputs):
            guard();ep=OLD/'episodes'/f'{i:02}';z=load(ep/'GEOMETRY.pt');ident=read(ep/'INPUT.json');done=read(ep/'COMPLETE.json')
            assert ident['key']==done['key']==row['key'];assert ident['common_F_sha']==done['common_F_sha']==tensor_sha(z['stock'])
            assert z['gradients']['event'].shape==z['gradients']['spatial'].shape==z['stock'].shape
            gradients={};full={}
            for name,key in [('T','event'),('S','spatial')]:gradients[name],full[name],_=compress(z['gradients'][key],u)
            result=dict(index=i,case=f'P{i:02}',common_F_sha=ident['common_F_sha'],shape=list(z['stock'].shape),
                gradient_capacity=gradient_capacity(gradients,full,b),directions={})
            for name in ['J','J_pass']:
                j,en,perp_energy=compress(z['deltas'][name],u,True);r=attribution(j,gradients,b)
                direct_dots={name2:float(-(z['gradients'][key].double()*z['deltas'][name].double()).sum()) for name2,key in [('T','event'),('S','spatial')]}
                perp_dots={k:direct_dots[k]+dot(g,j) for k,g in gradients.items()}
                r.update(total_energy=en,union_energy=dot(j,j),perpendicular=dict(energy=perp_energy,local_descent=perp_dots),
                         total_local_descent=direct_dots,perpendicular_relative_L2=math.sqrt(perp_energy/max(en,1e-30)))
                assert r['perpendicular_relative_L2']<cfg['reconstruction_relative']
                for convention in ['T_first','S_first','direct_sum']:
                    c=r[convention];assert c['reconstruction_L2']/max(math.sqrt(en),1e-30)<cfg['reconstruction_relative']
                    assert np.isclose(sum(v['energy'] for v in c['components'].values())+c['twice_energy_cross_term']+perp_energy,en,rtol=2e-7,atol=1e-8)
                    for k in gradients:
                        assert np.isclose(sum(v['local_descent'][k] for v in c['components'].values())+perp_dots[k],direct_dots[k],rtol=2e-7,atol=1e-8)
                # This explains construction, not an independent effect test.
                h={k:g/math.sqrt(full[k]) for k,g in gradients.items()};h_sum=h['T']+h['S'];scale=-dot(j,h_sum)/max(dot(h_sum,h_sum),1e-30)
                pieces={k:-scale*v for k,v in h.items()};rem=j-pieces['T']-pieces['S']
                r['gradient_origin']=dict(scale=scale,components={k:dict(energy=dot(v,v),local_descent={n:-dot(g,v) for n,g in gradients.items()}) for k,v in pieces.items()},
                    residual_relative_L2=float(np.linalg.norm(rem)/max(math.sqrt(en),1e-30)),residual_local_descent={k:-dot(g,rem)+perp_dots[k] for k,g in gradients.items()},
                    twice_energy_cross_term=2*dot(pieces['T'],pieces['S']))
                assert r['gradient_origin']['residual_relative_L2']<cfg['reconstruction_relative']
                result['directions'][name]=r
            jp=z['deltas']['J_pass'].double();jj=z['deltas']['J'].double()/math.sqrt(2)
            result['saved_pass_scaling_relative_error']=float((jp-jj).norm()/jp.norm())
            assert result['saved_pass_scaling_relative_error']<cfg['reconstruction_relative']
            result['inherited_native']={}
            for a,x in old['arms'].items():
                rr=x['rows'][i];assert rr['key']==row['key'] and rr['source']==row['source']
                result['inherited_native'][a]={k:rr['metrics'][k] for k in ['tIoU','sIoU','vIoU']}
            result['inherited_format_ok']={a:x['rows'][i]['format_ok'] for a,x in old['arms'].items()}
            for a in ['joint','joint_pass_matched']:
                result['inherited_native'][a+'_minus_decomposed_pp']={k:100*(result['inherited_native'][a][k]-result['inherited_native']['decomposed'][k]) for k in ['tIoU','sIoU','vIoU']}
            assert old['finite_objectives'][i]['key']==row['key']
            result['inherited_finite_CE']=old['finite_objectives'][i]['finite']
            write(DEST/'cases'/f'{i:02}.json',result);rows.append(result);del z,jp,jj
            print(json.dumps(dict(case=f'P{i:02}',completed=i+1)),flush=True)
        def summary(vals,tolerances=None):
            x=np.asarray(vals,float);tol=np.asarray(tolerances) if tolerances is not None else np.maximum(1e-10,1e-10*np.abs(x))
            return dict(mean=float(x.mean()),median=float(np.median(x)),minimum=float(x.min()),maximum=float(x.max()),
                        positive=int((x>tol).sum()),negative=int((x< -tol).sum()),zero=int((abs(x)<=tol).sum()))
        stats={}
        for name in ['J','J_pass']:
            stats[name]={}
            for order,component in [('T_first','T'),('T_first','S_given_T'),('S_first','S'),('S_first','T_given_S'),('direct_sum','T'),('direct_sum','S')]:
                for objective in ['T','S']:
                    key=f'{order}/{component}/objective_{objective}'
                    vals=[x['directions'][name][order]['components'][component]['local_descent'][objective] for x in rows]
                    tolerances=[max(1e-10,1e-10*(sum(abs(v['local_descent'][objective]) for v in x['directions'][name][order]['components'].values())+abs(x['directions'][name]['perpendicular']['local_descent'][objective]))) for x in rows]
                    stats[name][key]=summary(vals,tolerances)
        capacities={b:{k:summary([r['gradient_capacity'][b][k] for r in rows]) for k in ['union_fraction_of_full','own_fraction_of_union','additional_fraction_of_union']} for b in ['T','S']}
        report=dict(status='CPU_attribution_completed',basis=b['report'],rows=rows,local_descent_summary=stats,
                    gradient_capacity_summary=capacities,old_report_sha=sha(OLD/'independent_readback_v1/REPORT.json'),
                    native_effects='Inherited whole-correction outcomes only; no component finite intervention',
                    finite_component_causality='untested',new_GT_read=False,new_model_calls=0,new_GPU_seconds=0,
                    candidate='Decomposed Evidence, Joint Correction; architecture and OPD unimplemented/unqualified')
        write(DEST/'REPORT.json',report);guard()
        assert sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())<cfg['storage_bytes_limit']
        write(DEST/'COMPLETE.json',dict(status='completed',queries=16,report_sha=sha(DEST/'REPORT.json'),
            case_seal={str(p.relative_to(DEST)):sha(p) for p in sorted((DEST/'cases').glob('*.json'))},basis_sha=sha(DEST/'AUDIT_BASIS.pt'),GPU_seconds=0))
        status='completed'
    except BaseException as e:
        write(DEST/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:
        write(DEST/'CPU_RECEIPT.json',dict(status=status,seconds=time.monotonic()-start,GPU_seconds=0,
            cumulative_GPU_seconds=cfg['cumulative_GPU_seconds'],cap=None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);a=p.parse_args();globals()[a.action]()
