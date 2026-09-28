"""Independent Torch full-feature check of the NumPy union-coordinate audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,time,math,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.external_qualification_io import read,write,sha,check_pins
OLD=ROOT/'artifacts/desta3d_v3/latent_oracle_v1/decomposition001'
DEST=ROOT/'artifacts/desta3d_v3/latent_oracle_v1/joint_components_v1'


def run():
    import torch,numpy as np
    torch.set_num_threads(4);start=time.monotonic();cfg=read(DEST/'CONFIG.json')
    assert not (DEST/'ROOT_RAW_READBACK.json').exists();check_pins(read(DEST/'LOCK.json')['pins'])
    complete=read(DEST/'COMPLETE.json');assert sha(DEST/'REPORT.json')==complete['report_sha']
    for p,h in complete['case_seal'].items():assert sha(DEST/p)==h
    report=read(DEST/'REPORT.json');old=read(OLD/'independent_readback_v1/REPORT.json')
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    b=load(OLD/'BASIS.pt');qt=torch.linalg.qr(b['T'].double()).Q;qs=torch.linalg.qr(b['S'].double()).Q
    cat=torch.cat([qt,qs],1);sv=torch.linalg.svdvals(cat);rank=int((sv>1e-10*sv[0]).sum());assert rank==256
    qu=torch.linalg.qr(cat).Q;inverse=torch.linalg.pinv(cat,rtol=1e-10)
    errors=[];rows=[]
    def ck(a,b):
        a=np.asarray(a);b=np.asarray(b);err=float(np.max(abs(a-b)));assert np.allclose(a,b,rtol=cfg['reduction_rtol'],atol=cfg['reduction_atol']),(a,b,err)
        errors.append(err)
    def norm2(a):return float((a*a).sum())
    def proj(a,q):return a@q@q.T
    def consume(storage,name,v,gt,gs):
        acc=storage.setdefault(name,dict(energy=0.,T=0.,S=0.));acc['energy']+=norm2(v)
        acc['T']-=float((gt*v).sum());acc['S']-=float((gs*v).sum())
    ck(sv.tolist(),report['basis']['union_singular_values'])
    ck(torch.linalg.svdvals(qt.T@qs).tolist(),report['basis']['principal_cosines'])
    try:
        for i,r in enumerate(report['rows']):
            assert time.monotonic()-start<cfg['cpu_seconds_limit']
            z=load(OLD/'episodes'/f'{i:02}'/'GEOMETRY.pt');g={k:z['gradients'][v].reshape(-1,2560) for k,v in [('T','event'),('S','spatial')]};n=len(g['T'])
            caps={k:dict(full_energy=0.,union_energy=0.,own_energy=0.,additional_energy=0.) for k in g}
            for k in range(0,n,256):
                for name,q in [('T',qt),('S',qs)]:
                    a=g[name][k:k+256].double();pu=proj(a,qu);po=proj(a,q)
                    for field,value in [('full_energy',a),('union_energy',pu),('own_energy',po),('additional_energy',pu-po)]:caps[name][field]+=norm2(value)
            for name,fields in caps.items():
                for key,value in fields.items():ck(value,r['gradient_capacity'][name][key])
            case={}
            for name in ['J','J_pass']:
                j=z['deltas'][name].reshape(-1,2560);acc={};cross={k:0. for k in ['T_first','S_first','direct_sum']};origin={}
                scale=r['directions'][name]['gradient_origin']['scale']
                for k in range(0,n,256):
                    x=j[k:k+256].double();gt=g['T'][k:k+256].double();gs=g['S'][k:k+256].double()
                    pu=proj(x,qu);pt=proj(x,qt);ps=proj(x,qs);coeff=x@inverse.T
                    parts={'T_first':{'T':pt,'S_given_T':pu-pt},'S_first':{'S':ps,'T_given_S':pu-ps},
                           'direct_sum':{'T':coeff[:,:128]@qt.T,'S':coeff[:,128:]@qs.T}}
                    for order,cp in parts.items():
                        for component,v in cp.items():consume(acc,order+'/'+component,v,gt,gs)
                        a,c=cp.values();cross[order]+=2*float((a*c).sum())
                    consume(acc,'perpendicular',x-pu,gt,gs);consume(acc,'total',x,gt,gs)
                    for label,gg in [('T',gt),('S',gs)]:consume(origin,label,-scale*proj(gg,qu)/math.sqrt(caps[label]['full_energy']),gt,gs)
                rr=r['directions'][name]
                for key,a in acc.items():
                    if '/' in key:
                        order,component=key.split('/');ref=rr[order]['components'][component]
                    elif key=='perpendicular':ref=rr['perpendicular']
                    else:ref=dict(energy=rr['total_energy'],local_descent=rr['total_local_descent'])
                    ck(a['energy'],ref['energy'])
                    for ob in ['T','S']:ck(a[ob],ref['local_descent'][ob])
                for k,value in cross.items():ck(value,rr[k]['twice_energy_cross_term'])
                for k,a in origin.items():
                    ref=rr['gradient_origin']['components'][k];ck(a['energy'],ref['energy'])
                    for ob in ['T','S']:ck(a[ob],ref['local_descent'][ob])
                case[name]=dict(components=acc,cross_terms=cross)
            for a,arm in old['arms'].items():
                for m in ['tIoU','sIoU','vIoU']:ck(r['inherited_native'][a][m],arm['rows'][i]['metrics'][m])
            rows.append(dict(index=i,case=f'P{i:02}',directions=case));del z
            print(f'P{i:02} raw components checked',flush=True)
        # A separate summary reduction, including signs under declared zero band.
        for direction,items in report['local_descent_summary'].items():
            for key,expected in items.items():
                order,component,ob=key.split('/');ob=ob[-1]
                vals=[];tols=[]
                for row in rows:
                    cp=row['directions'][direction]['components'];vals.append(cp[order+'/'+component][ob])
                    tols.append(max(1e-10,1e-10*(sum(abs(x[ob]) for k,x in cp.items() if k.startswith(order+'/'))+abs(cp['perpendicular'][ob]))))
                x=np.array(vals);t=np.array(tols)
                for k,v in [('mean',x.mean()),('median',np.median(x)),('minimum',x.min()),('maximum',x.max())]:ck(v,expected[k])
                assert int((x>t).sum())==expected['positive'] and int((x< -t).sum())==expected['negative'] and int((abs(x)<=t).sum())==expected['zero']
        out=dict(status='passed',report_sha=sha(DEST/'REPORT.json'),queries=16,scalar_checks=len(errors),
            max_absolute_error=max(errors),atol=cfg['reduction_atol'],rtol=cfg['reduction_rtol'],
            verification='Torch QR/pseudoinverse and full2560-dimensional chunks vs NumPy SVD union coordinates; complete old gradients/deltas',
            rows=rows,CPU_seconds=time.monotonic()-start,GPU_seconds=0,new_GT_read=False)
        write(DEST/'ROOT_RAW_READBACK.json',out);print({k:v for k,v in out.items() if k!='rows'})
    except BaseException as e:
        write(DEST/'ROOT_READBACK_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),CPU_seconds=time.monotonic()-start));raise

if __name__=='__main__':run()
