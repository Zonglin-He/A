"""Second aggregation implementation; no model forward, GT inference, or new data."""
import sys,json,statistics,math,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a05_signal import D,SIGNALS,GAP,A0,A3,load
from vg_tta.desta3d_v3_oracle_io import read,write,sha

def main():
    import torch
    import numpy as np
    torch.set_num_threads(4);start=time.monotonic();r=read(D/'ROOT_SIGNAL_READBACK.json');assert r['status']=='passed'
    rows=read(D/'INPUTS.json');maxerr=0.;decisions=[];formats=[];counts={};check=0
    original=read(A0/'DEV64.json')
    digest=lambda s:hashlib.sha256(s.encode()).hexdigest()
    parents=sorted({x['source'] for x in original},key=lambda x:digest('DESTA-A05-v1|parent|'+str(x)))
    selected=[min((i for i,x in enumerate(original) if x['source']==p),key=lambda i:digest('DESTA-A05-v1|query|'+str(original[i]['key']))) for p in parents]
    assert len(parents)==16 and selected==[x['dev64_index'] for x in rows]
    qc=load(D/'QC.pt').double();q=load(A0/'BASIS.pt').double();b=torch.from_numpy(np.load(A3/'TRAIN_BASIS.npz')['basis'][:,:16].copy()).double()
    basis_error=float((qc-(q@b).float().double()).abs().max())
    gram_error=float((qc.T@qc-torch.eye(16,dtype=torch.float64)).abs().max())
    assert basis_error<=8*torch.finfo(torch.float32).eps and gram_error<2e-6
    for i,row in enumerate(rows):
        base=load(GAP/'episodes'/f"{row['gap_index']:04}"/'B1.pt')
        formats.append(dict(index=i,format_ok=bool(base['format_ok']),invalid_geometry=int((~base['geometry_valid']).sum()),native_positions=len(base['positions'])))
    for signal in SIGNALS:
        vals=[];positive={'event':0,'spatial':0};available=dict(positive)
        for i,x in enumerate(r['results'][signal]):
            ep=D/'episodes'/f'{i:02}';raw=load(ep/'RAW_SIGNALS.pt')[signal];ref=load(D/'references'/f'{i:02}.pt')
            gs=[raw['gradients'][b].double() for b in ['event','spatial']]
            units=[g/g.norm() if g.norm()>0 else torch.zeros_like(g) for g in gs];d=-(units[0]+units[1]);d=d/d.norm() if d.norm()>0 else torch.zeros_like(d)
            oracle=ref['oracle_direction'].double();c=float((d*oracle).sum()) if d.norm()>0 else 0.
            maxerr=max(maxerr,abs(c-x['gate_cosine']));vals.append(c);check+=1
            for b in positive:
                if ref['GT_available'][b]:
                    available[b]+=1;dot=float(-(ref['GT_gradients_C'][b].double()*d).sum());positive[b]+=int(dot>0)
                    maxerr=max(maxerr,abs(dot-x['local_descent'][b]));check+=1
                else:assert x['local_descent'][b] is None
        a=r['aggregates'][signal];median=statistics.median(vals);mean=statistics.fmean(vals)
        maxerr=max(maxerr,abs(mean-a['cosine_mean']),abs(median-a['cosine_median']))
        assert positive==a['positive_GT_descent'] and available==a['available_GT']
        passed=median>=.1 and all(available[b]>0 and positive[b]*100>=65*available[b] for b in positive)
        assert passed==a['passed'];check+=8
        if passed:decisions.append(signal)
        counts[signal]=dict(negative_cosines=sum(v<0 for v in vals),positive_cosines=sum(v>0 for v in vals),cosine_gt_point1=sum(v>.1 for v in vals),cosine_gt_point3=sum(v>.3 for v in vals))
    assert decisions==r['passing_signals'] and maxerr<1e-9
    out=dict(status='passed',checks=check,max_torch_geometry_aggregate_error=maxerr,selection_independently_reconstructed=True,
      basis_reconstruction_max_abs=basis_error,basis_gram_max_abs=gram_error,passing_signals=decisions,counts=counts,B1_format_support=formats,
      CPU_seconds=time.monotonic()-start,source_report_sha=sha(D/'ROOT_SIGNAL_READBACK.json'))
    write(D/'ROOT_SUMMARY_CROSSCHECK.json',out);print(json.dumps({k:v for k,v in out.items() if k!='B1_format_support'},indent=2))
if __name__=='__main__':main()
