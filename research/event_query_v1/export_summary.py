"""Aggregate saved optimization traces and anonymize public parent identities."""
import sys,json,shutil
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent));from run import R,D,P,read,put,sha
report=read(P/'report.json');original=D/'scored_private_report.json'
if not original.exists():shutil.copy2(P/'report.json',original)
parents=sorted({r['parent']for r in report['rows']});mapping={p:f'p{i+1:03d}'for i,p in enumerate(parents)}
for row in report['rows']:row['parent']=mapping[row['parent']]
put(P/'report.json',report)
receipt=read(D/'score_receipt.json');saved=D/'scoring_recovery_v1/original_success_receipt.json'
if not saved.exists():put(saved,receipt)
receipt.update(report_sha256=sha(P/'report.json'),private_original_report_sha256=sha(original),public_parent_anonymization='bijective p001..; identical numeric results and cluster membership');put(D/'score_receipt.json',receipt)
summary={};details={}
for kind in ['Raw','Predictive']:
    rows=[]
    for f in sorted((D/'traces').glob('*.pt')):
        tr=torch.load(f,map_location='cpu',weights_only=False);p=tr['q']['NativeG6'];q=tr['q'][kind+'Query'];t=tr['q'][kind+'Direct'];nz=t>0;k0=float((t[nz]*np.log(t[nz]/np.maximum(p[nz],1e-300))).sum());k1=float((t[nz]*np.log(t[nz]/np.maximum(q[nz],1e-300))).sum())
        rows.append(dict(query=f.stem,teacher_TV=float(abs(t-p).sum()/2),query_TV=float(abs(q-p).sum()/2),initial_KL=k0,final_KL=k1,KL_fraction_removed=(k0-k1)/k0 if k0>1e-12 else None,delta_RMS=tr['paths'][kind]['path'][-1]['normalized_delta_RMS']))
    ci={}
    for field in ['teacher_expected_fixed_v','query_expected_fixed_v']:
        v=np.array([100*(r['quality'][kind][field]-r['quality'][kind]['native_expected_fixed_v'])for r in report['rows']]);ix=np.random.default_rng(20261022).integers(0,len(v),(10000,len(v)));ci[field]=dict(mean_pp=float(v.mean()),CI95_pp=np.quantile(v[ix].mean(1),[.025,.975]).tolist())
    summary[kind]=dict(means={k:float(np.mean([r[k]for r in rows if r[k]is not None]))for k in rows[0]if k!='query'},at_radius=sum(r['delta_RMS']>.099999 for r in rows),expected_quality=ci);details[kind]=rows
put(P/'optimization_transfer.json',dict(summary=summary,rows=details,scope='saved trace readback only; no new fitting'))
print(json.dumps(summary))
