"""Postseal numerical coupling proof and saved outcome audit; no new GT read."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_coupling_v1.run import D,P,read,put,sha,records
import torch,numpy as np

def main():
 s=read(D/'sealed.json');assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json');rows=[]
 for row in records():
  k=row['id'];e=s['files'][k];assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu')
  if not e['supported']:continue
  ids=np.asarray(row['metadata']['frame_ids']);length=np.maximum(0,ids[None,:]+1-ids[:,None])/row['metadata']['fps'];qs=tr['q'];c=tr['coupling'];rows.append(dict(query=k,marginal_max_error=max(z[v]for z in c.values()for v in ['marginal_error_first','marginal_error_last']),self_q_max_error=float(abs(qs['SelfCoupledFusion']-qs['NativeG6']).max()),nonoverlap_real=c['real']['nonoverlapping_pair_mass'],nonoverlap_anti=c['anti']['nonoverlapping_pair_mass'],expected_interval_seconds={a:float((q*length).sum())for a,q in qs.items()}))
 out=dict(queries=52,max_marginal_error=max(x['marginal_max_error']for x in rows),max_self_identity_error=max(x['self_q_max_error']for x in rows),mean_nonoverlap_real=float(np.mean([x['nonoverlap_real']for x in rows])),mean_nonoverlap_anti=float(np.mean([x['nonoverlap_anti']for x in rows])),self_changed_decisions=read(P/'report.json')['changed']['SelfCoupledFusion'],no_new_labels_read=True,rows=rows);put(P/'coupling_audit.json',out);put(D/'coupling_audit_receipt.json',dict(code_sha256=sha(Path(__file__)),seal_sha256=sha(D/'sealed.json'),report_sha256=sha(P/'report.json'),audit_sha256=sha(P/'coupling_audit.json')));print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
if __name__=='__main__':main()
