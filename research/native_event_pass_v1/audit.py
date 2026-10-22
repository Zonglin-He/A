"""Postseal native-pass differences and saved good/failure cases, no extra GT read."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_pass_v1.run import D,P,read,put,sha
import numpy as np,torch

def main():
 s=read(D/'sealed.json');r=read(P/'report.json');assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json');vals={x['query']:x for x in r['rows']};rows=[]
 for k,e in s['files'].items():
  assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');a=tr['intervened_q']['First'];b=tr['intervened_q']['Second'];take=b>0;mid=(a+b)/2;JS=.5*np.sum(a[take]*np.log(a[take]/mid[take]))+.5*np.sum(b[take]*np.log(b[take]/mid[take]));h1=torch.cat([x.reshape(-1,256)for x in tr['first_hidden']]);h2=torch.cat([x.reshape(-1,256)for x in tr['second_hidden']]);rel=float(torch.linalg.vector_norm(h1-h2)/torch.linalg.vector_norm(h2));v=vals[k]
  rows.append(dict(query=k,posterior_TV=float(abs(a-b).sum()/2),posterior_JS=float(JS),relative_hidden_delta=rel,delta_t_first_native_pp=100*(v['values']['DirectFirst6']['t']-v['values']['NativeG6']['t']),delta_t_mixture_native_pp=100*(v['values']['RefinementMixture']['t']-v['values']['NativeG6']['t']),duration_group=v['duration_group']))
 out=dict(scope='Postseal no new labels/model; actual first vs second native states and saved task outcomes',queries=64,mean_posterior_TV=float(np.mean([x['posterior_TV']for x in rows])),median_posterior_TV=float(np.median([x['posterior_TV']for x in rows])),mean_relative_hidden_delta=float(np.mean([x['relative_hidden_delta']for x in rows])),same_first_decisions=64-r['changed']['DirectFirst6'],same_mixture_decisions=64-r['changed']['RefinementMixture'],rows=rows);put(P/'pass_audit.json',out);put(D/'pass_audit_receipt.json',dict(code_sha256=sha(Path(__file__)),seal_sha256=sha(D/'sealed.json'),report_sha256=sha(P/'report.json'),audit_sha256=sha(P/'pass_audit.json'),new_GT=False));print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
if __name__=='__main__':main()
