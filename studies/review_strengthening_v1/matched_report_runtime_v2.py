"""Only adapt legacy report baseline naming; old predictions/math remain frozen."""
from review_common import *
import matched_cpu as old
import inspect,copy,argparse

def validate_runtime():
 p=read(D/'execution_v2_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for k in ['code_sha256','input_sha256']:
  for f,h in p[k].items():assert sha(R/f)==h,f
 validate();return p

def matched_report(rows,arms,pairs,folder):
 # Report-only alias for the legacy helper's fixed severe-vs-Source calculation.
 # This alias NEVER changes box arrays, temporal intervals or official metrics.
 rr=[{**r,'values':{**r['values'],'Source':r['values']['SourceMatchedTime']}}for r in rows]
 report=selected.suite().parent_report(rr,arms,pairs,folder)
 report['severe_vs_SourceMatchedTime']=report.pop('severe_vs_Source');report['baseline_semantics']='Source boxes scored on the exact common selected current q1 MAP interval; NOT native Source temporal readout'
 atomic(folder/'report.json',report);return report

def score(ds):
 validate_runtime();src=inspect.getsource(old.score);needle='selected.suite().parent_report(rr,ARMS,PAIRS,folder)';assert src.count(needle)==1
 changed=src.replace(needle,'matched_report(rr,ARMS,PAIRS,folder)');assert changed.replace('matched_report(rr,ARMS,PAIRS,folder)',needle)==src
 scope={**vars(old),'matched_report':matched_report};exec(compile(changed,'<report-only_baseline_adapter_v2>','exec'),scope);scope['score'](ds)

def audit(ds):validate_runtime();old.audit(ds)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['score','audit']);ap.add_argument('--dataset',choices=COUNTS,required=True);a=ap.parse_args();globals()[a.command](a.dataset)
