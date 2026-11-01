"""Read-only CPU consistency audit; no feature / label / model accesses."""
import json,hashlib,collections
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;A=B/'attempt001'
def read(p):return json.loads(p.read_text())
s=read(A/'SEALED.json')
for p,h in s['sha256'].items():assert hashlib.sha256((A/p).read_bytes()).hexdigest()==h,p
rows=read(A/'OBSERVATION_ROWS.json');within=read(A/'WITHIN_CANDIDATE_ROWS.json');summary=read(A/'SUMMARY.json');assert len(rows)==len(within)==1200 and len(summary)==18
counts=collections.Counter((r['platform'],r['split'],r['key']) for r in rows)
for r in rows:
 assert r['split'] in ['A208','DEV64','T128']
 for k in ['all_query_row_margin','all_query_row_k4_boundary_margin','all_query_within_chosen_row_candidate_margin']:assert r[k]['min']>=-1e-12,(r['alias'],k)
 assert r['all_query_nearest_A_cosine']['min']>=-1.00001 and r['all_query_nearest_A_cosine']['max']<=1.00001
 for k in ['CPU32_vs_CPU64_row_changes','CPU32_vs_CPU64_row_or_candidate_changes','CPU32_vs_CPU64_k1_gate_changes','CPU32_vs_CPU64_k4_gate_changes']:assert 0<=r[k]<=256
for x in summary:
 rr=[r for r in rows if (r['platform'],r['split'],r['key'])==(x['platform'],x['split'],x['key'])];assert len(rr)==x['observations']
 for k in ['CPU32_vs_CPU64_row_changes','CPU32_vs_CPU64_row_or_candidate_changes','CPU32_vs_CPU64_k1_gate_changes','CPU32_vs_CPU64_k4_gate_changes']:assert sum(r[k] for r in rr)==x[k]
 assert np.isclose(np.median([r['all_query_row_margin']['median'] for r in rr]),x['median_nearest_A_row_margin_rowmedians'])
out={'state':'PASSED','sealed_checksums_verified':len(s['sha256']),'observation_key_rows':len(rows),'summary_rows':len(summary),'candidate_queries':sum(x['queries'] for x in summary),'split_contract':'A208 DEV64 T128 only','margins_and_precision_counts_consistent':True,'no_feature_GT_model_or_GPU_access_in_audit':True}
(B/'CPU_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
