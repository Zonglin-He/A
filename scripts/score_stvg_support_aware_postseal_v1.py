"""Root initial final-readout metric audit; requires both eight-case seals."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,statistics,json
from scripts.stvg_temporal_caption_query_v1 import ROOT,BASE as OLD,sha,read,roster,catalog,write
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal,_metric_functions,score_tube_xyxy
B=ROOT/'artifacts/stvg_support_aware_temporal_spatial_update_v1';torch.set_num_threads(2)
for p,h in read(B/'ROOT_READBACK_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h
for n in ['TEMPORAL_GLOBAL_BARRIER.json','STUDENT_GLOBAL_BARRIER.json']:
 z=read(B/n);assert z['status']=='complete'and z['count']==8 and z['GT_read'] is False
 for f in z['files']:assert sha(ROOT/f['path'])==f['sha256']
cases={r['case_id']:catalog(r)[0]for r in roster()};truths,bindings=_truths_after_seal(cases);off,ind,_=_metric_functions();rows=[];errors=[]
for r in roster():
 cid=r['case_id'];c=cases[cid];g=truths[cid];t=torch.load(B/'temporal'/f'{cid}.pt',map_location='cpu',weights_only=False);s=torch.load(B/'student'/f'{cid}.pt',map_location='cpu',weights_only=False);old=torch.load(OLD/'student'/f'{cid}.pt',map_location='cpu',weights_only=False)
 def score(box,iv):
  v=score_tube_xyxy(box,s['frame_ids'],c['width'],c['height'],iv,g['truth'],g['span'],r['dataset'],official=off,independent=ind);errors.append(v['official_independent_max_abs_error']);return v['metrics']
 arms={k:score(v['boxes'],v['interval']) for k,v in old['arms'].items()}
 for mode,v in t['results'].items():arms['T_'+mode]=score(s['arms']['S0_source']['boxes'],v['path'][-1]['physical_interval'])
 for arm,v in s['arms'].items():
  arms[arm]=score(v['boxes'],v['interval']);arms['trust_'+arm]=score(v['boxes'],t['results']['trust']['path'][-1]['physical_interval'])
 rows.append(dict(case_id=cid,dataset=r['dataset'],arms=arms))
groups=[]
for ds in ['hc2','vidstg']:
 rr=[r for r in rows if r['dataset']==ds]
 for arm in rr[0]['arms']:
  metrics={m:statistics.mean(r['arms'][arm][m]for r in rr)*100 for m in rr[0]['arms'][arm]};groups.append(dict(dataset=ds,arm=arm,metrics_percent=metrics))
write(B/'ROOT_PRELIMINARY_POSTSEAL_METRICS.json',dict(status='actual_postseal_metrics_pending_complete_source_readback',rows=rows,groups=groups,GT_read=True,GT_update=False,independent_metric_max_abs_error=max(errors),metric_count=len(errors),GT_source_bindings=bindings))
for ds in ['hc2','vidstg']:print(ds,{g['arm']:round(g['metrics_percent']['m_vIoU'],6)for g in groups if g['dataset']==ds})
print('max metric error',max(errors))
