"""Independent whole-domain official CPU scoring and cold sufficient-state audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from cg_common import *
import argparse,subprocess,traceback
def score(ds):
 import numpy as np,torch
 from full_score import metric
 torch.set_num_threads(1);validate();seal=read(D/f'{ds}_sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=labels(ds);out=[];checks=0;maxerr=0.
 for row in legacy().rows(ds):
  key=row['id'];e=committed(ds,key);assert e==seal['files'][key];g=truth[key];values={}
  with np.load(D/e['path'])as z:
   inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for arm in ARMS:
    values[arm],err=metric(z[arm],inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);assert err<2e-10;maxerr=max(maxerr,err);checks+=1
  out.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
 folder=P/ds;report=suite().parent_report(out,ARMS,PAIRS,folder);suite().signed_tails(out,PAIRS,folder);atomic(D/ds/'metrics.json',dict(private_rows=out,report=report));rec=dict(status='scored',queries=len(out),checks=checks,max_error=maxerr,seal_sha256=sha(D/f'{ds}_sealed.json'),whole_domain_sealed_before_GT=True);atomic(D/f'{ds}_score_receipt.json',rec);atomic(folder/'score_receipt.json',rec)
def audit(ds):
 import torch,numpy as np
 from cg_core import replay,checked_parent
 torch.set_num_threads(1);validate();seal=read(D/f'{ds}_sealed.json');sp=D/f'{ds}_score_receipt.json';assert read(sp)['seal_sha256']==sha(D/f'{ds}_sealed.json');checks=0;segments={}
 for row in legacy().rows(ds):
  key=row['id'];e=committed(ds,key);assert e==seal['files'][key]and e['input_binding']==legacy().row_binding(row);c=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);old,oe=checked_parent(ds,key);assert c['proof']==e['proof']and c['input_binding']==e['input_binding'];assert c['proof']['original_CG_compact_sha256']==oe['compact_sha256'];ac=c['acquisition'];assert sha(ac['teacher_path'])==ac['teacher_sha256'];assert not c['GT_read']and c['first_aux_DP_in_selected_path']is False
  a=replay(c)
  with np.load(D/e['path'])as z:assert all(np.array_equal(z[k],a[k])for k in a)
  assert np.array_equal(c['Source'],old['Source'])and np.array_equal(c['q0'],old['q0']);hs=fixed(parent.D/f'CG_{ds}_sealed.json')['files'][key];assert sha(parent.D/hs['path'])==hs['sha256']and sha(parent.D/hs['receipt_path'])==hs['receipt_sha256']
  with np.load(parent.D/hs['path'])as z:assert np.array_equal(z['FullMAP'],a['HistoricalTwoDP'])and np.array_equal(z['FullMAP_interval'],a['HistoricalTwoDP_interval'])
  checks+=3;w=e['worker'];ident=f"{w['pid']}:{w['startticks']}";s=segments.setdefault(ident,dict(queries=0,legacy_raw_reused=0,new_Source=0,new_expert=0,Source_seconds=0.,pipeline_seconds=0.,compatibility_seconds=0.,selected_finish_seconds=0.));s['queries']+=1;s['legacy_raw_reused']+=int(e['legacy_raw_reused']);s['new_Source']+=int(not e['legacy_raw_reused'])
  for k in ['Source_seconds','pipeline_seconds','compatibility_seconds','selected_finish_seconds']:s[k]+=e[k]
 rr=read(D/ds/'metrics.json')['private_rows'];report=read(P/ds/'report.json')
 for a,mm in report['means'].items():
  for k,v in mm.items():
   vals=[x['values'][a][k]for x in rr if x['values'][a][k]is not None];assert(v is None and not vals)or abs(v-float(np.mean(vals)))<1e-12
 rec=dict(status='verified_whole_domain',queries=len(rr),SHA_checks=checks,all_compact_arrays_cold_rederived_exact=True,all_available_Source_native_logits_q0_original_exact=True,actual_E_ROI_pool_checked_before_raw_disposal=True,all_means_recomputed=True,score_sha256=sha(sp),worker_segments=segments,model_load=seal['model_load'],actual_counts=seal['counts'],inherited_expert_and_Source_acquisition_cost_nonzero=True,new_GT_scoring=0,scope='Actual acquisition and selected finish/compatibility/pipeline/load are nested or separate as recorded; not deployment latency. Owned new raw grids discarded only after committed exact cold replay; sufficient states do not permit arbitrary later repooling.');atomic(D/f'{ds}_reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec)
def domain(ds):
 validate();atomic(D/f'{ds}_CPU_worker.json',identity())
 try:
  for action in ['score','audit']:
   atomic(D/f'{ds}_CPU_state.json',dict(status=action));log=(D/f'{ds}_{action}.log').open('ab');c=subprocess.Popen([sys.executable,'-B',__file__,action,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);atomic(D/f'{ds}_{action}_child.json',identity(c.pid));rc=c.wait();log.close();atomic(D/f'{ds}_{action}_exit.json',dict(code=rc));assert rc==0,'Failure preserved; no retry'
  atomic(D/f'{ds}_CPU_state.json',dict(status='complete_scored_audited_pending_root_analysis'))
 except BaseException:atomic(D/f'{ds}_CPU_state.json',dict(status='failed_preserved',error=traceback.format_exc()));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['score','audit','domain']);p.add_argument('--dataset',choices=COUNTS,required=True);z=p.parse_args();globals()[z.command](z.dataset)
