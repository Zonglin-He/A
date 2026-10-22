"""Publish/audit an already sealed and scored full run; no model or GT access.

Copies hash-bound public scores, verifies anonymous signed aggregation and raw
Final/Untuned aliases, and summarizes actual per-receipt cost across process resets.
"""
import argparse,csv,hashlib,json,sys,shutil
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def stat(values):
 a=np.asarray(values,dtype=float)
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),total=float(a.sum()))if len(a)else dict(n=0,mean=None,median=None,total=0.)
def aggregate(rows,report):
 n=report['queries'];assert len(rows)==n and len({r['query']for r in rows})==n
 parents=sorted({r['parent']for r in rows});assert len(parents)==report['parents']
 lookup={p:i for i,p in enumerate(parents)};weights=np.random.default_rng(20261022).multinomial(len(parents),[1/len(parents)]*len(parents),size=10000)
 error=0.
 def close(a,b):
  nonlocal error
  e=abs(float(a)-float(b));error=max(error,e);assert e<1e-10,(a,b,e)
 fields={'v':'m_vIoU','t':'m_tIoU','observed_actual_sIoU':'m_sIoU_observed_actual','unobserved_actual_sIoU':'m_sIoU_unobserved_actual','boundary_MAE_seconds':'boundary_MAE_seconds','tIoU_ge_05':'tIoU_ge_05'}
 for arm in ['Source','Untuned','Final']:
  for f,k in fields.items():close(np.mean([float(r[arm+'_'+f])for r in rows if r[arm+'_'+f]!='']),report['means'][arm][k])
  v=np.array([float(r[arm+'_v'])for r in rows]);close((v>.3).mean(),report['means'][arm]['vIoU@0.3']);close((v>.5).mean(),report['means'][arm]['vIoU@0.5'])
 for f,scale in [('v',100.),('t',100.),('observed_actual_sIoU',100.),('unobserved_actual_sIoU',100.),('boundary_MAE_seconds',1.)]:
  sums=np.zeros(len(parents));counts=sums.copy();delta=[]
  for r in rows:
   x,y=r['Final_'+f],r['Source_'+f]
   if x==''or y=='':continue
   d=scale*(float(x)-float(y));delta.append(d);i=lookup[r['parent']];sums[i]+=d;counts[i]+=1
  delta=np.array(delta);den=weights@counts;boot=(weights@sums)[den>0]/den[den>0];p=report['paired']['Final-Source'][f];close(delta.mean(),p['mean'])
  for a,b in zip(np.quantile(boot,[.025,.975]),p['CI95']):close(a,b)
  assert len(delta)==p['queries'];assert (int((delta>1e-10).sum()),int((delta < -1e-10).sum()),int((abs(delta)<=1e-10).sum()))==(p['increased'],p['decreased'],p['unchanged'])
  if scale==100.:assert int((delta < -5).sum())==p['declined_over5pp']
 for r in rows:
  for f in fields:assert r['Final_'+f]==r['Untuned_'+f]
  dv=100*(float(r['Final_v'])-float(r['Source_v']));close(dv,float(r['Final_minus_Source_v_pp']));assert (dv < -5)==(r['severe_Final_vs_Source']=='True')
 return dict(signed_rows=n,parents=len(parents),aliases=n,mean_threshold_count_and_parent_cluster_CI_checks=True,max_absolute_aggregation_error=error)

def costs(rr):
 fresh=[r for r in rr if not r['reused']['complete_capture_hit']];capture=[r for r in rr if r['reused']['complete_capture_hit']]
 result=dict(queries=len(rr),fresh_captures=len(fresh),complete_capture_hits=len(capture),query_wall_seconds=stat([r['query_wall_seconds']for r in rr]),fresh_query_wall_seconds=stat([r['query_wall_seconds']for r in fresh]),capture_reuse_wall_seconds=stat([r['query_wall_seconds']for r in capture]))
 hits=misses=0;transfer=0.;last=None
 for r in rr:
  b,a=r['visual_cache_before'],r['visual_cache_after']
  assert a['hits']-b['hits']==r['visual_hits']>=0 and a['misses']-b['misses']==r['visual_misses']>=0
  if last is not None:assert b['hits']==last['hits']and b['misses']==last['misses'],'Unexpected cache counter reset inside an explicit worker segment'
  last=a;hits+=r['visual_hits'];misses+=r['visual_misses'];x=a['key_and_transfer_seconds']-b['key_and_transfer_seconds'];assert x>=0;transfer+=x
 result.update(visual_hits=hits,visual_misses=misses,visual_hit_fraction=hits/(hits+misses)if hits+misses else None,visual_key_transfer_seconds=transfer,RGB_hits=sum(r.get('RGB',{}).get('hit',False)for r in fresh),RGB_misses=sum(not r.get('RGB',{}).get('hit',False)for r in fresh),actual_ReferDINO_query_forwards=sum(r['capture_cost'].get('ReferDINO_query_forwards',0)for r in rr),reused_expert_evidence_queries=sum(bool(r['capture_cost'].get('ReferDINO_cache_hit',False))for r in fresh))
 groups={'RGB_cache_hit':[r for r in fresh if r['RGB']['hit']],'RGB_cache_miss':[r for r in fresh if not r['RGB']['hit']],'visual_all_hits':[r for r in fresh if r['visual_misses']==0],'visual_partial_hits':[r for r in fresh if r['visual_hits']>0 and r['visual_misses']>0],'visual_no_hits':[r for r in fresh if r['visual_hits']==0]}
 result['query_wall_by_cache_group']={k:stat([r['query_wall_seconds']for r in v])for k,v in groups.items()}
 result['fresh_stage_seconds']={}
 for k in ['Source224_seconds','expert_current_wall_seconds','evidence_seconds','ERef_seconds','Source_to_GPU_seconds','Source_to_CPU_seconds','parse_seconds']:
  result['fresh_stage_seconds'][k]=stat([r['capture_cost'][k]for r in fresh if k in r['capture_cost']])
 result['fresh_stage_seconds'].update(RGB_decode_seconds=stat([r['RGB']['seconds']for r in fresh]),Global6_posterior_seconds=stat([r['posterior_seconds']['Untuned']for r in fresh]),BoxFixed_DP_seconds=stat([r['DP_seconds']['Untuned']for r in fresh]),six_layer_head_compute_seconds=stat([r['capture_cost']['six_layer_head']['head_seconds']for r in fresh]),six_layer_head_transfer_seconds=stat([r['capture_cost']['six_layer_head']['hidden_transfer_seconds']for r in fresh]),source_initial_load_seconds=stat([r['Source_initial_load_seconds']for r in fresh]))
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results/boxfixed_full_v1'));a=p.parse_args();root=a.runtime_root.resolve();run=root/'outputs/boxfixed_full_v1';plan=read(run/'plan.json');state=read(run/'state.json');assert state['status']in ['awaiting_publication','complete_published'];assert all(state['jobs'][ds]=='scored'for ds in ['hc2','vidstg'])
 assert sha(run/'plan.json')=='85d13696c99b0de93a0cd2b49201ab51bf35fe1baaa485a14e5e4594cc0c3b32';assert plan['fingerprint']=='2ebd134dfd055c8e17e01859b706d56d187b15b8907a8129c49669539d84ea6e'
 assert all(sha(root/k)==v for k,v in plan['sources'].items())
 audit={};eff={};reports={};public_files={}
 for ds,n in [('hc2',1902),('vidstg',10303)]:
  src=run/ds;seal=read(src/'sealed.json');receipt=read(src/'score_receipt.json');assert seal['queries']==n and len(seal['files'])==n and seal['fingerprint']==plan['fingerprint'];assert receipt['status']=='scored'and receipt['seal_sha256']==sha(src/'sealed.json')
  entries=[]
  for rel,h in receipt['public_files'].items():
   q=root/rel;assert sha(q)==h;dest=a.output/ds/q.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dest);public_files[str(Path(ds)/q.name)]=h
   if q.suffix=='.tsv':entries.extend(csv.DictReader(q.open(),delimiter='\t'))
  report=read(root/'results/boxfixed_full_v1'/ds/'report.json');assert report['queries']==n and report['proof']['identical_Final_Untuned_aliases']==n
  assert report['proof']['all_predictions_sealed_before_GT']and report['proof']['plan_sha256']==sha(run/'plan.json')and report['proof']['seal_sha256']==sha(src/'sealed.json')
  records=[json.loads(s)for s in (root/plan['jobs'][ds]['manifest']).read_text().splitlines()];ids=[r['id']for r in records];assert len(ids)==n and set(ids)==set(seal['files']);del records
  rr=[];aliases=0
  for key in ids:
   item=seal['files'][key];pred=src/item['path'];rp=src/item['receipt_path'];assert sha(pred)==item['sha256']and sha(rp)==item['receipt_sha256'];r=read(rp);assert r['prediction_sha256']==sha(pred)and r['fingerprint']==plan['fingerprint']and r['GT_read']is False and r['Final_aliases_Untuned']
   assert r['params']==plan['profile']['datasets'][ds]and r['updates']==r['Source448_calls']==r['extra_decoder_calls']==0
   assert r['DP_seconds']['Final']==r['posterior_seconds']['Final']==0
   with np.load(pred,allow_pickle=False)as z:assert np.array_equal(z['Final'],z['Untuned'])and np.array_equal(z['Final_interval'],z['Untuned_interval']);aliases+=1
   rr.append(r)
  checks=aggregate(entries,report);checks.update(predictions_and_receipts_hashed=n,raw_Final_Untuned_aliases=aliases,seal_sha256=sha(src/'sealed.json'),score_receipt_sha256=sha(src/'score_receipt.json'),official_dense_checks=report['proof']['official_dense_checks'],official_dense_max_error=report['proof']['official_dense_max_error']);audit[ds]=checks
  segments={'original_HC_worker':rr}if ds=='hc2'else {'before_pause_6088':rr[:6088],'resumed_remaining_4215':rr[6088:]}
  eff[ds]=dict(segments={k:costs(v)for k,v in segments.items()},combined_actual_receipts=dict(queries=n,query_wall_seconds=stat([r['query_wall_seconds']for r in rr]),fresh_query_wall_seconds=stat([r['query_wall_seconds']for r in rr if not r['reused']['complete_capture_hit']])),Final_aliases_Untuned=True,single_posterior_and_DP_computation_per_query=True,Source448_calls=0,extra_decoder_calls=0,updates=0)
  for k in ['visual_hits','visual_misses','RGB_hits']:assert sum(v[k]for v in eff[ds]['segments'].values())==report['efficiency'][k],(ds,k)
  severe=[r for r in entries if r['severe_Final_vs_Source']=='True'];fp=a.output/ds/'severe_declines.tsv'
  with fp.open('w')as f:
   w=csv.DictWriter(f,fieldnames=list(entries[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(severe)
  reports[ds]=report
 write(a.output/'verification.json',dict(status='verified',datasets=audit,source_files_hashed=len(plan['sources']),public_score_files=public_files,GT_read=False,new_scoring=False,new_model_forwards=0,new_expert_calls=0))
 write(a.output/'efficiency.json',dict(datasets=eff,scope='Actual saved per-query work; original complete-capture cost excluded. Worker segments kept separate across pause/resume. Stage times overlap and must not be summed. Query walls include shared Source/expert/evidence plus one Global6/DP because Final aliases Untuned; process/service startup and loop housekeeping outside the receipt are not a standalone cold deployment benchmark. Frozen scorer scope text saying BOTH Final/Untuned executed is generic and incorrect here; zero Final stage costs and all raw aliases explicitly verified.',old_cost_not_relabelled=True,cache_counter_reset_handled_without_negative_delta=True))
 write(a.output/'protocol.json',dict(method='BoxFixed+Global6 old hull+original physical MBR',plan_fingerprint=plan['fingerprint'],plan_sha256=sha(run/'plan.json'),runtime_fingerprint=plan['runtime_fingerprint'],profile=plan['profile']['datasets'],source_resolution=224,offsets=[0,1],native_temporal_layers=6,experts=1,observations=8,Source448_calls=0,extra_decoder_calls=0,updates=0,Final_Untuned='identical aliases, no repeated posterior/DP',source_snapshot=plan['sources'],datasets={ds:dict(queries=r['queries'],parents=r['parents'],manifest_sha256=plan['jobs'][ds]['manifest_sha256'],labels_sha256=r['proof']['labels_sha256'],seal_sha256=r['proof']['seal_sha256'])for ds,r in reports.items()},frozen_controller='https://github.com/Zonglin-He/A/blob/c310af5888dca78d6ea7b0a280cb67d209a90337/scripts/boxfixed_tuning/full.py',frozen_scorer='https://github.com/Zonglin-He/A/blob/c310af5888dca78d6ea7b0a280cb67d209a90337/scripts/boxfixed_tuning/full_score.py',exposure=reports['hc2']['exposure']))
 print(json.dumps(dict(status='verified',queries=sum(r['queries']for r in reports.values()),datasets=audit)))
if __name__=='__main__':main()
