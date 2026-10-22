"""Independent CPU scoring of the user-prioritized63 native448 captures.

Explicit new version: same original scalar scorer, extra cached-resolution/layer
controls on the matched63; no production or old deferred-plan mutation.
"""
import os,sys,csv
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.highres_priority.run import SUP,OUT,validate,read,write,sha
from scripts.interval_pool.score import score_boxes,summarize
from scripts.joint_boundary_readout.score import paired
from scripts.branch_decomposition.run import utility
PUBLIC=ROOT/'results/highres_priority_v2';COMMON=['Source','Only224six','Only448','Fusion'];LIGHT=['SingleHigh0','SingleHigh1','SingleFusion0','SingleFusion1','Only224Last','High6','High6Fusion']
def evaluate():
 torch.set_num_threads(1);os.nice(15);p=validate();seal=read(SUP/'sealed.json');assert seal['fingerprint']==p['fingerprint']and seal['GT_read']is False and seal['queries']==63;assert len(seal['files'])==63;assert not(PUBLIC/'report.json').exists()
 for r in seal['files']:assert sha(SUP/r['prediction_path'])==r['prediction_sha256']and sha(SUP/r['posterior_path'])==r['posterior_sha256']
 labelpath=ROOT/p['labels']['path'];assert sha(labelpath)==p['labels']['sha256'];import json
 labels={r['id']:r for r in [json.loads(l)for l in labelpath.read_text().splitlines()if l.strip()]};records={r['id']:r for r in [json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l.strip()]};old=read(ROOT/'results/tube_fusion_hc64/report.json');rows=[];checks=0;error=0.;signed=[];posterior_diagnostics=[]
 entries=list(seal['files']);first=read(OUT/'sealed.json')['files'][0];first=dict(first,id='RC001',cached_one=True,prediction_path=first['path'],prediction_sha256=first['sha256']);entries=[first,*entries]
 for receipt in entries:
  key=receipt['query'];single=receipt.get('cached_one',False);base=OUT if single else SUP;path=base/receipt['prediction_path'];assert sha(path)==receipt['prediction_sha256'];z=np.load(path);truth=labels[receipt['id']];record=records[receipt['id']];boxes={int(k):v for k,v in truth['boxes'].items()};names=COMMON if single else COMMON+LIGHT;values={};fixed={}
  for name in names:
   span=z[name+'_interval'].tolist();values[name],e=score_boxes(z[name],record,boxes,span,truth['interval'],z['positions8'].tolist());error=max(error,e);checks+=1;fixed[name],e=score_boxes(z['FixedSpatial'],record,boxes,span,truth['interval'],z['positions8'].tolist());error=max(error,e);checks+=1
  oldrow=next(r for r in old['queries_anonymous']if r['query']==key)
  for name,reference in [('Only224six','Global6'),('Fusion','Strong_Reference')]:
   for metric in ['v','t','start_error_seconds','end_error_seconds']:assert abs(values[name][metric]-oldrow['values'][reference][metric])<2e-10
  row=dict(query=key,parent=truth['parent'],values=values,fixed_spatial=fixed);rows.append(row)
  for name in names:
   a,b=values[name],values['Only224six'];signed.append(dict(query=key,method=name,vIoU_delta_pp=100*(a['v']-b['v']),tIoU_delta_pp=100*(a['t']-b['t']),fixed_space_vIoU_delta_pp=100*(fixed[name]['v']-fixed['Only224six']['v']),start_error_delta_seconds=a['start_error_seconds']-b['start_error_seconds'],end_error_delta_seconds=a['end_error_seconds']-b['end_error_seconds'],rescued_half=b['t']<.5<=a['t'],harmed_half=a['t']<.5<=b['t'],severe_v_loss= a['v']-b['v'] < -.05))
  qs=np.load(OUT/receipt['posterior_path'])if single else torch.load(SUP/receipt['posterior_path'],map_location='cpu',weights_only=False)['posteriors'];ids=z['frame_ids'];details={}
  for name in ['Only224six','Only448','Fusion']:
   q=qs[name];i,j=np.nonzero(q);a,b=truth['interval'];ov=np.maximum(0,np.minimum(b,ids[j]+1)-np.maximum(a,ids[i]));t=ov/((b-a)+(ids[j]+1-ids[i])-ov);qpos=q[q>0];details[name]=dict(entropy=float(-(qpos*np.log(qpos)).sum()),mass_tIoU_half=float(q[i,j][t>=.5].sum()),mass_tIoU_seven=float(q[i,j][t>=.7].sum()),expected_tIoU_against_GT=utility(ids,q,truth['interval']))
  posterior_diagnostics.append(dict(query=key,details=details))
 assert len(rows)==len({r['parent']for r in rows})==64 and checks==1394
 # Paired helper uses the explicit baseline name; no per-query branch selection.
 all_comparisons={name:paired(rows,name,'Only224six')for name in ['Only448','Fusion']};rr=[r for r in rows if all(name in r['values']for name in LIGHT)];assert len(rr)==63
 light_comparisons={name:{base:paired(rr,name,base)for base in ['Only224six','Fusion','Source']}for name in LIGHT};patterns={}
 for tag,condition in dict(high_rescues=lambda g,h,f:g<.5<=h,fusion_only_rescues=lambda g,h,f:g<.5 and h<.5<=f,high_harms=lambda g,h,f:h<.5<=g,fusion_harms=lambda g,h,f:f<.5<=g).items():
  selected=[r for r in rows if condition(r['values']['Only224six']['t'],r['values']['Only448']['t'],r['values']['Fusion']['t'])];patterns[tag]=dict(queries=len(selected),anonymous=[r['query']for r in selected])
 result=dict(status='completed_authorized448_branch_supplement',exposure='All64 repeatedly exposed development; no untouched confirmation. Single high-view arms have63, common arms64; never compare unmatched means.',queries64=len(rows),light_queries63=len(rr),means64=summarize(rows,COMMON,'values'),fixed_spatial_means64=summarize(rows,COMMON,'fixed_spatial'),paired64=all_comparisons,means63=summarize(rr,COMMON+LIGHT,'values'),fixed_spatial_means63=summarize(rr,COMMON+LIGHT,'fixed_spatial'),paired_light63=light_comparisons,patterns=patterns,posterior_diagnostics=posterior_diagnostics,queries_anonymous=[{k:v for k,v in r.items()if k!='parent'}for r in rows],cost=dict(mean448_seconds=float(np.mean([r['Source448_seconds']for r in seal['files']]))),proof=dict(all63_sealed_before_labels=True,queries448_new=63,previous_448_reused=1,Source224_new_calls=0,new_expert_calls=0,updates=0,official_dense_checks=checks,official_dense_maxerror=error,plan_fingerprint=p['fingerprint'],seal_sha256=sha(SUP/'sealed.json')),decision=dict(fixed_primary='SingleFusion0',offset1='prespecified sensitivity; not selectable by queryGT',production_changed=False,automatic_adoption=False,one_view_deployment_timing_verified=False,reason='Precision panel only. Root must inspect actual rescue/harm, timing scope and compatibility before selecting further work; never claim a measured deployment speedup from discarded-view cached readout.'))
 validate();assert not torch.cuda.is_initialized();PUBLIC.mkdir(parents=True,exist_ok=True);write(PUBLIC/'report.json',result)
 with(PUBLIC/'signed_queries.tsv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(signed[0]),delimiter='\t');w.writeheader();w.writerows(signed)
 write(SUP/'score_receipt.json',dict(status='scored',report_sha256=sha(PUBLIC/'report.json'),signed_sha256=sha(PUBLIC/'signed_queries.tsv')));state=read(SUP/'state.json');state['status']='scored_awaiting_root_analysis_and_publication';write(SUP/'state.json',state);print(json.dumps(dict(means64=result['means64'],patterns=patterns),indent=2))
if __name__=='__main__':evaluate()
