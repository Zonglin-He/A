"""Offline reviewer diagnostics from audited scalars, saved evidence and logits.

No model construction, optimizer, acquisition or method/parameter selection.
"""
from common_v1 import *
import numpy as np,math
EPS=1e-10
def dump_table(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w')as f:
  w=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def stats(rr,fields):
 if not rr:return dict(queries=0,parents=0,statistics={})
 parents=sorted({r['parent']for r in rr});ix={p:i for i,p in enumerate(parents)};n=len(parents);m=len(fields);s=np.zeros((n,m));c=np.zeros((n,m))
 for r in rr:
  for j,k in enumerate(fields):
   if r.get(k)is not None:s[ix[r['parent']],j]+=r[k];c[ix[r['parent']],j]+=1
 w=np.random.default_rng(20261022).multinomial(n,[1/n]*n,size=10000);num=w@s;den=w@c;boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);out={}
 for j,k in enumerate(fields):
  vals=[r[k]for r in rr if r.get(k)is not None];b=boot[:,j];b=b[np.isfinite(b)]
  out[k]=dict(mean=float(np.mean(vals))if vals else None,CI95=np.quantile(b,[.025,.975]).tolist()if len(b)else None,queries=len(vals),parents=int((c[:,j]>0).sum()))
 return dict(queries=len(rr),parents=n,statistics=out)
def pair_stats(rr,a,b):
 z=[]
 for r in rr:
  x=r['values'][a];y=r['values'][b];dt=100*(x['t']-y['t']);dv=100*(x['v']-y['v'])
  z.append(dict(parent=r['parent'],delta_t_pp=dt,delta_v_pp=dv,improved_v_percent=100.*(dv>EPS),harmed_v_percent=100.*(dv< -EPS),severe_t_percent=100.*(dt< -5),severe_v_percent=100.*(dv< -5),severe_union_percent=100.*(dt< -5 or dv< -5),rescue_union_percent=100.*(dt>5 or dv>5)))
 return stats(z,list(z[0])[1:])if z else stats([],[])
def inputs():
 files=[D/'plan_v1.json',R/'outputs/p4_cached_risk_v1/plan_v1.json']
 for ds in COUNTS:
  for root,names in [(R/f'outputs/review_strengthening_v1/offline_current/{ds}',['derived.json','audit_receipt.json','sealed.json']),(R/f'outputs/review_strengthening_v1/matched_cpu/{ds}',['metrics.json','reporting_receipt.json','score_receipt.json']),(selected.D/'components'/ds,['metrics.json']),(R/'outputs/temporal_feedback_risk_v1'/ds,['metrics.json','reporting_receipt.json']),(R/'outputs/p4_cached_risk_v1'/ds,['private_scalar_rows.json','audit_receipt.json'])]:files += [root/f for f in names]
  files += [R/'outputs/tta_full_v1'/f'{ds}_sealed.json',R/'outputs/tta_full_v1'/f'{ds}_reporting_receipt.json',R/'outputs/tta_full_v1'/ds/'metrics.json']
 return files
def prepare():
 assert not(D/'statistics_plan_v1.json').exists();p=dict(schema='offline_reviewer_diagnostics_v1',authorization='2026-11-03 all reviewer diagnostics; offline annotations allowed for analysis only, never inference/tuning',tasks=['C: query-equal spatial IoU distance bins all/predicted-event/GT-event; same Table10 interval','D: declarative/interrogative x fixed expert IoU thirds and unscorable; spatial/temporal/final gains separated','E: saved native and actually post-update Tent/SAR entropy/logit/state norms and interval changes, matched native/joint readouts','F: per-query event/background evidence means and AUROC; source tIoU [0,.2),[.2,.5),[.5,.8),[.8,1]; Source quality/control gain/harm','risk: dV>+1,dV<-1,dV<=-10; Source vIoU>=.5 and >=.8 and tIoU>=.8','robustness: reuse all36 clean-vs-corrupt degradation and paired difference of degradation'],distance_bins='[0,.5),[.5,1),[1,2),[2,4),[4,inf); native actually GT-supported unobserved frames only; half-open event intervals matching official scorer; empty strata explicit, never synthetic zeros',quality_bins='low[0,1/3),medium[1/3,2/3),high[2/3,1],unscorable; preserve continuous original GT-supported anchor IoU; not quantile or performance-selected',AUROC='Within-query tie-aware Mann Whitney rank AUROC, only queries containing both GT event/background native sampled frames; retain one-class/missing as unscorable; query/parent CI, frames not independent',bootstrap='10000parent clusters seed20261022, query-equal; pointwise not familywise; full development exposure',no_model=True,no_optimizer=True,no_acquisition=True,no_GPU=True,code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),S/'common_v1.py']},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs()});p['fingerprint']=digest(p);atomic(D/'statistics_plan_v1.json',p);atomic(P/'statistics_protocol_v1.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'])
def check():
 p=read(D/'statistics_plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for name in ['code_sha256','input_sha256']:
  for f,h in p[name].items():assert sha(R/f)==h,f
 sys.addaudithook(deny_gpu_media);return p
def offline(ds):
 root=R/f'outputs/review_strengthening_v1/offline_current/{ds}';a=read(root/'audit_receipt.json');assert a['derived_sha256']==sha(root/'derived.json')and a['all_GT_supported_anchors_distances_IoUs_exact'];return read(root/'derived.json')['rows']
def scalar(ds):
 root=R/f'outputs/review_strengthening_v1/matched_cpu/{ds}';a=read(root/'reporting_receipt.json');assert a['all_arrays_cold_rederived_exact']and a['score_sha256']==sha(root/'score_receipt.json');return read(root/'metrics.json')['private_rows']
def c_d_risk(ds):
 check();start=time.perf_counter();oo=offline(ds);matched={r['id']:r for r in scalar(ds)};truth=selected.labels(ds);p4=read(R/f'outputs/p4_cached_risk_v1/{ds}/private_scalar_rows.json')['rows'];forms={x['id']:x['query_form']for x in p4};outdir=P/'offline'/ds
 # Existing frame IoUs are already independently audited; only membership masks and aggregation change.
 cells={};joined=[];public=[]
 for i,x in enumerate(oo,1):
  cur=fixed(selected.D/f'components_{ds}_sealed.json')['files'][x['id']];assert sha(selected.D/cur['path'])==cur['sha256']
  with np.load(selected.D/cur['path'])as z:iv=z['FullMAP_interval'].copy()
  gtiv=truth[x['id']]['interval'];assert x['parent']==matched[x['id']]['parent'];allframes=[f for b in x['distance']for f in b]
  for mode in ['all_unobserved','predicted_event','GT_event']:
   span=None if mode=='all_unobserved'else iv if mode=='predicted_event'else gtiv
   for bi in range(-1,5):
    frames=allframes if bi<0 else x['distance'][bi];ff=[f for f in frames if span is None or span[0]<=f['frame']<span[1]]
    name=f'{mode}/'+('all_distances'if bi<0 else f'bin{bi}');cells.setdefault(name,[])
    if ff:
     our=float(np.mean([f['values']['Ours']for f in ff]));inter=float(np.mean([f['values']['Interpolation']for f in ff]));s=dict(parent=x['parent'],ours_IoU_percent=100*our,interpolation_IoU_percent=100*inter,delta_IoU_pp=100*(our-inter),annotated_frames=len(ff),sum_ours_IoU=sum(f['values']['Ours']for f in ff),sum_interpolation_IoU=sum(f['values']['Interpolation']for f in ff));cells[name].append(s)
     public.append(dict(query=f'Q{i:05d}',group=name,frames=len(ff),ours_IoU_percent=100*our,interpolation_IoU_percent=100*inter,delta_IoU_pp=100*(our-inter)))
  q=x['quality'];qg='unscorable'if q is None else'low'if q<1/3 else'medium'if q<2/3 else'high';m=matched[x['id']];vals=m['values'];sf=vals['Ours']['unobserved_actual_sIoU'];si=vals['ExpertInterpolation']['unobserved_actual_sIoU']
  joined.append(dict(id=x['id'],parent=x['parent'],form=forms[x['id']],quality_group=qg,values=vals,unobserved_delta_pp=None if sf is None or si is None else 100*(sf-si),duration_group=m['duration_group']))
 distance={}
 for name,rr in cells.items():
  v=stats(rr,['ours_IoU_percent','interpolation_IoU_percent','delta_IoU_pp']);n=sum(r['annotated_frames']for r in rr);v.update(annotated_frames=n,frame_weighted_ours_IoU_percent=100*sum(r['sum_ours_IoU']for r in rr)/n if n else None,frame_weighted_interpolation_IoU_percent=100*sum(r['sum_interpolation_IoU']for r in rr)/n if n else None);distance[name]=v
 quality={}
 for form in ['all']+(['declarative','interrogative']if ds=='vidstg'else[]):
  for qg in ['all','low','medium','high','unscorable']:
   rr=[r for r in joined if(form=='all'or r['form']==form)and(qg=='all'or r['quality_group']==qg)];quality[f'{form}/{qg}']=dict(spatial_v=pair_stats(rr,'Ours','ExpertInterpolation'),unobserved=stats(rr,['unobserved_delta_pp']))
 riskrows=read(R/f'outputs/temporal_feedback_risk_v1/{ds}/metrics.json')['private_rows'];groups={'all':riskrows}
 for f,cut in [('v',.5),('v',.8),('t',.8)]:groups[f'Source_{f}_ge_{cut}']=[r for r in riskrows if r['values']['Source'][f]>=cut]
 risk={}
 for name,rr in groups.items():
  vals=[]
  for r in rr:
   dv=100*(r['values']['FullMAP']['v']-r['values']['Source']['v']);vals.append(dict(parent=r['parent'],delta_v_pp=dv,gain_gt1_percent=100.*(dv>1),loss_lt_minus1_percent=100.*(dv< -1),loss_le_minus10_percent=100.*(dv<=-10),negative_percent=100.*(dv< -EPS)))
  risk[name]=stats(vals,['delta_v_pp','gain_gt1_percent','loss_lt_minus1_percent','loss_le_minus10_percent','negative_percent'])
 atomic(outdir/'distance_event_membership.json',distance);atomic(outdir/'query_form_expert_quality_spatial.json',quality);atomic(outdir/'risk_thresholds.json',risk);dump_table(outdir/'distance_event_all_signed.tsv',public)
 atomic(D/'offline'/ds/'joined_private.json',dict(rows=joined));receipt=dict(status='derived_offline_membership_quality_risk_pending_independent_audit',queries=len(oo),code_sha256=sha(Path(__file__)),elapsed_wall_seconds=time.perf_counter()-start,worker=identity(),reused_frame_IoUs=True,new_predictions=0,new_GT_metric_scoring=0,new_model=0);atomic(outdir/'derive_receipt.json',receipt);print(ds,'C/D/risk derived',flush=True)
def entropy(z):
 import torch
 return torch.cat([-(x[0].T.softmax(1)*x[0].T.log_softmax(1)).sum(1)for x in z]).double().numpy()
def tta(ds):
 check();import torch
 torch.set_num_threads(1);root=R/'outputs/tta_full_v1';a=read(root/f'{ds}_reporting_receipt.json');assert a['status']=='verified_full_domain_Tent_SAR';seal=read(root/f'{ds}_sealed.json');metric={r['id']:r for r in read(root/ds/'metrics.json')['private_rows']};results={m:[]for m in ['Tent','SAR']};public=[];start=time.perf_counter()
 for i,(key,e)in enumerate(seal['files'].items(),1):
  assert sha(root/e['logits_path'])==e['logits_sha256']and sha(root/e['receipt_path'])==e['receipt_sha256'];v=torch.load(root/e['logits_path'],map_location='cpu',weights_only=False,mmap=True);native=v.get('native_logits')
  if native is None:
   idx=fixed(root/'context_index.json')[ds][key];assert sha(idx['path'])==idx['sha256'];ctx=torch.load(idx['path'],map_location='cpu',weights_only=False,mmap=True);ctx=ctx['temporal']if idx['kind']=='shared_native'else ctx;native=[x['native_logits']for x in ctx['offsets']]
  for m in results:
   trace=e['traces'][m];assert trace['reset_exact'];after=v['adapted_logits'][m];before_entropy=entropy(native);after_entropy=entropy(after);before_trace=np.asarray(trace['steps'][0]['forwards'][0]['entropy']);assert np.allclose(before_entropy,before_trace,rtol=0,atol=2e-6)
   delta=sum(float((x.double()-y.double()).square().sum())for x,y in zip(after,native))**.5;centered=sum(float(((x.double()-x.double().mean(1,keepdim=True))-(y.double()-y.double().mean(1,keepdim=True))).square().sum())for x,y in zip(after,native))**.5;vals=metric[key]['values'];d=dict(parent=metric[key]['parent'],entropy_before=float(before_entropy.mean()),entropy_after=float(after_entropy.mean()),entropy_delta=float((after_entropy-before_entropy).mean()),entropy_decreased_percent=100.*(after_entropy.mean()<before_entropy.mean()),parameter_update_L2=float(trace['parameter_delta']),logit_change_L2=delta,centered_logit_change_L2=centered,changed_logits_percent=100.*(delta>0),changed_native_interval_percent=100.*(vals[m+'Native']['t']!=vals['Source']['t']or vals[m+'Native']['v']!=vals['Source']['v']),native_delta_t_pp=100*(vals[m+'Native']['t']-vals['Source']['t']),native_delta_v_pp=100*(vals[m+'Native']['v']-vals['Source']['v']),joint_delta_t_pp=100*(vals[m+'JointMAP']['t']-vals['SourceJointMAP']['t']),joint_delta_v_pp=100*(vals[m+'JointMAP']['v']-vals['SourceJointMAP']['v']))
   # Interval changes use exact saved endpoints, not equality of scored IoUs.
   with np.load(root/e['path'])as z:d['changed_native_interval_percent']=100.*(not np.array_equal(z[m+'Native_interval'],z['Source_interval']))
   results[m].append(d);public.append(dict(query=f'Q{i:05d}',method=m,**{k:x for k,x in d.items()if k!='parent'}))
  if i%250==0:atomic(D/'tta'/ds/'progress.json',dict(completed=i,total=len(seal['files']),worker=identity()))
 report={m:stats(rr,[k for k in rr[0]if k!='parent'])for m,rr in results.items()};atomic(P/'tta'/ds/'diagnostics.json',dict(methods=report,original_verified_counts=a['diagnostics'],actual_post_all_updates_entropy=True,scope='Pinned official Tent/SAR algorithms on temporal-decoder LN affine only; fixed Source boxes, five episodic steps, no expert; not canonical BN/fullnetwork/continual ranking. Score-matched native/joint controls separated; no new optimizer.'));dump_table(P/'tta'/ds/'all_saved_optimizer_diagnostics.tsv',public);atomic(D/'tta'/ds/'private_rows.json',results);atomic(P/'tta'/ds/'derive_receipt.json',dict(status='derived_saved_logits_diagnostics_pending_independent_audit',queries=len(seal['files']),all_logit_receipt_SHA=True,elapsed_wall_seconds=time.perf_counter()-start,worker=identity(),new_model=0,new_optimizer=0,new_GT_scoring=0));print(ds,'TTA diagnostics derived',flush=True)
def auc(scores,labels):
 scores=np.asarray(scores);labels=np.asarray(labels,bool);p=int(labels.sum());n=len(labels)-p
 if p==0 or n==0:return None
 order=np.argsort(scores,kind='stable');rank=np.empty(len(scores),float);j=0
 while j<len(scores):
  k=j+1
  while k<len(scores)and scores[order[k]]==scores[order[j]]:k+=1
  rank[order[j:k]]=(j+1+k)/2;j=k
 return float((rank[labels].sum()-p*(p+1)/2)/(p*n))
def temporal_evidence(ds):
 check();dest=D/'temporal'/ds;audit=read(dest/'reporting_receipt.json');assert audit['all_arrays_cold_rederived_exact']and audit['metrics_sha256']==sha(dest/'metrics.json');seal=read(dest/'sealed.json');metrics={r['id']:r for r in read(dest/'metrics.json')['private_rows']};truth=selected.labels(ds);rows_out=[];pub=[]
 for i,row in enumerate(rows('temporal',ds),1):
  key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256'];g=truth[key];vals=metrics[key]['values'];x=vals['Source']['t'];group='tIoU_0_02'if x<.2 else'tIoU_02_05'if x<.5 else'tIoU_05_08'if x<.8 else'tIoU_08_10'
  with np.load(D/e['path'])as z:
   ids=z['frame_ids'];label=(ids>=g['interval'][0])&(ids<g['interval'][1]);q=z['q0'];pp=q[q>0];r=dict(parent=metrics[key]['parent'],id=key,source_t_group=group,source_joint_peak=float(q.max()),source_joint_entropy=float(-(pp*np.log(pp)).sum()),prototype_separation_SourceROI=float(z['prototype_separation'][0]),prototype_separation_CorrectedROI=float(z['prototype_separation'][1]),delta_t_pp=100*(vals['Ours']['t']-vals['Source']['t']),delta_v_pp=100*(vals['Ours']['v']-vals['Source']['v']),improved_t_percent=100.*(vals['Ours']['t']-vals['Source']['t']>EPS/100),harmed_t_percent=100.*(vals['Ours']['t']-vals['Source']['t']< -EPS/100),severe_t_percent=100.*(vals['Ours']['t']-vals['Source']['t']< -.05),new_zero_from_Source_ge05_percent=100.*(vals['Source']['t']>=.5 and vals['Ours']['t']==0))
   for name in ['SourceROI','CorrectedROI']:
    L=z['evidence_'+name];r[name+'_event_mean']=float(L[label].mean())if label.any()else None;r[name+'_background_mean']=float(L[~label].mean())if(~label).any()else None;r[name+'_AUROC']=auc(L,label)
   r['Corrected_minus_Source_AUROC']=None if r['CorrectedROI_AUROC']is None else r['CorrectedROI_AUROC']-r['SourceROI_AUROC'];r['native_frames']=len(ids);r['event_frames']=int(label.sum());r['background_frames']=int((~label).sum())
  rows_out.append(r);pub.append(dict(query=f'Q{i:05d}',**{k:v for k,v in r.items()if k not in ['id','parent']}))
 fields=[k for k in rows_out[0]if k not in ['parent','id','source_t_group','native_frames','event_frames','background_frames']];report={}
 groups={'all':rows_out,**{g:[r for r in rows_out if r['source_t_group']==g]for g in ['tIoU_0_02','tIoU_02_05','tIoU_05_08','tIoU_08_10']},'temporal_improved':[r for r in rows_out if r['delta_t_pp']>EPS],'temporal_harmed':[r for r in rows_out if r['delta_t_pp']< -EPS]}
 for name,rr in groups.items():report[name]=dict(**stats(rr,fields),native_frames=sum(r['native_frames']for r in rr),event_frames=sum(r['event_frames']for r in rr),background_frames=sum(r['background_frames']for r in rr),AUROC_unscorable_queries=sum(r['CorrectedROI_AUROC']is None for r in rr))
 atomic(P/'temporal_evidence'/ds/'report.json',report);dump_table(P/'temporal_evidence'/ds/'all_query_evidence.tsv',pub);atomic(D/'temporal_evidence'/ds/'private_rows.json',rows_out)
 # Connect expert quality with separated spatial and temporal comparisons, never an online gate.
 quality=read(D/'offline'/ds/'joined_private.json')['rows'];by={r['id']:r for r in quality};joined=[]
 for key,m in metrics.items():
  r=by[key];joined.append(dict(parent=m['parent'],form=r['form'],quality_group=r['quality_group'],values=m['values']))
 qq={}
 for form in ['all']+(['declarative','interrogative']if ds=='vidstg'else[]):
  for qg in ['all','low','medium','high','unscorable']:
   rr=[r for r in joined if(form=='all'or r['form']==form)and(qg=='all'or r['quality_group']==qg)];qq[f'{form}/{qg}']={a+'-'+b:pair_stats(rr,a,b)for a,b in [('ExpertInitTR','FixedE_SourceTR'),('Ours','ExpertInitTR'),('Ours','Source')]}
 atomic(P/'offline'/ds/'query_form_quality_separated_temporal_spatial.json',qq);atomic(P/'temporal_evidence'/ds/'derive_receipt.json',dict(status='offline_evidence_quality_analysis_derived_pending_independent_audit',queries=len(rows_out),AUROC_definition='tie-aware within query, no independent frame bootstrap',prediction_audit_sha256=sha(dest/'reporting_receipt.json'),code_sha256=sha(Path(__file__)),worker=identity(),new_model=0,new_GT_scoring=0));print(ds,'evidence diagnostics derived',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('task',choices=['prepare','offline','tta','evidence']);ap.add_argument('--dataset',choices=COUNTS);x=ap.parse_args();{'prepare':prepare,'offline':c_d_risk,'tta':tta,'evidence':temporal_evidence}[x.task](*([]if x.task=='prepare'else[x.dataset]))
