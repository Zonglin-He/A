"""Full-list companion of the user-selected original128 log-size/nearest-tail table."""
from pathlib import Path
import os,sys,json,hashlib,time,csv,importlib.util
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
R=Path(__file__).resolve().parents[3];SIX=R/'outputs/spatial_temporal_six_v1';F=R/'outputs/shared_time_spatial_recalc_v2';D=R/'outputs/spatial_full_requested_table_v1';P=R/'results/spatial_full_requested_table_v1'
spec=importlib.util.spec_from_file_location('original128_table_export',R/'results/spatial128_requested_table_v1/report_code/export.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
read=h.read;sha=h.sha;write=h.write;cluster=h.cluster
def main():
 start=time.perf_counter();sel=read(SIX/'matching_pilot/selection.json');out={};sources={}
 sys.path[:0]=[str(R/'studies/reviewer_key_controls_v1'),str(R)]
 from common_v1 import selected,deny_gpu_media
 from stvg.runtime import guard
 sys.addaudithook(deny_gpu_media);sys.addaudithook(guard)
 for ds,count in [('hc2',1902),('vidstg',10303)]:
  base=SIX/'pure_full'/ds;pub=R/'results/spatial_temporal_six_v1/pure_full'/ds;ack=read(base/'reporting_receipt.json');assert ack['all_arrays_cold_exact']and ack['report_sha256']==sha(pub/'report.json')and ack['metrics_sha256']==sha(base/'metrics.json');rep=read(pub/'report.json');rr=read(base/'metrics.json')['private_rows'];seal=read(base/'sealed.json');fresh=read(F/ds/'sealed.json');rowmap={r['id']:r for r in selected.records(ds)};lookup={r['id']:r for r in rr};assert len(rr)==len(lookup)==len(seal['files'])==len(fresh['files'])==count and set(lookup)==set(rowmap)==set(seal['files'])==set(fresh['files']);mx=0.;checks=0
  for r in rr:
   key=r['id'];e=seal['files'][key];f=fresh['files'][key];assert sha(SIX/e['path'])==e['sha256']and sha(SIX/e['receipt_path'])==e['receipt_sha256']and sha(F/f['path'])==f['sha256'];meta=rowmap[key]['metadata']
   with np.load(SIX/e['path'])as z,np.load(F/f['path'])as cur:
    ids=z['frame_ids'];pos=z['positions8'];assert np.array_equal(ids,cur['frame_ids'])and np.array_equal(pos,cur['positions8'])and np.array_equal(z['Current_interval'],cur['shared_interval'])and np.array_equal(z['PureMatched_interval'],z['Current_interval'])and np.array_equal(z['Current'],cur['Full'])
    xy=cur['raw_expert_xyxy']/np.array([meta['width'],meta['height']]*2);raw=np.column_stack([(xy[:,0]+xy[:,2])/2,(xy[:,1]+xy[:,3])/2,xy[:,2]-xy[:,0],xy[:,3]-xy[:,1]]);pure=np.repeat(raw[:1],len(ids),axis=0);pure[pos[-1]+1:]=raw[-1];inside=(ids>=ids[pos[0]])&(ids<=ids[pos[-1]])
    for k in range(2):pure[inside,k]=np.interp(ids[inside],ids[pos],raw[:,k])
    for k in range(2,4):pure[inside,k]=np.exp(np.interp(ids[inside],ids[pos],np.log(raw[:,k])))
    pure[pos]=raw;err=float(np.max(abs(pure-z['PureMatched'])));assert err<2e-14,(key,err);mx=max(mx,err);checks+=1
   assert r['values']['Current']['t']==r['values']['PureMatched']['t'];assert(r['values']['Current']['unobserved_actual_sIoU']is None)==(r['values']['PureMatched']['unobserved_actual_sIoU']is None)
   if checks%1000==0:write(D/(ds+'_progress.json'),dict(completed=checks,total=count,new_GT=0,new_GPU=0))
  prior={r['id']:r for r in read(SIX/'matching_pilot'/ds/'metrics.json')['private_rows']};assert set(prior)=={r['id']for r in sel[ds]}and len(prior)==128
  for key,r in prior.items():
   for a in ['PureMatched','Current']:
    for k in ['t','v','unobserved_actual_sIoU']:assert lookup[key]['values'][a][k]==r['values'][a][k]
  means={a:{k:100*float(np.mean([r['values'][a][k]for r in rr if r['values'][a][k]is not None]))for k in ['t','v','unobserved_actual_sIoU']}for a in ['PureMatched','Current']}
  for a,m in means.items():
   for k,v in m.items():assert abs(v-100*rep['means'][a][k])<1e-10
  paired={k:cluster(rr,k)for k in ['v','unobserved_actual_sIoU','t']}
  for k,z in paired.items():ref=rep['paired']['Current-PureMatched'][k];assert abs(z['mean']-ref['mean'])<1e-10 and np.max(abs(np.array(z['CI95'])-ref['CI95']))<1e-10
  support={a:sum(r['values'][a]['unobserved_actual_sIoU']is not None for r in rr)for a in ['PureMatched','Current']};out[ds]=dict(queries=count,parents=len({r['parent']for r in rr}),means_percent=means,paired_pp=paired,scorable_Unobs_queries=support,original128_all_t_v_unobserved_scalars_exact_subset=True,all_inputs_arrays_SHA_interval_anchors_nearest_tails_verified=True,independent_rebuild_max_roundoff=mx)
  for p in [base/'sealed.json',base/'reporting_receipt.json',base/'metrics.json',pub/'report.json',F/ds/'audit_receipt.json']:sources[str(p.relative_to(R))]=sha(p)
  pm={p:f'P{i:04}'for i,p in enumerate(sorted({r['parent']for r in rr}),1)}
  with(P/(ds+'_all_signed.tsv')).open('w')as f:
   w=csv.writer(f,delimiter='\t');w.writerow(['query','parent','Interpolation_vIoU','Our_vIoU','Interpolation_Unobs_IoU','Our_Unobs_IoU','delta_v_pp','delta_Unobs_pp'])
   for i,r in enumerate(rr,1):a,b=r['values']['PureMatched'],r['values']['Current'];w.writerow([f'Q{i:05}',pm[r['parent']],100*a['v'],100*b['v'],None if a['unobserved_actual_sIoU']is None else 100*a['unobserved_actual_sIoU'],None if b['unobserved_actual_sIoU']is None else 100*b['unobserved_actual_sIoU'],100*(b['v']-a['v']),None if a['unobserved_actual_sIoU']is None else 100*(b['unobserved_actual_sIoU']-a['unobserved_actual_sIoU'])])
 table=['| Spatial trajectory | HC vIoU | HC Unobs. IoU | Vid vIoU | Vid Unobs. IoU |','|---|---:|---:|---:|---:|']
 for label,a in [('Expert interpolation','PureMatched'),('Our path','Current')]:table.append('|'+label+'|'+'|'.join(f'{out[ds]["means_percent"][a][k]:.2f}'for ds in ['hc2','vidstg']for k in ['v','unobserved_actual_sIoU'])+'|')
 table.append('|Δ, pp|'+'|'.join(f'{out[ds]["paired_pp"][k]["mean"]:+.2f}'for ds in ['hc2','vidstg']for k in ['v','unobserved_actual_sIoU'])+'|')
 text='# Full-list companion of the original128 comparison\n\n'+'\n'.join(table)+'\n\nHC1902/227parents; Vid10303/732parents. Same complete query roster, original8ReferDINO observations and actual Full interval per query. Exact user-selected definition is center-linear/log-width-height interpolation with nearest rawexpert tails, no E tails and no corner-linear substitution. Original128/domain are exact scalar subsets of these complete results, verified per-query t/v/unobserved IoU.\n\nUnobserved IoU follows the same ORIGINAL128 definition: GT-supported native sampled frames excluding exact8observed frames, query-equal over scorable queries, not restricted to predicted event. HC1902 and Vid10300queries have support; Vid3null queries are excluded identically from both conditional IoU means, but all10303remain in primaryv. No dense/native support mixing.\n\nAll means and10000parent pointwise CIs independently reproduced from sealed previously audited full predictions/scores. Every original interpolation and Current file SHA re-read, sameFullinterval checked, nearest-tail log-size interpolation independently reconstructed, Full equals the latest separate full-recalculation arrays. No new Source/expert/model/GPU/GT scoring/prediction, no hardanchor fix or manuscript change. Existing full negative/positive results are preserved. Full v gain positiveHC; Vid v point estimate-.146243pp CI[-.294717,-.000487], upper bound very close0; Unobs+.243630CI[-.305862,.790250]crosses0. Exposed query roster/pointwise statistics, no target-based tuning. Cached verification wall is not deployment time; inherited acquisition nonzero.\n'
 (P/'TABLE.md').write_text(text);write(P/'report.json',dict(status='complete_same_definition_full_table_reused_verified_exported',datasets=out,new_Source=0,new_expert=0,new_GPU=0,new_GT_scoring=0,scope='Exact full-list companion to original128 log-size/nearest-rawexpert-tail table; no new algorithm variant.'));write(D/'completion.json',dict(status='complete_reuse_export_all12205_same_definition',source_sha256=sources,report_sha256=sha(P/'report.json'),table_sha256=sha(P/'TABLE.md'),code_sha256=sha(Path(__file__)),helper_sha256=sha(R/'results/spatial128_requested_table_v1/report_code/export.py'),new_Source=0,new_expert=0,new_GPU=0,new_GT_scoring=0,new_predictions=0,elapsed_seconds=time.perf_counter()-start,original_source_results_already_published=True,this_convenience_export_publication='local; source results retain existing publication'));print('\n'.join(table))
if __name__=='__main__':P.mkdir(parents=True,exist_ok=True);main()
