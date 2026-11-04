"""Re-export only the original fixed128 nearest-expert-tail log-size comparison."""
from pathlib import Path
import os,json,hashlib,csv,time
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
R=Path(__file__).resolve().parents[3];D=R/'outputs/spatial128_requested_table_v1';P=R/'results/spatial128_requested_table_v1';M=R/'outputs/spatial_temporal_six_v1/matching_pilot';MP=R/'results/spatial_temporal_six_v1/matching_pilot';F=R/'outputs/shared_time_spatial_recalc_v2'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def cluster(rows,key):
 parents=sorted({r['parent']for r in rows});index={p:i for i,p in enumerate(parents)};n=len(parents);s=np.zeros(n);count=np.zeros(n);delta=[]
 for r in rows:
  a,b=r['values']['Current'][key],r['values']['PureMatched'][key]
  if a is None or b is None:continue
  v=100*(a-b);j=index[r['parent']];s[j]+=v;count[j]+=1;delta.append(v)
 w=np.random.default_rng(20261022).multinomial(n,[1/n]*n,size=10000);den=w@count;z=(w@s)[den>0]/den[den>0];delta=np.array(delta)
 return dict(mean=float(delta.mean()),CI95=np.quantile(z,[.025,.975]).tolist(),queries=len(delta),improved=int((delta>1e-10).sum()),harmed=int((delta< -1e-10).sum()),unchanged=int((abs(delta)<=1e-10).sum()))
def main():
 start=time.perf_counter();selection=read(M/'selection.json');assert sha(M/'selection.json')==sha(R/'outputs/matched_pipeline_timing_v1/selection.json');all_reports={};bound={str(M/'selection.json'):sha(M/'selection.json')}
 original=read(R/'outputs/pure_expert_interpolation_v1/inventory.json');suite=R/'outputs/simplepath_full_v2'
 for ds in ['hc2','vidstg']:
  ack=read(M/ds/'reporting_receipt.json');assert ack['metrics_sha256']==sha(M/ds/'metrics.json')and ack['report_sha256']==sha(MP/ds/'report.json')and ack['distance_report_sha256']==sha(MP/ds/'distance_report.json');report=read(MP/ds/'report.json');distance=read(MP/ds/'distance_report.json');rr=read(M/ds/'metrics.json')['private_rows'];lookup={r['id']:r for r in rr};chosen=selection[ds];assert len(chosen)==len(rr)==len(lookup)==128 and len({r['parent']for r in rr})==128 and set(lookup)=={r['id']for r in chosen};seal=read(M/ds/'sealed.json');full=read(F/ds/'sealed.json');checks=0;err=0.
  for row in chosen:
   key=row['id'];e=seal['files'][key];f=full['files'][key];parent=original[ds][key]['parent'];assert sha(M/e['path'])==e['sha256']and sha(M/e['receipt_path'])==e['receipt_sha256']and sha(F/f['path'])==f['sha256']and sha(suite/ds/parent['path'])==parent['sha256']
   with np.load(M/e['path'])as z,np.load(F/f['path'])as current,np.load(suite/ds/parent['path'])as source:
    ids=z['frame_ids'];pos=z['positions8'];raw=source['native_boxes'];assert np.array_equal(ids,source['frame_ids'])and np.array_equal(ids,current['frame_ids'])and np.array_equal(pos,current['positions8']);assert np.array_equal(z['Current_interval'],z['PureMatched_interval'])and np.array_equal(z['Current_interval'],current['shared_interval'])and np.array_equal(z['Current'],current['Full']);assert np.array_equal(z['PureMatched'][pos],raw)
    rebuilt=np.repeat(raw[:1],len(ids),axis=0);rebuilt[pos[-1]+1:]=raw[-1];inside=(ids>=ids[pos[0]])&(ids<=ids[pos[-1]])
    for k in range(2):rebuilt[inside,k]=np.interp(ids[inside],ids[pos],raw[:,k])
    for k in range(2,4):rebuilt[inside,k]=np.exp(np.interp(ids[inside],ids[pos],np.log(raw[:,k])))
    rebuilt[pos]=raw;diff=float(np.max(abs(rebuilt-z['PureMatched'])));assert diff<2e-14;err=max(err,diff);assert np.array_equal(rebuilt[:pos[0]],z['PureMatched'][:pos[0]])and np.array_equal(rebuilt[pos[-1]+1:],z['PureMatched'][pos[-1]+1:]);checks+=1
   assert lookup[key]['values']['Current']['t']==lookup[key]['values']['PureMatched']['t']
  means={arm:{k:100*float(np.mean([r['values'][arm][k]for r in rr]))for k in ['v','unobserved_actual_sIoU','t']}for arm in ['PureMatched','Current']}
  for arm,m in means.items():
   for k,v in m.items():assert abs(v-100*report['means'][arm][k])<1e-10
  paired={key:cluster(rr,key)for key in ['v','unobserved_actual_sIoU','t']}
  for key,z in paired.items():
   ref=report['paired']['Current-PureMatched'][key];assert abs(z['mean']-ref['mean'])<1e-10 and np.max(abs(np.array(z['CI95'])-ref['CI95']))<1e-10
  support=distance['all_unobserved/all_distances'];all_reports[ds]=dict(queries=128,parents=128,means_percent=means,paired_pp=paired,annotated_native_unobserved_support=support,all128_original_prediction_SHA_checked=True,all128_interpolation_anchors_nearest_tails_and_Current_Full_arrays_verified=True,independent_rebuild_max_coordinate_roundoff=err,source_reporting_receipt_sha256=sha(M/ds/'reporting_receipt.json'))
  for path in [M/ds/'sealed.json',M/ds/'metrics.json',M/ds/'reporting_receipt.json',MP/ds/'report.json',MP/ds/'distance_report.json']:bound[str(path.relative_to(R))]=sha(path)
  pmap={p:f'P{i:03}'for i,p in enumerate(sorted({r['parent']for r in rr}),1)}
  with (P/(ds+'_all_signed.tsv')).open('w')as f:
   w=csv.writer(f,delimiter='\t');w.writerow(['query','parent','Interpolation_vIoU_percent','Full_vIoU_percent','Interpolation_unobserved_IoU_percent','Full_unobserved_IoU_percent','delta_v_pp','delta_unobserved_pp'])
   for i,r in enumerate(rr,1):a,b=r['values']['PureMatched'],r['values']['Current'];w.writerow([f'Q{i:03}',pmap[r['parent']],100*a['v'],100*b['v'],100*a['unobserved_actual_sIoU'],100*b['unobserved_actual_sIoU'],100*(b['v']-a['v']),100*(b['unobserved_actual_sIoU']-a['unobserved_actual_sIoU'])])
 table=['| Spatial trajectory | HC vIoU | HC Unobs. IoU | Vid vIoU | Vid Unobs. IoU |','|---|---:|---:|---:|---:|']
 for name,arm in [('Expert interpolation','PureMatched'),('Our path','Current')]:table.append('|'+name+'|'+'|'.join(f'{all_reports[ds]["means_percent"][arm][k]:.2f}'for ds in ['hc2','vidstg']for k in ['v','unobserved_actual_sIoU'])+'|')
 table.append('|Δ (Our path − interpolation)|'+'|'.join(f'{all_reports[ds]["paired_pp"][k]["mean"]:+.2f}'for ds in ['hc2','vidstg']for k in ['v','unobserved_actual_sIoU'])+'|')
 text='# Original fixed128 spatial comparison\n\n'+ '\n'.join(table)+'\n\nEach dataset uses exactly its original fixed128queries/128distinctparents and same eight original ReferDINO observations, native sampled frame IDs and Current Full interval. Per-query tIoU is equal between paths. The user explicitly selected center-linear/log-width-height interpolation, nearest rawexpert frame outside coverage, exact8expert outputs. This is not the initialized-E-tail historical Interpolation and not the ordinary corner-linear variant. No E/Source/proposed geometry inputs enter this selected interpolation estimator.\n\nUnobserved IoU retains the ORIGINAL audited column definition: query-equal IoU on GT-supported native sampled frames excluding exact eight observed positions, across the video rather than restricted to the predicted event. All128 queries/domain have support. This must not be mixed with dense inside-event IoU. The same native GT frame support is shared by both methods; inherited frame audit and receipt hashes verified, no new GT evaluation.\n\nBoth point-estimate gains are positive on this existing panel. HC paired v/unobserved CIs are positive; both Vid CIs cross0. Original fully exposed panel, no reselection, parameter tuning, new Source/expert/GPU/model/GT score or method/anchor change. These128results do not replace the separately preserved full-list results. All signed outcomes remain in TSVs.\n'
 (P/'TABLE.md').write_text(text);write(P/'report.json',dict(status='original128_table_reused_verified_exported',datasets=all_reports,definition='center-linear/log-size/nearest-rawexpert tails',Unobs_scope='all GT-supported native sampled frames minus exact8observed, queryequal; not predicted-event-only',new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_GT_scoring=0));write(D/'completion.json',dict(status='complete_original128_requested_table_reused_verified_exported',source_hashes=bound,report_sha256=sha(P/'report.json'),table_sha256=sha(P/'TABLE.md'),code_sha256=sha(Path(__file__)),queries_per_domain=128,new_Source=0,new_expert=0,new_GPU=0,new_GT_scoring=0,old_frozen_files_or_predictions_changed=False,source_results_already_published=True,this_convenience_export_publication='local only; inherited source reports already published',elapsed_seconds=time.perf_counter()-start));print('\n'.join(table))
if __name__=='__main__':P.mkdir(parents=True,exist_ok=True);main()
