"""Postsealed matched original-source candidate and exact teacher-repeat control."""
import sys,json,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stvg_temporal_caption_query_v1 import ROOT,read,write,sha
B=ROOT/'artifacts/stvg_box_scale_global_evidence_v1';C=B/'source_position_control001'
def main():
 seal=read(C/'TEACHER_GLOBAL_BARRIER.json');assert seal['status']=='complete'and seal['count']==24 and seal['GT_read']is False and seal['runtime_sha256']==sha(C/'RUNTIME.json')
 for p,h in read(C/'RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h
 for f in seal['files']:assert sha(ROOT/f['path'])==f['sha256']
 prior=read(B/'ROOT_PRELIMINARY_POSTSEAL_METRICS.json');jobs=read(C/'TEACHER_JOB_LOCK.json')['records'];data={}
 for j in jobs:
  p=C/'teacher'/f"{j['case_id']}_{j['kind']}_{j['window_id']:03d}.json";r=read(p);assert r['GT_read']is False and r['input_file_sha256']==j['input_file_sha256']and r['prompt_sha256']==j['prompt_sha256'];data[(j['case_id'],j['kind'])]=r
 rows=[]
 for r in read(B/'GRID_INPUT_BARRIER.json')['records']:
  cid=r['case_id'];alias=('HC'if cid.startswith('hc2')else 'V')+f"{int(cid.split('_')[-1])+1:02d}";old=read(B/'teacher'/f'{cid}_original_000.json');repeat=data[(cid,'original')];exact=all(old[k]==repeat[k]for k in ['label_logits','probabilities','referent_support_log_odds','preferred_label']);assert exact,'Original frozen teacher repeated image logits must match exactly'
  base=next(z for z in prior['rows']if z['case']==alias);masked=data[(cid,'source_masked')];marked=data[(cid,'source_marked')];source_mask=repeat['referent_support_log_odds']-masked['referent_support_log_odds'];source_mark=marked['referent_support_log_odds'];mc=next(z for z in prior['candidates']if z['case']==alias and z['window_id']==base['marked_selected_window_id']);mark_max=mc['marked_score'];keep_mask=source_mask>=base['mask_max_response'];keep_mark=source_mark>=mark_max;mi=base['source_IoU']if keep_mask else base['mask_selected_IoU'];marki=base['source_IoU']if keep_mark else base['marked_selected_IoU'];available=base['GT_anchor_available']
  rows.append(dict(case=alias,dataset=base['dataset'],GT_anchor_available=available,source_IoU=base['source_IoU'],oracle_grid_only_IoU=base['oracle_IoU'],oracle_grid_plus_source_IoU=max(base['oracle_IoU'],base['source_IoU'])if available else None,source_mask_response=source_mask,source_mark_score=source_mark,mask_grid_plus_source_keeps_source=keep_mask,marked_grid_plus_source_keeps_source=keep_mark,mask_grid_plus_source_selected_IoU=mi,marked_grid_plus_source_selected_IoU=marki,mask_grid_plus_source_delta=None if not available else mi-base['source_IoU'],marked_grid_plus_source_delta=None if not available else marki-base['source_IoU'],source_marked_preferred_label=marked['preferred_label'],original_teacher_repeat_bitwise_scalar_match=exact))
 cost=dict(actual_teacher_requests=24,original_repeat_requests=8,source_mask_requests=8,source_mark_requests=8,model_forward_plus_sync_seconds=sum(r['total_seconds']for r in data.values()),input_tokens=sum(r['input_tokens']for r in data.values()),visual_tokens=sum(r['visual_tokens']for r in data.values()),output_tokens=0,GPU_peak_bytes=max(r['GPU_peak_bytes']for r in data.values()),worker_launch_to_exit_seconds=read(C/'TEACHER_WORKER_EXIT.json')['time']-read(C/'TEACHER_LAUNCH.json')['time'],per_request_full_latency_seconds=None)
 out=dict(status='pass',count=8,rows=rows,cost=cost,source_tie_priority_prelocked=True,control_added_after_first_P0_readback_disclosed=True,first160_results_not_modified=True,GT_for_source_geometry_generation_or_teacher_selection=False,GT_read=True,new_student_fits=0)
 write(C/'ROOT_READBACK.json',out)
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds and r['GT_anchor_available']];print(ds,{k:statistics.mean(r[k]for r in rr)for k in ['oracle_grid_plus_source_IoU','mask_grid_plus_source_selected_IoU','marked_grid_plus_source_selected_IoU','mask_grid_plus_source_delta','marked_grid_plus_source_delta']})
 print('source controls/repeats complete',cost)
if __name__=='__main__':main()
