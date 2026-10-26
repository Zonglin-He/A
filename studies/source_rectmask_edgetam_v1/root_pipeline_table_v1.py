"""Crossjoin all audited single-expert variants; no new metric or model call."""
from mask_common import *
import csv

def run():
 ref=R/'outputs/fresh_external_comparison_v1/referformer_v4';rp=read(ref/'complete_published_small_v1.json');assert rp['all_remote_verified']and sha(Path(rp['completion_path']))==rp['completion_sha256'];table={};proof={};lines=['# Single external-expert complete tubes: eight Source prompts and expert substitution','', 'Same original confirm128parents/domain, all source/current scalar rows crossjoined exactly. Source-box and Source-rectangle-mask EdgeTAM both use original Source event/eight positions, the exact same Source boxes, RGB/grid, checkpoint, direction and observed-box writeback. The mask version rasterizes boxes into binary rectangles and uses official add_new_mask, never referring-expert masks. It tests prompt representation, not extra segmentation annotation. Current and ReferFormer each have their own temporal update. One external expert count is controlled, not pretraining, resolution, processed frames, compute or isolated propagation quality. Old ReferDINO-to-EdgeTAM stacked extension is retained separately, not placed in this table.','']
 for ds in ['hc2','vidstg']:
  folder=D/'pilot_v1'/ds;assert read(folder/'reporting_receipt.json')['status']=='verified_whole128_Source_prompt_only_all_mask_input_arrays_and_means';assert read(folder/'root_all_mask_prompt_contract_v1.json')['queries']==128;assert read(D/'GPU_exit.json')['code']==0
  nr=read(folder/'metrics.json')['private_rows'];rr=read(ref/'small_v1'/ds/'metrics.json')['private_rows'];rb={x['id']:x for x in rr};old=R/'outputs/source_prompt_edgetam_v1/pilot_v1'/ds;bb={x['id']:x for x in read(old/'metrics.json')['private_rows']};rep=read(P/ds/'report.json');assert len(nr)==len(rb)==len(bb)==128;aliases={'Source':('Source',None),'ReferDINO_current':('CurrentOurs',None),'Source_box_EdgeTAM':('BoxPromptEdgeTAM',None),'Source_rectangle_mask_EdgeTAM':('EdgeTAM',None),'ReferFormer_current':('AlternativeOurs','ref')};means={}
  for x in nr:
   for a in ['Source','CurrentSimple','CurrentOurs']:assert x['values'][a]==rb[x['id']]['values'][a]==bb[x['id']]['values'][a]
   assert x['values']['BoxPromptEdgeTAM']==bb[x['id']]['values']['EdgeTAM'];assert x['parent']==rb[x['id']]['parent']==bb[x['id']]['parent']and x['duration_group']==rb[x['id']]['duration_group']==bb[x['id']]['duration_group']
  import numpy as np
  for alias,(a,origin)in aliases.items():means[alias]={k:100*float(np.mean([(rb[x['id']]if origin=='ref'else x)['values'][a][k]for x in nr]))for k in ['t','v','vIoU_ge_03','vIoU_ge_05']}
  table[ds]=means;lines +=[f'## {ds}','','|Complete pipeline|tIoU %|vIoU %|R.3 %|R.5 %|','|---|---:|---:|---:|---:|']
  for a,m in means.items():lines.append(f'|{a}|{m["t"]:.6f}|{m["v"]:.6f}|{m["vIoU_ge_03"]:.6f}|{m["vIoU_ge_05"]:.6f}|')
  for k in ['EdgeTAM-BoxPromptEdgeTAM','CurrentOurs-EdgeTAM']:
   z=rep['paired'][k];lines +=['',f'{k}: v{z["v"]["mean"]:+.6f}pp parent-paired10000-bootstrap95%CI{z["v"]["CI95"]}; t{z["t"]["mean"]:+.6f}CI{z["t"]["CI95"]}.']
  proof[ds]=dict(queries=128,parents=128,all_shared_metrics_parent_duration_and_old_box_Edge_exact=True,mask_audit_sha256=sha(folder/'reporting_receipt.json'),all_mask_pixel_contract_sha256=sha(folder/'root_all_mask_prompt_contract_v1.json'),box_audit_sha256=sha(old/'reporting_receipt.json'),ReferFormer_audit_sha256=sha(ref/'small_v1'/ds/'reporting_receipt.json'),mask_root_sha256=sha(folder/'root_export_receipt_v1.json'))
 lines +=['','All signed effects/severe union/rescues/shortlong and actual cost are preserved in the two per-domain root reports. Rectangular masks do not provide native object contours; no second expert or ground truth produces them. Pointwise CIs crossing zero do not prove equivalence. Cohorts previously exposed; no score-selected checkpoint/prompt/direction/threshold, no full12205 result or complete uncached deployment speed. Model load one actual PID/start shared by domains, old Source acquisition nonzero, old ReferDINO acquisition not charged to the two Source-only EdgeTAM branches. Stacked-extra-capacity experiment remains separately available. Main profile/paper unchanged.']
 (P/'SINGLE_EXPERT_PROMPT_VARIANTS_v1.md').write_text('\n'.join(lines)+'\n');atomic(P/'single_expert_prompt_means_v1.json',table)
 with(P/'single_expert_prompt_means_v1.tsv').open('w',newline='')as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['dataset','pipeline','tIoU_percent','vIoU_percent','R03_percent','R05_percent'])
  for ds,x in table.items():
   for a,m in x.items():w.writerow([ds,a,*[m[k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']]])
 rec=dict(status='root_all_single_external_expert_prompt_variants_crossjoined_exact',datasets=proof,code_sha256=sha(Path(__file__)),new_Source=0,new_expert=0,new_model=0,new_GT_scoring=0,full12205_complete=False);atomic(D/'root_pipeline_table_receipt_v1.json',rec);atomic(P/'root_pipeline_table_receipt_v1.json',rec)
if __name__=='__main__':run()
