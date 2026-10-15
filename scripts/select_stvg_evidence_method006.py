"""Apply prelocked global development selection; confirmation never selects a recipe."""
import sys,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
def freeze(v,preservation,selection):
 code=['scripts/stvg_evidence_temporal_fit_v1.py','scripts/stvg_evidence_spatial_inputs005.py','scripts/stvg_weightfree_coordinate_fit_v1.py','scripts/stvg_evidence_spatial_preservation_v1.py','scripts/stvg_table1_temporal_adapter_v1.py','scripts/run_stvg_evidence_teacher_v1.py','scripts/score_stvg_evidence_v1.py']
 write(BASE/'FROZEN_METHOD_LOCK.json',dict(status='locked',temporal_variant=read(BASE/'TEMPORAL_SELECTION_LOCK.json')['variant'],spatial_variant=v,output_preservation=bool(preservation),spatial_configs=read(C/'SPATIAL_SELECTION_LOCK.json')['configs'],code_pins={p:sha(ROOT/p)for p in code},selection_sha256=sha(selection),design_sha256=sha(BASE/'DESIGN_LOCK.json'),confirmation_roster_sha256=sha(BASE/'CONFIRMATION_ROSTER_LOCK.json'),selection_data='sealed exposed development32 only',confirmation_used_for_selection=False,search_finished=True,CURRENT_promoted=False,bootstrap_executed=False,time=time.time()))
 b=BASE/'confirmation';b.mkdir(exist_ok=True);write(b/'VARIANT_LOCK.json',dict(status='locked',variants=[v],temporal_selection_sha256=sha(BASE/'TEMPORAL_SELECTION_LOCK.json'),frozen_method_sha256=sha(BASE/'FROZEN_METHOD_LOCK.json'),GT_read=False));print('FROZEN',v,preservation,flush=True)
def main():
 import torch,numpy as np
 from scripts.review_stvg_instance_step_diagnostic_v1 import iou_cxcywh
 phase=sys.argv[1];b=BASE/phase;summary=read(b/'POSTSEAL_SUMMARY.json')['summary'];m={(r['dataset'],r['arm']):r['mean']for r in summary if r['metric']=='m_vIoU'};assert read(b/'GLOBAL_PREDICTION_BARRIER.json')['count']==32
 if phase=='round2':
  preference=['new_scene','new_crop','old_scene','old_crop'];scores={v:math.fsum(m[d,'TS_'+v]for d in ['hc2','vidstg'])/2 for v in preference};winner=max(preference,key=lambda v:scores[v]);rows=read(b/'POSTSEAL_ROWS.json')['rows'];lookup={(r['case_id'],r['arm']):r for r in rows};diag=[]
  for rec in roster():
   cid=rec['probe_alias'];d=torch.load(b/'student'/(cid+'.pt'),map_location='cpu',weights_only=False);aa=d['reference_records'][winner]['anchors'];f=d['fits'].get(winner);qdelta=None;u0=lookup[cid,'T']['metrics']['m_sIoU_unsupervised_GT'];u1=lookup[cid,'TS_'+winner]['metrics']['m_sIoU_unsupervised_GT']
   if aa and f is not None:
    pp=[a['position']for a in aa];rr=np.asarray([a['box_cxcywh']for a in aa]);ww=np.asarray([a['weight']for a in aa]);initial=next(x['boxes']for x in f['path']if not x.get('skipped'));qdelta=float(np.average(iou_cxcywh(f['final_cached_decoder_boxes'].numpy()[pp],rr)-iou_cxcywh(initial.numpy()[pp],rr),weights=ww))
   harmed=(u0 is not None and u1 is not None and (u1-u0)*100<=-1.);trigger=bool(f is not None and f['optimizer_steps']>0 and qdelta is not None and qdelta>1e-10 and harmed);diag.append(dict(case_id=cid,dataset=rec['dataset'],reference_agreement_delta=qdelta,unsupervised_sIoU_delta_pp=None if u0 is None or u1 is None else(u1-u0)*100,trigger=trigger))
  count=sum(x['trigger']for x in diag);dest=BASE/'SPATIAL_SELECTION_LOCK.json';write(dest,dict(status='locked',variant=winner,equal_domain_mean_joint_vIoU=scores,development_GT_selection=True,confirmation_used_for_selection=False,round2_seal_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'),preservation_rule_sha256=sha(BASE/'PRESERVATION_RULE_LOCK.json'),conditional_preservation_trigger_count=count,conditional_preservation_required=count>=4,records=diag,time=time.time()))
  print('SELECT',winner,scores,'preservation cases',count,flush=True)
  if count<4:freeze(winner,False,dest)
 elif phase=='preservation':
  parent=read(BASE/'SPATIAL_SELECTION_LOCK.json');assert parent['conditional_preservation_required'];v=parent['variant'];before=math.fsum(m[d,'TS_'+v]for d in ['hc2','vidstg'])/2;after=math.fsum(m[d,'TS_preserved']for d in ['hc2','vidstg'])/2;dest=BASE/'PRESERVATION_SELECTION_LOCK.json';write(dest,dict(status='locked',original_mean=before,preserved_mean=after,chosen_preservation=after>before,development_GT_selection=True,confirmation_used_for_selection=False,GT_read=False));freeze(v,after>before,dest)
 else:raise ValueError(phase)
if __name__=='__main__':main()
