"""Measured layer trajectories, fixed final slot and actual initial reference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_dino_mask_align32_v1 import D,P,verify
from vg_tta.stvg_dino_feedback_cache32_v1 import iou,public_value


def mean(v):
    v=[float(x)for x in v if x is not None];return float(np.mean(v))if v else None


def main():
    p=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32;rows=[];examples=[];totals=dict(same_mask=0,different_mask=0,newly_admitted=0,newly_rejected=0)
    for rec in p['records']:
        z=load(D/'predictions'/(rec['alias']+'.pt'));vectors={k:[[]for _ in range(6)]for k in ['native_high_score','A_score','B_score','A_native_layer_IoU','B_native_layer_IoU']};initial=[];c=dict(same_mask=0,different_mask=0,newly_admitted=0,newly_rejected=0)
        for pos,fr in z['traces'].items():
            native=z['native'][pos];slot=native['high_slot'];a,b=fr['A_unaligned'],fr['B_aligned'];inv=fr['inverse']
            for j,ix in enumerate(fr['candidate_indices']):
                h=inv[j];same=a['mask_slots'][h]==b['mask_slots'][h];c['same_mask'if same else'different_mask']+=1;c['newly_admitted']+=a['scores'][-1,h]<.35<=b['scores'][-1,h];c['newly_rejected']+=b['scores'][-1,h]<.35<=a['scores'][-1,h]
                ini=float(iou(a['initial_reference'][h],native['initial_references'][slot])[0,0]);initial.append(ini)
                for l in range(6):
                    vectors['native_high_score'][l].append(native['layer_scores'][l,slot]);vectors['A_score'][l].append(a['scores'][l,h]);vectors['B_score'][l].append(b['scores'][l,h]);vectors['A_native_layer_IoU'][l].append(float(iou(a['boxes'][l,h],native['layer_boxes'][l,slot])[0,0]));vectors['B_native_layer_IoU'][l].append(float(iou(b['boxes'][l,h],native['layer_boxes'][l,slot])[0,0]))
                if rec['alias']=='H010'and pos==20 and int(ix)==5 or rec['alias']=='V010'and pos==36 and int(ix)==11:
                    examples.append(dict(alias=rec['alias'],dataset=rec['dataset'],position=pos,sample=int(ix),native_high_slot=slot,mask_A=int(a['mask_slots'][h]),mask_B=int(b['mask_slots'][h]),
                        student_to_native_initial_IoU=ini,native_scores=native['layer_scores'][:,slot].tolist(),A_scores=a['scores'][:,h].tolist(),B_scores=b['scores'][:,h].tolist(),
                        A_final_to_native_final_IoU=float(iou(a['boxes'][-1,h],native['layer_boxes'][-1,slot])[0,0]),B_final_to_native_final_IoU=float(iou(b['boxes'][-1,h],native['layer_boxes'][-1,slot])[0,0])))
        rows.append(dict(alias=rec['alias'],dataset=rec['dataset'],counts=c,mean_student_to_native_initial_IoU=mean(initial),layers={k:[mean(x)for x in v]for k,v in vectors.items()}))
        for k in totals:totals[k]+=int(c[k])
    groups={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(queries=16,queries_with_returned_refs=sum(r['layers']['A_score'][0]is not None for r in rr),
            counts={k:sum(int(r['counts'][k])for r in rr)for k in totals},layers={k:[mean([r['layers'][k][l]for r in rr])for l in range(6)]for k in rows[0]['layers']},
            mean_student_to_native_initial_IoU=mean([r['mean_student_to_native_initial_IoU']for r in rr]))
    put(P/'MEASURED_LAYER_DIAGNOSIS.json',public_value(dict(status='complete_measured_six_layers',count=32,groups=groups,rows=rows,examples=examples,
        native_slot_selected_by_final_native_target_score=True,not_best_layer_selection=True,propagated_references_and_head_readouts_saved_separately=True,
        original_native_initial_not_final_proxy=True,no_GT_needed_for_layer_diagnosis=True)))
    print(public_value(dict(groups=groups,examples=examples)))


if __name__=='__main__':main()
