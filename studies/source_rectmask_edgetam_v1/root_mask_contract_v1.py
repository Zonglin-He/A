"""Additional independent Source-to-rectangle pixel contract; no float replay claim."""
from mask_common import *
import argparse

def run(ds,first_two=False):
 import numpy as np,torch
 from adapter_v1 import rectangular_masks
 torch.set_num_threads(1);validate();parent.gt_guard();release();folder=D/'pilot_v1'/ds
 if first_two:entries=read(folder/'bridge_waiting.json')['files'];assert read(folder/'root_actual_bridge.json')['status']=='passed_first_two_actual_EdgeTAM_queries'
 else:assert read(folder/'reporting_receipt.json')['status']=='verified_whole128_Source_prompt_only_all_mask_input_arrays_and_means';entries=list(read(folder/'sealed.json')['files'].values())
 by={r['id']:r for r in rows(ds)};start=time.perf_counter()
 for e in entries:
  row=by[Path(e['path']).stem];cap,origin=parent.selected.suite().load_parent(ds,row);pos,iv,native=source_fields({k:cap[k]for k in ['Source','native_logits','records','frame_ids','positions8']});assert origin==e['input_binding']['original_Source_parent'];assert sha(D/e['support_path'])==e['support_sha256'];support=torch.load(D/e['support_path'],map_location='cpu',weights_only=False,mmap=True);N,H,W=support['native_shape'];masks=rectangular_masks(native,H,W);proof=e['input_binding'];assert hashlib.sha256(masks.tobytes()).hexdigest()==proof['Source_rectangle_mask_sha256'];assert[int(m.sum())for m in masks]==proof['rectangle_mask_pixels']==e['protocol']['rectangular_mask_pixels'];assert np.array_equal(native,support['native_boxes'])and np.array_equal(pos,support['positions8']);assert e['protocol']['all_eight_rectangular_mask_prompts']and e['protocol']['no_box_prompt_API_used'];assert not proof['ReferDINO_input_used']and not proof['current_method_event_used']
 rec=dict(status='root_actual_Source_rectangle_mask_pixels_prompt_API_and_eight_boxes_exact',queries=len(entries),all_mask_pixels_SHA_from_actual_Source_exact=True,all_eight_mask_not_box_prompt_API=True,not_a_second_float_replay_after_disposal=True,first_two_full_float_original_bridge_sha256=sha(folder/'root_actual_bridge.json'),code_sha256=sha(Path(__file__)),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_expert=0,new_GT_scoring=0);atomic(folder/('root_first_two_mask_contract_v1.json'if first_two else'root_all_mask_prompt_contract_v1.json'),rec)
 if first_two:atomic(P/(ds+'_first_two_Source_rectangular_mask_actual_bridge_v1.json'),rec)
 else:atomic(P/ds/'root_all_mask_prompt_contract_v1.json',rec)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hc2','vidstg'],required=True);p.add_argument('--first-two',action='store_true');a=p.parse_args();run(a.dataset,a.first_two)
