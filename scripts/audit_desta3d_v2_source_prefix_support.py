"""CPU-only readback of the already sealed source28199 task/native supports."""
import sys,re,time,json
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
BASE=ROOT/'artifacts/desta3d_v2/tta_v2';OLD=BASE/'source_cast_probe_v1';OUT=BASE/'source_input_vjp_v1'


def main():
    from transformers import AutoTokenizer
    assert not (OUT/'PREFIX_SUPPORT_READBACK.json').exists()
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    t=AutoTokenizer.from_pretrained(ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B',local_files_only=True)
    d=torch.load(OLD/'RAW_ENDPOINTS.pt',map_location='cpu',weights_only=False)
    rec=read(OLD/'SOURCE_RECORD.json');support=d['task_support']
    start=int(torch.nonzero(support['labels'][0]!=-100).flatten()[0]);prefix=int(support['ptd_prefix_lengths'][0])
    gt=t.decode(support['input_ids'][0,start:prefix],skip_special_tokens=False)
    assert gt==rec['response']+'<|im_end|>\n'
    phase=re.compile(r'<\|time_start\|><t(\d+)><t(\d+)><\|time_end\|>')
    gt_span=list(map(int,phase.search(gt).groups()))
    rows=list(map(int,re.findall(r'<t(\d+)><\|box_start\|>',gt)))
    assert rows==rec['response_box_positions_1based'] and rows==list(range(gt_span[0],gt_span[1]+1))
    active=np.flatnonzero(rec['event_active']).tolist();assert [x-1 for x in rows]==active
    parent=BASE/'source_task_control_v2/episodes/00'
    original=torch.load(parent/'no_update.pt',map_location='cpu',weights_only=False)
    after=torch.load(parent/'supervised.pt',map_location='cpu',weights_only=False)
    ns=list(map(int,phase.search(original['event_completion']).groups()))
    nr=list(map(int,re.findall(r'<t(\d+)><\|box_start\|>',original['spatial_completion'])))
    reference=lambda s:s.split('<|object_ref_end|>')[0]+'<|object_ref_end|>'
    assert reference(gt)==reference(original['event_completion'])==reference(after['event_completion'])
    assert ns==[x+1 for x in original['interval']] and nr==[x+1 for x in original['positions']]
    equal={k:(torch.equal(original[k],after[k]) if torch.is_tensor(original[k]) else original[k]==after[k])
           for k in ['boxes_cxcywh','geometry_valid','interval','positions','format_ok','event_completion','spatial_completion']}
    assert all(equal.values())
    logits={}
    for name,a,b in [('time',original['time_distribution']['endpoint_logits'],after['time_distribution']['endpoint_logits']),
                     ('coordinates',original['readout']['coordinate_logits'],after['readout']['coordinate_logits'])]:
        assert a.shape==b.shape
        x=a.double().numpy();y=b.double().numpy()
        def logp(z):
            z=z-z.max(-1,keepdims=True);return z-np.log(np.exp(z).sum(-1,keepdims=True))
        lp,lq=logp(x),logp(y)
        logits[name]={'shape':list(a.shape),'changed_elements':int(np.count_nonzero(x-y)),
            'maxabs_change':float(np.max(np.abs(x-y))),
            'KL_B1_to_supervised3_mean':float(np.mean(np.sum(np.exp(lp)*(lp-lq),axis=-1)))}
    report={'time':time.time(),'status':'passed','source_key':rec['key'],'source_parents':1,
       'new_GPU_calls':0,'optimizer_steps':0,'new_labels_read':False,'existing_source_training_record_reused':True,
       'target_data':False,'GT_record_response_equals_actual_training_prefix':True,'training_response_start':start,
       'training_prefix_length':prefix,'reference_exact_between_GT_and_both_native_states':True,
       'GT_time_tokens_1based':gt_span,'native_time_tokens_1based':ns,
       'GT_box_rows_1based':rows,'native_box_rows_1based':nr,
       'source_GT_physical_interval_half_open':rec['event_interval'],
       'GT_active_frame_ids':[rec['frame_ids'][i] for i in active],
       'native_extra_frame_ids':[rec['frame_ids'][x-1] for x in nr if x not in rows],
       'native_saved_B1_vs_supervised3_equal':equal,'native_logits':logits,
       'task_branch_NTP_MTP_counts':{b:{k:d['states']['B1']['branches'][b]['info'][k] for k in ['ntp_count','mtp_count']} for b in ['event','spatial']},
       'official_contract':'sft_dataset adds spatial query blocks for GT start..end; ptd_generation uses predicted time anchors. Training GT full-sequence CE and native cached decoding are different conditions by design.',
       'interpretation':'prefix difference matters for CE-to-native interpretation, but cannot explain the local VJP-versus-finite-CE discrepancy because both of those task measurements use the same exact GT prefix.',
       'pins':{str(p):sha(p) for p in [Path(__file__),parent/'no_update.pt',parent/'supervised.pt',
           ROOT/'external/ParallelTubeDecoding/src/dataset/sft_dataset.py',ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py']}}
    save_once(OUT/'PREFIX_SUPPORT_READBACK.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
