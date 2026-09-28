"""Targets must preserve the actual (possibly augmented) processed video."""
import ast,types
from pathlib import Path
import torch
from vg_tta.desta3d_v3_training import training_inputs

def test_training_inputs_passes_exact_processed_pixels(monkeypatch):
    # Isolate construction with a tokenizer/target-builder fixture; no media,
    # target GT, processor/model weights or CUDA in this CPU contract.
    import scripts.ptd_8b_teacher_feasibility_v1 as helper
    t=types.SimpleNamespace(encode=lambda text,**kw:[11,12,13]);processor=types.SimpleNamespace(tokenizer=t)
    # Tokenizer fixture supports the original parsed marker sequence.
    prompt=dict(input_ids=torch.tensor([[80000,80001]]),video_grid_thw=torch.tensor([[7,2,2]]),
        pixel_values_videos=torch.arange(28.).reshape(7,4),mm_token_type_ids=torch.tensor([[1,0]]))
    captured={}
    def original_shape_contract(pr,data,start):
        captured['pixels']=data['pixel_values_videos'].clone();captured['labels']=data['labels'].clone()
        # Isolate construction, not tokenizer semantics (real target semantics
        # are separately exercised by test_desta3d_v3_source).
        data.update(ptd_prefix_length=torch.tensor(len(data['input_ids'])),
          attention_mask=torch.ones_like(data['input_ids']),ptd_position_ids=torch.arange(len(data['input_ids'])),
          ptd_context_limits=torch.arange(len(data['input_ids'])))
        return data
    monkeypatch.setattr(helper,'append_targets',original_shape_contract)
    record=dict(response_eligible=True,response='<|object_ref_start|>person<|object_ref_end|>')
    out=training_inputs(processor,prompt,record)
    assert torch.equal(out['pixel_values_videos'],prompt['pixel_values_videos'])
    assert torch.equal(captured['pixels'],prompt['pixel_values_videos'])
    assert out['input_ids'].shape[0]==1 and out['ptd_prefix_lengths'].shape==(1,)
    assert captured['labels'][:2].tolist()==[-100,-100]
    assert training_inputs(processor,prompt,dict(response_eligible=False)) is None
