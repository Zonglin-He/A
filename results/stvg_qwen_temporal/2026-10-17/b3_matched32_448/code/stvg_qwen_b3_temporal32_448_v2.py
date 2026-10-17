"""Correct native Qwen total-video budget to preserve all eight448 input frames."""
from pathlib import Path
import inspect,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_qwen_b3_temporal32_v1 as old
BASE=old.BASE/'processor448002';read,write,sha=old.read,old.write,old.sha
def verify():
    old.verify()
    for p,h in read(BASE/'EXECUTION_RUNTIME002.json')['pins'].items():assert sha(ROOT/p)==h,p
    assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']
def configure448(p):
    from scripts.stvg_table1_temporal_adapter_v1 import configure_original_processor
    configure_original_processor(p)
    # Qwen3-VL smart_resize multiplies frame_count*H*W before comparing
    # size.longest_edge. R1's per-frame448^2 cannot be copied as total budget.
    p.video_processor.size=dict(shortest_edge=8*3136,longest_edge=8*448**2)
    p.video_processor.cap_pixels_per_frame=False
    return p
def init():
    old.verify();assert not BASE.exists();BASE.mkdir()
    for name in ['ROSTER_LOCK.json','INPUT_BARRIER.json']:(BASE/name).write_bytes((old.BASE/name).read_bytes())
    write(BASE/'ACTIVE_EXECUTION.json',dict(status='locked_remaining_original_budget',deadline_epoch=read(old.BASE/'ACTIVE_EXECUTION.json')['deadline_epoch'],original_start_epoch=read(old.BASE/'ACTIVE_EXECUTION.json')['start_epoch'],budget_not_reset=True,GT_read=False))
    write(BASE/'METHOD_LOCK.json',dict(status='locked_before_corrected_model',count=32,per_domain=16,only_correction='Qwen size.longest_edge is total video pixels: 8*448^2 preserves original eight448 frames',expected_native_video_grid=[4,28,28],expected_video_tokens=784,prior_low_resolution_grid=[4,8,8],prior_video_tokens=64,I1_I2_raw_pixels_times_Prompt_rewards_Student_unchanged=True,original_deadline_not_reset=True,correction_selected_from_processor_code_and_token_audit_not_scores=True,previous_calls_and_all_outputs_preserved=True,new_captions_max=64,new_prefills_max=64,extra_prefix_qualifications_max=2,Student_parameters=512,GT_read=False))
    paths=[Path(__file__),ROOT/'scripts/continue_stvg_qwen_b3_temporal32_448_v2.py',ROOT/'scripts/report_stvg_qwen_b3_temporal32_448_v2.py',old.BASE/'EXECUTION_RUNTIME.json',old.BASE/'ACTUAL_TOKEN_READBACK.json',BASE/'METHOD_LOCK.json',BASE/'ROSTER_LOCK.json',BASE/'INPUT_BARRIER.json'];write(BASE/'EXECUTION_RUNTIME002.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False));write(BASE/'STATUS.json',dict(status='prepared',count=32,GT_read=False),mutable=True)
def run(stage):
    verify();source=inspect.getsource(getattr(old,stage));ns=dict(old.__dict__,BASE=BASE,verify=verify)
    if stage=='teacher':
        assert source.count('configure_original_processor(p)')==1;source=source.replace('configure_original_processor(p)','configure448(p)');source=source.replace('assert int(grid[0])==4','assert grid.tolist()==[4,28,28]');ns['configure448']=configure448
    exec(compile(source,__file__,'exec'),ns);return ns[stage]()
if __name__=='__main__':
    if sys.argv[1]=='init':init()
    else:run(sys.argv[1])
