"""First DINO-free support/critic experiment: source-only state, sealed labels."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_mllm_candidate_feedback_v1'
PAPER=ROOT/'artifacts/stvg_opd_p7a_paper_revalidation_v1'
PYTHON=ROOT/'.venv/videochat_r15_v1/bin/python'
CHECKPOINT=ROOT/'checkpoints/VideoChat-R1_5-7B'


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,d,mutable=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not mutable:raise FileExistsError(p)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)


def save(p,d):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    temp=p.with_name(p.name+'.tmp');torch.save(d,temp);temp.replace(p)
    write(p.with_suffix('.json'),dict(status='accepted_complete_source_only_case',sha256=sha(p),bytes=p.stat().st_size,GT_read=False,time=time.time()))


def load(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)


def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
        path=os.fsdecode(args[0])
        banned=['/annos/','/annotations/','labels_diagnostic','GT_LABEL','GT_EXPOSURE','GT_SUBSET',
                '/ROWS.json','/SUMMARY.json','test_annotations.json','valv2_proc.json','val_v2.json','vidstd-test-anno',
                '/stvg_opd_p7a_paper_revalidation_v1/captures/',
                '/stvg_opd_p7a_paper_revalidation_v1/P1/readouts/',
                'models--IDEA-Research--grounding-dino','grounding_dino_http_v1']
        if any(x in path for x in banned):raise PermissionError('DINO-free preseal worker denied reference/label/active-paper payload')


def verify():
    d=read(BASE/'RUNTIME.json')
    for path,h in d['pins'].items():assert sha(ROOT/path)==h,path
    for path,h in d['protected'].items():assert sha(ROOT/path)==h,path
    assert sha(BASE/'DESIGN_LOCK.json')==d['design_sha256']
    assert sha(BASE/'FIXED_ROSTER.json')==d['roster_sha256']
    return d


def standby():
    from scripts.pause_stvg_opd_for_geometry_v1 import process
    pause=read(BASE/'PAPER_PRIORITY_PAUSE_RECEIPT.json')
    for prior in [pause['paper_process']]+pause['baseline_standby']:
        now=process(prior['pid'])
        assert now['start_ticks']==prior['start_ticks'] and now['command']==prior['command'] and now['state']=='T'


def raw_row(record):
    plan=ROOT/'artifacts/decota_paper_experiments_v1'/record['dataset']/'PLAN.json'
    assert sha(plan)==record['plan_sha256']
    row=read(plan)['rows'][record['query_ordinal']]
    assert row['source']==record['parent'] and row['frame_ids']==record['input']['frame_ids']
    subject_path=ROOT/'artifacts/decota_paper_experiments_v1'/record['dataset']/'subjects'/f'{record["query_ordinal"]:05}.json'
    subject=read(subject_path)
    assert subject['ordinal']==record['query_ordinal'] and subject['GT_read'] is False
    assert subject['caption_sha256']==hashlib.sha256(row['input']['caption'].encode()).hexdigest()
    row={**row,'parses':subject['parses'],'subject_sha256':sha(subject_path)}
    return row


def prepare_roster():
    source=ROOT/'artifacts/stvg_opd_temporal_guided_spatial_candidate_v1/DESIGN_LOCK.json'
    old=read(source);selected=[x for x in old['captures'] if x['previous_P6']]
    assert len(selected)==64
    records=[]
    for ds in ('hc2','vidstg'):
        candidates=sorted((x for x in selected if x['dataset']==ds),key=lambda x:x['query_ordinal'])
        assert len(candidates)==32
        planpath=ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json'
        plan=read(planpath);parents=set()
        for i,x in enumerate(candidates):
            row=plan['rows'][x['query_ordinal']];inp=row['input']
            parent=row['source'];assert parent not in parents;parents.add(parent)
            bounds=[float(inp.get('start_frame',0)),float(inp.get('end_frame',inp['frame_count']))]
            assert bounds[0]<bounds[1] and bounds[0]<=row['frame_ids'][0]<row['frame_ids'][-1]<bounds[1]
            records.append(dict(case_id=f'{ds}_{i:03}',dataset=ds,source=x['source'],parent=parent,
                query_ordinal=x['query_ordinal'],input=inp,video_bounds=bounds,
                plan_sha256=sha(planpath),origin_capture_index=x['index'],origin_cohort_sha256=sha(source),
                origin_binding_bytes_only=x['original_Full'],historical_target_exposure=True,
                original_source_initialization=True,DINO_state_initialization=False))
    write(BASE/'FIXED_ROSTER.json',dict(status='locked_before_new_teacher_and_GT',records=records,
          count=64,parents_per_target=32,selection='exact historical previous-P6 clean cohort; sorted ordinal; no new effect filtering',
          original_cohort_sha256=sha(source),GT_read=False,time=time.time()))
    write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_source_and_teacher_measurement',scope='first candidate support plus critic screen only',
          parents=64,source_checkpoint_state_only=True,DINO_state_or_reference_used=False,
          spatial_candidates=4,small_sigmas=dict(hc2=.025,vidstg=.1),large_factor=4.,correlation=.5,
          time_candidates=['Native','Projection','expand_0p1','contract_0p1'],
          context_frames=16,image_views=8,side=224,max_tokens=8192,
          dimensions=dict(spatial=['referent','event','box'],temporal=['referent','event','extent'],global_caption=['event']),
          score='p_SUPPORTED/(p_SUPPORTED+p_CONTRADICTED)',uncertainty='p_UNCERTAIN',
          branch_reward='product of the three candidate-specific dimension scores',
          uncertainty_abstention=.5,MLLM_fixed='OpenGVLab/VideoChat-R1_5-7B',
          MLLM_revision='a80aa31b2fcfcb9ebb342fdc4069c7c8402a123d',precision='BF16',attention='sdpa',
          GPU_weight_budget='8GiB',CPU_weight_budget='12GiB',quantization=False,
          student_qualification_cases=[records[i]['case_id'] for i in [0,1,32,33]],
          critic_qualification_cases=[records[i]['case_id'] for i in [0,32]],
          max_validation_requests_per_query=24,formal_requests_per_query_at_most=10,
          student_optimizer_steps=0,student_backwards=0,MLLM_training_steps=0,
          GT_only_after_all64_candidate_and_critic_seal=True,automatic_dual_branch_TTA=False,
          production_promoted=False,paper_method_replaced=False,seed=20261013,
          total_teacher_wall_limit_seconds=28800,time=time.time()))
    write(BASE/'STATUS.json',dict(status='prepared_source_roster_teacher_download_in_progress',stage='first_screen',GT_read=False,time=time.time()),mutable=True)
    print('DINO_FREE_ROSTER_LOCKED',64,flush=True)


if __name__=='__main__':prepare_roster()
