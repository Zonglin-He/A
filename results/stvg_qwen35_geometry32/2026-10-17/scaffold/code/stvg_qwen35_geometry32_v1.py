"""Qwen direct geometry32 CPU scaffold. No model/scorer/automatic launcher entry."""
from pathlib import Path
import argparse,copy,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write
from scripts.stvg_qwen35_geometry_contract_v1 import MODEL_ID,REVISION,COORDINATES,PROMPT_VERSION,prompt,qualification_ok
BASE=ROOT/'artifacts/stvg_qwen35_geometry32_v1'
PREVIOUS=ROOT/'artifacts/stvg_r1_geometry32_v1'
COMPLEMENT=ROOT/'artifacts/stvg_r1_dino_complement32_v1'
CHECKPOINT=ROOT/'checkpoints/Qwen3.5-9B'

def relative(p):return str(p.relative_to(ROOT))
def canonical_sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def download_status():
    control=CHECKPOINT/'.download-control';manifest=read(control/'expected_side001.json');state=read(control/'status_side001.json')
    assert manifest['repo']==MODEL_ID and manifest['revision']==state['revision']==REVISION
    missing=[];wrong=[]
    for r in manifest['files']:
        p=CHECKPOINT/r['name']
        if not p.is_file():missing.append(r['name'])
        elif p.stat().st_size!=r['bytes']:wrong.append(r['name'])
    verified={r['name']:r['sha256'] for r in state.get('LFS_checks',[])}
    lfs_ok=all(verified.get(r['name'])==r['sha256'] for r in manifest['files'] if r['sha256'])
    ready=state['status']=='complete' and not missing and not wrong and lfs_ok
    return dict(status='download_verified' if ready else 'download_not_ready',revision=REVISION,
                downloader_state=state['status'],missing=missing,size_mismatch=wrong,
                LFS_receipt_complete=lfs_ok,weights_hashed_by_this_scaffold=False,
                total_bytes=manifest['total_bytes'],snapshot_for_model_ready=ready)

def verify_scaffold():
    d=read(BASE/'CPU_SCOPE_LOCK.json')
    for p,h in d['pins'].items():assert sha(ROOT/p)==h,p
    return d

def prepare():
    assert not BASE.exists(),'preserve an existing scaffold; do not overwrite'
    assert read(PREVIOUS/'FINAL_COMPLETION.json')['status']=='complete' and read(COMPLEMENT/'FINAL_COMPLETION.json')['status']=='complete'
    rr=read(PREVIOUS/'ROSTER_LOCK.json');jobs=copy.deepcopy(read(PREVIOUS/'INPUT_JOB_LOCK.json'))
    assert rr['count']==32 and len(rr['records'])==32 and len({r['parent_sha256'] for r in rr['records']})==32
    assert {ds:sum(r['dataset']==ds for r in rr['records']) for ds in ['hc2','vidstg']}=={'hc2':16,'vidstg':16}
    allowed={'job_id','case_id','dataset','slot','position','physical_frame_id','physical_timestamp','image',
             'original_image_shape','original_pixel_sha256','context','query_text','source_case_sha256','weight','GT_read'}
    assert len(jobs['records'])==128 and len({j['job_id'] for j in jobs['records']})==128
    BASE.mkdir();files={};prompts=[]
    for j in jobs['records']:
        assert set(j)==allowed and j['GT_read'] is False
        for k in ['image','context']:
            p=ROOT/j[k]['path'];assert sha(p)==j[k]['sha256'];files[relative(p)]=j[k]['sha256']
        assert j['context']['tensor_layout']=='TCHW' and len(j['context']['actual_timestamps_seconds'])==4
        assert j['context']['clip_role']=='preserved_unmarked_event_context'
        prompts.append(dict(job_id=j['job_id'],prompt_sha256=hashlib.sha256(prompt(j).encode()).hexdigest()))
    for name in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','BASELINE_BARRIER.json','TEMPORAL_LOCK.json']:
        (BASE/name).write_bytes((PREVIOUS/name).read_bytes())
    write(BASE/'INPUT_JOB_LOCK.json',jobs)
    write(BASE/'PROMPT_LOCK.json',dict(status='draft_for_GPU_interface_qualification',version=PROMPT_VERSION,coordinate_protocol=COORDINATES,
          native_enable_thinking=False,formal_max_new_tokens=256,variants=1,records=prompts,
          native_processor_and_coordinate_convention_not_yet_verified=True,GT_read=False))
    probes=[]
    for ds in ['hc2','vidstg']:
        for r in [r for r in rr['records'] if r['dataset']==ds][:2]:
            j=next(j for j in jobs['records'] if j['case_id']==r['probe_alias'] and j['slot']==0);probes.append(j['job_id'])
    write(BASE/'QUALIFICATION_PLAN.json',dict(status='pending_actual_GPU',fixed_parent_probe_jobs=probes,
        parent_probes=4,synthetic_transform_probes_max=2,formal_parent_probe_requests_reused_only_if_exact_final_input_and_prompt_match=True,
        formal_requests=128,no_outcome_retries=True,no_prompt_sweep=True,
        actual_checks=['native model class/BF16','same unmarked4context+standalone anchor identity',
          'actual processor grid/resize/crop/pad and normalized1000 mapping','native template enable_thinking=False',
          'complete answer/truncation/no-target/abstention','video timestamp metadata and resource peak'],GT_read=False))
    baseline_files={};baseline_metadata=[]
    specifications=[('Current_WF',PREVIOUS/'BASELINE_BARRIER.json'),
        ('R1_DIRECT',PREVIOUS/'arms/DIRECT/trials/G32/GLOBAL_PREDICTION_BARRIER.json'),
        ('R1_HEADS',PREVIOUS/'arms/HEADS/trials/G32/GLOBAL_PREDICTION_BARRIER.json'),
        ('D0',PREVIOUS/'arms/DINO/trials/G32/GLOBAL_PREDICTION_BARRIER.json'),
        ('D1',COMPLEMENT/'arms/D1/trials/G32/GLOBAL_PREDICTION_BARRIER.json'),
        ('D1_D2',COMPLEMENT/'D2_GLOBAL_PREDICTION_BARRIER.json')]
    for arm,p in specifications:
        b=read(p);assert b['status']=='complete' and b['count']==32
        assert {r['case_id'] for r in b['files']}=={r['probe_alias'] for r in rr['records']}
        for r in b['files']:assert sha(ROOT/r['path'])==r['sha256'];baseline_files[r['path']]=r['sha256']
        baseline_metadata.append(dict(arm=arm,barrier_path=relative(p),sha256=sha(p),new_model_calls=0))
    write(BASE/'MATCHED_BASELINES.json',dict(status='CPU_hash_bound',count=32,records=baseline_metadata,old_cost_not_zero=True,no_rescoring=True))
    write(BASE/'INPUT_BARRIER.json',dict(status='CPU_bound',planned_sources=32,planned_anchors=128,files=files,
        identical_previous_input_job_sha256=sha(PREVIOUS/'INPUT_JOB_LOCK.json'),no_new_RGB_decode=True,GT_read=False))
    design=read(ROOT/'artifacts/stvg_paper_t2_b_v1/arms/B/trials/G32/DESIGN_LOCK.json')
    write(BASE/'METHOD_PLAN.json',dict(status='scaffold_only_not_execution_lock',model=MODEL_ID,revision=REVISION,dtype='bfloat16',
        temporal='sealed T2 and R1 Caption feedback unchanged',Student=dict(parameters=1792,recipe='lite_fullstep',configs=design['configs'],response_settings=design['response_settings'],episodic=True),
        new_arms=['QWEN_OPD','QWEN_D2'],same_exact_sealed_references_for_both_arms=True,
        D2='reuse unchanged previous Source-relative center/log-size interpolation in physical time, Source outside T2',
        maximum_formal_requests=128,maximum_extra_synthetic_qualification_requests=2,maximum_new_Student_fits=32,
        source_and_Temporal_models_new=0,metadata_processes_may_be_concurrent_but_models_serial=True,
        proposed_resource_limit=dict(wall_hours=8,host_high_GiB=16,host_max_GiB=24,swap_GiB=1,disk_floor_GiB=30,actual_GPU_qualification_pending=True),
        selection='manual multidimensional review: reference coverage/quality, Full-minus-T2, paired tails, nonanchor spatial quality, cost and structural simplicity; no mandatory t95-excludes-zero gate on development32',
        no_per_domain_OPD_D2_splicing=True,no_Temporal_Teacher_migration=True,no_single_Teacher_claim=True,
        no_auto_larger_models_Prompt_heads_optimizer_search=True,CURRENT_unchanged=True,GT_read=False))
    write(BASE/'EXECUTION_POLICY.json',dict(status='scaffold_only',model_execution_authorized=False,GT_scoring_authorized=False,
        auto_start_on_download=False,new_waiters=False,new_units=False,download_owned_by_existing_unit='qwen35-9b-download-side-001.service'))
    paths=[Path(__file__),ROOT/'scripts/stvg_qwen35_geometry_contract_v1.py',ROOT/'scripts/stvg_qwen35_geometry_adapters_v1.py',ROOT/'scripts/test_stvg_qwen35_geometry_scaffold_v1.py',
        ROOT/'protocols/stvg_qwen35_geometry32_v1.md',ROOT/'scripts/stvg_r1_dino_complement32_v1.py',ROOT/'scripts/stvg_r1_geometry32_measure_v1.py',ROOT/'scripts/stvg_overnight_student_v1.py',
        ROOT/'methods/CURRENT_METHOD.json',ROOT/'methods/PAPER_METHOD.json',CHECKPOINT/'config.json',CHECKPOINT/'chat_template.jinja',
        CHECKPOINT/'.download-control/expected_side001.json']
    paths += [BASE/n for n in ['ROSTER_LOCK.json','INPUT_JOB_LOCK.json','PROMPT_LOCK.json','METHOD_PLAN.json','TEMPORAL_LOCK.json','SOURCE_CAPTURE_BARRIER.json','BASELINE_BARRIER.json','QUALIFICATION_PLAN.json','MATCHED_BASELINES.json','EXECUTION_POLICY.json']]
    pins={relative(p):sha(p) for p in paths};pins.update(files);pins.update(baseline_files)
    write(BASE/'CPU_SCOPE_LOCK.json',dict(status='CPU_preparation_locked',pins=pins,planned_sources=32,planned_anchors=128,measurement_validity='untested',claim='untested',GT_read=False))
    write(BASE/'DOWNLOAD_PREFLIGHT.json',download_status());write(BASE/'STATUS.json',dict(status='scaffold_prepared_waiting_download_and_qualification',actual_model_calls=0,actual_Student_fits=0,new_GT_reads=0,automatic_continuation=False))
    print('CPU_SCAFFOLD_BOUND_32_SOURCES_128_ANCHORS_NO_MODEL_NO_GT')

def execution_gate(stage):
    verify_scaffold();policy=read(BASE/'EXECUTION_POLICY.json')
    if not (BASE/'RUN_AUTHORIZATION.json').exists():raise RuntimeError('scaffold-only scope: model execution not authorized; no automatic continuation')
    permit=read(BASE/'RUN_AUTHORIZATION.json')
    if permit.get('scope')!=relative(BASE) or permit.get('status')!='explicitly_authorized' or permit.get('CPU_scope_sha256')!=sha(BASE/'CPU_SCOPE_LOCK.json') or permit.get('allow_model_execution') is not True:
        raise RuntimeError('missing scope/hash-bound explicit future run authorization')
    if not time.time()<permit.get('deadline_epoch',0):raise RuntimeError('run authorization expired')
    if stage=='score' and permit.get('allow_GT_scoring') is not True:raise RuntimeError('GT scoring not authorized')
    if not download_status()['snapshot_for_model_ready']:raise RuntimeError('checkpoint download/verification incomplete')
    if stage!='interface_qualification':
        q=read(BASE/'QUALIFICATION_COMPLETION.json');m=read(BASE/'MODEL_SNAPSHOT_LOCK.json')
        if not qualification_ok(q,input_sha=sha(BASE/'INPUT_JOB_LOCK.json'),model_sha=sha(BASE/'MODEL_SNAPSHOT_LOCK.json'),
             template_sha=m['chat_template_sha256'],prompt_sha=sha(BASE/'PROMPT_LOCK.json')):raise RuntimeError('actual Qwen interface qualification not passed')
    return permit

def plan():
    verify_scaffold()
    return dict(status='plan_not_started',steps=[
      'existing downloader completes verified checkpoint','separate compatible Qwen environment + CPU processor/template checks',
      'explicit future execution authorization + resource lock','4 fixed parent interface probes / max2 synthetic coordinate probes',
      'freeze actual adapter/version/one Prompt; 128 unique logical localization requests',
      'independent parse/input/coordinate readback and complete QWEN_REFERENCE_LOCK',
      'same refs: original OPD -> complete Student root; unchanged D2 -> complete propagation root',
      'all predictions sealed -> GT reference/Full/nonanchor/tails/cost evaluation',
      'anonymous scalar+actual pixels+privacy+remote publication+archive closure'],
      automatic_start=False,planned_new_Student_fits_max=32,planned_requests=128,
      runtime_adapter_status='draft_unverified_until_actual_checkpoint_processor_qualification')

def saved_diagnostic():
    """Existing sealed metrics only; no new metric evaluator, GT, tensor or pixels."""
    verify_scaffold();d=read(COMPLEMENT/'POSTSEAL_RESULTS.json');idx={(r['case_id'],r['arm']):r for r in d['rows']};rr=read(BASE/'ROSTER_LOCK.json')['records'];rows=[];summary=[]
    import math,statistics
    for ds in ['hc2','vidstg']:
        ids=[r['probe_alias'] for r in rr if r['dataset']==ds]
        for metric in ['m_vIoU','m_sIoU_dense_GT','m_sIoU_unsupervised_GT']:
            vals={cid:None if any(idx[cid,a]['metrics'][metric] is None for a in ['D1','D2']) else 100*(idx[cid,'D2']['metrics'][metric]-idx[cid,'D1']['metrics'][metric]) for cid in ids}
            old=next(p for p in d['paired'] if p['dataset']==ds and p['metric']==metric and p['comparison']=='D2_minus_D1')
            assert all(abs(v-old['per_source_delta_pp'][cid])<1e-12 for cid,v in vals.items() if v is not None)
            known=[v for v in vals.values() if v is not None];mean=math.fsum(known)/len(known) if known else None
            assert mean==old['mean_delta_pp'];summary.append(dict(dataset=ds,metric=metric,comparison='D2_minus_D1',mean_delta_pp=mean,
                paired_t95_pp=old['paired_source_t95_interval_pp'],positive=old['positive'],negative=old['negative'],neutral=old['neutral'],unknown=old['unknown'],
                median_delta_pp=statistics.median(known) if known else None,min_delta_pp=min(known) if known else None,max_delta_pp=max(known) if known else None))
            for cid,v in vals.items():rows.append(dict(source_alias=f'R{1+next(i for i,r in enumerate(rr) if r["probe_alias"]==cid):03d}',dataset=ds,metric=metric,D2_minus_D1_pp=v))
    out=dict(status='complete_saved_scalar_diagnostic_only',count=32,source_result_sha256=sha(COMPLEMENT/'POSTSEAL_RESULTS.json'),rows=rows,summary=summary,
        all_extreme_sources_retained=True,same_T2_and_same_D1_references=True,no_new_model_or_GT_reader=True,no_new_scores=True,selection_not_performed=True)
    write(BASE/'D1_D2_SAVED_DIAGNOSTIC.json',out);print('SAVED_SCALAR_DIAGNOSTIC_COMPLETE_NO_NEW_SCORING')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','verify','download-status','plan','saved-diagnostic']);a=parser.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='verify':verify_scaffold();print('CPU_SCOPE_PINS_PASS')
    elif a.command=='download-status':print(json.dumps(download_status(),indent=2))
    elif a.command=='plan':print(json.dumps(plan(),indent=2))
    else:saved_diagnostic()
if __name__=='__main__':main()
