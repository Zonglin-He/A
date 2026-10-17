"""Post-seal anonymous reporting for the fixed single-Qwen main method.

No model is created. This opens evaluation targets only after the complete
locked population and every saved prediction hash have passed the seal.
"""
from pathlib import Path
import argparse,copy,hashlib,json,math,re,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def report(scope):
    from main_method.pipeline import path,load_pt
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    import numpy as np
    out=path(scope);base=ROOT/'artifacts/stvg_main_qwen_v1';rr=read(out/'ROSTER_LOCK.json')['records'];n=len(rr)
    root=read(out/'ROOT_READBACK.json');cpu=read(out/'CPU_COMPLETION.json');bar=read(out/'GLOBAL_PREDICTION_BARRIER.json')
    assert root['status']=='pass'and root['count']==n and cpu['status']=='complete'and cpu['count']==n and bar['status']=='complete'and bar['count']==n
    saved=read(out/'POSTSEAL_RESULTS.json');assert sha(out/'POSTSEAL_RESULTS.json')==cpu['result_sha256']
    sources=read(out/'SOURCE_BARRIER.json')['files'];assert len(sources)==len(bar['files'])==n
    aliases={r['probe_alias']:f'R{i+1:05d}'for i,r in enumerate(rr)}
    cs={r['probe_alias']:load_pt(f)for r,f in zip(rr,sources)}
    assert len(cs)==n
    for r in rr:
        c=cs[r['probe_alias']];assert c['query_key_sha256']==r['query_key_sha256']and c['parent_sha256']==r['parent_sha256']and c['dataset']==r['dataset']
    pred=[load_pt(f)for f in bar['files']];assert {p['case_id']for p in pred}==set(cs)
    targets,bindings=truth._truths_after_seal(cs);official,independent,_=truth._metric_functions()
    rows=[];checked=0;support_rows=[]
    old={(x['case_id'],x['arm']):x['metrics']for x in saved['rows']}
    for r,p in zip(rr,pred):
        cid=r['probe_alias'];assert p['case_id']==cid
        for arm,a in p['arms'].items():
            m,_=metric._score_student_arm(a,cs[cid],targets[cid],set(),official,independent)
            m.update(R_vIoU_at_0p3=float(m['m_vIoU']>.3),R_vIoU_at_0p5=float(m['m_vIoU']>.5))
            assert m==old[cid,arm],(cid,arm)
            assert all(re.fullmatch('[A-Za-z][A-Za-z0-9_]*',k)and(v is None or isinstance(v,(int,float)))for k,v in m.items())
            checked+=len(m);rows.append(dict(case_id=cid,dataset=r['dataset'],arm=arm,metrics=m))
        # Geometry uses the SAME final Temporal output for every arm here.
        # Native decoder positions and benchmark-interpolated GT positions are
        # separate supports; a changed coordinate is not a semantic label.
        reference=set(p['observed_positions']['Full'])
        gt=metric._truth_map(targets[cid]);ids=list(map(int,cs[cid]['frame_ids']))
        interpolated={i:b for i,b in gt.items()if i not in set(ids)}
        for arm in ['Temporal_only','Full','same_reference_D2']:
            assert p['arms'][arm]['interval']==p['arms']['Full']['interval']
            sm,_=metric._score_student_arm(p['arms'][arm],cs[cid],targets[cid],reference,official,independent)
            dense_interp=None
            if interpolated:
                dense_interp=truth.score_tube_xyxy(p['arms'][arm]['boxes'],ids,cs[cid]['input']['width'],cs[cid]['input']['height'],p['arms']['Full']['interval'],interpolated,targets[cid]['span'],r['dataset'],official=official,independent=independent)['metrics']['m_sIoU_dense_GT']
            delta=np.max(np.abs(np.asarray(p['arms'][arm]['boxes'])-np.asarray(p['arms']['Temporal_only']['boxes'])),axis=1)
            support_rows.append(dict(source_alias=aliases[cid],dataset=r['dataset'],arm=arm,admitted_reference_count=len(reference),
                known_native_observed_GT_positions=sum(ids[i]in gt for i in reference),known_native_nonobserved_GT_positions=sum(fid in gt and i not in reference for i,fid in enumerate(ids)),
                benchmark_interpolated_GT_positions=len(interpolated),
                changed_native_observed_positions=int(sum(delta[i]>1e-9 for i in reference)),changed_native_nonobserved_positions=int(sum(v>1e-9 and i not in reference for i,v in enumerate(delta))),
                native_observed_sIoU=sm['m_sIoU_supervised_GT'],native_nonobserved_sIoU=sm['m_sIoU_unsupervised_GT'],benchmark_interpolated_sIoU=dense_interp))
    summary,paired=subset_aggregate(rows);assert summary==saved['summary']and paired==saved['paired']
    public=ROOT/'results/stvg_main_qwen/2026-10-17'/scope
    anon=[{**x,'source_alias':aliases[x['case_id']]}for x in rows]
    for x in anon:del x['case_id']
    paired=copy.deepcopy(paired)
    for p in paired:p['per_source_delta_pp']={aliases[k]:v for k,v in p['per_source_delta_pp'].items()}
    scientific=dict(status='complete',count=n,cohort='exposed development engineering32; not independent confirmation'if scope=='engineering32'else'official locked complete evaluation',rows=anon,summary=summary,paired=paired,
        model='one frozen Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        temporal='original pairwise Query-Caption merged-interval KL512 + marginal MAP',spatial='original Direct4 and episodic Gaussian OPD1792',
        all_negative_neutral_unknown_retained=True,CURRENT_and_historical_PAPER_unchanged=True,
        no_official_or_independent_confirmation_claim=scope=='engineering32')
    write(public/'RESULTS.json',scientific)
    write(public/'SPARSE_PROPAGATION.json',dict(status='complete',count=n,rows=support_rows,same_final_time_for_all_arms=True,reference_support='actual admitted Full anchors; no GT selection',native_decoder_and_benchmark_interpolation_separated=True,coordinate_change_is_not_semantic_correctness=True,raw_pixels_GT_coordinates_and_source_identities_excluded=True))
    keep=['status','teacher','revision','Teacher_frozen','Teacher_native_non_thinking','Teacher_dtype','source_checkpoints','temporal','spatial','arms','extra_controls','official_count','official_domains','official_parents','target_GT_use','no_external_Actionness','no_DINO','no_new_parameter_or_Prompt_search']
    design=read(base/'METHOD_LOCK.json');write(public/'FROZEN_METHOD.json',{k:design[k]for k in keep})
    for name in ['GPU_EQUIVALENCE_COMPLETION.json','CPU_EQUIVALENCE_COMPLETION.json','PROCESSOR_EQUIVALENCE_COMPLETION.json']:
        v=read(base/name);write(public/name,{k:v for k,v in v.items()if k not in ['records','files']})
    usage={}
    for stage,filename in [('Temporal','TEMPORAL_TEACHER_BARRIER.json'),('Spatial','SPATIAL_TEACHER_BARRIER.json')]:
        if (out/filename).exists():
            b=read(out/filename);usage[stage]=dict(source_count=b['count'],actual_new_model_calls_this_worker=b.get('new_calls'),exact_input_reuse_does_not_count_as_new_calls=True)
    write(public/'EXECUTION_COUNTS.json',dict(status='complete',count=n,Teacher=usage,new_Source_and_Spatial_fits_for_GPU_qualification=dict(Source=2,Spatial=2),qualification_is_not_an_independent_performance_cohort=True))
    write(out/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=n,scalar_values_independently_recomputed=checked,all_summary_and_signed_paired_values_exact=True,prediction_hashes_and_unique_roster_checked=True,GT_read=True,public_result_sha256=sha(public/'RESULTS.json')))
    columns=['m_vIoU','m_tIoU','m_sIoU_dense_GT','R_vIoU_at_0p3','R_vIoU_at_0p5']
    idx={(x['dataset'],x['arm'],x['metric']):x['mean']for x in summary}
    text=['# Frozen single-Qwen main method: '+scope,'',
        'All complete predictions were sealed before GT evaluation. An independent CPU readback recomputed every scalar and all signed paired aggregates. This group does not alter the frozen recipe or choose whether the authorized official evaluation starts.', '',
        'Engineering32 is the prelocked exposed HC16/Vid16 development cohort. It is not official full-scale measurement or an independent confirmation.'if scope=='engineering32'else'Complete official population: HC2 3482/237 parent videos and VidSTG 10303/732 parent videos.', '',
        'One frozen Qwen3.5 supplies new two-hypothesis Captions, both-order pairwise preferences and Direct geometry at the actual Native/Adapted observation clocks. Temporal uses the original 512-parameter optimizer and marginal MAP. Spatial uses the unchanged 1792-parameter episodic Gaussian OPD, full-grid response, acceptance/rollback and original domain settings. S-only is an actual Native-clock Teacher/Student run; Full reacquires evidence at the adapted clock.', '',
        '| Dataset | Arm | vIoU % | tIoU % | dense-GT sIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|']
    arms=list(dict.fromkeys(x['arm']for x in rows))
    for ds in ['hc2','vidstg']:
        for arm in arms:
            values=[idx.get((ds,arm,k))for k in columns];cells=['unknown'if v is None else f'{100*v:.6f}'for v in values];text.append('| '+ds+' | '+arm+' | '+' | '.join(cells)+' |')
    text+=['','Source_MAP is the shared no-update marginal-decoder control. Source_original_decoder is the original Source readout. The same-reference D2 control performs no parameter adaptation. Full/Temporal final time is exact-equal, as is S-only/Source_MAP final time. The JSON retains every source, arm, metric, signed difference and unknown; no domain-specific selection or deletion is applied.','',
        'The two-domain GPU qualification proves Source/complete gradient-Adam-response/complete Decoder engineering equivalence to the original path; it is not additional performance evidence. Caption/pairwise preference correctness and adaptation benefit are evaluated from actual outcomes, not assumed from legal output or accepted updates. Existing failed confirmations and negative results remain in the archive.','',
        'Raw video, image pixels, GT coordinates, actual Query/Caption text, raw Teacher responses, tensors, weights, source identities, local paths and credentials are excluded from this export. Actual private input/output pixel review and independent privacy/remote verification are required separately before publication is closed.']
    p=public/'REPORT.md';p.write_text('\n'.join(text)+'\n')
    # Exact public allowlist and hashes, without source bindings or payload paths.
    files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)for p in sorted(public.glob('*'))if p.is_file()]
    write(out/'ANONYMOUS_EXPORT_RECEIPT.json',dict(status='pass',count=n,files=files,raw_private_payloads_excluded=True,GT_read=True))
    print(json.dumps(dict(status='export_complete_requires_pixel_privacy_remote_review',count=n,scalar_readbacks=checked,files=len(files),public_path=str(public.relative_to(ROOT)))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope',choices=['engineering32','official_clean']);report(p.parse_args().scope)
