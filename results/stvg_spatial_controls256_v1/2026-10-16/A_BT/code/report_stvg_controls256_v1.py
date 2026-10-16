"""Per-group scalar export; no model, media, raw prompt or GT-coordinate export."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import read,put,sha,BASE as CORE
SCOPES={'core256':'stvg_spatial_controls256_v1','global32':'stvg_global32_context_v1'}
def main(kind,group):
    base=ROOT/'artifacts'/SCOPES[kind];count=256 if kind=='core256' else 32;per=count//2
    if kind=='core256':
        assert group in ('Full_E1','A','B','C','A_BT','B_BT')
        from scripts.score_stvg_spatial_controls256_v1 import validate_barrier,aggregate
        validate_barrier(group);b=base/'trials'/group;cpu=read(b/'CPU_COMPLETION002.json');p=b/'POSTSEAL_RESULTS002.json'
        assert cpu['status']=='complete' and cpu['count']==count and cpu['GT_read'] and cpu['result_sha256']==sha(p)
        d=read(p);teacher={'A_BT':'A','B_BT':'B'}.get(group,group)
        if group=='C':
            from scripts.audit_stvg_spatial_controls256_edges_v1 import main as edge_audit
            if not(base/'EDGE_AUDIT_CPU_COMPLETION.json').exists():edge_audit()
    else:
        assert group=='G32'
        from scripts.score_stvg_global32_context_v1 import aggregate
        from scripts.run_stvg_global32_student_v1 import check
        check('G32')
        b=base/'trials/G32';cpu=read(base/'CPU_COMPLETION.json');p=base/'POSTSEAL_RESULTS.json'
        assert cpu['status']=='complete' and cpu['count']==count and cpu['GT_read'] and cpu['result_sha256']==sha(p)
        root=read(b/'ROOT_READBACK.json');bar=read(b/'GLOBAL_PREDICTION_BARRIER.json');student=read(b/'STUDENT_GLOBAL_BARRIER.json')
        assert root['status']=='pass' and root['count']==count and not root['GT_read']
        assert bar['status']==student['status']=='complete' and bar['count']==student['count']==count and not bar['GT_read'] and not student['GT_read']
        assert bar['root_sha256']==sha(b/'ROOT_READBACK.json') and root['barrier_sha256']==sha(b/'STUDENT_GLOBAL_BARRIER.json') and bar['files']==student['files']
        for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256'] and sha(ROOT/f['receipt_path'])==f['receipt_sha256']
        d=read(p);teacher='G32'
    assert d['status']=='complete' and d['count']==count and not d['bootstrap']
    s,pairs=aggregate(d['rows']);assert s==d['summary'] and pairs==d['paired']
    ss,pp=aggregate(d['actual_support_rows']);assert ss==d['actual_support_summary'] and pp==d['actual_support_paired']
    aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(read(base/'ROSTER_LOCK.json')['records'],1)};assert len(aliases)==count
    def anon(v):
        if isinstance(v,list):return [anon(x) for x in v]
        if isinstance(v,dict):return {(aliases[k] if k in aliases else k):(aliases[x] if k=='case_id' else anon(x)) for k,x in v.items() if k not in ('GT_bindings','teacher_sha256','teacher_output_sha256')}
        return v
    out=ROOT/'results'/SCOPES[kind]/'2026-10-16'/group;out.mkdir(parents=True,exist_ok=False)
    put(out/'RESULTS.json',dict(status='complete',count=count,per_domain=per,rows=anon(d['rows']),summary=s,paired=anon(pairs),actual_support_rows=anon(d['actual_support_rows']),actual_support_summary=ss,actual_support_paired=anon(pp),GT_read=True,all_positive_negative_neutral_unknown_preserved=True))
    put(out/'UPDATE_AND_GEOMETRY.json',dict(status='complete',count=count,usage=anon(d['usage']),geometry=anon(d['geometry']),semantic_status=d['semantic_status'],GT_read=True))
    requests=[]
    for j in read(base/'TEACHER_JOB_LOCK.json')['records']:
        r=read(base/'teacher'/teacher/(j['job_id']+'.json'))
        requests.append(dict(case_id=aliases[j['case_id']],dataset=j['dataset'],slot=j['center_slot'],parse_status=r['parsed']['status'],selected_candidate_id=r['parsed'].get('selected_candidate_id'),visual_token_count=r['visual_token_count'],output_count=r['output_count'],truncated=r['truncated'],wall_seconds=r['wall_seconds'],sequence_readback=anon(r['sequence_readback']),**{k:r[k] for k in ('new_vision_call','new_teacher_call','new_global_encoder_call','new_local_encoder_calls','processor_seconds','global_cache_seconds','peak_GPU_allocated_bytes','peak_GPU_reserved_bytes') if k in r}))
    config=read(b/'DESIGN_LOCK.json');tb=read(base/'teacher'/teacher/'TEACHER_GLOBAL_BARRIER.json')
    cost=dict(status='complete',count=count,configs=config['configs'],response_settings=config['response_settings'],recipe=config['recipe'],response_scope=config['response_scope'],trainable_parameters=1792,episodic=True,cross_query_LN_writeback=False,requests=requests,teacher={k:v for k,v in tb.items() if k not in ('files',)},actual_fits=sum(x['actual_fit'] for x in d['usage']),accepted_steps=sum(x['accepted_steps'] for x in d['usage']),nonzero_final_updates=sum(x['nonzero_final'] for x in d['usage']),reference_support='four common prospective E1 positions; actual adopted support separately reported; missing cases retained',shared_Source_Temporal_and_visual_preparation_cost_not_zero=True,GT_read=False)
    if group.endswith('_BT'):cost.update(new_teacher_calls_for_this_group=0,exact_teacher_group_reused=teacher)
    if kind=='core256':
        q=read(base/'NATIVE_QUALIFICATION.json');cost['qualification']={k:v for k,v in q.items() if k not in ('records','clock_proof')}
        e=base/'execution/native256_core6'/group;cost['execution_seconds']={mode:read(e/mode/'WORKER_EXIT.json')['wall_seconds'] for mode in ('student_prepare','student_worker','student_root','score')}
    else:
        cost['global_caches']=[dict(case_id=aliases[p.parent.name],**{k:v for k,v in read(p).items() if k in ('encoder_seconds','selection_and_store_seconds')}) for p in sorted((base/'global_features').glob('*/GLOBAL_FEATURE_LOCK.json'))]
        cost['qualification_extra_encodings']=[{k:v for k,v in r.items() if k in ('dataset','status','global_cache_new_encoder_calls','extra_joint_qualification_encoder_calls','original_generate_extra_encoder_calls','joint_encoding_pack_seconds')} for r in read(base/'NATIVE_QUALIFICATION.json')['records']]
        cost['execution_seconds']={mode:read(base/'execution'/mode/'WORKER_EXIT.json')['wall_seconds'] for mode in ('qualification','teacher','teacher_root','student_prepare','student_worker','student_root','score')}
        cost['visual_sequence_ratio_vs_A']=3.8;cost['ratio_is_not_wall_time_ratio']=True
    put(out/'CONFIG_AND_COST.json',anon(cost))
    if kind=='core256':
        protocol=ROOT/'protocols/stvg_spatial_controls256_scope003.md';put(out/'EXECUTION_SCOPE.json',read(base/'EXECUTION_SCOPE_LOCK003.json'))
        if group=='C':
            edge=read(base/'EDGE_AUDIT_CPU_COMPLETION.json');put(out/'ACTUAL_BC_EDGE_PATHS.json',anon(edge))
    else:protocol=ROOT/'protocols/stvg_global32_context_v1.md';put(out/'DESIGN.json',read(base/'DESIGN_LOCK.json'))
    (out/'PROTOCOL.md').write_text(protocol.read_text())
    put(out/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=count,all_aggregate_and_support_scalar_readbacks=True,all_signs_and_unknowns_retained=True,source_result_sha256=sha(p),GT_read=True,semantic_review_separate=True))
    lines=[f'# {SCOPES[kind]} / {group}','',f'{count} fixed development parents, {per} per domain, one query per parent. Membership was locked before cache inventory and new outcomes; historical project exposure remains. This is not an untouched official-fullset result. No Bootstrap or CURRENT promotion.','',
    'The group completed full Student decoding, independent saved-arithmetic/readout roots and complete prediction barriers before GT scoring. All parents, rejected/no-reference cases, positive/negative/neutral differences and unknown sparse-support metrics are preserved. Common prospective anchor support and arm-specific adopted support are reported separately.','',
    'Original rejection, rollback and failed-update stopping remain. A_BT/B_BT change only allowed backtracking scales with identical A/B Teacher references. Full_E1 is an unpruned evidence control; it is not the historical DINO online Spatial OPD and cannot establish a matched optimizer comparison with that method.','',
    'Candidate identity/role/full-extent review remains a separate prelocked arm-blind pixel audit unless a later semantic receipt is supplied. Geometry scores or increased accepted updates alone do not prove semantic correctness. This measurement batch can be published before that later review, with its status explicitly pending.','']
    if kind=='global32':lines += ['The only intervention is global context: full legal input range32 sampled frames, 4096 projected tokens pruned to512, plus exactly the same local5-frame candidate video/96 A indices. Four independent original WHO requests remain. Global features are encoded once per qualified pixel/model/clock binding; extra qualification encodes are itemized. Four LM prefills remain and input sequence ratio3.8 is not a wall-time ratio. This jointly changes global temporal range, density and budget; it does not isolate a compression-algorithm effect.','']
    lines += ['| Student arm | HC vIoU % | VidSTG vIoU % |','|---|---:|---:|']
    for a in dict.fromkeys(r['arm'] for r in d['rows']):lines.append('| '+a+' | '+' | '.join(f"{100*next(x['mean'] for x in s if x['dataset']==ds and x['arm']==a and x['metric']=='m_vIoU'):.6f}" for ds in ('hc2','vidstg'))+' |')
    contrasts=[group+'_minus_Temporal',group+'_minus_Frozen',group+'_minus_'+('A' if kind=='global32' else 'Full_E1')]
    lines+=['','| Domain | Contrast | Delta pp | Positive / negative / unchanged / unknown |','|---|---|---:|---|']
    for r in pairs:
        if r['metric']=='m_vIoU' and r['comparison'] in contrasts:lines.append(f"| {r['dataset']} | {r['comparison']} | {r['mean_delta_pp']:+.6f} | {r['positive']} / {r['negative']} / {r['neutral']} / {r['unknown']} |")
    lines+=['','Shared Source/Temporal/candidate acquisition and earlier visual encoding are not free. Reused outputs are labeled as reuse. New full-model fits/readouts, teacher requests, qualification work and model-load costs are itemized rather than counted as independent repetitions.','',
    'Inference prompts treat scene text, query and descriptions as untrusted data, not executable instructions. Public export excludes media, GT coordinates, raw teacher text, real query/caption strings, weights, raw tensors/optimizer states, personal conversations and credentials. No component is selected per dataset from these outcomes. A group publication is not whole-study FINAL.']
    doc=ROOT/'docs'/f'STVG_{kind.upper()}_{group}_001.md';doc.write_text('\n'.join(lines)+'\n')
    names=['report_stvg_controls256_v1.py','publish_stvg_controls256_v1.py','prepare_stvg_spatial_controls256_v1.py','continue_stvg_spatial_controls256_source002.py','continue_stvg_spatial_controls256_native002.py','continue_stvg_spatial_controls256_native003.py','run_stvg_spatial_controls256_student003.py','run_stvg_spatial_controls256_native_v1.py','run_stvg_spatial_controls256_student_v1.py','score_stvg_spatial_controls256_v1.py','score_stvg_spatial_controls256002.py','audit_stvg_spatial_controls256_policies_v1.py','audit_stvg_spatial_controls256_edges_v1.py','run_stvg_spatial_controls256_temporal_v1.py','run_stvg_spatial_controls256_inputs_v1.py','stvg_instance_spatial32_v1.py','stvg_referent_abc_native002.py','stvg_referent_abc_tokens002.py','stvg_qwen_uniform_clock002.py','stvg_spatial_lite_fit_v1.py','audit_stvg_spatial_lite_fit002.py','run_stvg_instance_student32_v1.py']
    if kind=='global32':names+=['prepare_stvg_global32_context_v1.py','stvg_global32_native_v1.py','run_stvg_global32_context_v1.py','run_stvg_global32_student_v1.py','score_stvg_global32_context_v1.py']
    from scripts.export_stvg_lite_batch001 import public_source_text
    bindings=[]
    for name in names:
        f=ROOT/'scripts'/name;target=out/'code'/name;target.parent.mkdir(exist_ok=True);target.write_text(public_source_text(f.read_bytes()));bindings.append(dict(name=name,source_sha256=sha(f),public_sha256=sha(target)))
    put(out/'SOURCE_BINDINGS.json',dict(status='pass',records=bindings,runtime_sha256=sha(base/('NATIVE_DISPATCH_RUNTIME003.json' if kind=='core256' else 'EXECUTION_RUNTIME.json'))))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,('hc2','vidstg')):
        for i,c in enumerate(dict.fromkeys(contrasts)):
            found=[r for r in pairs if r['dataset']==ds and r['comparison']==c and r['metric']=='m_vIoU']
            if not found:continue
            r=found[0];v=list(r['per_source_delta_pp'].values());ax.scatter([i]*len(v),v,s=11,alpha=.5);ax.scatter([i],[r['mean_delta_pp']],marker='_',s=400,c='black')
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(len(contrasts)),[x.replace('_minus_',' vs ') for x in contrasts],rotation=12);ax.set_title(f'{ds}: all{per} parents');ax.set_ylabel('Student vIoU difference (pp)')
    fig.suptitle(group+'; whole cohort retained');fig.tight_layout();fig.savefig(out/'paired_effects.png',dpi=150);plt.close(fig)
    paths={str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT))}
    put(base/'publication'/group/'EXPORT_MANIFEST.json',dict(status='complete',count=count,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]))
    print(json.dumps(dict(status='exported_pending_actual_pixel_privacy_remote',kind=kind,group=group,count=count,files=len(paths))))
if __name__=='__main__':main(*sys.argv[1:3])
