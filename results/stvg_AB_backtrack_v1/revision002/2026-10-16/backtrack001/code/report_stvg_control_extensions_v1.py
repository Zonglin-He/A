"""Complete scalar-only exports for bounded controlled research extensions."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referent_abc002 import read,put,sha
from scripts.score_stvg_referent_abc_v1 import aggregate
SCOPES={'who':'stvg_spatial_who_order_v1','ABT':'stvg_AB_backtrack_v1/revision002','edges':'stvg_BC_retained_edges_v1','transfer':'stvg_B_transfer_v1'}
def main(kind,group):
    base=ROOT/'artifacts'/SCOPES[kind];out=ROOT/'results'/SCOPES[kind]/'2026-10-16'/group;out.mkdir(parents=True,exist_ok=False)
    if kind=='who':d=read(base/'trials'/group/'POSTSEAL_RESULTS_EXTENDED.json');assert read(base/'trials'/group/'EXTENDED_CPU_COMPLETION.json')['result_sha256']==sha(base/'trials'/group/'POSTSEAL_RESULTS_EXTENDED.json')
    elif kind in ('ABT','transfer'):d=read(base/'STUDY_READBACK.json');assert d['status']=='pass'
    else:d=read(base/'CPU_COMPLETION.json');assert d['status']=='complete'and not d['GT_read']
    rows=d['rows'];aliases={cid:f'R{i:03}'for i,cid in enumerate(sorted({x['case_id']for x in rows}),1)}
    def anon(x):
        if isinstance(x,list):return[anon(v)for v in x]
        if isinstance(x,dict):return{(aliases[k]if k in aliases else'gradient_max_abs_error'if k=='gradient'else k):(aliases[v]if k=='case_id'else anon(v))for k,v in x.items()if k not in ('GT_bindings','teacher_sha256','teacher_output_sha256')}
        return x
    if kind!='edges':
        s,p=aggregate(rows);assert s==d['summary']and p==d['paired'];put(out/'RESULTS.json',dict(status='complete',count=16,rows=anon(rows),summary=s,paired=anon(p),interactions=anon(d.get('interactions',[])),all_positive_negative_neutral_unknown_preserved=True,GT_read=True))
    else:put(out/'RESULTS.json',anon(d))
    lines=['# '+SCOPES[kind]+' / '+group,'','Sixteen already exposed development parents, eight HC and eight Vid, one query per parent. No untouched confirmation or fullset claim. All source signs and unknowns are retained. No Bootstrap or CURRENT promotion.','']
    if kind=='who':
        put(out/'SEMANTIC_AND_GEOMETRY.json',dict(status='complete',semantic=anon(d['semantic']),geometry=anon(d['geometry']),usage=anon(d['usage']),GT_read=True));b=base/'trials'/group;t=read(base/'teacher'/group/'TEACHER_GLOBAL_BARRIER.json');dd=read(b/'DESIGN_LOCK.json');req=[]
        for j in read(base/'TEACHER_JOB_LOCK.json')['records']:
            a=read(base/'teacher'/group/(j['job_id']+'.json'));req.append(dict(case_id=aliases[j['case_id']],dataset=j['dataset'],slot=j['center_slot'],parse_status=a['parsed']['status'],visible_id=a['parsed'].get('visible_id'),truncated=a['truncated'],generated_tokens=a['output_count'],wall_seconds=a['wall_seconds']))
        put(out/'CONFIG_AND_COST.json',dict(status='complete',design=read(base/'DESIGN_LOCK.json'),configs=dd['configs'],response_settings=dd['response_settings'],recipe=dd['recipe'],response_scope=dd['response_scope'],visual_tokens_kept=1280 if group=='FULL_ORDER'else 160,visual_tokens_original=1280,max_new_tokens=96,repetition_penalty=1.05,requests=req,qualification=read(base/'NATIVE_QUALIFICATION.json'),new_visual_encodings=t['new_vision_calls'],new_generations=t['new_greedy_generation_calls'],model_load_seconds=t['model_load_seconds'],loop_seconds=t['worker_loop_seconds'],worker_seconds={m:read(base/'execution'/group/m/'WORKER_EXIT.json')['wall_seconds']for m in ('teacher','teacher_root','student_prepare','student_worker','student_root')},actual_fits=sum(x['actual_fit']for x in d['usage']),accepted_steps=sum(x['accepted_steps']for x in d['usage']),nonzero_updates=sum(x['nonzero_parameter_update']for x in d['usage']),inherited_inputs_and_captions_not_free=True,GT_read=False))
        lines+=['One prelocked ordered Referent Identity -> Relation/Role -> Visible Extent -> Abstention protocol. Original video explanations, query, allowed IDs, system message and JSON-only generation budget remain fixed. There is no Temporal prompt change, token selector change, coordinate expansion or OPD/backtracking change. Full retains original1280 visual tokens including old padding; A retains exactly the old160 indices. This is a prompt intervention, not a new compression method.','',
          'Original Full_E1 and A controls reuse their exact sealed predictions. New WHO generations, reference admission, actual1792-parameter episodic Student updates and complete final readouts are performed. The native bridge passed first8 unpruned generate-token equality on both domains and A cached/uncached next-token checks. Projected visual features are reused after bytes/tensor/policy bindings, not raw output reuse. Qualification incurs two fresh encoder calls; main groups incur none. Inherited preprocessing/Source/caption costs are not zero.','',
          'Identity and role use only the same sixteen previously locked private panels, one project-exposed annotator, not new GT-driven relabeling. Same entity includes partial target, complete target does not. Unknown roles and labels remain unknown. The pool may lack any complete target. Known-anchor geometric IoU and selected-minus-Source are separate offline diagnostics. Neither semantic labels, known-anchor IoU nor more accepted updates alone establishes a Student gain.','']
        lines+=['| Domain | Arm | Known identity windows | Same entity | Complete target | Role supported / contradicted / unknown |','|---|---|---:|---:|---:|---|']
        for r in d['semantic']['summary']:
            v=r['role_counts'];lines.append(f"| {r['dataset']} | {r['arm']} | {r['identity_known_windows']} | {r['same_entity_known']} | {r['complete_target_known']} | {v.get('supported',0)} / {v.get('contradicted',0)} / {v.get('unknown',0)} |")
        names=['run_stvg_spatial_who_order_v1.py','score_stvg_spatial_who_order_v1.py','stvg_referent_abc_native002.py','stvg_referent_abc_student002.py','score_stvg_referent_abc_v1.py','stvg_qwen_uniform_clock002.py','stvg_spatial_lite_fit_v1.py','audit_stvg_spatial_lite_fit002.py'];protocol=ROOT/'protocols/stvg_spatial_who_order_v1.md';comparisons=[group+'_minus_'+d['paired_original_control'],group+'_minus_Temporal',group+'_minus_Frozen']
    elif kind=='edges':
        lines+=['Saved real graph edges are counted only when both actual selected endpoints survive. Local three-cell paths require both consecutive edges and all three vertices. Positive weights are normalized by all available positive graph weights; missing graphs and nonapplicable context paths remain unknown. Per-window ratios are averaged within each parent before equal-parent aggregation. Source aliases refer to the same16 parents, not512 independent samples. Numeric ID glyphs were excluded from graph edges.','',
          'These are reciprocal same-candidate projected-feature correspondences, not semantic object identity, attention CS, or proof of correct role. Preserving more edges may retain background or wrong-object structure. Both B and C use static input text embeddings; permuting the same embedding multiset does not change the mean-top-two cosine score. No new model, token selection, GT read or Student fit is executed by this audit.','',
          'The prelocked one-shot paired-endpoint C trigger requires local source-macro C-minus-B <=0 in at least one fully evaluable domain. Actual trigger: '+str(d['paired_C_triggered'])+'. No extra paired-C generation is run when the trigger is false. A positive preservation increase without a Student benefit directs interpretation toward edge quality and downstream use, not an absent retained-edge increase.','', '| Domain | View | C-minus-B retained weighted-edge ratio |','|---|---|---:|']
        for r in d['paired']:lines.append(f"| {r['dataset']} | {r['view']} | {r['C_minus_B']:+.8f} |")
        names=['audit_stvg_BC_retained_edges_v1.py','stvg_referent_abc_tokens002.py'];protocol=None;comparisons=[]
    elif kind=='ABT':
        ab=read(base/'trials/ABT/POSTSEAL_RESULTS.json');put(out/'UPDATE_USE.json',dict(status='complete',usage=anon(ab['usage']),prefix=anon(read(base/'trials/ABT/PREFIX_READBACK.json')),GT_read=False));put(out/'CONFIG_AND_COST.json',dict(status='complete',design=read(base/'DESIGN_LOCK.json'),configs=read(base/'trials/ABT/DESIGN_LOCK.json')['configs'],worker_seconds={m:read(base/'execution/ABT'/m/'WORKER_EXIT.json')['wall_seconds']for m in ('worker','root')},actual_new_fits=sum(x['actual_fit']for x in ab['usage']if x['arm']=='ABT'),new_Teacher_calls=0,B_BT_exact_R01_reuse=True,GT_read=False))
        lines+=['A-Full/B-Full/A-BT/B-BT uses exact original references within each visual arm. Only A-BT requires new16 Student fit/readout trials; B-BT is the already completed exact R01, not an additional GPU run. Both backtracking arms keep the original finite/nondecreasing mean-Teacher-IoU acceptance, Adam rollback and early stop after all four scales fail. Original full proposals are tried first, then1/.5/.25/.125. No rejected proposal is forced through and no learning rate or loss is retuned.','',
          'The interaction (B_BT-B_Full)-(A_BT-A_Full) checks whether step reachability selectively restores the value of B evidence rather than generally helping both evidence types. Increased reference agreement is not GT correctness. Saved original full-proposal common-prefix arithmetic is audited before rescue changes trajectories; it is not a second complete backward.','']
        names=['run_stvg_AB_backtrack002.py','score_stvg_AB_backtrack002.py','run_stvg_AB_backtrack_v1.py','score_stvg_AB_backtrack_v1.py','audit_stvg_B_transfer_v1.py','score_stvg_B_transfer_v1.py','run_stvg_B_transfer_v1.py','stvg_spatial_lite_fit_v1.py','audit_stvg_spatial_lite_fit002.py'];protocol=None;comparisons=['A_BT_minus_A_Full','B_BT_minus_B_Full','B_BT_minus_A_BT']
    else:
        put(out/'FACTORIAL_DIAGNOSTICS.json',anon({k:v for k,v in d.items()if k not in ('rows','summary','paired','actual_support_rows','actual_support_summary','actual_support_paired')}));lines+=['Original B reference extent x guarded reachability factorial. G10 changes reference width/height by fixed1.25 at identical centers and IDs, R01 uses original boxes with original four allowed backtracking scales, GR11 combines them. The original acceptance gate is preserved throughout. Expansion is a diagnostic perturbation, not a certified complete target. Known-anchor coverage/precision and final full-Tube effects have different support and cannot be numerically substituted.','',
          'Three new groups each contain16 actual fits and complete model readouts. Exact B00 control is reused, not a fourth independent cohort. Only three original B cases reject all original proposals; the others already update. The predefined originally-rejected subgroup is descriptive, not a posthoc accepted-sample metric. All parents are included in the primary equal-parent means.','']
        names=['compare_stvg_B_transfer_v1.py','run_stvg_B_transfer_v1.py','audit_stvg_B_transfer_v1.py','score_stvg_B_transfer_v1.py'];protocol=ROOT/'protocols/stvg_B_transfer_v1.md';comparisons=['G10_minus_B00','R01_minus_B00','GR11_minus_B00']
    if kind!='edges':
        lines+=['','| Actual Student arm | HC vIoU % | VidSTG vIoU % |','|---|---:|---:|']
        for a in dict.fromkeys(r['arm']for r in rows):lines.append('| '+a+' | '+' | '.join(f"{100*next(x['mean']for x in s if x['dataset']==ds and x['arm']==a and x['metric']=='m_vIoU'):.6f}"for ds in ('hc2','vidstg'))+' |')
        lines+=['','| Domain | Contrast | Mean delta pp | Positive / negative / unchanged |','|---|---|---:|---|']
        for r in p:
            if r['metric']=='m_vIoU'and r['comparison']in comparisons:lines.append(f"| {r['dataset']} | {r['comparison']} | {r['mean_delta_pp']:+.6f} | {r['positive']} / {r['negative']} / {r['neutral']} |")
        lines+=['','All five official/independent metrics and nulls are supplied after complete cohort prediction barriers and independent saved-arithmetic/readout roots pass. Four common prospective observation positions are not necessarily the actually adopted supervision support. Original Frozen/Temporal controls and complete Student predictions remain matched. Full_E1 is native unpruned R1 evidence, not the historical DINO online method.','']
    if protocol:(out/'PROTOCOL.md').write_text(protocol.read_text())
    else:put(out/'PROTOCOL.json',read(base/'DESIGN_LOCK.json'))
    put(out/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=16,all_scalar_aggregates_recomputed=kind!='edges',all_signs_and_unknowns_exported=True,source_result_sha256=sha(base/'CPU_COMPLETION.json')if kind=='edges'else sha(base/'STUDY_READBACK.json')if kind in ('ABT','transfer')else sha(base/'trials'/group/'POSTSEAL_RESULTS_EXTENDED.json'),GT_read=kind!='edges'))
    from scripts.export_stvg_lite_batch001 import public_source_text
    bindings=[]
    for n in names+['report_stvg_control_extensions_v1.py','publish_stvg_control_extensions_v1.py']:
        f=ROOT/'scripts'/n;q=out/'code'/n;q.parent.mkdir(exist_ok=True);q.write_text(public_source_text(f.read_bytes()));bindings.append(dict(name=n,source_sha256=sha(f),public_sha256=sha(q)))
    put(out/'SOURCE_BINDINGS.json',dict(status='pass',records=bindings,runtime_sha256=sha(base/'RUNTIME.json')))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,('hc2','vidstg')):
        if kind=='edges':
            for i,view in enumerate(('context','local')):
                for k,a in enumerate(('B','C')):
                    r=next(x for x in d['summary']if x['dataset']==ds and x['view']==view and x['arm']==a);ax.bar(i+(k-.5)*.3,r['mean_retained_edge_ratio'],width=.28,label=a if i==0 else None)
            ax.set_xticks([0,1],['context','local']);ax.set_ylabel('Actual retained positive edge weight ratio');ax.legend()
        else:
            for i,c in enumerate(comparisons):
                r=next(x for x in p if x['dataset']==ds and x['comparison']==c and x['metric']=='m_vIoU');ax.scatter([i]*8,list(r['per_source_delta_pp'].values()),s=25,alpha=.7);ax.scatter([i],[r['mean_delta_pp']],marker='_',s=450,c='black');ax.set_ylabel('Actual Student vIoU delta (pp)')
            ax.set_xticks(range(len(comparisons)),[c.replace('_minus_',' vs ')for c in comparisons],rotation=10);ax.axhline(0,c='gray',lw=.8)
        ax.set_title(ds+' / 8 exposed parents')
    fig.suptitle(kind+' / '+group+'; all parents retained');fig.tight_layout();fig.savefig(out/'paired_effects.png',dpi=150);plt.close(fig)
    lines+=['Private media, real query/caption strings, raw Teacher responses, GT coordinates, feature tensors, parameter/optimizer payloads and credentials are excluded. This complete group is published independently; a group receipt does not establish method freeze or broader study completion.']
    doc=ROOT/'docs'/('STVG_CONTROL_'+kind.upper()+'_'+group+'_001.md');doc.write_text('\n'.join(lines)+'\n');paths={str(f.relative_to(ROOT))for f in out.rglob('*')if f.is_file()}|{str(doc.relative_to(ROOT))};put(base/'publication'/group/'EXPORT_MANIFEST.json',dict(status='complete',paths=sorted(paths),files=[dict(path=f,sha256=sha(ROOT/f),bytes=(ROOT/f).stat().st_size)for f in sorted(paths)]));print(json.dumps(dict(status='exported_pending_pixel_privacy_remote',kind=kind,group=group,files=len(paths))))
if __name__=='__main__':main(*sys.argv[1:3])
