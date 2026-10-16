"""Final two-by-two prompt readout, preserving discordant mechanisms."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_spatial_who_order_v1 import BASE,read,put,sha
from scripts.score_stvg_referent_abc_v1 import aggregate
def main():
    d=read(BASE/'STUDY_READBACK.json');s,p=aggregate(d['rows']);assert s==d['summary']and p==d['paired'];assert read(BASE/'CONTROL_INPUT_READBACK.json')['status']=='pass';aliases={cid:f'R{i:03}'for i,cid in enumerate(sorted({x['case_id']for x in d['rows']}),1)}
    def anon(x):
        if isinstance(x,list):return[anon(v)for v in x]
        if isinstance(x,dict):return{(aliases[k]if k in aliases else k):(aliases[v]if k=='case_id'else anon(v))for k,v in x.items()if k not in ('GT_bindings','teacher_sha256','teacher_output_sha256')}
        return x
    out=ROOT/'results/stvg_spatial_who_order_v1/2026-10-16/closing001';out.mkdir(parents=True,exist_ok=False);put(out/'RESULTS.json',anon(d));sem=[];decisions=[]
    for ds in ('hc2','vidstg'):
        for a in ('Full_E1','FULL_ORDER','A','A_ORDER'):
            rr=[x for x in d['semantic_rows']if x['dataset']==ds and x['arm']==a];known=[x for x in rr if x['identity_known']];sem.append(dict(dataset=ds,arm=a,known=len(known),same=sum(x['category']in ('target_full','target_partial')for x in known),complete=sum(x['category']=='target_full'for x in known),role_counts=dict(collections.Counter(x['role_status']for x in rr))))
        for a in ('FULL_ORDER','A_ORDER'):
            rr=[x for x in d['Teacher_decisions']if x['dataset']==ds and x['arm']==a];u=[x for x in d['usage']if x['dataset']==ds and x['arm']==a];decisions.append(dict(dataset=ds,arm=a,windows=len(rr),abstain=sum(x['abstained']for x in rr),invalid=sum(x['parse_status']!='valid'for x in rr),truncated=sum(x['truncated']for x in rr),actual_fits=sum(x['actual_fit']for x in u),anchors=sum(x['anchors']for x in u),accepted_steps=sum(x['accepted_steps']for x in u),nonzero_updates=sum(x['nonzero_parameter_update']for x in u)))
    put(out/'SEMANTIC_AND_UPDATE_READBACK.json',dict(status='pass',summary=sem,update_use=decisions,limits='fixed small exposed single-annotator panel, HC known7 and Vid known4 only; not a population benchmark; no complete-candidate guarantee',GT_read=False));put(out/'CONFIG.json',dict(status='locked',design=read(BASE/'DESIGN_LOCK.json'),control_input_readback=read(BASE/'CONTROL_INPUT_READBACK.json'),native_qualification=read(BASE/'NATIVE_QUALIFICATION.json'),new_main_WHO_generations=128,new_main_visual_encoder_passes=0,qualification_shared_visual_passes=2,qualification_short_generation_decodes=6,actual_new_fits=sum(x['actual_fit']for x in d['usage']),no_Temporal_prompt_or_token_or_OPD_change=True,GT_read=False))
    from scripts.export_stvg_lite_batch001 import public_source_text
    bindings=[]
    for n in ('compare_stvg_spatial_who_order_v1.py','report_stvg_who_order_closing_v1.py','run_stvg_spatial_who_order_v1.py','score_stvg_spatial_who_order_v1.py'):
        f=ROOT/'scripts'/n;q=out/'code'/n;q.parent.mkdir(exist_ok=True);q.write_text(public_source_text(f.read_bytes()));bindings.append(dict(name=n,source_sha256=sha(f),public_sha256=sha(q)))
    put(out/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=16,all_scalar_aggregates_and_pairs_recomputed=True,all_source_signs_unknowns_preserved=True,source_result_sha256=sha(BASE/'STUDY_READBACK.json'),source_bindings=bindings,GT_read=True));(out/'PROTOCOL.md').write_text((ROOT/'protocols/stvg_spatial_who_order_v1.md').read_text())
    lines=['# One ordered Spatial WHO prompt: matched two-by-two results','','One prelocked Referent -> Role -> Visible Extent -> Abstain decision protocol, sixteen exposed development parents, eight per domain, original Full_E1/A controls reused exactly. No Temporal prompt, compression indices, candidates, clock, optimizer, acceptance or hyperparameter change. No further wording search.','','| Actual combined Student | HC vIoU % | VidSTG vIoU % |','|---|---:|---:|']
    for a in ('Frozen','Temporal','Full_E1','FULL_ORDER','A','A_ORDER'):lines.append('| '+a+' | '+' | '.join(f"{100*next(x['mean']for x in s if x['dataset']==ds and x['arm']==a and x['metric']=='m_vIoU'):.6f}"for ds in ('hc2','vidstg'))+' |')
    lines+=['','| Domain | Fixed visual evidence | New minus old prompt (pp) | Positive / negative / unchanged |','|---|---|---:|---|']
    for ds in ('hc2','vidstg'):
        for a,b in (('FULL_ORDER','Full_E1'),('A_ORDER','A')):
            r=next(x for x in p if x['dataset']==ds and x['comparison']==a+'_minus_'+b and x['metric']=='m_vIoU');lines.append(f"| {ds} | {b} | {r['mean_delta_pp']:+.6f} | {r['positive']} / {r['negative']} / {r['neutral']} |")
    lines+=['','| Domain | Arm | Known identity windows | Same entity | Complete target | Role supported / contradicted / unknown (8 windows) |','|---|---|---:|---:|---:|---|']
    for r in sem:
        q=r['role_counts'];lines.append(f"| {r['dataset']} | {r['arm']} | {r['known']} | {r['same']} | {r['complete']} | {q.get('supported',0)} / {q.get('contradicted',0)} / {q.get('unknown',0)} |")
    lines+=['','These semantic labels come from the SAME original fixed sixteen panels, one exposed annotator. Unknown/mixed labels remain unknown; same entity includes partial target. No relabeling after seeing prompt outcomes. Prompt-compatible selected labels do not reveal hidden model reasoning, and more complete targets in a few inspected windows do not prove that every adopted reference is geometrically sound.','',
      'The interaction (A_ORDER-A)-(FULL_ORDER-Full_E1) is descriptive and exported for all five metrics. It separates a prompt effect under each visual input from the compression main effect. Full retains original padding/context and1280 tokens; A retains exact original160 native tokens and valid-area/ID safeguards. This is not a pure padding or runtime-speed contrast.','',
      'All16 prediction streams and saved Student arithmetic/final-readout roots were sealed before each group GT scoring. The old Full_E1 and A baselines are reused only with matching Source, candidate, physical clock, original prompt and configuration. All160 unique arm-parent rows, metric nulls, every paired sign, sparse geometry, accepted/nonzero and zero/reference-absent cases are retained. No Bootstrap, fullset, LN carry or CURRENT change.','',
      'This bounded protocol ends here. A cross-domain method or unified final prompt is not selected from a per-dataset development best score. Conditional larger confirmation would require a separately locked version and cohort; this study does not launch it. Discordant semantics and Student gains are evidence for a remaining transfer bottleneck, not a solved mechanism. The public prompt template and actual configuration are included; private media/GT coordinates/query-caption values/raw responses/tensors/parameters remain excluded.']
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for aa,ds in zip(ax,('hc2','vidstg')):
        for i,(a,b)in enumerate((('FULL_ORDER','Full_E1'),('A_ORDER','A'))):
            r=next(x for x in p if x['dataset']==ds and x['comparison']==a+'_minus_'+b and x['metric']=='m_vIoU');aa.scatter([i]*8,list(r['per_source_delta_pp'].values()),s=25);aa.scatter([i],[r['mean_delta_pp']],marker='_',s=400,c='black')
        aa.axhline(0,c='gray',lw=.8);aa.set_xticks([0,1],['Full1280','A160']);aa.set_ylabel('New minus old WHO / Student vIoU (pp)');aa.set_title(ds+' / 8 exposed parents')
    fig.suptitle('One fixed WHO prompt under two visual budgets');fig.tight_layout();fig.savefig(out/'paired_effects.png',dpi=150);plt.close(fig)
    doc=ROOT/'docs/STVG_SPATIAL_WHO_ORDER_CLOSING001.md';doc.write_text('\n'.join(lines)+'\n');paths={str(f.relative_to(ROOT))for f in out.rglob('*')if f.is_file()}|{str(doc.relative_to(ROOT))};put(BASE/'publication/closing001/EXPORT_MANIFEST.json',dict(status='complete',paths=sorted(paths),files=[dict(path=f,sha256=sha(ROOT/f),bytes=(ROOT/f).stat().st_size)for f in sorted(paths)]));print(json.dumps(dict(status='exported_pending_actual_pixel_privacy_remote',semantic=sem,update_use=decisions,files=len(paths))))
if __name__=='__main__':main()
