"""Root caption-input repair report, anonymous outputs only; no GT reader."""
from pathlib import Path
import json,sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_spatial_caption_repair_v1 import BASE,CORE,CONF,NATIVE,read,sha,put,verify
from scripts.export_stvg_lite_batch001 import public_source_text
from scripts.publish_stvg_soft_ep_v1 import verify_privacy
def main():
    verify();done=read(BASE/'CAPTION_COMPLETION.json');assert done['status']=='complete' and done['count']==54;diag=read(BASE/'DIAGNOSIS.json');original={};new={(x['cohort'],x['case_id']):x for x in done['files']};rows=[];aliases={(r['cohort'],r['case_id']):f'C{i:03d}' for i,r in enumerate(diag['records'],1)}
    for label,b in [('core256',CORE),('confirmation128',CONF)]:
        caps=[]
        for c in read(b/'CAPTION_LOCK.json')['records']:
            if c['available']:assert sha(ROOT/c['path'])==c['sha256'];caps.append(c);continue
            f=new[label,c['case_id']];p=ROOT/f['path'];assert sha(p)==f['sha256'];v=read(p);assert v['raw_caption'].strip() and v['candidate_id']=='time_Native' and not v['GT_read'];caps.append(dict(case_id=c['case_id'],candidate_id='time_Native',path=f['path'],sha256=f['sha256'],available=True,individual_clip_repair=True,GT_read=False));rows.append(dict(alias=aliases[label,c['case_id']],cohort=label,output_status=v['status'],caption_sha256=v['caption_sha256'],caption_output_tokens=v['caption_output_tokens'],output_truncated_at_cap=v['output_truncated_at_cap'],input_sha256=v['input_sha256'],GT_read=False))
        assert all(c['available'] for c in caps)
        overlay=BASE/'overlays'/label/'CAPTION_LOCK.json'
        if not overlay.exists():put(overlay,dict(status='locked',count=len(caps),records=caps,old_lock_sha256=sha(b/'CAPTION_LOCK.json'),no_original_lock_or_prediction_changed=True,not_reconfirmation=True,GT_read=False))
        else:assert read(overlay)['records']==caps
        original[label]=dict(count=len(caps),original_missing=sum(not c['available'] for c in read(b/'CAPTION_LOCK.json')['records']),overlay_available=sum(c['available'] for c in caps),old_Temporal_root_sha256=sha(b/'temporal/ROOT_READBACK.json'),old_prediction_and_result_protocol_preserved=True)
    assert len(rows)==54
    out=ROOT/'results/stvg_spatial_caption_repair_v1/2026-10-17/caption54';out.mkdir(parents=True,exist_ok=False)
    summary=dict(status='complete',count=54,scope='Caption input engineering repair only; no corrected Student effect is claimed',cohorts=original,missing_reasons=[{k:v for k,v in x.items()} for x in diag['summary']],root_cause=diag['root_cause'],new_plain_caption_calls=54,extra_qualification_caption_calls=done['engineering_extra_caption_calls'],extra_qualification_alignment_calls=done['engineering_alignment_calls'],original_caption_generator_exact_qualification_domains=done['exact_original_caption_qualification_domains'],new_Temporal_fits=0,new_GT_reader_calls=0,old_results_and_confirmation_retained=True,records=rows,GT_read=False)
    put(out/'REPAIR_RESULTS.json',summary);put(out/'CPU_CONTRACT_COMPLETION.json',read(BASE/'CPU_CONTRACT_COMPLETION.json'));put(out/'REPAIR_PROTOCOL.json',read(BASE/'REPAIR_PLAN_LOCK.json'))
    text='''# Spatial Caption acquisition repair:54 valid clips were never requested

Core256 had35 missing Captions (HC8/Vid27). Confirmation128 had19 (HC3/Vid16). All54 were absent because the entire four-candidate Temporal workflow was ineligible, not because an executed R1 generation was empty or its output file was lost. Core35 and18 confirmation cases had fewer than four distinct candidate intervals after valid duplicate removal; one confirmation case had a sample-binding mismatch in the expanded non-Native candidate. Each affected Native clip independently retained eight legal distinct Source frames.

The fix separates individual Spatial Caption eligibility from permission to run the four-candidate Temporal parameter update. The four-distinct-candidate Temporal gate remains unchanged, as do its fallback intervals,512-parameter fit and historical predictions. The new API validates one existing candidate's interval, canonical physical-frame clock and exact Source sampling binding. No invented fourth interval, repeated frame, altered radius or relaxed rejection rule is introduced.

All54 repaired clips use the original eight448x448 marked frames, FPS, Caption Prompt, greedy160-token cap, frozen R1 weights and corrected original clock. Every decoded RGB stream matches the old canonical pixel hash. Two domains additionally compare the actual generated Caption bit-for-bit with the original full adapter's Caption generation; the two engineering Caption calls and two alignment calls are counted separately from54 repair calls. Alignment and Temporal fitting are not prerequisites for Spatial Caption. All actual Caption text and media stay private.

New immutable overlays provide256/256 and128/128 available Caption bindings. Original locks, negative results, noCoverage results and failed Confirmation128 are preserved. This is input repair on already exposed data, not a new independent confirmation and not a claim that B now improves Student.

The active Native factorial is handed over before its first WHO/Student stage. Existing feature work is allowed to exit normally; no live GPU worker is signalled. A new version consumes the repaired Caption bindings. Where Native and Adapted intervals are identical (43 parents), it uses the exact existing E1 control inputs, avoiding unrelated fresh-proposal/render differences. Spatial parameters, Prompt, visual budget, seeds, acceptance rules and Source/Temporal outputs remain fixed. The corrected Full-B comparison reuses an executed complete Native-B prediction only for the35 Caption-missing queries whose intervals, E1 jobs, Source and recipe exactly match; it does not splice frames, windows or states. Historical original Full-B remains a separate row. The corrected factorial is still pending until its real dual-arm roots, predictions and postseal evaluation finish.

Public exports exclude private queries, Caption/Teacher response text, media, annotations/GT coordinates, raw features/tensors, model weights, optimizer states, credentials and conversation records. No CURRENT change or official fullset.
'''
    doc=ROOT/'docs/STVG_SPATIAL_CAPTION_REPAIR_001.md';doc.write_text(text)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([-.17,.83],[35,19],.34,label='Original missing');ax.bar([.17,1.17],[0,0],.34,label='After individual Caption repair');ax.set(xticks=[0,1],xticklabels=['core256','confirmation128'],ylabel='Missing Caption sources');ax.legend();fig.tight_layout();fig.savefig(out/'caption_missing.png',dpi=150);plt.close(fig)
    for name in ('stvg_spatial_caption_repair_v1.py','continue_stvg_captionfix_native002.py','score_stvg_factorial_captionfix002.py','report_stvg_caption_repair_v1.py','publish_stvg_caption_repair_v1.py'):
        p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
    put(out/'ROOT_REPAIR_READBACK.json',dict(status='pass',count=54,all_original_missing_causes_traced=True,actual_nonempty_Captions_input_bound=True,all_original384_bindings_retained_or_explicitly_repaired=True,Temporal_gate_unchanged=True,Student_effect_not_claimed=True,new_GT_reader_calls=0,GT_read=False))
    paths={str(doc.relative_to(ROOT))}|{str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()};verify_privacy(paths)
    put(BASE/'publication/caption54/EXPORT_MANIFEST.json',dict(status='complete',count=54,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));put(BASE/'REPAIR_ROOT_COMPLETION.json',dict(status='pass',count=54,core_overlay_available=256,confirmation_overlay_available=128,old_results_untouched=True,GT_read=False));print('CAPTION54_REPAIR_EXPORTED_PENDING_ACTUAL_PIXELS_REMOTE')
if __name__=='__main__':main()
