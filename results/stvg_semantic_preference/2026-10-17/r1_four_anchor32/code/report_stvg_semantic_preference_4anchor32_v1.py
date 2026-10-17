"""Independently verify sealed32 scalars, anonymous export and actual fixed pixel readout panel."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_semantic_preference_4anchor32_v1';TEACHER=BASE;OUT=ROOT/'results/stvg_semantic_preference/2026-10-17/r1_four_anchor32'
def report():
    import numpy as np
    from scipy.stats import t
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    d=read(BASE/'POSTSEAL_RESULTS.json');assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json')
    rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)};idx={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};checks=0
    for s in d['summary']:
        ids=[r['probe_alias'] for r in rr if r['dataset']==s['dataset']];v=[idx[c,s['arm']][s['metric']] for c in ids if idx[c,s['arm']][s['metric']] is not None];assert s['valid']==len(v) and s['unknown']==16-len(v)
        assert not v or abs(s['mean']-math.fsum(v)/len(v))<1e-12;checks+=1
    for p in d['paired']:
        a,b=p['comparison'].split('_minus_');v={r['probe_alias']:100*(idx[r['probe_alias'],a][p['metric']]-idx[r['probe_alias'],b][p['metric']]) for r in rr if r['dataset']==p['dataset'] and idx[r['probe_alias'],a][p['metric']] is not None and idx[r['probe_alias'],b][p['metric']] is not None}
        assert v==p['per_source_delta_pp'];vs=list(v.values());n=len(vs);mean=math.fsum(vs)/n if n else None;assert mean==p['mean_delta_pp'] and p['valid']==n and p['unknown']==16-n
        if n>1:
            e=float(t.ppf(.975,n-1)*np.std(vs,ddof=1)/math.sqrt(n));assert np.allclose([mean-e,mean+e],p['paired_source_t95_interval_pp'],atol=1e-11,rtol=0)
        assert [sum(x>1e-6 for x in vs),sum(x< -1e-6 for x in vs),sum(abs(x)<=1e-6 for x in vs)]==[p[k] for k in ['positive','negative','neutral']];checks+=n
    def anon(x):
        if isinstance(x,dict):return {aliases.get(k,k):anon(v) for k,v in x.items() if k!='GT_bindings'}
        if isinstance(x,list):return [anon(v) for v in x]
        return aliases.get(x,x) if isinstance(x,str) else x
    OUT.mkdir(parents=True,exist_ok=False);write(OUT/'RESULTS.json',dict(anon(d),scope='exposed32 four-anchor fixed-support matched comparison; not independent confirmation'))
    dsg=read(BASE/'arms/GEOM/trials/G32/DESIGN_LOCK.json');write(OUT/'ACTUAL_METHOD.json',dict(count=32,Student_backbone='TA-STVG',source_training_domain_by_target={'hc2':'VidSTG','vidstg':'HC-STVG2'},source_checkpoint_sha256_by_target={ds:sorted({r['source_checkpoint_sha256'] for r in rr if r['dataset']==ds}) for ds in ['hc2','vidstg']},Teacher='frozen STVG-R1-7B',revision='c0db00d019bf746f54b67770b82261b2161fed59',anchors_per_Query=4,slots=[0,1,2,3],candidate_actions=4,raw_underlying_input_exact_Direct=True,preference_only_numbered_candidate_overlay_and_its_fixed_Prompt=True,policy='fixed_support_Gaussian_logit_distillation_not_on_policy',configs=dsg['configs'],response_settings=dsg['response_settings'],trainable_parameters=1792,recipe='lite_fullstep',common_hard_reference='permutation_averaged_semantic_argmax',same_admission_and_reference_acceptance=True,GEOM='(1-p0)*softmax(IoU(candidate,hard_reference)/tau)',PREF='permutation_averaged_p_i_keeping_abstention_mass',CURRENT_changed=False,development_not_confirmation=True))
    write(OUT/'COST.json',dict(new_R1_prefill_calls=192,reused_R1_prefill_calls=64,total_actual_R1_prefill_calls=256,free_generations=0,new_Temporal_or_DINO=0,
        actual_new_Student_fits=sum(u['fit'] for u in d['usage']),stages=[dict(stage=p.parent.name,**read(p)) for p in sorted((BASE/'execution').glob('*/WORKER_EXIT.json'))],original_source_candidate_and_T2_cost_not_zero=True))
    write(OUT/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,all_signed_unknown_pairs_checked=True,no_new_GT_reader=True))
    def value(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
    lines=['# R1 semantic preference versus geometric weights: matched32 Student effect','',
        'Complete exposed HC16/Vid16, all four prelocked anchors per Query, exact original Direct32 observation positions/raw source pixels/context/query; preference images additionally carry the pre-existing numbered candidate overlays, while Direct uses the raw anchor image. All32 previous slot0 outputs are exactly reused;96 new anchor inputs/192 new native forwards,256 total real forwards. Direct used128 coordinate generations; these are different computation types and counts. Remaining Direct differences are continuous boxes vs fixed candidate support and on-policy32 sampling vs fixed-support distillation. The causal mechanism control is same-support GEOM, not Direct. No GT enters input/candidate/Teacher/Student/acceptance. Both Student arms and parameter-free D2 are complete and rooted before evaluation.', '',
        'The first four-anchor GEOM worker completed all32 prediction seals before a pre-existing speed-profile receipt name collision at final I/O (FileExistsError). Original profile, completed predictions, failed-worker log/runtime and its full cost are preserved. Recovery uses a new unique profile label for PREF; GEOM predictions are only rooted, never refit. An initial recovery CPU preflight looked for the later post-root barrier instead of the Student output seal, so runtime creation failed and the launcher exited before any model/priority hold/GT; journal and original source are preserved before the corrected prelaunch. No model, support, parameter, numeric precision, reward, acceptance, or update changed. '+
        'GEOM and PREF use the SAME four frozen WF candidate atoms, SAME hard reference (argmax of two-order averaged R1 preference, with0 abstention), SAME total weight1-p0, original1792 episodic Student/Adam/domain sigma-lr-tau-steps/full-grid response/fullstep reference-IoU acceptance, rollback/reject-stop and full-frame decoder readout. GEOM uses softmax candidate IoU to that hard reference; PREF uses the actual candidate preference mass. This is fixed-support Gaussian distillation, NOT on-policy32 sampled-action OPD. No clamping: an invalid open-logit fixed atom or incoming Source anchor rejects that anchor; abstention rejects that anchor too. All remaining admissible anchors are retained equally in GEOM/PREF/D2; a Query with no admitted anchor is Source/no-fit in all three. This per-anchor handling is the multi-anchor extension of the old single-anchor rule and is disclosed separately. All32 stay in the denominators.', '',
        'Root reconstructs support/weights/likelihood/mean gradient independently and checks saved Adam/response/acceptance/state chain. The prior one-anchor root-only absolute-moment audit failure and recovery are preserved in that separate study. This four-anchor root directly uses the independently validated audit with pointwise gamma8 float32 operation-roundoff bounds for the moments, retains strict5e-6 parameter-delta error, and rejects a deliberate incorrect-variance control. No model, update, precision, acceptance or prediction changed; failed root cost remains recorded. Source and temporal logits are preserved; final cached and actual complete decoder trajectories must be exact. D2 uses same hard reference/admission and Source-relative center/log-size physical-time residual propagation inside T2, Source outside, with zero parameter update.', '',
        '| Domain | Arm | vIoU % | dense common-time sIoU % | tIoU % | nonanchor sIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        for arm in ['Frozen','T2','GEOM','PREF','D2']:lines.append('|'+ds+'|'+arm+'|'+'|'.join('unknown' if (v:=value(ds,arm,k)) is None else f'{100*v:.6f}' for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU','m_sIoU_unsupervised_GT','R_vIoU_at_0p3','R_vIoU_at_0p5'])+'|')
    lines+=['','| Domain | Contrast | vIoU delta pp | paired source t95 pp | + / - / neutral / unknown |','|---|---|---:|---|---|']
    for p in d['paired']:
        if p['metric']=='m_vIoU' and p['comparison'] in ['PREF_minus_GEOM','PREF_minus_T2','GEOM_minus_T2','D2_minus_PREF','D2_minus_T2']:lines.append(f"|{p['dataset']}|{p['comparison']}|{p['mean_delta_pp']:.6f}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
    lines+=['','All accepted/rejected/no-fit reasons and per-source signed/unknown deltas are in RESULTS.json. Paired intervals describe exposed-development uncertainty; no posthoc source exclusions, domain splicing, threshold tuning, new independent confirmation or method promotion. Candidate identity/role correctness remains unlabelled; posterior calibration and candidate coverage are not inferred from this effect. Original semantic512 panel remains44 model-assisted/468 pending. The full two-Teacher2x2 is not complete until Qwen matched cells actually run.']
    from scipy.stats import t as dist
    from collections import Counter
    old=read(ROOT/'artifacts/stvg_semantic_preference_student32_v1/POSTSEAL_RESULTS.json');direct=read(ROOT/'artifacts/stvg_r1_direct_reliable32_v1/POSTSEAL_RESULTS.json')
    oldidx={(r['case_id'],r['arm']):r['metrics'] for r in old['rows']};didx={(r['case_id'],r['arm']):r['metrics'] for r in direct['rows']};cross=[];baseline_checks=0
    for ds in ['hc2','vidstg']:
        ids=[r['probe_alias'] for r in rr if r['dataset']==ds]
        for cid in ids:
            for a in ['Frozen','T2']:
                for k in ['m_vIoU','m_tIoU','m_sIoU_dense_GT']:
                    assert idx[cid,a][k]==didx[cid,a][k]==oldidx[cid,a][k];baseline_checks+=1
        for label,against,aa,scope in [('PREF4_minus_Direct4',didx,'R1_FIXED_OPD','end_to_end_different_feedback_support'),('PREF4_minus_PREF1',oldidx,'PREF','four_vs_one_anchor_with_per_anchor_admission'),('GEOM4_minus_Direct4',didx,'R1_FIXED_OPD','end_to_end_different_feedback_support')]:
            for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU']:
                vals={cid:100*(idx[cid,'GEOM' if label.startswith('GEOM') else 'PREF'][k]-against[cid,aa][k]) for cid in ids};v=list(vals.values());m=math.fsum(v)/len(v);e=float(dist.ppf(.975,len(v)-1)*np.std(v,ddof=1)/math.sqrt(len(v)))
                cross.append(dict(dataset=ds,comparison=label,metric=k,mean_delta_pp=m,paired_source_t95_interval_pp=[m-e,m+e],positive=sum(x>1e-6 for x in v),negative=sum(x< -1e-6 for x in v),neutral=sum(abs(x)<=1e-6 for x in v),unknown=0,per_source_delta_pp=vals,comparison_scope=scope))
    write(OUT/'DIRECT_AND_ONE_ANCHOR_COMPARISONS.json',anon(dict(count=32,records=cross,baseline_exact_scalar_checks=baseline_checks,Direct_receipt_sha256=sha(ROOT/'artifacts/stvg_r1_direct_reliable32_v1/publication/measured32/FINAL_GITHUB_RECEIPT.json') if (ROOT/'artifacts/stvg_r1_direct_reliable32_v1/publication/measured32/FINAL_GITHUB_RECEIPT.json').exists() else None)))
    sup=read(BASE/'SUPPORT_LOCK.json');admission={ds:dict(anchor_reasons=dict(Counter(a['admission_reason'] for a in sup['anchor_decisions'] if a['dataset']==ds)),Query_admitted_anchor_histogram=dict(Counter(str(len(a['active'])) for a in sup['records'] if a['dataset']==ds))) for ds in ['hc2','vidstg']}
    cost=read(OUT/'COST.json');outputs=[read(ROOT/f['path']) for f in read(BASE/'teachers/R1/QUALIFICATION_COMPLETION.json')['files']];slot0=set(read(BASE/'REUSE_LOCK.json')['files'][i]['path'].rsplit('/',1)[-1][:-5] for i in range(32));cost.update(new_prefill_forward_wall_seconds=math.fsum(e['wall_seconds'] for d in outputs if d['job_id'] not in slot0 for e in d['evidence']),reused_prefill_forward_wall_seconds_historical_not_zero=math.fsum(e['wall_seconds'] for d in outputs if d['job_id'] in slot0 for e in d['evidence']),Teacher_model_load_seconds=read(BASE/'teachers/R1/QUALIFICATION_COMPLETION.json')['load_seconds'],scientific_fit_count_by_arm={a:sum(u['fit'] for u in read(BASE/'POSTSEAL_RESULTS.json')['usage'] if u['arm']==a) for a in ['GEOM','PREF']},failed_GEOM_receipt_finalizer_cost_included=True);write(OUT/'COST.json',cost,mutable=True)
    write(OUT/'ANCHOR_ADMISSION.json',dict(count=32,planned_anchors=128,by_domain=admission,all_Query_denominators_preserved=True));lines+=['','## Comparisons with prior one-anchor and Direct four-anchor','','End-to-end Direct contrasts retain different action supports, numbered-image/Prompt versus direct-coordinate interfaces, feedback forms and real calls; they do not isolate semantic likelihood. One-anchor outputs remain unchanged. The inherited private scorer schema scope string is superseded by the four-anchor protocol/input alignment; metric computations and sealed rows are unchanged.','', '| Domain | Contrast | vIoU delta pp | paired source t95 pp | + / - / neutral |','|---|---|---:|---|---|']
    for c in cross:
        if c['metric']=='m_vIoU':lines.append(f"|{c['dataset']}|{c['comparison']}|{c['mean_delta_pp']:.6f}|{c['paired_source_t95_interval_pp']}|{c['positive']}/{c['negative']}/{c['neutral']}|")
    lines+=['','The comparison file contains all source-paired signs and intervals; ANCHOR_ADMISSION.json reports all128 decisions and per-Query reference counts. No selection on GT, no new independent confirmation, no parameter or Prompt tuning.']
    doc=ROOT/'docs/STVG_SEMANTIC_PREFERENCE_R1_FOUR_ANCHOR32_001.md';doc.write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,ds in zip(axes,['hc2','vidstg']):
        aa=['GEOM','PREF','D2'];ax.bar(aa,[100*(value(ds,a,'m_vIoU')-value(ds,'T2','m_vIoU')) for a in aa]);ax.axhline(0,color='black',lw=.7);ax.set_title(ds+' all16 parents');ax.set_ylabel('vIoU change from T2 (pp)')
    fig.tight_layout();fig.savefig(OUT/'effects.png',dpi=170);plt.close(fig)
    for n in ['stvg_semantic_preference_4anchor32_v2.py','continue_stvg_semantic_preference_4anchor32_002.py','report_stvg_semantic_preference_4anchor32_v1.py','stvg_semantic_preference_4anchor32_v1.py','continue_stvg_semantic_preference_4anchor32_v1.py','test_stvg_semantic_preference_4anchor32_v1.py','stvg_semantic_preference_r1_collect32_v1.py','stvg_semantic_preference_student32_v1.py','continue_stvg_semantic_preference_student32_v1.py','test_stvg_semantic_preference_student32_v1.py','report_stvg_semantic_preference_student32_v1.py','audit_stvg_fixed_support_adam002.py','root_stvg_semantic_preference_student32_002.py','continue_stvg_semantic_preference_student32_002.py']:
        p=OUT/'code'/n;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/n).read_bytes()))
    paths={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'protocols/stvg_semantic_preference_student32_v1.md','protocols/stvg_semantic_preference_student32_audit002.md','protocols/stvg_semantic_preference_4anchor32_v1.md','protocols/stvg_semantic_preference_4anchor32_recovery002.md'};verify_privacy(paths)
    write(BASE/'publication/r1_four_anchor32/EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]))
    write(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,GT_read=False));print('STUDENT32_EXPORT_READY',checks)
def render():
    import numpy as np,torch
    from PIL import Image,ImageDraw
    assert read(BASE/'CPU_COMPLETION.json')['count']==32
    rr=read(BASE/'ROSTER_LOCK.json')['records'];jobs=read(BASE/'INPUT_JOB_LOCK.json')['records'];ss={r['case_id']:r for r in read(BASE/'SUPPORT_LOCK.json')['records']};maps={a:{f['case_id']:f for f in read(BASE/'arms'/a/'trials/G32/GLOBAL_PREDICTION_BARRIER.json')['files']} for a in ['GEOM','PREF']};maps['D2']={f['case_id']:f for f in read(BASE/'D2_GLOBAL_PREDICTION_BARRIER.json')['files']};views=[]
    for ds in ['hc2','vidstg']:
        rec=next(r for r in rr if r['dataset']==ds);cid=rec['probe_alias'];preds={}
        for a in ['GEOM','PREF','D2']:
            f=maps[a][cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
            if a=='D2':preds[a]=d['arm']
            else:preds[a]=d['arms']['TS_M2'];preds['T2']=d['arms']['T']
        canvas=Image.new('RGB',(1600,1600),(20,20,20));text=ImageDraw.Draw(canvas);bindings=[]
        for row,j in enumerate(sorted([j for j in jobs if j['case_id']==cid],key=lambda j:j['slot'])):
            assert sha(ROOT/j['image']['path'])==j['image']['sha256'];rgb=np.load(ROOT/j['image']['path'],allow_pickle=False);H,W=rgb.shape[:2];pos=j['position'];ref=next((a for a in ss[cid]['common_reference'] if a['position']==pos),None)
            for col,a in enumerate(['T2','GEOM','PREF','D2']):
                im=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(im);x,y,u,v=map(float,preds[a]['boxes'][pos]);draw.rectangle([x*W,y*H,u*W,v*H],outline=(0,255,255),width=2)
                if ref:
                    x,y,w,h=ref['box_cxcywh'];draw.rectangle([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H],outline=(255,220,0),width=2)
                im.thumbnail((390,360));canvas.paste(im,(col*400,row*400+30));text.text((col*400+6,row*400+8),ds+f' fixed first source slot{j["slot"]} / '+a,fill='white')
            bindings.append(dict(case_id=cid,position=pos,input_sha256=j['image']['sha256']))
        p=BASE/('private_actual_output_panel_'+ds+'.png');canvas.save(p);views.append(dict(dataset=ds,path=str(p.relative_to(ROOT)),sha256=sha(p),records=bindings));print(p)
    write(BASE/'ROOT_PIXEL_VIEW_BINDING.json',dict(status='prepared_pending_actual_view',count=8,files=views,GT_read=False))
if __name__=='__main__':{'report':report,'render':render}[sys.argv[1]]()
