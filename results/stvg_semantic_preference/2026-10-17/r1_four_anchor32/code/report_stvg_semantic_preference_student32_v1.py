"""Independently verify sealed32 scalars, anonymous export and actual fixed pixel readout panel."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_semantic_preference_student32_v1';TEACHER=ROOT/'artifacts/stvg_semantic_preference_r1_32_v1';OUT=ROOT/'results/stvg_semantic_preference/2026-10-17/r1_student32'
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
    OUT.mkdir(parents=True,exist_ok=False);write(OUT/'RESULTS.json',anon(d))
    dsg=read(BASE/'arms/GEOM/trials/G32/DESIGN_LOCK.json');write(OUT/'ACTUAL_METHOD.json',dict(count=32,Student_backbone='TA-STVG',source_training_domain_by_target={'hc2':'VidSTG','vidstg':'HC-STVG2'},source_checkpoint_sha256_by_target={ds:sorted({r['source_checkpoint_sha256'] for r in rr if r['dataset']==ds}) for ds in ['hc2','vidstg']},Teacher='frozen STVG-R1-7B',revision='c0db00d019bf746f54b67770b82261b2161fed59',anchors_per_Query=1,slot=0,candidate_actions=4,policy='fixed_support_Gaussian_logit_distillation_not_on_policy',configs=dsg['configs'],response_settings=dsg['response_settings'],trainable_parameters=1792,recipe='lite_fullstep',common_hard_reference='permutation_averaged_semantic_argmax',same_admission_and_reference_acceptance=True,GEOM='(1-p0)*softmax(IoU(candidate,hard_reference)/tau)',PREF='permutation_averaged_p_i_keeping_abstention_mass',CURRENT_changed=False,development_not_confirmation=True))
    write(OUT/'COST.json',dict(new_R1_prefill_calls=48,reused_R1_prefill_calls=16,total_actual_R1_prefill_calls=64,free_generations=0,new_Temporal_or_DINO=0,
        actual_new_Student_fits=sum(u['fit'] for u in d['usage']),stages=[dict(stage=p.parent.name,**read(p)) for p in sorted((BASE/'execution').glob('*/WORKER_EXIT.json'))],original_source_candidate_and_T2_cost_not_zero=True))
    write(OUT/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,all_signed_unknown_pairs_checked=True,no_new_GT_reader=True))
    def value(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
    lines=['# R1 semantic preference versus geometric weights: matched32 Student effect','',
        'Complete exposed HC16/Vid16, fixed slot0 one anchor per Query. Original32/4-anchor Direct is a different observation setting and is not used as this matched control. Existing first8 preference outputs are exactly reused; only24 more parents/48 additional native forwards,64 total real R1 forwards. No GT enters input/candidate/Teacher/Student/acceptance. Both Student arms and parameter-free D2 are complete and rooted before evaluation.', '',
        'GEOM and PREF use the SAME four frozen WF candidate atoms, SAME hard reference (argmax of two-order averaged R1 preference, with0 abstention), SAME total weight1-p0, original1792 episodic Student/Adam/domain sigma-lr-tau-steps/full-grid response/fullstep reference-IoU acceptance, rollback/reject-stop and full-frame decoder readout. GEOM uses softmax candidate IoU to that hard reference; PREF uses the actual candidate preference mass. This is fixed-support Gaussian distillation, NOT on-policy32 sampled-action OPD. No clamping: an invalid open-logit fixed atom or incoming Source anchor makes the entire Query a common Source no-op; abstentions remain Source in all three methods. All32 stay in the denominators.', '',
        'Root reconstructs support/weights/likelihood/mean gradient independently and checks saved Adam/response/acceptance/state chain. The original root-only fixed5e-6 absolute Adam moment comparison failed on large fixed-support gradients; all32 GEOM predictions and Teacher outputs were preserved. The corrected independent audit uses pointwise gamma8 float32 operation-roundoff bounds for the moments, retains strict5e-6 parameter-delta error, and rejects a deliberate incorrect-variance control. No model, update, precision, acceptance or prediction changed; failed root cost remains recorded. Source and temporal logits are preserved; final cached and actual complete decoder trajectories must be exact. D2 uses same hard reference/admission and Source-relative center/log-size physical-time residual propagation inside T2, Source outside, with zero parameter update.', '',
        '| Domain | Arm | vIoU % | dense common-time sIoU % | tIoU % | nonanchor sIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        for arm in ['Frozen','T2','GEOM','PREF','D2']:lines.append('|'+ds+'|'+arm+'|'+'|'.join('unknown' if (v:=value(ds,arm,k)) is None else f'{100*v:.6f}' for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU','m_sIoU_unsupervised_GT','R_vIoU_at_0p3','R_vIoU_at_0p5'])+'|')
    lines+=['','| Domain | Contrast | vIoU delta pp | paired source t95 pp | + / - / neutral / unknown |','|---|---|---:|---|---|']
    for p in d['paired']:
        if p['metric']=='m_vIoU' and p['comparison'] in ['PREF_minus_GEOM','PREF_minus_T2','GEOM_minus_T2','D2_minus_PREF','D2_minus_T2']:lines.append(f"|{p['dataset']}|{p['comparison']}|{p['mean_delta_pp']:.6f}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
    lines+=['','All accepted/rejected/no-fit reasons and per-source signed/unknown deltas are in RESULTS.json. Paired intervals describe exposed-development uncertainty; no posthoc source exclusions, domain splicing, threshold tuning, new independent confirmation or method promotion. Candidate identity/role correctness remains unlabelled; posterior calibration and candidate coverage are not inferred from this effect. Original semantic512 panel remains44 model-assisted/468 pending. The full two-Teacher2x2 is not complete until Qwen matched cells actually run.']
    doc=ROOT/'docs/STVG_SEMANTIC_PREFERENCE_R1_STUDENT32_001.md';doc.write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,ds in zip(axes,['hc2','vidstg']):
        aa=['GEOM','PREF','D2'];ax.bar(aa,[100*(value(ds,a,'m_vIoU')-value(ds,'T2','m_vIoU')) for a in aa]);ax.axhline(0,color='black',lw=.7);ax.set_title(ds+' all16 parents');ax.set_ylabel('vIoU change from T2 (pp)')
    fig.tight_layout();fig.savefig(OUT/'effects.png',dpi=170);plt.close(fig)
    for n in ['stvg_semantic_preference_r1_collect32_v1.py','stvg_semantic_preference_student32_v1.py','continue_stvg_semantic_preference_student32_v1.py','test_stvg_semantic_preference_student32_v1.py','report_stvg_semantic_preference_student32_v1.py','audit_stvg_fixed_support_adam002.py','root_stvg_semantic_preference_student32_002.py','continue_stvg_semantic_preference_student32_002.py']:
        p=OUT/'code'/n;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/n).read_bytes()))
    paths={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'protocols/stvg_semantic_preference_student32_v1.md','protocols/stvg_semantic_preference_student32_audit002.md'};verify_privacy(paths)
    write(BASE/'publication/r1_student32/EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]))
    write(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,GT_read=False));print('STUDENT32_EXPORT_READY',checks)
def render():
    import numpy as np,torch
    from PIL import Image,ImageDraw
    assert read(BASE/'CPU_COMPLETION.json')['count']==32
    rr=read(BASE/'ROSTER_LOCK.json')['records'];jj={j['case_id']:j for j in read(BASE/'INPUT_JOB_LOCK.json')['records']};ss={r['case_id']:r for r in read(BASE/'SUPPORT_LOCK.json')['records']};maps={a:{f['case_id']:f for f in read(BASE/'arms'/a/'trials/G32/GLOBAL_PREDICTION_BARRIER.json')['files']} for a in ['GEOM','PREF']};maps['D2']={f['case_id']:f for f in read(BASE/'D2_GLOBAL_PREDICTION_BARRIER.json')['files']}
    rows=[r for ds in ['hc2','vidstg'] for r in [r for r in rr if r['dataset']==ds][:2]];canvas=Image.new('RGB',(1600,1600),(20,20,20));text=ImageDraw.Draw(canvas);bindings=[]
    for i,r in enumerate(rows):
        cid=r['probe_alias'];j=jj[cid];pos=j['position'];assert sha(ROOT/j['image']['path'])==j['image']['sha256'];rgb=np.load(ROOT/j['image']['path'],allow_pickle=False);H,W=rgb.shape[:2];preds={}
        for a in ['GEOM','PREF','D2']:
            f=maps[a][cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
            if a=='D2':preds[a]=d['arm']
            else:preds[a]=d['arms']['TS_M2'];preds['T2']=d['arms']['T']
        for col,a in enumerate(['T2','GEOM','PREF','D2']):
            im=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(im);x,y,u,v=map(float,preds[a]['boxes'][pos]);draw.rectangle([x*W,y*H,u*W,v*H],outline=(0,255,255),width=2)
            anchors=ss[cid]['common_reference']
            if anchors:
                x,y,w,h=anchors[0]['box_cxcywh'];draw.rectangle([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H],outline=(255,220,0),width=2)
            im.thumbnail((390,360));canvas.paste(im,(col*400,i*400+30));text.text((col*400+6,i*400+8),r['dataset']+f' fixed source {i+1} / '+a,fill='white')
        bindings.append(dict(case_id=cid,position=pos,input_sha256=j['image']['sha256']))
    p=BASE/'private_actual_output_panel.png';canvas.save(p);write(BASE/'ROOT_PIXEL_VIEW_BINDING.json',dict(status='prepared_pending_actual_view',count=4,path=str(p.relative_to(ROOT)),sha256=sha(p),records=bindings,GT_read=False));print(p)
if __name__=='__main__':{'report':report,'render':render}[sys.argv[1]]()
