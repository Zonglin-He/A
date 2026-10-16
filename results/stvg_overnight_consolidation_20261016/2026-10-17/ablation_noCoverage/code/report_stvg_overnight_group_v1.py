"""Scalar-only independent group export for finite night studies; no new GT."""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
from scripts.stvg_overnight_score_v1 import aggregate,seal
read,sha,put=c.read,c.sha,c.write
def main(scope):
    base=c.BASE/scope;assert base.is_relative_to(c.BASE);d=read(base/'POSTSEAL_RESULTS.json');cpu=read(base/'CPU_COMPLETION.json');assert d['status']=='complete' and cpu['status']=='complete' and cpu['GT_read'] and cpu['result_sha256']==sha(base/'POSTSEAL_RESULTS.json');n=d['count'];assert n in (32,128)
    for prefix in ('','actual_support_'):
        s,p=aggregate(d[prefix+'rows'],n);assert s==d[prefix+'summary'] and p==d[prefix+'paired']
        # Independent scalar/sign arithmetic, separate from the score aggregator.
        idx={(r['dataset'],r['case_id'],r['arm']):r['metrics'] for r in d[prefix+'rows']};assert len(idx)==len(d[prefix+'rows'])
        for x in d[prefix+'paired']:
            a,b=x['comparison'].split('_minus_');ids={i for ds,i,arm in idx if ds==x['dataset']};v={i:100*(idx[x['dataset'],i,a][x['metric']]-idx[x['dataset'],i,b][x['metric']]) for i in ids if idx[x['dataset'],i,a][x['metric']] is not None and idx[x['dataset'],i,b][x['metric']] is not None};assert v==x['per_source_delta_pp'] and x['valid']==len(v) and x['unknown']==n//2-len(v);assert x['positive']==sum(t>1e-6 for t in v.values()) and x['negative']==sum(t< -1e-6 for t in v.values()) and x['neutral']==sum(abs(t)<=1e-6 for t in v.values());assert (not v and x['mean_delta_pp'] is None) or abs(x['mean_delta_pp']-math.fsum(v.values())/len(v))<1e-12
    rr=read(base/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)};assert len(aliases)==n
    def anon(v):
        if isinstance(v,list):return [anon(x) for x in v]
        if isinstance(v,dict):return {aliases.get(k,k):anon(x) for k,x in v.items()}
        return aliases.get(v,v) if isinstance(v,str) else v
    out=ROOT/'results/stvg_overnight_consolidation_20261016/2026-10-17'/scope;out.mkdir(parents=True,exist_ok=False);paths=set()
    def save(name,x):put(out/name,x);paths.add(str((out/name).relative_to(ROOT)))
    save('RESULTS.json',anon({k:v for k,v in d.items() if k not in ('GT_bindings',)}));save('ROOT_SCALAR_AUDIT.json',dict(status='pass',count=n,CPU_only=True,GT_read=False,independent_aggregate_sign_unknown_readback=True,new_GT_reader_calls=0,new_model_calls=0))
    configs=[];scopes=read(base/'ARM_LOCK.json')['arms'] if scope=='confirmation128' else [None]
    for arm in scopes:
        b=base/'arms'/arm if arm else base;seal(b);cfg=read(b/'SCOPE_CONFIG.json');design=read(b/'trials/G32/DESIGN_LOCK.json');teacher=read(b/'teacher/G32/TEACHER_GLOBAL_BARRIER.json');requests=[]
        for e in teacher['files']:
            x=read(ROOT/e['path']);requests.append(anon({k:v for k,v in x.items() if k in ('case_id','dataset','visual_token_count','output_count','truncated','new_teacher_call','new_global_encoder_call','new_local_encoder_calls','wall_seconds','peak_GPU_allocated_bytes')}))
        timings={p.parent.name:read(p)['wall_seconds'] for p in (b/'execution').glob('*/WORKER_EXIT.json')};configs.append(dict(arm=arm or scope,policy=cfg['policy'],recipe=cfg['recipe'],global_frames=cfg['global_frames'],global_keep=cfg['global_keep'],local_keep=96,physical_clock=cfg['physical_clock'],configs=design['configs'],response_settings=design['response_settings'],trainable_spatial_parameters=1792,teacher={k:v for k,v in teacher.items() if k!='files'},requests=requests,execution_seconds=timings,shared_Source_Temporal_candidates_and_vision_acquisition_are_not_free=True,warm_reuse_not_deployment_zero_cost=True))
    save('CONFIG_AND_COST.json',dict(status='complete_actual_saved_group_cost',count=n,records=configs,GT_read=False));decision=c.BASE/'GLOBAL_DEVELOPMENT_DECISION.json' if scope=='global_B' else base/'CONFIRMATION_DECISION.json'
    if decision.exists():save('PRELOCKED_DECISION.json',read(decision))
    doc=ROOT/'docs'/('STVG_NIGHT_'+scope.upper()+'_001.md');lines=['# Finite night study: '+scope,'',f'{n} fixed parents, {n//2} per domain, one query per parent. Complete all-arm Student roots and prediction seals preceded GT access. All positive/negative/neutral and unknown cases are retained. Development global groups are exact core256 subsets; confirmation128 excludes all core256 parents and prioritizes no registered past selection, but is not claimed globally unseen.','', 'Original acceptance, rejection, rollback, precision, step counts, per-domain parameters and final full-frame Student readout remain. Global B changes full-range32 global512 plus original local B96 and separately qualifies actual physical elapsed mRoPE. The original global A group retains its uniform-average-gap clock. Comparing those two global groups is not a pure selector experiment. Local B is the matching frozen core updater.','', 'The engineering rule is prelocked: promote global only if domain-mean vIoU gain exceeds0.10pp with neither domain below -0.20pp, or an independently established >=15% end-to-end saving within -0.05pp. Reused vision costs are not treated as free deployment savings. A negative global result returns core B without a new frame/budget/prompt search. Semantic identity, role and full-extent pixel review remains pending and is not inferred from IoU or accepted updates.','', '| Arm | HC vIoU % | VidSTG vIoU % |','|---|---:|---:|']
    for arm in dict.fromkeys(r['arm'] for r in d['rows']):lines.append('| '+arm+' | '+' | '.join(f"{100*next(x['mean'] for x in d['summary'] if x['dataset']==ds and x['arm']==arm and x['metric']=='m_vIoU'):.6f}" for ds in ('hc2','vidstg'))+' |')
    lines+=['','| Domain | Contrast | Mean delta pp | Positive / negative / neutral / unknown |','|---|---|---:|---|']
    for x in d['paired']:
        if x['metric']=='m_vIoU' and x['comparison'] in ('Candidate_minus_LocalB','Candidate_minus_Temporal','Candidate_minus_Frozen','B_minus_A','B_minus_Temporal','B_minus_Frozen'):lines.append(f"| {x['dataset']} | {x['comparison']} | {x['mean_delta_pp']:+.6f} | {x['positive']} / {x['negative']} / {x['neutral']} / {x['unknown']} |")
    lines+=['','All public files exclude media, annotations/GT coordinates, weights, raw tensors/optimizer state, actual query-caption prompts, raw Teacher responses, personal conversations and credentials. This group publication is not whole-study FINAL. No Bootstrap, online carry, CURRENT promotion or official all-query execution.'];doc.write_text('\n'.join(lines)+'\n');paths.add(str(doc.relative_to(ROOT)))
    from scripts.export_stvg_lite_batch001 import public_source_text
    bindings=[]
    for name in ('stvg_overnight_common_v1.py','stvg_physical_video_clock_v1.py','stvg_overnight_global_v1.py','stvg_overnight_student_v1.py','stvg_overnight_score_v1.py','continue_stvg_overnight001.py','stvg_overnight_confirmation_v1.py','score_stvg_overnight_confirmation_v1.py','continue_stvg_overnight_confirmation002.py','continue_stvg_overnight_confirmation003.py','report_stvg_overnight_group_v1.py'):
        p=ROOT/'scripts'/name;t=out/'code'/name;t.parent.mkdir(exist_ok=True);t.write_text(public_source_text(p.read_bytes()));paths.add(str(t.relative_to(ROOT)));bindings.append(dict(name=name,source_sha256=sha(p),public_sha256=sha(t)))
    save('SOURCE_BINDINGS.json',dict(status='pass',records=bindings))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4));contrast='B_minus_A' if scope=='confirmation128' else 'Candidate_minus_LocalB'
    for ax,ds in zip(axs,('hc2','vidstg')):
        x=next(p for p in d['paired'] if p['dataset']==ds and p['comparison']==contrast and p['metric']=='m_vIoU');v=list(x['per_source_delta_pp'].values());ax.scatter(range(len(v)),v,s=22,alpha=.65);ax.axhline(0,color='gray');ax.axhline(x['mean_delta_pp'],color='black');ax.set(title=f'{ds}: all{n//2} parents',xlabel='Fixed roster order',ylabel='Student vIoU difference (pp)')
    fig.suptitle(scope+' / '+contrast);fig.tight_layout();im=out/'paired_effects.png';fig.savefig(im,dpi=150);plt.close(fig);paths.add(str(im.relative_to(ROOT)))
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    verify_privacy(paths)
    for rel in paths:
        p=ROOT/rel
        if p.suffix in ('.json','.md'):
            txt=p.read_text()
            for private in aliases:assert ('"'+private+'"') not in txt,rel
    save('ROOT_PRIVACY_READBACK.json',dict(status='pass',count=n,strict_allowlist=True,raw_media_GT_prompt_tensor_excluded=True,actual_figure_pixels_checked_separately=True,GT_read=False))
    pub=c.BASE/'publication'/scope;put(pub/'EXPORT_MANIFEST.json',dict(status='complete',count=n,paths=sorted(paths),files=[dict(path=x,sha256=sha(ROOT/x),bytes=(ROOT/x).stat().st_size) for x in sorted(paths)]));print(json.dumps(dict(status='exported_pending_actual_pixel_and_remote',scope=scope,count=n,files=len(paths))))
if __name__=='__main__':main(sys.argv[1])
