"""One frozen, source-only sigma-start change; no optimizers or new models."""
from pathlib import Path
import argparse,collections,json,math,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_posterior_repair_v1 import BASE as REPAIR,OLD,SEED,read,put,load,save,sha,condition,metric
PARENT=ROOT/'artifacts/stvg_joint_native_path_diagnosis_v1'
BASE=ROOT/'artifacts/stvg_joint_sigma213_source24_v1'
OUT=ROOT/'results/stvg_joint_posterior/2026-10-18/sigma213_source24'

def prepare():
    import torch
    assert not(BASE/'PLAN_LOCK.json').exists()
    rr=read(PARENT/'PLAN_LOCK.json')['source'];assert len(rr)==24
    pins={}
    for f in ['scripts/run_stvg_joint_sigma213_source24_v1.py','vg_tta/stvg_joint_sigma_start_v1.py','vg_tta/stvg_joint_posterior_repair_v1.py','vg_tta/stvg_joint_posterior_v1.py','scripts/run_stvg_joint_posterior_repair_v1.py','protocols/stvg_joint_sigma213_source24_v1.md','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']:
        pins[f]=sha(ROOT/f)
    for ds in ['hc2','vidstg']:
        for folder in ['models','statistics']:
            f=REPAIR/folder/(ds+'.pt');load(f);pins[str(f.relative_to(ROOT))]=sha(f)
    for r in rr:
        for f in [OLD/'features/source'/(r['alias']+'.pt'),OLD/'source_supervision'/(r['alias']+'.pt'),REPAIR/'selected_predictions'/(r['alias']+'.pt'),PARENT/'states'/(r['alias']+'.pt')]:
            pins[str(f.relative_to(ROOT))]=sha(f)
    put(BASE/'PLAN_LOCK.json',dict(status='locked_before_new_inference',source=rr,new_models=0,new_training=0,new_Student_Teacher_calls=0,new_target_inference=0,sigma_start=.213,sigma_end=.03,levels=16,samples=4,seed=SEED,checkpoints={'hc2':30,'vidstg':20},observation_guidance=False,primary='sigma213 minus raw Native Source',secondary='sigma213 minus sealed sigma2',codec_control='Native encoded/decoded32 knots',source_validation_not_established_unseen_to_pretrained_Student=True,source_count=24,previous_GPU_reserved_seconds=read(PARENT/'GPU_COMPLETION.json')['cumulative_with_previous_repair_seconds'],new_GPU_limit_seconds=300,cumulative_GPU_limit_seconds=14400,followup='none; no automatic observation/target/OPD phase'))
    put(BASE/'RUNTIME_LOCK.json',dict(pins=pins))
    ss=torch.logspace(math.log10(.213),math.log10(.03),16);assert abs(float(ss[0])-.213)<1e-7 and abs(float(ss[-1])-.03)<1e-7 and bool((ss[:-1]>ss[1:]).all())
    x=torch.randn(4,33,4);d=torch.randn_like(x);s=ss[0];sn=ss[1]
    assert torch.allclose(d+(sn/s)*(x-d),x+(sn-s)*(x-d)/s,atol=5e-7,rtol=1e-6)
    assert torch.equal(x+(sn/s)*(x-x),x)
    put(BASE/'CPU_CONTRACT_COMPLETION.json',dict(status='pass',synchronized_initial_sigma_and_schedule=True,levels=16,end_sigma=.03,Euler_equivalence=True,identity_preserves_input=True,new_training=0,GPU_initialized=False))
    assert not torch.cuda.is_initialized()

def infer():
    import torch
    from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior
    from vg_tta.stvg_joint_sigma_start_v1 import frozen_rollout
    from scripts.run_decota_paper_main_v1 import gpu
    from main_method.runtime import resource_guard
    p=read(BASE/'PLAN_LOCK.json')
    for f,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f
    assert not(BASE/'INFERENCE_STARTED.json').exists();put(BASE/'INFERENCE_STARTED.json',dict(status='started',pid=os.getpid(),new_training=0))
    lease=gpu();started=time.monotonic();files=[];qualification=[]
    try:
        torch.cuda.set_per_process_memory_fraction(.5)
        def budget():
            torch.cuda.synchronize();elapsed=time.monotonic()-started;resource_guard();assert elapsed<300 and elapsed+p['previous_GPU_reserved_seconds']<14400
        for ds in ['hc2','vidstg']:
            st=load(REPAIR/'statistics'/(ds+'.pt'));net=PreconditionedJointPrior(st['mean'],st['scale']).cuda().eval().requires_grad_(False);net.load_state_dict(load(REPAIR/'models'/(ds+'.pt'))['state'])
            rr=[r for r in p['source']if r['dataset']==ds]
            for i,r in enumerate(rr):
                budget();z=load(OLD/'features/source'/(r['alias']+'.pt'));c=condition(z);cc={k:v.expand((4,)+v.shape[1:])for k,v in c.items()}
                old=load(REPAIR/'selected_predictions'/(r['alias']+'.pt'));saved_epsilon=load(PARENT/'states'/(r['alias']+'.pt'))['epsilon']
                if i==0:
                    qual=frozen_rollout(net,cc,SEED+1000+i,sigma_start=2.)
                    error=float((qual['final']-old['samples']).abs().max());assert error<2e-5 and torch.equal(qual['epsilon'],saved_epsilon)
                    qualification.append(dict(dataset=ds,alias=r['alias'],sigma2_terminal_max_error=error,epsilon_exact=True))
                new=frozen_rollout(net,cc,SEED+1000+i,sigma_start=.213)
                assert torch.equal(new['epsilon'],saved_epsilon)
                expected=cc['native'].cpu()+new['steps'][0]['sigma']*saved_epsilon;expected[:,0,2:]=0;assert torch.equal(expected,new['initial'])
                files.append(save(BASE/'predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=ds,path=new,new_training=0,source_GT_input=False,old_sigma2_reference_sha256=sha(REPAIR/'selected_predictions'/(r['alias']+'.pt')),epsilon_exact=True)))
                put(BASE/'STATUS.json',dict(status='running',stage='sigma213_frozen_inference',done=len(files),total=24,pid=os.getpid()),True)
            del net;torch.cuda.empty_cache()
        put(BASE/'GPU_QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,rows=qualification,only_original_sigma2_interface_replay=True,new_training=0))
        put(BASE/'PREDICTION_BARRIER.json',dict(status='complete',count=24,files=files,predictions_sealed_before_source_scoring=True,new_Student_Teacher_calls=0,new_target_predictions=0))
    finally:
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        put(BASE/'GPU_COMPLETION.json',dict(status='complete',new_reserved_seconds=elapsed,cumulative_reserved_seconds=elapsed+p['previous_GPU_reserved_seconds'],limit_seconds=14400,includes_two_original_sigma2_interface_replays=True,new_training=0));lease.close()

def audit_export():
    import torch,numpy as np
    from scipy.stats import t
    from scripts.run_stvg_joint_posterior_v1 import decode_state
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
    torch.set_num_threads(4);assert not torch.cuda.is_initialized();assert read(BASE/'PREDICTION_BARRIER.json')['count']==24
    checks=0;rows=[];p=read(BASE/'PLAN_LOCK.json')
    for f,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f;checks+=1
    old_ref=read(REPAIR/'SOURCE_VALIDATION_POSTERIOR.json')if(REPAIR/'SOURCE_VALIDATION_POSTERIOR.json').exists()else None
    for r in p['source']:
        z=load(OLD/'features/source'/(r['alias']+'.pt'));y=load(OLD/'source_supervision'/(r['alias']+'.pt'));new=load(BASE/'predictions'/(r['alias']+'.pt'));old=load(REPAIR/'selected_predictions'/(r['alias']+'.pt'));c=condition(z,'cpu')
        def score_state(x):
            nonlocal checks
            d=decode_state(x,z);return score_decoded(d)
        def score_decoded(d):
            nonlocal checks
            row=dict(frame_ids=z['frame_ids'],input=z['input']);a=official(d['boxes'],row,y['dense_source_GT'],y['span'],d['interval'],r['dataset']);b=DenseTube(d['boxes'],row,y['dense_source_GT'],y['span'],clip=r['dataset']=='hc2').score(d['interval'])
            for k in ['v','t','s']:assert abs(a[k]-b[k])<2e-9;checks+=1
            a.update(R03=float(a['v']>=.3),R05=float(a['v']>=.5),duration_fraction=(d['interval'][1]-d['interval'][0])/(z['frame_ids'][-1]+1-z['frame_ids'][0]))
            return a
        raw=score_decoded(dict(boxes=z['boxes'][0].numpy(),interval=z['native_interval']));codec=score_state(c['native'][0]);old_samples=[score_state(x)for x in old['samples']];samples=[score_state(x)for x in new['path']['final']]
        origin=next(a for a in read(REPAIR/'qualification'/(r['dataset']+'.json'))['rows']if a['alias']==r['alias'])
        for j,a in enumerate(old_samples):
            for k in ['v','t','s']:assert abs(a[k]-origin['samples'][j][k])<2e-6;checks+=1
        for k in ['v','t','s']:assert abs(raw[k]-origin['Source'][k])<1e-12;checks+=1
        rows.append(dict(alias=r['alias'],dataset=r['dataset'],Native=raw,Native_codec=codec,sigma2=old_samples,sigma213=samples))
    keys=['v','t','s','R03','R05','duration_fraction'];means={};paired={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];assert len(rr)==12;means[ds]={};paired[ds]={}
        for support in ['sample0','mean_per_sample']:
            values={}
            for arm in ['Native','Native_codec','sigma2','sigma213']:
                values[arm]={k:[r[arm][k]if arm in ['Native','Native_codec']else(r[arm][0][k]if support=='sample0'else float(np.mean([a[k]for a in r[arm]])))for r in rr]for k in keys}
            means[ds][support]={arm:{k:float(np.mean(v))for k,v in aa.items()}for arm,aa in values.items()}
            pp={}
            for arm in ['Native','Native_codec','sigma2']:
                dd={}
                for k in keys:
                    d=np.array(values['sigma213'][k])-np.array(values[arm][k]);rad=t.ppf(.975,11)*d.std(ddof=1)/np.sqrt(12)
                    dd[k]=dict(mean=float(d.mean()),CI95=[float(d.mean()-rad),float(d.mean()+rad)],values=d.tolist(),positive=int((d>1e-8).sum()),negative=int((d< -1e-8).sum()),neutral=int((np.abs(d)<=1e-8).sum()))
                    assert dd[k]['positive']+dd[k]['negative']+dd[k]['neutral']==12;assert abs(dd[k]['mean']-(means[ds][support]['sigma213'][k]-means[ds][support][arm][k]))<1e-12;checks+=2
                pp['sigma213 - '+arm]=dd
            paired[ds][support]=pp
    result=dict(status='complete',count=24,new_models=0,new_training_steps=0,source_only=True,no_observations=True,means=means,paired=paired,rows=rows,source_validation_exposed=True,no_automatic_next_stage=True,selection='none; exact user-requested sigma.213, frozen checkpoints and sample0; all path metrics reported')
    put(BASE/'RESULTS.json',result);OUT.mkdir(exist_ok=True,parents=True)
    # Per-case GT-event coordinates/durations are not public. Actual output
    # states/intervals remain in the private prediction seal, not in this JSON.
    def redact(v):
        if isinstance(v,dict):return {k:redact(a)for k,a in v.items()if k!='duration_fraction'}
        if isinstance(v,list):return [redact(a)for a in v]
        return v
    public=dict(result);public['rows']=redact(rows);put(OUT/'RESULTS.json',public)
    root=dict(status='pass',count=24,independent_checks=checks,official_DenseTube_agreement=True,all_original_reference_metrics_reproduced=True,all_weights_conditions_registries_and_old_predictions_unchanged=True,epsilon_exact=True,all24_sealed_before_source_GT_score=True,new_training=0)
    put(BASE/'ROOT_COMPLETION.json',root);put(OUT/'ROOT_COMPLETION.json',root)
    for name in ['CPU_CONTRACT_COMPLETION.json','GPU_QUALIFICATION_COMPLETION.json','GPU_COMPLETION.json']:
        put(OUT/name,read(BASE/name))
    put(OUT/'PROTOCOL.json',{k:v for k,v in p.items()if k!='source'})
    lines=['# Frozen source24: synchronized sigma .213 initialization and schedule','','Fixed repaired HC epoch30 / Vid epoch20, original source-val12 each, original4 epsilon paths. No training, observation guidance, new Teacher/Student or target runs. Only initial sigma and its16-level log schedule start changed from2 to exactly.213, end.03 and Euler rule unchanged. Old Source/sigma2 references reused and independently verified.','','Source origins are official train and were held out only for the auxiliary prior; not an independent evaluation unseen to pretrained Student. This is a fixed source-development inference comparison, not target confirmation or method promotion.','','## Fixed first-path results','','Percent; entire12-source denominator/domain. R03/R05 are query-level vIoU thresholds.','','| Method | HC vIoU | HC tIoU | HC sIoU | HC R03 | HC R05 | Vid vIoU | Vid tIoU | Vid sIoU | Vid R03 | Vid R05 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['Native','Native_codec','sigma2','sigma213']:
        vals=[means[ds]['sample0'][arm][k]*100 for ds in ['hc2','vidstg']for k in ['v','t','s','R03','R05']];lines.append('| '+arm+' | '+' | '.join(f'{x:.6f}'for x in vals)+' |')
    lines+=['','## Paired first-path vIoU differences','','Mean and source-query paired95% interval in percentage points. Confidence intervals do not define an automatic next stage.','','| Comparison | HC mean / CI95 / positive-negative-neutral | Vid mean / CI95 / positive-negative-neutral |','|---|---|---|']
    for arm in ['Native','Native_codec','sigma2']:
        vals=[]
        for ds in ['hc2','vidstg']:
            a=paired[ds]['sample0']['sigma213 - '+arm]['v'];vals.append(f"{a['mean']*100:+.6f} / [{a['CI95'][0]*100:+.6f},{a['CI95'][1]*100:+.6f}] / {a['positive']}-{a['negative']}-{a['neutral']}")
        lines.append('| sigma213 - '+arm+' | '+' | '.join(vals)+' |')
    lines+=['','All4 per-path metrics and mean-of-path metrics are in RESULTS.json. No coordinate averaging, GT path selection, source removal or checkpoint reselection. Native_codec is the same32-knot readout control; raw Native is primary. Noisy initialization is correctly synchronized to the same new .213 starting sigma used by the model and update schedule. It is not the earlier no-initial-noise/sigma2 counterfactual.','','## Scope and cost','','This is still a finite Native-warm-start approximation, not an exact conditional posterior sampler. No new claims about identity reliability, dense reward, target adaptation or pretrained-Student-unseen generalization. No current-method promotion or automatic observation/target/OPD follow-up. Cached feature inference GPU reservation: '+f"{read(BASE/'GPU_COMPLETION.json')['new_reserved_seconds']:.6f}"+' seconds including2 fixed sigma2 interface replays; cumulative '+f"{read(BASE/'GPU_COMPLETION.json')['cumulative_reserved_seconds']:.6f}"+' seconds under the original14400s cap. CPU audit/publication and original upstream video/feature/expert work are separate; not online E2E latency.','',f'Independent scalar/reference/codec/mean/paired checks: {checks}. All old model/reference seals preserved. Private source-pixel inspection is engineering only, not semantic identity labeling.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('ROOT',checks)
    for ds in ['hc2','vidstg']:print(ds,json.dumps(means[ds]))

def render():
    from PIL import Image,ImageDraw
    import numpy as np
    from scripts.run_decota_paper_main_v1 import frames_for
    from scripts.run_stvg_joint_posterior_v1 import decode_state
    files=[]
    for ds in ['hc2','vidstg']:
        r=next(r for r in read(BASE/'PLAN_LOCK.json')['source']if r['dataset']==ds);z=load(OLD/'features/source'/(r['alias']+'.pt'));new=load(BASE/'predictions'/(r['alias']+'.pt'));old=load(REPAIR/'selected_predictions'/(r['alias']+'.pt'))
        frames,ids,_=frames_for(ds,dict(input=r['input'],frame_ids=r['frame_ids']),collections.OrderedDict());assert ids==z['frame_ids']
        arms=[('Native',dict(boxes=z['boxes'][0].numpy(),interval=z['native_interval'])),('sigma2',decode_state(old['samples'][0],z)),('sigma213',decode_state(new['path']['final'][0],z))]
        pos=np.linspace(0,len(ids)-1,4).round().astype(int);canvas=Image.new('RGB',(1120,576),(22,22,22));draw=ImageDraw.Draw(canvas)
        for j,(name,a)in enumerate(arms):
            for k,idx in enumerate(pos):
                im=Image.fromarray(frames[idx]);im.thumbnail((280,155));ox,oy=k*280,j*192+34;canvas.paste(im,(ox,oy));cx,cy,w,h=a['boxes'][idx]
                draw.rectangle([ox+(cx-w/2)*im.width,oy+(cy-h/2)*im.height,ox+(cx+w/2)*im.width,oy+(cy+h/2)*im.height],outline=(255,210,40),width=2)
                draw.text((ox+4,j*192+4),f"{r['alias']} {name} slot{k+1} event={a['interval'][0]<=ids[idx]<a['interval'][1]}",fill='white')
        f=BASE/'root_view'/(ds+'.png');f.parent.mkdir(exist_ok=True);canvas.save(f);files.append(str(f))
    put(BASE/'ROOT_VIEW_PLAN.json',dict(status='rendered_pending_actual_view',files=files,source_only=True,GT_overlays=False,identity_labels=False));print('\n'.join(files))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','infer','audit_export','render']);args=a.parse_args();globals()[args.command]()
