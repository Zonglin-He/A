"""Finite fixed-DINO shared text-memory adaptation, sealing, scoring and audit."""
from pathlib import Path
import os,sys,time,gc,collections,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_shared_text_query32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/shared_text_query32';PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1'
CODE=['vg_tta/stvg_shared_text_query32_v1.py','scripts/run_stvg_shared_text_query32_v1.py','scripts/continue_stvg_shared_text_query32_v1.py','protocols/stvg_shared_text_query32_v1.md']
VERIFIER=None

def verify(live=False,full=False):
    global VERIFIER
    p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
    from main_method.runtime import PinVerifier,resource_guard
    if VERIFIER is None:VERIFIER=PinVerifier(l['pins'])
    VERIFIER.verify(full=full);resource_guard()
    if live:assert time.time()<p['deadline_epoch']
    return p

def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)

def prepare():
    from vg_tta.stvg_shared_text_query32_v1 import properties,ARMS,token_spec
    from transformers import RobertaTokenizerFast
    old=read(PRE/'PLAN_LOCK.json');assert read(PRE/'FINAL_COMPLETION.json')['status']=='complete'
    from methods.decota_final_simplified_v1._tastvg_load import ROBERTA_CACHE
    snapshot=ROBERTA_CACHE/'snapshots'/(ROBERTA_CACHE/'refs/main').read_text().strip()
    tokenizer=RobertaTokenizerFast.from_pretrained(str(snapshot),local_files_only=True)
    specs={};files=CODE+['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']
    for r in old['records']:
        ds=r['dataset'];q=r['query_ordinal'];plan=ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json';sf=ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/subjects/{q:05}.json';row=read(plan)['rows'][q];row={**row,'parses':read(sf)['parses']}
        specs[r['alias']]=token_spec(tokenizer,row);files +=[str(sf.relative_to(ROOT)),str(plan.relative_to(ROOT))]
        files +=[str((PRE/k/(r['alias']+'.pt')).relative_to(ROOT))for k in ['source','E','predictions']]
    put(D/'PLAN_LOCK.json',dict(status='locked',records=old['records'],count=32,per_domain=16,arms=ARMS,steps=2,lr=.01,regularization=.001,
        source_commit='520484a53d4e1874e552662d470f61569fe21586',target_noun_tokens=specs,threshold_and_E_unchanged=True,
        no_new_DINO=True,no_dynamic_OPD=True,GT_after_full32_seal=True,cohort_development_exposed=True,independent_confirmation=False,
        GPU_seconds_cap=900,created_epoch=time.time(),deadline_epoch=time.time()+3600,read_final_step=2,no_parameter_search=True))
    pins={**read(PRE/'RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in set(files)}}
    put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));put(D/'CPU_PROPERTIES.json',properties());status('prepared',0,GT_read=False)
    print({k:dict(selected=v['selected_count'],tokens=v['token_count'],original_noun=v['original_unique_noun'])for k,v in specs.items()},flush=True)

def distribution(records,logits):
    import numpy as np
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
    g=probabilities(records,logits);ids,M=mass_table(g);assert abs(M-independent_cdf(g,ids)).max()<1e-12
    return ids,M,final_mbr(ids,M)['interval']

def measure():
    import torch,numpy as np
    from scripts.run_decota_paper_main_v1 import gpu,model_for
    from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture
    from vg_tta.stvg_shared_text_query32_v1 import input_pool
    from methods.decota_final_simplified_v1.tensors import state_hash,detached
    from vg_tta.stvg_shared_text_query32_v1 import fit,token_spec
    p=verify(True);lease=None
    while lease is None:
        verify(True)
        try:lease=gpu()
        except BlockingIOError:status('waiting_GPU_lease',0,GT_read=False);time.sleep(15)
    start=time.perf_counter();files=[];domain=None;model=None;cache=collections.OrderedDict()
    def check():
        verify(True);assert time.perf_counter()-start+3<p['GPU_seconds_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
    try:
        put(D/'GPU_RUNTIME.json',dict(status='active',pid=os.getpid(),start_epoch=time.time(),shared_lease=True))
        for ix,r in enumerate(p['records']):
            check();tick=time.perf_counter();a=r['alias'];ds=r['dataset'];status('shared_text_two_step_fit',ix,alias=a,GT_read=False)
            if domain!=ds:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;mh=state_hash(model.state_dict());cache.clear()
            s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));old=load(PRE/'predictions'/(a+'.pt'))
            row,records,base,native,expert,inp,cap_seconds=capture(model,ds,r['query_ordinal'],cache)
            assert torch.equal(native['boxes'].cpu(),s['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],s['logits']))
            spec=token_spec(model.text_encoder.tokenizer,row);assert spec==p['target_noun_tokens'][a]
            for v in base.views:
                h,w=v['info']['fea_map_size'];assert v['prefix'].shape[0]-2*h*w==spec['token_count']
            frames=input_pool(expert);assert len(frames)==len(e['estimate']['selected']);positions=[];targets=[]
            for f,k in zip(frames,e['estimate']['selected']):
                if k>=0:positions.append(f['position']);targets.append(f['boxes'][k])
            assert len(positions)==e['estimate']['support_frames']
            selected_positions=list(positions)
            if not spec['indices']:positions=[];targets=[]
            E=torch.as_tensor(old['outputs']['E_MBR'],dtype=torch.float64);assert torch.equal(E,torch.as_tensor(e['E']))
            save(D/'prefix'/(a+'.pt'),dict(views=detached(base.views,'cpu'),source=s['Source'],logits=s['logits'],records=records,spec=spec,GT_read=False))
            ft=time.perf_counter();f=fit(base,s['Source'],E,spec['indices'],positions,torch.as_tensor(np.asarray(targets).reshape(-1,4),dtype=torch.float32),p['lr'],p['regularization'],check);torch.cuda.synchronize();fit_seconds=time.perf_counter()-ft
            ids,M0,I0=distribution(records,f['path'][0]['logits']);ids1,M1,I1=distribution(records,f['path'][-1]['logits']);assert ids==ids1==old['frame_ids']and I0==old['interval']
            out=f['path'][-1]['boxes'].numpy();outputs=dict(E_MBR=E.numpy(),Temporal_only=E.numpy(),Spatial_only=out,Joint=out)
            intervals=dict(E_MBR=I0,Temporal_only=I1,Spatial_only=I0,Joint=I1)
            assert state_hash(model.state_dict())==mh
            files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=ids,planned_positions=old['planned_positions'],
                active=bool(positions),E_selected_positions=selected_positions,token_spec=spec,teacher_candidate_ids=[int(k)for k in e['estimate']['selected']],fit=f,outputs=outputs,intervals=intervals,
                distributions=dict(original=M0,adapted=M1),compute=dict(new_Source_captures=1,new_DINO_calls=0,Source_capture_seconds=cap_seconds,fit_seconds=fit_seconds,case_seconds=time.perf_counter()-tick),GT_read=False,pretrained_weights_frozen=True)))
            if ix==1:put(D/'PILOT2_TIMING.json',dict(count=2,elapsed_seconds=time.perf_counter()-start,rough32_seconds=16*(time.perf_counter()-start),not_a_guarantee=True))
            print(a,ix+1,round(time.perf_counter()-start,2),flush=True)
            del base,native,f,s,e,old,E,expert;gc.collect();torch.cuda.empty_cache()
        check();verify(full=True);put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_outputs_before_GT=True,GT_read=False,GPU_allocation_seconds=time.perf_counter()-start+3,engineering_initialization_seconds_upper_bound=3,final_worker_GPU_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        status('all32_sealed_before_GT',32,GT_read=False)
    finally:lease.close()

def metric(bb,iv,row,gt,span,ds,ids,pos):
    import numpy as np
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
    from scripts.run_stvg_dino_local_residual32_v1 import mean
    tube=DenseTube(bb,row,gt,span,clip=ds=='hc2');m=official(bb,row,gt,span,iv,ds);assert max(abs(m[k]-tube.score(iv)[k])for k in ['v','t','s'])<2e-10
    known=np.array([int(f)in gt for f in ids]);observed=np.isin(np.arange(len(ids)),pos);si=np.array([iou_gt(bb[j],gt[int(f)],row,ds)if known[j]else np.nan for j,f in enumerate(ids)])
    fps=row['input']['fps'];dsig=float(iv[0]-span[0]);deig=float(iv[1]-span[1]);duration=iv[1]-iv[0];truthduration=span[1]-span[0]
    return dict(**m,observed_actual_s=mean(si[known&observed]),unobserved_actual_s=mean(si[known&~observed]),
        start_signed_seconds=dsig/fps,end_signed_seconds=deig/fps,start_abs_seconds=abs(dsig)/fps,end_abs_seconds=abs(deig)/fps,
        duration_seconds=duration/fps,duration_ratio=duration/truthduration,duration_frames=duration)

def score():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,numpy as np
    torch.set_num_threads(4)
    from vg_tta.stvg_shared_text_query32_v1 import ARMS
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_stvg_dino_local_residual32_v1 import mean
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
    prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/dino_endpoint_confirm32/RESULTS_PUBLIC.json')['rows']}
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));row=plans[ds][q]
        values={arm:metric(b,o['intervals'][arm],row,gt[ds][0][q],gt[ds][1][q],ds,o['frame_ids'],o['planned_positions'])for arm,b in o['outputs'].items()}
        for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
            x,y=values['E_MBR'][k],prior[a]['values']['E_MBR'][k];assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12
        f=o['fit'];M0=o['distributions']['original'];M1=o['distributions']['adapted'];tiny=np.finfo(float).tiny
        rows.append(dict(alias=a,dataset=ds,active=o['active'],values=values,token_count=o['token_spec']['token_count'],selected_tokens=o['token_spec']['selected_count'],
            temporal=dict(changed_interval=o['intervals']['E_MBR']!=o['intervals']['Joint'],start_change_frames=o['intervals']['Joint'][0]-o['intervals']['E_MBR'][0],
                end_change_frames=o['intervals']['Joint'][1]-o['intervals']['E_MBR'][1],duration_change_frames=values['Joint']['duration_frames']-values['E_MBR']['duration_frames'],
                total_variation=float(abs(M1-M0).sum()/2),KL_adapted_to_original=float((M1*np.log(np.maximum(M1,tiny)/np.maximum(M0,tiny))).sum()),
                logit_max_change=max(float((x-y).abs().max())for x,y in zip(f['path'][0]['logits'],f['path'][-1]['logits']))),
            fit=dict(updates=sum(t['updated']for t in f['updates']),delta_norm=float(f['path'][-1]['delta'].norm()),loss=[v['spatial_objective']for v in f['path']],gradient_norm=[None if t['gradient']is None else float(t['gradient'].norm())for t in f['updates']],
                step_norm=[float(t['delta_step'].norm())for t in f['updates']],output_L2=float((f['path'][-1]['boxes']-f['path'][0]['boxes']).norm())),compute=o['compute']))
    endpoints=list(rows[0]['values']['E_MBR']);groups={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds]
        groups[ds]=dict(count=len(rr),active=sum(r['active']for r in rr),metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in endpoints}for a in ARMS},
            paired={a+'_minus_E_MBR':{k:paired([None if r['values'][a][k]is None or r['values']['E_MBR'][k]is None else 100*(r['values'][a][k]-r['values']['E_MBR'][k])for r in rr])for k in ['v','t','s','observed_actual_s','unobserved_actual_s']}for a in ARMS[1:]},
            boundary_changes=dict(changed=sum(r['temporal']['changed_interval']for r in rr),expanded=sum(r['temporal']['duration_change_frames']>0 for r in rr),contracted=sum(r['temporal']['duration_change_frames']<0 for r in rr),
                same_duration=sum(r['temporal']['duration_change_frames']==0 for r in rr)),new_Source_captures=16,new_DINO_calls=0,fit_seconds=sum(r['compute']['fit_seconds']for r in rr))
    put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,rows=rows,groups=groups,all32_sealed_before_GT=True,pretrained_weights_frozen=True,shared_text_offset256=True,
        fixed_DINO_E_associations=True,final_step=2,historical_development_exposure=True,independent_confirmation=False,no_parameter_search=True,compute={k:bar[k]for k in ['GPU_allocation_seconds','peak_allocated_bytes']}))
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_after_full_seal=True));print({ds:{a:(g['metrics'][a]['v'],g['metrics'][a]['t'])for a in ARMS}for ds,g in groups.items()},flush=True)

def audit():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,numpy as np
    torch.set_num_threads(4)
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.stvg_shared_text_query32_v1 import objective,ARMS
    p=verify();result=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in result['rows']};checks=collections.Counter()
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));f=o['fit']
        assert torch.equal(f['path'][0]['native_boxes'],s['Source'])and torch.equal(f['path'][0]['boxes'],torch.as_tensor(e['E']))and torch.count_nonzero(f['path'][0]['delta'])==0;checks['zero_Source_E_exact']+=1
        assert o['E_selected_positions']==[v['position']for v,k in zip(__import__('scripts.run_stvg_joint_robust_correction32_v1',fromlist=['input_pool']).input_pool(s['expert']),e['estimate']['selected'])if k>=0];checks['fixed_E_association']+=1
        assert np.array_equal(o['outputs']['E_MBR'],o['outputs']['Temporal_only'])and np.array_equal(o['outputs']['Spatial_only'],o['outputs']['Joint']);checks['matched_four_readouts']+=1
        for v in f['path']:
            got=float(objective(v['boxes'],f['positions'],f['teacher'].double(),v['delta'],p['regularization']))
            assert abs(got-v['spatial_objective'])<2e-7;checks['saved_actual_objective']+=1
            ids,M,iv=distribution(s['records'],v['logits'])
            if v['step']in [0,2]:
                key='original'if v['step']==0 else'adapted';arm='E_MBR'if v['step']==0 else'Joint'
                assert np.array_equal(M,o['distributions'][key])and iv==o['intervals'][arm];checks['full_physical_distribution_MBR']+=1
        for j,t in enumerate(f['updates']):
            assert torch.equal(f['path'][j+1]['delta']-f['path'][j]['delta'],t['delta_step']);checks['actual_delta_step']+=1
            if not t['updated']:assert torch.count_nonzero(t['delta_step'])==0;checks['no_support_noop']+=1
        # Independent physical-frame official evaluation from interpolation.
        row=plans[ds][q];gg=gt[ds][0][q];span=gt[ds][1][q];fid=np.array(sorted(gg));truth=np.array([gg[int(i)]for i in fid]);ids=np.array(o['frame_ids'])
        for arm,b in o['outputs'].items():
            iv=o['intervals'][arm];xy=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            pred=np.column_stack([np.interp(fid,ids,xy[:,j])for j in range(4)]);inter=np.maximum(0,np.minimum(pred[:,2:],truth[:,2:])-np.maximum(pred[:,:2],truth[:,:2])).prod(1)
            union=np.maximum(0,pred[:,2:]-pred[:,:2]).prod(1)+np.maximum(0,truth[:,2:]-truth[:,:2]).prod(1)-inter;iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);iou[(fid<ids[0])|(fid>ids[-1])]=0
            overlap=max(0,min(iv[1],span[1])-max(iv[0],span[0]));den=max(max(iv[1],span[1])-min(iv[0],span[0]),1);inside=(fid>=max(iv[0],span[0]))&(fid<min(iv[1],span[1]))
            tests=dict(v=float(iou[inside].sum()/den),t=overlap/den,s=float(iou.mean()))
            for key,mask in [('observed_actual_s',np.isin(fid,ids[o['planned_positions']])),('unobserved_actual_s',np.isin(fid,ids)&~np.isin(fid,ids[o['planned_positions']]))]:tests[key]=float(iou[mask].mean())if mask.any()else None
            for k,v in tests.items():
                got=rows[a]['values'][arm][k];assert got is None if v is None else abs(got-v)<2e-10;checks['independent_metric_scalars']+=1
    verify(full=True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),four_readouts_same_fit=True,only_shared_text256=True,all32_before_GT=True,MBR_full_distribution=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),flush=True)

def report():
    import numpy as np
    from vg_tta.stvg_shared_text_query32_v1 import ARMS
    z=read(P/'RESULTS_PUBLIC.json');cfg=read(D/'PLAN_LOCK.json')
    publiccfg={k:v for k,v in cfg.items()if k not in ['records','target_noun_tokens','created_epoch','deadline_epoch']};publiccfg['token_policy']='prepended subject and unique original referent noun, fast-tokenizer offsets; only shared text-memory insertion';put(P/'CONFIGURATION.json',publiccfg)
    provenance={k:v for k,v in read(PRE/'RUNTIME_LOCK.json')['pins'].items()if 'model' in k and ('safetensors'in k or k.endswith('.pth'))};put(P/'MODEL_PROVENANCE.json',dict(pins=provenance,all_pretrained_weights_unchanged=True,new_DINO_calls=0,new_Source_captures=32))
    lines=['# Shared target-text memory adaptation32','', 'Completed mechanism development experiment; not independent confirmation, not dynamic OPD, no Current promotion.','',
        'One disposable 256D offset, two fixed Adam steps/query, fixed E-selected DINO references. All pretrained parameters frozen. Native routing/initial decoder queries fixed. Shared text cross-attention memory changes in both decoders; action/relation tokens receive no direct insertion.', '',
        '| Domain | Output | vIoU % | tIoU % | sIoU % | Unobserved actual sIoU % |','|---|---|---:|---:|---:|---:|']
    for ds,g in z['groups'].items():
        for a in ARMS:
            m=g['metrics'][a];lines.append(f"| {ds} | {a} | {100*m['v']:.6f} | {100*m['t']:.6f} | {100*m['s']:.6f} | {100*m['unobserved_actual_s']:.6f} |")
    lines+=['','Primary comparison: Temporal_only minus E_MBR; boxes identical. Spatial_only retains original time; Joint changes both. All results from the same fit.']
    for ds,g in z['groups'].items():
        lines+=['',f"## {ds}",f"Active {g['active']}/16; boundary changes {g['boundary_changes']}; two-step fitting {g['fit_seconds']:.3f}s."]
        for a in ARMS[1:]:lines.append(a+': '+str(g['paired'][a+'_minus_E_MBR']))
        lines.append('Temporal boundary metrics, seconds and ratio (all16): '+str({a:{k:v for k,v in g['metrics'][a].items()if k not in ['v','t','s','observed_actual_s','unobserved_actual_s']}for a in ['E_MBR','Temporal_only']}))
    lines+=['','All32 outputs sealed before GT. GT is only final offline evaluation. Unknown frame labels stay null, not failures or successes. Query=parent. Signed rows, 10000-replicate paired bootstrap seed20261018 and loss tails are retained in RESULTS_PUBLIC.json. Boundary expansion alone is not proof of semantic improvement.',
        '',f"GPU allocation {z['compute']['GPU_allocation_seconds']:.3f}s, peak allocated {z['compute']['peak_allocated_bytes']/2**30:.3f}GiB; new Source captures32, DINO calls0. Cached Source/E and original expert generation costs excluded; not first-online latency.",
        '', 'This tests spatial-to-temporal transfer through shared memory. It does not replicate DSE-VTG cosine pseudo labels, prove identity disambiguation, remove native internal actionness routing or establish OPD gains. No automatic OPD, wider cohort or parameter search follows this run. Conceptual source: [DSE-VTG](https://arxiv.org/html/2609.08850v1).']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for panel,ds in zip(ax,['hc2','vidstg']):
        rr=[r for r in z['rows']if r['dataset']==ds];panel.axhline(0,color='k',lw=.7)
        for a in ARMS[1:]:panel.plot(range(1,17),[100*(r['values'][a]['v']-r['values']['E_MBR']['v'])for r in rr],marker='o',label=a)
        panel.set(title=ds,xlabel='Fixed query alias order',ylabel='vIoU difference vs E (pp)');panel.legend()
    fig.tight_layout();fig.savefig(P/'COMPARISON.svg');fig.savefig(D/'COMPARISON.png',dpi=150);plt.close(fig)
    cost=dict(total_Source_captures_completed=32,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,**z['compute']);put(P/'COST_BREAKDOWN.json',cost)
    files=CODE+[str(f.relative_to(ROOT))for f in P.iterdir()if f.is_file()];put(D/'PUBLIC_FILES_FINAL.json',dict(files=sorted(files)));put(D/'REPORT_COMPLETION.json',dict(status='complete',count=32))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score','audit','report']);args=ap.parse_args()
    try:globals()[args.stage]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=args.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
