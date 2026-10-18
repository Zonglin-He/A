"""Disjoint-parent retest: frozen Source/E, then center and24 aligned queries."""
from pathlib import Path
import os,sys,time,hashlib,collections,gc,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
D=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_endpoint_confirm32'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
EXCLUSIONS=['artifacts/stvg_dino_supported_source256_v1/PLAN_LOCK.json','artifacts/stvg_dino_rollback32_v1/PLAN_LOCK.json',
 'artifacts/stvg_query_e_opd64_v1/PLAN_LOCK.json','artifacts/stvg_soft_correspondence32_v1/PLAN_LOCK.json',
 'artifacts/stvg_spatial_readout32_v1/PLAN_LOCK.json','artifacts/stvg_spatial_controls256_v1/ROSTER_LOCK.json',
 'artifacts/stvg_external_opd20_online64_v1/PLAN_LOCK.json','artifacts/stvg_joint_robust_correction32_v1/PLAN_LOCK.json']
CODE=['vg_tta/stvg_dino_endpoint_confirm32_v1.py','scripts/run_stvg_dino_endpoint_confirm32_v1.py',
 'scripts/score_stvg_dino_endpoint_confirm32_v1.py','scripts/audit_stvg_dino_endpoint_confirm32_v1.py',
 'scripts/report_stvg_dino_endpoint_confirm32_v1.py','scripts/continue_stvg_dino_endpoint_confirm32_v1.py',
 'protocols/stvg_dino_endpoint_confirm32_v1.md']
_verifier=None

def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),timestamp=time.time(),**kw),True)
def used_gpu():
    return sum(read(D/f)['GPU_allocation_seconds']for f in ['SOURCE_BARRIER_hc2.json','SOURCE_BARRIER_vidstg.json']if (D/f).exists())
def verify(live=False,full=False):
    global _verifier
    p=read(D/'PLAN_LOCK.json');lock=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==lock['plan_sha256']
    if live:assert time.time()<p['deadline_epoch'],'original2400s deadline reached'
    from main_method.runtime import PinVerifier,resource_guard
    if _verifier is None:_verifier=PinVerifier(lock['pins'])
    _verifier.verify(full=full);resource_guard();return p

def prepare():
    from vg_tta.stvg_dino_endpoint_confirm32_v1 import ARMS,properties
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')for ds in ['hc2','vidstg']}
    excluded={ds:set()for ds in plans};audit=[]
    for f in EXCLUSIONS:
        doc=read(ROOT/f);counts={ds:0 for ds in plans}
        for r in doc.get('records',doc.get('target',[])):
            ds='hc2'if r.get('dataset')in ['hc2','hcstvg2']else r.get('dataset')
            if ds not in plans:continue
            parent=plans[ds]['rows'][r['query_ordinal']]['input']['source'];excluded[ds].add(parent);counts[ds]+=1
        audit.append(dict(path=f,sha256=sha(ROOT/f),records=counts))
    design=read(OLD/'DESIGN_LOCK.json')
    for ds in plans:excluded[ds].update(plans[ds]['rows'][q]['input']['source']for q in design['datasets'][ds]['development_query_ordinals'])
    records=[];counts={}
    def h(s):return hashlib.sha256(('endpoint_confirm32_v1|20261018|'+s).encode()).hexdigest()
    for ds,plan in plans.items():
        byparent=collections.defaultdict(list)
        for q,row in enumerate(plan['rows']):byparent[row['input']['source']].append(q)
        eligible=set(byparent)-excluded[ds];assert len(eligible)>=16
        selected=sorted(eligible,key=lambda parent:h(ds+'|parent|'+parent))[:16]
        for i,parent in enumerate(selected):
            q=min(byparent[parent],key=lambda q:h(ds+'|query|'+parent+'|'+str(q)))
            records.append(dict(alias=('NH'if ds=='hc2'else'NV')+f'{i+1:03}',dataset=ds,parent=parent,query_ordinal=q,
                prior_input=str((OLD/'inputs'/('vidstg_to_hc2'if ds=='hc2'else'hcstvg2_to_vidstg')/'clean'/f'{q:05}.pt').relative_to(ROOT))))
        counts[ds]=dict(total_parents=len(byparent),excluded_parents=len(excluded[ds]),eligible_parents=len(eligible),selected_parents=16,
            parent_overlap_with_exclusions=0,PLAN_sha256=sha(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json'))
    # Membership is immutable before cache existence, tensor or quality inventory.
    put(D/'ROSTER_LOCK.json',dict(status='locked_before_cache_inventory',count=32,records=records,domains=counts,
        selection='fixed salted SHA256 parent then query order; seed20261018',exclusions=audit,
        prediction_quality_selection=False,GT_read=False,historical_full_exposure=True,globally_unseen=False))
    inputs={};checks=[]
    barriers={ds:read(OLD/'stages'/('P1_'+ds)/'PREDICTION_BARRIER.json')for ds in plans}
    for r in records:
        f=ROOT/r['prior_input'];assert f.is_file(),('locked input missing; do not replace sample',r['alias'])
        digest=sha(f);assert digest==barriers[r['dataset']]['inputs'][str(f.relative_to(OLD))];inputs[r['prior_input']]=digest
        q=r['query_ordinal'];sub=ROOT/f'artifacts/decota_paper_experiments_v1/{r["dataset"]}/subjects/{q:05}.json';assert sub.is_file();inputs[str(sub.relative_to(ROOT))]=sha(sub)
        checks.append(dict(alias=r['alias'],original_input_hash_exact=True,subject_present=True))
    cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')
    scales={ds:cal['source_domains']['vidstg'if ds=='hc2'else'hc2']['scale']for ds in plans}
    p=dict(status='locked',records=records,count=32,per_domain=16,seed=20261018,arms=ARMS,scale=scales,threshold=.35,
        denominator=24,beta=1/16,frozen_candidate_sha256=sha(ROOT/'results/stvg_error_state/2026-10-18/dino_endpoint32/FROZEN_CANDIDATE.json'),
        roster_sha256=sha(D/'ROSTER_LOCK.json'),GT_after_all32_outputs=True,historical_full_exposure=True,
        no_dynamic_OPD=True,no_parameter_search=True,no_network_updates=True,created_epoch=time.time(),deadline_epoch=time.time()+2400,
        GPU_seconds_cap=900,Source_capture_max=32,anchor_encoding_max=128)
    put(D/'PLAN_LOCK.json',p)
    deps=['vg_tta/stvg_dino_mask_align32_v1.py','vg_tta/stvg_dino_content_match32_v1.py','vg_tta/stvg_dino_conditional_opd32_v1.py',
        'vg_tta/stvg_error_state_correction_v1.py','vg_tta/stvg_dino_supported_correction32_v1.py','vg_tta/stvg_joint_robust_correction32_v1.py',
        'scripts/run_stvg_joint_robust_correction32_v1.py','scripts/run_stvg_dino_supported_expanded256_recovery002.py',
        'scripts/run_decota_paper_main_v1.py','scripts/run_spatial_regression_alignment_v1.py',
        'scripts/stvg_temporal_marginal_decode_util256_v1.py','vg_tta/stvg_posterior_decision_tree_v1.py',
        'vg_tta/stvg_opd_frozen_mask_cache012.py','main_method/runtime.py','methods/CURRENT_METHOD.json',
        'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json',
        '.conda/tubedetr/lib/python3.11/site-packages/transformers/models/grounding_dino/modeling_grounding_dino.py']
    deps += [str(f.relative_to(ROOT))for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py')]
    from methods.decota_final_simplified_v1.config import MethodConfig,EXPERT_SNAPSHOT,EXPERT_SHA256
    weightfiles={}
    for direction in ['vid_to_hc1','hc2_to_vid']:
        cfg=MethodConfig.for_direction(direction);assert sha(ROOT/cfg.checkpoint)==cfg.checkpoint_sha256;weightfiles[cfg.checkpoint]=cfg.checkpoint_sha256
    assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors')==EXPERT_SHA256;weightfiles[str(Path(EXPERT_SNAPSHOT)/'model.safetensors')]=EXPERT_SHA256
    pins={f:sha(ROOT/f)for f in CODE+deps+EXCLUSIONS};pins.update(inputs);pins.update(weightfiles)
    pins[str((OLD/'DESIGN_LOCK.json').relative_to(ROOT))]=sha(OLD/'DESIGN_LOCK.json')
    put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins))
    put(D/'INPUT_READBACK.json',dict(status='pass',count=32,membership_before_inventory=True,checks=checks,GT_read=False,new_Source_captures_required=32))
    put(D/'CPU_PROPERTIES.json',properties());status('prepared',0,GT_read=False)
    print(counts,flush=True)

def lease_gpu():
    from scripts.run_decota_paper_main_v1 import gpu
    while True:
        verify(True)
        try:return gpu()
        except BlockingIOError:
            status('waiting_shared_GPU_lease',0,GT_read=False);time.sleep(15)

def capture(ds):
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for
    from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture as fresh
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    p=verify(True);lease=lease_gpu();start=time.perf_counter();prior=used_gpu();files=[];cache=collections.OrderedDict()
    try:
        model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict())
        for j,r in enumerate([r for r in p['records']if r['dataset']==ds]):
            verify(True);assert prior+time.perf_counter()-start<p['GPU_seconds_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30;tick=time.perf_counter()
            row,records,base,native,expert,inp,seconds=fresh(model,ds,r['query_ordinal'],cache)
            files.append(save(D/'source'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=ds,query_ordinal=r['query_ordinal'],
                frame_ids=row['frame_ids'],Source=native['boxes'].detach().cpu(),logits=detached(base.zero['logits'],'cpu'),
                records=detached(records,'cpu'),expert=detached(expert,'cpu'),Native_interval=native['physical_interval'],
                input=inp,GT_read=False,source_frozen=True,Student_updates=0,
                compute=dict(Source_capture_seconds=seconds,case_seconds=time.perf_counter()-tick,new_Source_forwards=1,new_original_DINO_calls=0))))
            base.restore(base.initial);del base,records,row,native,expert;gc.collect();torch.cuda.empty_cache()
            status('Source_capture_'+ds,j+1,GT_read=False);print(ds,r['alias'],round(time.perf_counter()-start,2),flush=True)
            if j==1:put(D/('SOURCE_PILOT2_'+ds+'.json'),dict(status='measured',count=2,seconds=time.perf_counter()-start,no_method_change=True))
        assert state_hash(model.state_dict())==mh;elapsed=time.perf_counter()-start;assert prior+elapsed<p['GPU_seconds_cap']
        put(D/('SOURCE_BARRIER_'+ds+'.json'),dict(status='complete',count=16,files=files,GT_read=False,GPU_allocation_seconds=elapsed,
            pretrained_state_sha256=mh,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    finally:lease.close()

def prepare_E():
    import numpy as np,torch
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
    from scripts.run_stvg_joint_robust_correction32_v1 import input_pool
    from vg_tta.stvg_error_state_correction_v1 import encode
    from vg_tta.stvg_dino_supported_correction32_v1 import estimate
    from vg_tta.stvg_joint_robust_correction32_v1 import apply
    p=verify(True);assert all(read(D/('SOURCE_BARRIER_'+ds+'.json'))['count']==16 for ds in ['hc2','vidstg']);files=[]
    cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')
    for r in p['records']:
        s=load(D/'source'/(r['alias']+'.pt'));b=s['Source'].double().numpy();ids=s['frame_ids'];U,q=mass_table(probabilities(s['records'],s['logits']))
        assert U==ids and abs(q-independent_cdf(probabilities(s['records'],s['logits']),U)).max()<1e-12
        m=final_mbr(ids,q);lo,hi=np.triu_indices(len(ids),1);k=np.flatnonzero((lo==m['indices'][0])&(hi==m['indices'][1]))[0]
        assert abs(brute_utility(ids,q,[k])[0]-m['utility'])<1e-12
        ff=input_pool(s['expert']);pools=[encode(f['boxes'])-encode(b)[f['position']]for f in ff]
        setting=cal['source_domains']['vidstg'if r['dataset']=='hc2'else'hc2'];est=estimate(pools,[f['scores']for f in ff],setting['scale'],setting['truncation'],1/16)
        files.append(save(D/'E'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],frame_ids=ids,Source=b,E=apply(b,est['delta']),
            estimate=est,planned_positions=s['expert']['positions4'],interval=m['interval'],GT_read=False,MBR_utility=m['utility'])))
    put(D/'E_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False));assert not torch.cuda.is_initialized()

def measure():
    import numpy as np,torch
    from scripts.run_decota_paper_main_v1 import gpu,read_row,frames_for
    from methods.decota_final_simplified_v1.observations import SpatialExpert,target_tokens
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_dino_mask_align32_v1 import MaskDecoder,feedback
    from vg_tta.stvg_dino_conditional_opd32_v1 import legal
    from vg_tta.stvg_dino_endpoint_confirm32_v1 import combine
    p=verify(True);assert read(D/'E_BARRIER.json')['count']==32;lease=lease_gpu();start=time.perf_counter();prior=used_gpu();files=[];cache=collections.OrderedDict();qualified=set()
    try:
        expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);mh=state_hash(expert.model.state_dict())
        for ix,r in enumerate(p['records']):
            verify(True);assert prior+time.perf_counter()-start<p['GPU_seconds_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30;tick=time.perf_counter()
            s=load(D/'source'/(r['alias']+'.pt'));e=load(D/'E'/(r['alias']+'.pt'));positions=e['planned_positions'];active=e['estimate']['support_frames']>0
            actions=np.empty((0,4));sample=dict(details=[],targets=actions);center=dict(legal=False,anchors={});traces={};native={};quals=[];caches=[]
            center_seconds=sample_seconds=0.;center_logical=sample_logical=unique_sample=0
            if active:
                row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,cache);assert list(ids)==list(s['frame_ids'])
                for pos in positions:
                    rc=s['expert']['observations'][('original',pos)]['receipt'];assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==rc['rgb_sha256']
                    text=rc['text'];inp=expert.processor(images=frames[pos],text=text,return_tensors='pt').to('cuda');offsets=expert.processor.tokenizer(text,return_offsets_mapping=True)['offset_mapping']
                    if rc['context_active']:ent=target_tokens(text,offsets,rc['target_span'])
                    else:
                        entity=row['parses']['old']['entity'];a=text.find(entity);assert a>=0;ent=target_tokens(text,offsets,[a,a+len(entity)])
                    c=MaskDecoder(expert.model,inp,ent);caches.append(c);native[pos]=c.native_trace()
                    if r['dataset']not in qualified:quals.append(c.qualify_mask())
                qualified.add(r['dataset']);scale=np.asarray(p['scale'][r['dataset']]);mu=np.asarray(e['estimate']['delta'])/scale
                rng=np.random.default_rng(p['seed']+int(hashlib.sha256(r['alias'].encode()).hexdigest()[:8],16));actions=mu+rng.standard_normal((24,4))
                center['legal']=bool(legal(e['E'][positions]).all());center_traces={}
                if center['legal']:
                    for pos,c in zip(positions,caches):
                        before=c.decoder_seconds;trace=c.run_trace(e['E'][pos:pos+1],True);center_seconds+=c.decoder_seconds-before;center_logical+=1
                        assert np.array_equal(trace['initial_reference'],e['E'][pos:pos+1].astype(np.float32));center_traces[pos]=trace
                        score=float(trace['scores'][-1,0]);center['anchors'][pos]=dict(box=trace['boxes'][-1,0],score=score,accepted=score>=.35,
                            slot=int(trace['content_slots'][0]),mask_slot=int(trace['mask_slots'][0]))
                before=sum(c.decoder_seconds for c in caches);f,traces=feedback(caches,e['Source'],actions,scale,positions,paired=False)
                sample=f['B_aligned'];sample_seconds=sum(c.decoder_seconds for c in caches)-before
                sample_logical=sum(len(tr['candidate_indices'])for tr in traces.values());unique_sample=sum(len(tr['B_aligned']['initial_reference'])for tr in traces.values())
                traces=dict(center=center_traces,samples=traces)
            assert state_hash(expert.model.state_dict())==mh
            result=combine(e['E'],positions,sample['details'],center,active)
            cost=dict(case_seconds=time.perf_counter()-tick,encoding_frames=len(caches),encoding_native_seconds=sum(c.encoding_full_forward_seconds for c in caches),
                center_decoder_seconds=center_seconds,sample_decoder_seconds=sample_seconds,center_logical_queries=center_logical,sample_logical_queries=sample_logical,
                unique_sample_queries=unique_sample,qualifier_decoder_calls=sum(c.decoder_calls for c in caches)-int(center['legal'])*len(caches)-len([t for t in traces.get('samples',{}).values()]),
                new_original_DINO_calls=0)
            files.append(save(D/'predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],frame_ids=s['frame_ids'],planned_positions=positions,
                interval=e['interval'],active=active,**result,actions=actions,sample_feedback=sample,center_feedback=center,traces=traces,native=native,
                qualifications=quals,scale=p['scale'][r['dataset']],compute=cost,GT_read=False,pretrained_weights_frozen_exact=True)))
            status('conditional_center_and24',ix+1,GT_read=False);print(r['alias'],ix+1,round(time.perf_counter()-start,2),flush=True)
            del caches,result,native,traces;gc.collect();torch.cuda.empty_cache()
        elapsed=time.perf_counter()-start;assert prior+elapsed<p['GPU_seconds_cap'];verify(True,True)
        put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_outputs_before_GT=True,GT_read=False,
            GPU_teacher_allocation_seconds=elapsed,GPU_total_allocation_seconds=prior+elapsed,pretrained_state_sha256=mh,
            qualified_domains=sorted(qualified),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    finally:lease.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','capture','E','measure']);a.add_argument('--dataset',choices=['hc2','vidstg']);args=a.parse_args()
    try:
        if args.stage=='capture':capture(args.dataset)
        elif args.stage=='E':prepare_E()
        else:globals()[args.stage]()
    except BaseException as e:
        put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',type=type(e).__name__,error=str(e),partials_preserved=True));raise
