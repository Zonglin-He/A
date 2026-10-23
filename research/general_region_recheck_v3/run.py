"""User-selected fixed class-free candidate: additional32 parents/direction."""
import os,sys,time,json,hashlib,traceback,gc
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));OLD=R.parent/'stvg-mainline';PROVIDER=R.parent/'stvg-opd'
_requested_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
from research.general_region_v1.run import read,put,sha,who,extract,ARMS,A,B
# The archived CPU helper hides CUDA on import. Restore the calling environment
# BEFORE torch/CUDA is imported/initialized. CPU scoring deliberately keeps ''.
if _requested_cuda is None:
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = _requested_cuda
from research.general_region_v1.core import PRIMARY,candidates
D=R/'outputs/general_region_recheck_v3';P=R/'results/general_region_recheck_v3'

def verify(assets=False):
    p=read(D/'plan.json')
    assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    for f,h in p['code'].items():assert sha(R/f)==h,f
    for f,h in p['bound_inputs'].items():assert sha(f)==h,f
    sys.path.insert(0,str(R/'tools'));from full import check
    check(assets);assert read(R/'outputs/k16_full_v1/state.json')['status']=='paused_by_user'
    return p

def prepare():
    assert not(D/'plan.json').exists();first=read(R/'outputs/general_region_v1/plan.json');full=read(R/'outputs/k16_full_v1/plan.json');old=read(OLD/'outputs/native_event_target_semantic_v1/plan.json');reuse=read(R/'outputs/k16_full_v1/reuse_index.json');items=[];counts={}
    for ds in ['hc2','vidstg']:
        blocked={str(x['parent'])for x in first['items']+old['items']if x['dataset']==ds};job=full['jobs'][ds];rows=[json.loads(x)for x in(PROVIDER/job['manifest']).read_text().splitlines()];parents={}
        for row in rows:
            parent=str(row['metadata']['source'])
            if parent not in blocked:parents.setdefault(parent,[]).append(row)
        rank=lambda s:hashlib.sha256((ds+'|'+s+'|general_region_recheck_v1|20261023').encode()).hexdigest()
        selected=sorted(parents,key=rank)[:32];assert len(selected)==32;seal=read(PROVIDER/'outputs/boxfixed_full_v1'/ds/'sealed.json')
        for i,parent in enumerate(selected):
            row=min(parents[parent],key=lambda r:rank(r['id']));items.append(dict(alias=('HC2'if ds=='hc2'else'VIDSTG')+f'_RECHECK_{i+1:03d}',dataset=ds,parent=parent,row=row,reference=seal['files'][row['id']],geometry_reference=reuse[ds].get(row['id'])))
        counts[ds]=dict(remaining_parent_pool=len(parents),selected=32,blocked=(227 if ds=='hc2' else 732)-len(parents),geometry_reuse=sum(x['geometry_reference']is not None for x in items if x['dataset']==ds))
    code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
    for f in (R/'research/general_region_v1').glob('*.py'):code[str(f.relative_to(R))]=sha(f)
    failed=read(R/'outputs/general_region_recheck_v2/plan.json')
    assert [(x['alias'],x['parent'],x['row']['id'])for x in items]==[(x['alias'],x['parent'],x['row']['id'])for x in failed['items']], 'Keep the exact predeclared64 inputs'
    p=dict(schema='general_region_recheck_v3',superseded_v2_plan_sha256=sha(R/'outputs/general_region_recheck_v2/plan.json'),engineering_migration='v1 zero-model CPUenv failure preserved. V2 originalSource firstquery verified but numpyaux passed to tensor-only K16comparison;0savedpredictions/0GT. Both immutable code/plan/logs retained,ownedPIDs exited/lockfree. V3 ONLY converts the same float64 aux values to a CPUtensor at the K16comparison call, and saves rawcapture before diagnostic readout. RegionKernel core/inputs/precision/seed/controls/tradeoff exactsame; nooldhashwaiver/no completedpredictions to rerun.',primary=PRIMARY,arms=ARMS,items=items,count=64,parents=counts,code=code,labels={ds:dict(path=j['labels'],sha256=j['labels_sha256'])for ds,j in full['jobs'].items()},
        authorization='User explicitly accepts small declines and prioritizes general class-free method story after seeing initial development means/tails. Freeze the exact candidate and do a fixed additional-parent dual recheck, no tuning. Original development qualification FAILED and remains false; this is user-selection with revised engineering tradeoff, NOT retroactively passing the old gate.',
        algorithm='EXACT general_region_v1/core.py RegionKernel, context1.5, otherparitycenter/unitnorm, K=(1+cos)/2, common16kernelshrink, ONEq0update, originalMBR and unchangedBoxFixed/eightanchors. No layer/kernel/scale/round/template changes.',
        selection='Source-only inputSHA seed20261023: exclude all expanded128 and earlierHC64/HC32/Vid32 parents.32 distinctparents/direction, onequeryparent SHAselected. No labels/outcomes consulted. Originalofficialfullhistoricallyexposed; prospective-for-current-candidate recheck, NOT untouched test.',
        qualification='EACHdirection: actual t/fullv mean beyondSource/G6 positive; mean t/fullv difference vsK16 >=-.25pp; Source short<=5s/long>5s t >=-.25pp; timewins>=harms vsG6; >5pp t andv harm counts no more thanK16+ONEquery per metric. This one-query tolerance doesnotclaim harms are small/calibrated; report ALL tails. RegionvsFullGrid/Reflected/WrongTime/Position/K16Matched CIs mandatory but are MECHANISM comparisons, not automatic admission; if targetlocation lacks matched advantage, mainclaim is generic native-representation feedback, NOT proved spatialcausation. No retune/controlpromotion afterrecheck. If any condition fails no newfull; preserve failed old/new gates.',
        tradeoff='The .25pp mean tolerance and at most1additional severequery per32panel are explicit prospective engineering choices under latest small-decline acceptance, not statisticalequivalence. Initial development severe t38 vsK16t19 and failedVidcontrols disclosed. Do not label all severe failures small.',
        acquisition='Acquire originalSource224 memory on missing input-bound caches; strict oldSource/E/Global6/nativeinterval/eightpositions bitwise for EVERYquery. Reuse oldgeometry whereavailable, otherwise exact Sourcecapture with cached oldReferDINO evidence only; _call hardfails. Same source/checkpoint/grid/precision/seed. OriginalASA head used ONLY to compute fixedK16comparison, never RegionKernel evidence. Fullnative path itself may contain original ASA; do not claim removal of Sourcearchitecture/training.',
        budget=dict(max_GPU_seconds=1200,seed=20260920,new_expert=0,new_observations=0,new_training=0,new_other_backbone=0),
        all64_seal_before_GT=True,GT_in_inference=False,bootstrap_seed=20261023,
        bound_inputs={str(f):sha(f)for f in [R/'outputs/general_region_v1/plan.json',R/'outputs/general_region_v1/sealed.json',R/'results/general_region_v1/report.json',R/'results/general_region_v1/QUALIFICATION.json',R/'outputs/k16_full_v1/plan.json',R/'outputs/k16_full_v1/reuse_index.json',OLD/'outputs/native_event_target_semantic_v1/plan.json']})
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p)
    put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['items','labels','bound_inputs']})
    for f in code:
        q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
    put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));verify(True);print(json.dumps(counts),p['fingerprint'],flush=True)

def predict():
    import numpy as np,torch
    from stvg.runtime import setup,lease,load_source
    from stvg.predictor import Predictor
    from stvg.teacher_bridge import ResidentTeacher,evidence_key
    from stvg.decode import decode
    from stvg.query import QuerySubjectParser
    from stvg.native import frozen_native_forward
    from stvg.temporal_hidden import cache_temporal
    from stvg.global6 import read_logits,posterior
    from stvg.temporal import probabilities,mass_table,final_mbr
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.semantic import target_semantics
    from stvg.prototype import one_step
    p=verify(True);setup();torch.set_num_threads(4);lock=lease();start=time.perf_counter();assert not(D/'predictions').exists();put(D/'state.json',dict(status='acquiring_Source_memory_and_predicting',fingerprint=p['fingerprint'],**who()))
    class CacheOnly(ResidentTeacher):
        def _call(self,*a,**kw):raise RuntimeError('New expert call prohibited; preserve failure')
        def __call__(self,frames,caption,*,positions,frame_ids,dataset):
            key,_=evidence_key(frames,caption,positions=positions,frame_ids=frame_ids,fingerprint=self.fingerprint,dataset=dataset)
            roots=[PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',OLD/'outputs/tastvg_paper_v1/teacher_cache'];folder=next((f for f in roots if(f/(key+'.json')).exists()),None)
            assert folder is not None,'Original expert evidence missing';self.cache_dir=folder
            out=super().__call__(frames,caption,positions=positions,frame_ids=frame_ids,dataset=dataset);assert out['cache_hit']and out['query_forwards']==0 and self.process is None;return out
    full=read(R/'outputs/k16_full_v1/plan.json');teacher=CacheOnly(fingerprint=full['provider_runtime_fingerprint']);files={};counts=dict(Source_queries=0,geometry_reuse=0,geometry_rebuild=0,K16_comparison_ASA_forwards=0,new_expert=0,new_observations=0,new_training=0)
    try:
        for ds in ['hc2','vidstg']:
            model=Predictor(ds);model.teacher_service=teacher
            for item in [x for x in p['items']if x['dataset']==ds]:
                assert time.perf_counter()-start<p['budget']['max_GPU_seconds'];tick=time.perf_counter();row=item['row'];key=item['alias'];torch.manual_seed(p['budget']['seed']);np.random.seed(p['budget']['seed']);entry=item['geometry_reference'];ref=item['reference'];original=PROVIDER/'outputs/boxfixed_full_v1'/ds/ref['path'];assert sha(original)==ref['sha256']
                if entry:
                    assert sha(entry['path'])==entry['sha256'];cap=torch.load(entry['path'],weights_only=False,map_location='cpu');g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids'];cap['positions8']=cap.get('positions8',cap.get('planned_positions'));tc=cap.get('temporal_cache',cap.get('native_hidden'));counts['geometry_reuse']+=1
                    if model.source is None:model.source=load_source(model.config)
                    if tc is None:
                        frames,ids=decode(row['metadata'],ds);model.source.cuda()
                        if model.parser is None:model.parser=QuerySubjectParser(R/'checkpoints/stanza')
                        rr=dict(input=row['metadata'],frame_ids=ids,parses={'subject':model.parser(row['metadata']['caption'])['subject']})
                        batch,recs,base=frozen_native_forward(model.source,frames,rr);assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source']);tc=cache_temporal(model.source,base,recs,ids);del batch,base,frames;model.source.cpu();counts['Source_queries']+=1
                    else:recs=cap['records']
                    cap.update(records=recs,temporal_cache=tc)
                else:
                    frames,ids=decode(row['metadata'],ds);native=model.capture(frames,ids,row['metadata']);assert native['cost']['ReferDINO_query_forwards']==0 and native['cost']['ReferDINO_cache_hit'];g=native['geometry'];tc=native['native_hidden'];cap=dict(Source=native['Source'],E=native['E'],frame_ids=ids,positions8=native['planned_positions'],records=native['records'],geometry=g.state_dict(),temporal_cache=tc,GT_read=False);counts['geometry_rebuild']+=1;counts['Source_queries']+=1;del native,frames
                head=model.source.temp_embed.cuda();logs,_=read_logits(head,[v['hidden']for v in tc['offsets']],[v['native_logits']for v in tc['offsets']]);head.cpu()
                if 'logits'in cap:assert all(torch.equal(a,b)for a,b in zip(cap['logits'],logs))
                cap['logits']=logs;q0=posterior(cap['records'],logs,ids);iv0=final_mbr(ids,q0)['interval'];source_ids,sourceq=mass_table(probabilities(cap['records'],[z[-1]for z in logs]));assert source_ids==ids;source_iv=final_mbr(ids,sourceq)['interval']
                global_boxes=evidence_dp(g.for_interval(iv0))['final'].numpy()
                with np.load(original,allow_pickle=False)as z:
                    for a,b in [(cap['Source'],z['Source']),(cap['E'],z['E_Ref']),(global_boxes,z['Final']),(np.asarray(iv0),z['Final_interval']),(np.asarray(source_iv),z['Source_interval']),(np.asarray(cap['positions8']),z['positions8'])]:assert np.array_equal(a,b),(key,'original Source/E/G6/native/8 evidence bitwise')
                cf=D/'captures'/f'{key}.pt';cf.parent.mkdir(parents=True,exist_ok=True);cap['GT_read']=False;torch.save(cap,cf)
                aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();model.source.t_spatial_clas.cuda()
                tensor_aux=torch.from_numpy(aux)
                assert np.array_equal(tensor_aux.numpy(),aux),'Value-preserving K16 comparison adapter'
                try:sem,_,_,_=target_semantics(model.source,cap,tensor_aux)
                finally:model.source.t_spatial_clas.cpu()
                counts['K16_comparison_ASA_forwards']+=2;kq,kstep=one_step(sem,q0);features,pooling=extract(cap,aux);qs,feedback=candidates(features,q0,kstep['L'],ids);qs.update(NativeG6=q0,K16=kq);ivs={a:final_mbr(ids,q)['interval']for a,q in qs.items()};arr={a:evidence_dp(g.for_interval(ivs[a]))['final'].numpy()for a in qs};arr['Source']=cap['Source'];ivs['Source']=source_iv
                f=D/'predictions'/f'{key}.npz';f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arr,**{a+'_interval':v for a,v in ivs.items()},frame_ids=ids,positions8=cap['positions8']);tr=D/'traces'/f'{key}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,feedback=feedback,features=features,pooling=pooling,GT_read=False),tr)
                cf=D/'captures'/f'{key}.pt';cf.parent.mkdir(parents=True,exist_ok=True);cap['GT_read']=False;torch.save(cap,cf);e=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),capture_path=str(cf.relative_to(D)),capture_sha256=sha(cf),old_outputs_bitwise=True,new_expert=0,seconds=time.perf_counter()-tick);files[key]=e;put(D/'receipts'/f'{key}.json',e);put(D/'progress.json',dict(completed=len(files),total=64,dataset=ds,counts=counts,seconds=time.perf_counter()-start,GT_read=False,**who()));print(key,round(e['seconds'],3),flush=True);del cap,tc,g,features,sem,qs,logs,arr;gc.collect();torch.cuda.empty_cache()
            del model;gc.collect();torch.cuda.empty_cache()
        verify();put(D/'sealed.json',dict(count=64,files=files,counts=counts,plan_sha256=sha(D/'plan.json'),GT_read=False,old_Source_E_G6_native_positions_all64_bitwise=True,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed_before_new_GT',fingerprint=p['fingerprint']));print('SEALED64',flush=True)
    finally:teacher.close();lock.close()
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));raise
