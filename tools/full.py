"""Frozen Native2D dual-domain full evaluation; old queues are read-only evidence."""
import argparse,fcntl,gc,hashlib,json,os,subprocess,sys,time,traceback
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OLD=ROOT.parent/'stvg-mainline';PROVIDER=ROOT.parent/'stvg-opd'
K16=ROOT.parent/'stvg-k16'
FUSED=ROOT.parent/'stvg-general'
OLDN2D=ROOT.parent/'stvg-native2d'
NATIVEPRIOR=ROOT.parent/'stvg-native2d-nativeprior'
PREVIOUS=NATIVEPRIOR/'outputs/native2d_nativeprior_full_v1'
RUN=ROOT/'outputs/simplepath_full_v2';PUBLIC=ROOT/'results/simplepath_full_v2'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def identity(pid=None):
    pid=os.getpid() if pid is None else pid
    return dict(pid=pid,startticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
def alive(x):
    try:return identity(x['pid'])=={k:x[k] for k in ['pid','startticks']}
    except (OSError,ValueError):return False
def binding(row,runtime):
    body=dict(dataset=row['dataset'],runtime_fingerprint=runtime,input={k:v for k,v in row['metadata'].items() if k!='video_path'})
    return hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def check(assets=False):
    p=read(RUN/'plan.json');assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    for root,k in [(ROOT,'code'),(NATIVEPRIOR,'nativeprior_code'),(OLDN2D,'previous_code'),(FUSED,'fused_code'),(K16,'k16_code'),(OLD,'old_paper_code'),(PROVIDER,'provider_sources')]:
        for f,h in p[k].items():assert sha(root/f)==h,(k,f)
    for f,h in p['input_bindings'].items():assert sha(f)==h,f
    if assets:
        for f,h in p['assets'].items():assert sha(PROVIDER/f)==h,f
    return p

def prepare():
    assert not (RUN/'plan.json').exists(), 'Never replace a frozen plan'
    parent=read(PREVIOUS/'plan.json')
    pause_dir=NATIVEPRIOR/'outputs/maintenance/switch_to_simplepath_20261023'
    pause=read(pause_dir/'pause_receipt.json')
    assert pause['counts']=={'hc2':1902,'vidstg':2388} and not pause['unbound'] and pause['gpu_lock_free']
    assert read(PREVIOUS/'state.json')['status']=='paused_by_user_switch_to_SimplePath'
    reuse=read(PREVIOUS/'reuse_index.json')
    # Old content vectors were pooled on a DIFFERENT auxiliary path: never reuse.
    for ds,entries in reuse.items():
        for key,entry in entries.items():
            entry['reuse_original_kind']=entry['kind'];entry['kind']='legacy_geometry_repool'
    manifest=read(pause_dir/'receipt_manifest.json');count=0;wall=0.
    for ds,entries in manifest.items():
        for key,proof in entries.items():
            rp=PREVIOUS/ds/'receipts'/(key+'.json');cp=PREVIOUS/ds/'captures'/(key+'.pt');npz=PREVIOUS/ds/'predictions'/(key+'.npz')
            rc=read(rp)
            assert sha(rp)==proof['receipt_sha256'] and sha(cp)==proof['capture_sha256'] and sha(npz)==proof['prediction_sha256']
            reuse[ds][key]=dict(kind='legacy_geometry_repool',path=str(cp),sha256=sha(cp),input_binding=rc['input_binding'],acquisition_receipt=str(rp),acquisition_receipt_sha256=sha(rp))
            count+=1;wall+=rc['query_wall_seconds']
    assert count==4290
    atomic(RUN/'reuse_index.json',reuse)
    promotion=read(PREVIOUS/'promotion_inputs.json');assert len(promotion)==256
    atomic(RUN/'promotion_inputs.json',promotion)
    cfg=read(ROOT/'configs/default.json')
    assert cfg['temporal']['kappa']==64 and cfg['temporal']['beta']==1 and cfg['temporal']['rounds']==1 and cfg['temporal']['prototype_fitting']=='all_video'
    sources=[*sorted((ROOT/'stvg').glob('*.py')),*sorted((ROOT/'interfaces').glob('*.py')),*sorted((ROOT/'tools').glob('full*.py')),ROOT/'tools/native2d_bridge.py',ROOT/'tools/adoption_verify.py',ROOT/'configs/default.json']
    code={str(f.relative_to(ROOT)):sha(f) for f in sources}
    bindings=dict(parent['input_bindings'])
    for f in [RUN/'reuse_index.json',RUN/'promotion_inputs.json',PREVIOUS/'plan.json',pause_dir/'pause_receipt.json',pause_dir/'receipt_manifest.json',NATIVEPRIOR/'outputs/algorithm_pruning_v2/plan.json',NATIVEPRIOR/'outputs/algorithm_pruning_v2/sealed.json',NATIVEPRIOR/'outputs/algorithm_pruning_v2/score_receipt.json']:
        bindings[str(f)]=sha(f)
    plan=dict(schema='simplepath_full_v2',method=cfg['method'],authorization='User accepts SimplePath, explicitly cancels offset-wise prototype fitting, authorizes safe pause/save and switch after coupled validation, and requests local/GitHub cleanup. No further tuning.',profile=cfg['temporal'],code=code,nativeprior_code=parent['code'],previous_code=parent['previous_code'],fused_code=parent['fused_code'],k16_code=parent['k16_code'],old_paper_code=parent['old_paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],jobs=parent['jobs'],provider_runtime_fingerprint=parent['provider_runtime_fingerprint'],input_bindings=bindings,
        spatial=dict(algorithm='SimplePath',observed=.5,unobserved=.5,link=.25,both_anchors=True,bidirectional_affine=True,cycle=False,extent_RBF=False,initialization_guard=False,mask_transport_link=False),
        new_expert_calls=0,new_training=0,GT_in_inference=False,main_arms=['Source','Interpolation','OursS','Global6','Full'],ablation_arms=['Full','NoLocal','Independent','OursS'],mechanism_arms=['Source','SharedInit','OursS'],aliases='NativeTemporal=OursS. Global6 only a sealed imported reference. Full uses native last-layer prior, ONE all-video shrunk prototype pair, ONE update, original MBR, SimplePath.',
        reuse='Reuse exact Source/E/native final logits/geometric candidates/eight observations. Recompute the SimplePath full-video auxiliary path. Every old pooled Native2D vector is invalidated: repool actual pre-text appearance grids. Bound prior+geometry uses original Source224 appearance prefix only; missing geometry uses one original Source224 with cached SAME teacher8. No extra expert.',
        exposure='Adoption after exposed modular67HC/128perdirection results, not new independent confirmation. SimplePath observed sIoU -1.155595pp; NoSplit Vid mean t-.052664/v-.043750pp, Source severe union +4 queries per direction and Vid short t-.006256pp retained. Prior full HC t+.747072pp failed Source+.9. No combined gain claimed before full.',
        output='HC1902/227 then Vid10303/732, queryequal; full domain seal before independent CPU official labels; HC CPU overlaps VidGPU. All signed/severe/CI/ablations/stages/extent/distance retained.',
        post_full=parent['post_full'],tta_audit_sha256=parent['tta_audit_sha256'],no_more_tuning=True,regression_queries=256)
    plan['fingerprint']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();atomic(RUN/'plan.json',plan)
    for f in code:
        target=RUN/'source_snapshot'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/f).read_bytes())
    atomic(RUN/'tta_audit.json',read(PREVIOUS/'tta_audit.json'))
    atomic(RUN/'cost_origin.json',dict(previous_method_capture_queries=count,previous_method_receipt_wall_sum=wall,reused_from=str(PREVIOUS),previous_origin_sha256=sha(PREVIOUS/'cost_origin.json'),note='Previous acquisition provenance only; no old pooled features or predictions migrated. New Source visual-prefix work counted explicitly per new worker; do not double-count old69 costs.'))
    atomic(RUN/'acceptance.json',dict(status='pending_full_evaluation',metric='query_equal_mean_tIoU',minimum_delta_Source_pp=.9,each_direction_required=True,no_tuning=True,not_guaranteed=True))
    private={'nativeprior_code','previous_code','fused_code','assets','jobs','old_paper_code','provider_sources','input_bindings'}
    atomic(PUBLIC/'protocol.json',{k:v for k,v in plan.items() if k not in private})
    atomic(RUN/'state.json',dict(status='prepared',fingerprint=plan['fingerprint'],old_queues_not_resumed=True));check(True);print(plan['fingerprint'])

def worker(ds):
    import torch,numpy as np
    from stvg.runtime import setup,lease,load_source
    from stvg.predictor import Predictor
    from stvg.visual_cache import VisualCache
    from stvg.teacher_bridge import ResidentTeacher,evidence_key
    from stvg.worker import resource_guard
    from stvg.decode import decode
    from stvg.native import frozen_native_forward
    from stvg.temporal_hidden import cache_temporal
    from stvg.prior import from_capture
    from stvg.temporal import probabilities,mass_table,final_mbr
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.representation import extract,observe_native_grid
    from stvg.paper_panel import derive
    from stvg.query import QuerySubjectParser
    from collections import OrderedDict
    class RGB:
        def __init__(self):self.entries=OrderedDict();self.bytes=0;self.hits=self.misses=0
        def get(self,m):
            key=(ds,m['video_sha256'],tuple(m['frame_ids']),m['width'],m['height'])
            if key in self.entries:
                assert sha(m['video_path'])==m['video_sha256'];v=self.entries.pop(key);self.entries[key]=v;self.hits+=1;return v.copy(),list(m['frame_ids'])
            a,ids=decode(m,ds);self.misses+=1
            if a.nbytes<=512*2**20:
                while self.entries and self.bytes+a.nbytes>512*2**20:_,v=self.entries.popitem(last=False);self.bytes-=v.nbytes
                self.entries[key]=a.copy();self.bytes+=a.nbytes
            return a,ids
    class CacheOnly(ResidentTeacher):
        def _call(self,*a,**kw):raise RuntimeError('No new expert forward authorized; preserve cache-miss failure')
        def __call__(self,frames,caption,*,positions,frame_ids,dataset):
            key,_=evidence_key(frames,caption,positions=positions,frame_ids=frame_ids,fingerprint=self.fingerprint,dataset=dataset)
            roots=[PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',OLD/'outputs/tastvg_paper_v1/teacher_cache']
            folder=next((f for f in roots if (f/(key+'.json')).exists()),None)
            if folder is None:raise RuntimeError('Existing expert evidence missing: '+key)
            self.cache_dir=folder;v=super().__call__(frames,caption,positions=positions,frame_ids=frame_ids,dataset=dataset)
            assert v['cache_hit'] and v['query_forwards']==0 and self.process is None;return v
    p=check();setup();gpu=lease();dest=RUN/ds;atomic(dest/'worker.json',dict(**identity(),fingerprint=p['fingerprint']))
    rows=[json.loads(s) for s in (PROVIDER/p['jobs'][ds]['manifest']).read_text().splitlines()];index=read(RUN/'reuse_index.json')[ds]
    oldseal=read(PROVIDER/'outputs/boxfixed_full_v1'/ds/'sealed.json');teacher=CacheOnly(fingerprint=p['provider_runtime_fingerprint']);model=Predictor(ds,check=resource_guard);model.teacher_service=teacher
    rgb=RGB();visual=VisualCache(4*2**30);files={};recent=[];start=time.perf_counter();counts=dict(feature_reuse=0,geometry_reuse=0,new_geometry=0,Source_queries=0,additional_classifier_forwards=0,appearance_prefix_queries=0,new_expert=0)
    try:
        for i,r in enumerate(rows):
            resource_guard();key=r['id'];b=binding(r,p['provider_runtime_fingerprint']);rp=dest/'receipts'/(key+'.json');npz=dest/'predictions'/(key+'.npz');cp=dest/'captures'/(key+'.pt')
            if rp.exists():
                rec=read(rp);assert rec['fingerprint']==p['fingerprint'] and rec['input_binding']==b and sha(npz)==rec['prediction_sha256'] and sha(cp)==rec['capture_sha256']
                files[key]=dict(path=str(npz.relative_to(dest)),sha256=sha(npz),receipt_path=str(rp.relative_to(dest)),receipt_sha256=sha(rp),capture_path=str(cp.relative_to(dest)),capture_sha256=sha(cp));continue
            assert not npz.exists() and not cp.exists(),'Preserve unbound partial files; do not overwrite'
            tick=time.perf_counter();vc=visual.stats();rc=(rgb.hits,rgb.misses);source_before=counts['Source_queries'];entry=index.get(key);cost={};sem=None;frames=None
            torch.manual_seed(20260920);np.random.seed(20260920)
            if entry:
                assert entry['input_binding']==b and sha(entry['path'])==entry['sha256'];cap=torch.load(entry['path'],weights_only=False,map_location='cpu')
                g=SpatialGeometry.from_cached_geometry(cap['geometry']);cap['geometry']=g.state_dict();ids=cap['frame_ids'];positions=cap.get('positions8',cap.get('planned_positions'))
                counts['geometry_reuse']+=1
                if sem is None:
                    cached_native_prior=('records'in cap and ('native_logits'in cap or 'logits'in cap))
                    if cached_native_prior:
                        from stvg.visual_prefix import native_appearance_grid
                        if model.source is None:model.source=load_source(model.config)
                        frames,ids=rgb.get(r['metadata']);st=time.perf_counter()
                        with visual.context(model.source,frames,ids):grid=native_appearance_grid(model.source,frames,ids,r['metadata'])
                        cap['native_local_grid']=grid;counts['appearance_prefix_queries']+=1;cost['native_appearance_prefix_seconds']=time.perf_counter()-st
                        cost['Source_prior_geometry_hash_reused']=True
                    else:
                        if model.source is None:model.source=load_source(model.config)
                        frames,ids=rgb.get(r['metadata']);model.source.cuda()
                        if model.parser is None:model.parser=QuerySubjectParser(ROOT/'checkpoints/stanza')
                        rr=dict(input=r['metadata'],frame_ids=ids,parses={'subject':model.parser(r['metadata']['caption'])['subject']})
                        st=time.perf_counter()
                        with observe_native_grid(model.source)as grid,visual.context(model.source,frames,ids):batch,recs,base=frozen_native_forward(model.source,frames,rr)
                        assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
                        logs=[v.detach().cpu().clone() for v in base.zero['logits']];del batch,base;model.source.cpu()
                        if 'logits'in cap:assert all(torch.equal(a,c[-1])for a,c in zip(logs,cap['logits']))
                        cap.update(records=recs,native_logits=logs,native_local_grid=grid)
                        counts['Source_queries']+=1;cost['Source_seconds']=time.perf_counter()-st

            else:
                if model.source is None:model.source=load_source(model.config)
                frames,ids=rgb.get(r['metadata']);st=time.perf_counter()
                with visual.context(model.source,frames,ids):native=model.capture(frames,ids,r['metadata'],collect_mask_diagnostics=True)
                assert native['cost']['ReferDINO_query_forwards']==0;counts['Source_queries']+=1;counts['new_geometry']+=1;cost=dict(native['cost'])
                cap=dict(Source=native['Source'],E=native['E'],frame_ids=ids,positions8=native['planned_positions'],records=native['records'],native_logits=native['native_logits'],geometry=native['geometry'].state_dict(),native_interval=native['intervals']['Native'],native_local_grid=native['native_local_grid'],GT_read=False)
                g=native['geometry'];positions=native['planned_positions'];del native
            if 'native_logits'not in cap:cap['native_logits']=[x[-1]for x in cap['logits']]
            cap['native_prior']=from_capture(cap)
            native_interval=final_mbr(ids,cap['native_prior'])['interval']
            if 'native_interval'in cap:assert native_interval==list(cap['native_interval'])
            cap['native_interval']=native_interval
            oldentry=oldseal['files'][key];oldpath=PROVIDER/'outputs/boxfixed_full_v1'/ds/oldentry['path'];assert sha(oldpath)==oldentry['sha256']
            with np.load(oldpath,allow_pickle=False)as z:
                cap['reference_Global6']=z['Final'].copy();cap['reference_Global6_interval']=z['Final_interval'].copy()
            cap['positions8']=positions
            if sem is None:
                aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();st=time.perf_counter()
                sem,pool_stats=extract(cap,aux)
                cost['region_pooling_seconds']=time.perf_counter()-st
                cost['region_pooling_fallback_frames']=sum(x['fallback']for x in pool_stats)
            cap['content_features']=sem;st=time.perf_counter();arr,trace=derive(cap);cost['panel_seconds']=time.perf_counter()-st
            oldentry=oldseal['files'][key];oldpath=PROVIDER/'outputs/boxfixed_full_v1'/ds/oldentry['path'];assert sha(oldpath)==oldentry['sha256']
            with np.load(oldpath,allow_pickle=False) as z:
                for a,c in [('Source','Source'),('SharedInit','E_Ref'),('Global6','Final')]:assert np.array_equal(arr[a],z[c]),(key,a)
                for a,c in [('Source_interval','Source_interval'),('Global6_interval','Final_interval'),('positions8','positions8')]:assert np.array_equal(arr[a],z[c]),(key,a)
            # Preserve compact geometry+pooled content features, not the large decoder-memory cache.
            compact={k:cap[k] for k in ['Source','E','frame_ids','positions8','records','native_logits','native_prior','geometry','native_interval','content_features','reference_Global6','reference_Global6_interval','GT_read']}
            npz.parent.mkdir(parents=True,exist_ok=True);cp.parent.mkdir(parents=True,exist_ok=True);torch.save(compact,cp);np.savez_compressed(npz,**arr)
            after=visual.stats();rec=dict(fingerprint=p['fingerprint'],input_binding=b,prediction_sha256=sha(npz),capture_sha256=sha(cp),worker=identity(),old_Source_E_Global6_bitwise=True,old_prediction_sha256=oldentry['sha256'],reuse_kind=entry['kind'] if entry else 'new_geometry_cached_teacher',complete_capture_reused=False,legacy_geometry_reused=bool(entry),old_pooled_features_reused=False,Source_recomputed=counts['Source_queries']-source_before,expert_forwards=0,capture_cost=cost,query_wall_seconds=time.perf_counter()-tick,visual_delta={k:after[k]-vc[k] for k in ['hits','misses']},RGB_delta=dict(hits=rgb.hits-rc[0],misses=rgb.misses-rc[1]),GT_read=False,trace=trace)
            atomic(rp,rec);files[key]=dict(path=str(npz.relative_to(dest)),sha256=sha(npz),receipt_path=str(rp.relative_to(dest)),receipt_sha256=sha(rp),capture_path=str(cp.relative_to(dest)),capture_sha256=sha(cp))
            recent.append(rec['query_wall_seconds']);recent=recent[-100:];atomic(dest/'progress.json',dict(completed=i+1,total=len(rows),recent_seconds=float(np.mean(recent)),ETA_seconds=float(np.mean(recent))*(len(rows)-i-1),counts=counts,visual=after,**identity(),GT_read=False))
            print(json.dumps(dict(dataset=ds,completed=i+1,total=len(rows),seconds=rec['query_wall_seconds'],reuse=rec['reuse_kind'])),flush=True)
            del cap,compact,arr,g,sem,frames
        check();atomic(dest/'sealed.json',dict(status='sealed',fingerprint=p['fingerprint'],queries=len(rows),files=files,GT_read=False));atomic(dest/'run.json',dict(status='completed_sealed',seconds=time.perf_counter()-start,counts=counts,**identity()))
    finally:teacher.close();gpu.close()

def controller(resume=False):
    p=check(True);bridge=read(RUN/'bridge.json');assert bridge['status']=='passed'and bridge['checks']==8 and bridge['new_aux_repool_verified'];proof=read(RUN/'regression.json');assert proof['status']=='passed' and proof['plan_sha256']==sha(RUN/'plan.json') and proof['queries']==p['regression_queries'] and proof['spatial_queries']==67;lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(RUN/'state.json');assert old['fingerprint']==p['fingerprint'];assert old['status']=='prepared' or (resume and old['status']!='complete_published' and not alive(old.get('controller',{'pid':-1,'startticks':-1})) and not any(alive(x) for x in old.get('children',[])))
    state=dict(status='running',fingerprint=p['fingerprint'],controller=identity(),children=[],jobs={});atomic(RUN/'state.json',state);pending=[]
    def launch(script,args,role,env=None):
        log=(RUN/(role+'.log')).open('ab');c=subprocess.Popen([sys.executable,'-B',str(ROOT/'tools'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,**(env or {})});log.close();who={**identity(c.pid), 'role':role};state['children'].append(who);atomic(RUN/'state.json',state);return c,who
    def finish(c,w):
        code=c.wait();state['children']=[x for x in state['children'] if x!=w];atomic(RUN/'state.json',state);assert code==0,f'{w["role"]} exit{code}; preserve evidence'
    try:
        for ds in ['hc2','vidstg']:
            if not(RUN/ds/'sealed.json').exists():
                state['jobs'][ds]='running';c,w=launch('full.py',['worker','--dataset',ds],ds+'-worker');finish(c,w)
            state['jobs'][ds]='sealed';atomic(RUN/'state.json',state)
            if not(RUN/ds/'score_receipt.json').exists():pending.append(launch('full_score.py',['--dataset',ds],ds+'-scorer',{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}))
        for c,w in pending:finish(c,w)
        state.update(status='awaiting_cost_qualitative_publication',jobs={'hc2':'scored','vidstg':'scored'});atomic(RUN/'state.json',state)
    except BaseException:state.update(status='failed',error=traceback.format_exc());atomic(RUN/'state.json',state);raise
    finally:lock.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','worker','run']);ap.add_argument('--dataset',choices=['hc2','vidstg']);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    controller(a.resume) if a.command=='run' else worker(a.dataset) if a.command=='worker' else prepare()
