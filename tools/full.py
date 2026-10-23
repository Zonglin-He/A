"""Frozen Native2D dual-domain full evaluation; old queues are read-only evidence."""
import argparse,fcntl,gc,hashlib,json,os,subprocess,sys,time,traceback
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OLD=ROOT.parent/'stvg-mainline';PROVIDER=ROOT.parent/'stvg-opd'
K16=ROOT.parent/'stvg-k16'
FUSED=ROOT.parent/'stvg-general'
PREVIOUS=ROOT/'outputs/native2d_full_v1'
RUN=ROOT/'outputs/native2d_full_v2';PUBLIC=ROOT/'results/native2d_full_v2'

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
    for root,k in [(ROOT,'code'),(PREVIOUS/'source_snapshot','native_v1_code'),(FUSED,'fused_code'),(K16,'k16_code'),(OLD,'old_paper_code'),(PROVIDER,'provider_sources')]:
        for f,h in p[k].items():assert sha(root/f)==h,(k,f)
    for f,h in p['input_bindings'].items():assert sha(f)==h,f
    if assets:
        for f,h in p['assets'].items():assert sha(PROVIDER/f)==h,f
    return p

def prepare():
    assert not(RUN/'plan.json').exists(),'Never replace frozen plan'
    original=read(FUSED/'outputs/region_kernel_full_v2/plan.json');paper=read(OLD/'outputs/tastvg_paper_v1/plan.json');kp=read(K16/'outputs/k16_full_v1/plan.json');old=read(PROVIDER/'outputs/boxfixed_full_v1/plan.json')
    study=read(FUSED/'outputs/native2d_risk_v2/plan.json');decision=read(FUSED/'results/native2d_risk_v2/MAIN_SELECTION.json');assert decision['status']=='engineering_selected'
    selected=decision['deployed'];cfg=read(ROOT/'configs/default.json');assert cfg['temporal']['kappa']==selected['kappa']and cfg['temporal']['beta']==selected['beta']
    assert read(FUSED/'outputs/region_kernel_full_v2/state.json')['status']=='paused_by_user_native2d_selected'
    pause=read(FUSED/'outputs/maintenance/fused_pause_native2d_promotion_20261023/pause_receipt.json');assert pause['gpu_lock_free']
    reuse=read(FUSED/'outputs/region_kernel_full_v2/reuse_index.json');bound=dict(original['input_bindings']);proof=[]
    for ds,j in paper['jobs'].items():
        dest=FUSED/'outputs/region_kernel_full_v2'/ds
        for rp in(dest/'receipts').glob('*.json'):
            rec=read(rp);cp=dest/'captures'/(rp.stem+'.pt');assert sha(cp)==rec['capture_sha256'];reuse[ds][rp.stem]=dict(kind='geometry',path=str(cp),sha256=sha(cp),input_binding=rec['input_binding'])
    ds1=FUSED/'outputs/local_visual_v2';ds2=FUSED/'outputs/local_visual_tune_v2';rs=FUSED/'outputs/native2d_risk_v2';s1=read(ds1/'sealed.json');s2=read(ds2/'confirm/sealed.json');s3=read(rs/'confirm/sealed.json');sd=read(rs/'dev/sealed.json');winner=read(rs/'frozen_config.json')['winner'];is_selected=selected=={k:read(rs/'frozen_config.json')[k]for k in ['kappa','beta']}
    for i in study['dev_items']:
        key=i['alias'];e=sd['files'][key];path=i['capture_path'];fp=i['trace_path'];fh=i['trace_sha256'];b=binding(i['row'],paper['provider_runtime_fingerprint']);entry=dict(kind='native2d_features',path=path,sha256=i['capture_sha256'],feature_path=fp,feature_sha256=fh,input_binding=b,reference_prediction=str(rs/e['path']),reference_sha256=e['sha256'],reference_arm=winner if is_selected else'K16_B100',reference_trace=str(rs/e['trace_path']),reference_trace_sha256=e['trace_sha256'],reference_q=winner if is_selected else'K16_B100');reuse[i['dataset']][i['row']['id']]=entry;proof.append(dict(dataset=i['dataset'],row=i['row'],**entry))
    for i in study['confirm_items']:
        key=i['alias'];e=s3['files'][key];c=i['capture_entry'];path=OLD/'outputs/target_semantic128_collect_v3'/c['capture_path'];b=binding(i['row'],paper['provider_runtime_fingerprint']);entry=dict(kind='native2d_features',path=str(path),sha256=c['capture_sha256'],feature_path=str(rs/e['trace_path']),feature_sha256=e['trace_sha256'],input_binding=b,reference_prediction=str(rs/e['path']),reference_sha256=e['sha256'],reference_arm='Selected'if is_selected else'K16',reference_trace=str(rs/e['trace_path']),reference_trace_sha256=e['trace_sha256'],reference_q='Selected'if is_selected else'K16');reuse[i['dataset']][i['row']['id']]=entry;proof.append(dict(dataset=i['dataset'],row=i['row'],**entry))
    assert len(proof)==256
    # Old pooled fused features are incompatible. Reuse ONLY original geometry there.
    for ds,entries in reuse.items():
        for key,e in entries.items():
            if e['kind']=='complete_features':entries[key]={k:v for k,v in e.items()if k in ['path','sha256','input_binding']};entries[key]['kind']='geometry'
    atomic(RUN/'reuse_index.json',reuse);atomic(RUN/'promotion_inputs.json',proof)
    for f in [RUN/'reuse_index.json',RUN/'promotion_inputs.json',rs/'plan.json',rs/'frozen_config.json',rs/'confirm/sealed.json',rs/'confirm/score_receipt.json',FUSED/'results/native2d_risk_v2/MAIN_SELECTION.json',FUSED/'outputs/region_kernel_full_v2/plan.json',FUSED/'outputs/maintenance/fused_pause_native2d_promotion_20261023/pause_receipt.json']:bound[str(f)]=sha(f)
    for f in [PREVIOUS/'plan.json',ROOT/'outputs/maintenance/native2d_visual_shortcut_20261023/pause_receipt.json',ROOT/'outputs/maintenance/native2d_visual_shortcut_20261023/prefix_probe.json']:bound[str(f)]=sha(f)
    code={str(f.relative_to(ROOT)):sha(f)for f in [*sorted((ROOT/'stvg').glob('*.py')),*sorted((ROOT/'tools').glob('full*.py')),ROOT/'configs/default.json']}
    p=dict(schema='native2d_full_v2',method='BoxFixed-Native2D',authorization='User explicitly prioritizes lower severely harmed sample fraction, authorizes finite Native2D calibration, Native2D sole main method, clean local/GitHub main trees and NEW dual-domainfull. Engineering selection is distinct from scientific qualification.',profile=cfg['temporal'],validation=decision,code=code,native_v1_code=read(PREVIOUS/'plan.json')['code'],fused_code=original['code'],k16_code=kp['code'],old_paper_code=paper['code'],provider_sources=old['sources'],assets=old['assets'],jobs=paper['jobs'],provider_runtime_fingerprint=paper['provider_runtime_fingerprint'],input_bindings=bound,spatial=dict(observed=.5,guard=.1,link=.25,unchanged=True),new_expert_calls=0,new_training=0,GT_in_inference=False,main_arms=['Source','Interpolation','OursS','Global6','Full'],ablation_arms=['Full','NoLocal','Independent','OursS'],mechanism_arms=['Source','SharedInit','OursS'],aliases='NativeTemporal=OursS; Source/SharedInit/Global6 reproduce sealed originalofficialfull. NewFull is Native2D, never fused/wordhead output alias.',reuse='256 same-input Native2D pooled features reused; compatible first69 v1 predictions/captures migrated bytewise with new explicit receipts, previouscost excluded; old fused pools incompatible. Original geometry/Source/E/eightpositions cached. Missing Native2D features with bound geometry+all6logits read exact original2D encoder+projection prefix ONLY; raw grids/pools bitwise tested. Otherwise originalSource224 ONCE builds missinggeometry; visual512MiBRGB/4GiBtensorLRU. Cache misses cannot call expert. Newtime/full-spatial-ablations recomputed, never import oldFull.',output='AllHC1902/227->Vid10303/732; queryequal. Complete domainseal BEFORE independentCPUofficiallabels, HCscore can overlap VidGPU. Allsigned/severe vsSource/G6, parentclusterCI, observed/unobserved support, boundaries/length/groups/extent/distance, workersegmentcosts retained. No full-result tuning.',post_full='Faircost16parents/direction,3repeats afterwarmup/no crossqueryreuse, originalexpert8; private continuousframe review viaimages, publish clean Native2Dcode and allsigns, archivecheck snapshot check, freshCAS main/research and ALLremote bytes/hash/tree verification.',exposure='Native2D dev64parents/direction and fixed32parent/direction calibration recheck. All originalfull/othermethod outcomes historicallyexposed; not untouchedtest/equivalence, negative qualification retained.',tta_audit_sha256=original['tta_audit_sha256'],no_more_tuning=True,regression_queries=256)
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(RUN/'plan.json',p)
    for f in code:q=RUN/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/f).read_bytes())
    atomic(RUN/'tta_audit.json',read(FUSED/'outputs/region_kernel_full_v2/tta_audit.json'));atomic(PUBLIC/'protocol.json',{k:v for k,v in p.items()if k not in ['fused_code','assets','jobs','old_paper_code','provider_sources','input_bindings']});atomic(RUN/'state.json',dict(status='prepared',fingerprint=p['fingerprint'],oldqueues_not_resumed=True));check(True);print(p['fingerprint'])

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
    from stvg.global6 import read_logits,posterior
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
                g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids'];positions=cap.get('positions8',cap.get('planned_positions'))
                if entry['kind']=='native2d_features':
                    assert sha(entry['feature_path'])==entry['feature_sha256'];oldtrace=torch.load(entry['feature_path'],weights_only=False,map_location='cpu');sem=oldtrace['features']['Native2D']if isinstance(oldtrace['features'],dict)else oldtrace['features'];counts['feature_reuse']+=1;del oldtrace
                else:counts['geometry_reuse']+=1
                if sem is None:
                    cached_native_prior=('records'in cap and'logits'in cap and len(cap['logits'])==2 and all(v.ndim==4 and v.shape[0]==6 for v in cap['logits']))
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
                        tc=cache_temporal(model.source,base,recs,ids);del batch,base
                        head=model.source.temp_embed;logs,_=read_logits(head,[v['hidden']for v in tc['offsets']],[v['native_logits']for v in tc['offsets']]);model.source.cpu()
                        if 'logits'in cap:assert all(torch.equal(a,c)for a,c in zip(logs,cap['logits']))
                        cap.update(records=recs,logits=logs,native_local_grid=grid)
                        counts['Source_queries']+=1;cost['Source_seconds']=time.perf_counter()-st

            else:
                if model.source is None:model.source=load_source(model.config)
                frames,ids=rgb.get(r['metadata']);st=time.perf_counter()
                with visual.context(model.source,frames,ids):native=model.capture(frames,ids,r['metadata'])
                assert native['cost']['ReferDINO_query_forwards']==0;counts['Source_queries']+=1;counts['new_geometry']+=1;cost=dict(native['cost'])
                head=model.source.temp_embed.cuda();tc=native['native_hidden'];logs,_=read_logits(head,[v['hidden'] for v in tc['offsets']],[v['native_logits'] for v in tc['offsets']]);head.cpu()
                cap=dict(Source=native['Source'],E=native['E'],frame_ids=ids,positions8=native['planned_positions'],records=native['records'],logits=logs,geometry=native['geometry'].state_dict(),native_interval=native['intervals']['Native'],native_local_grid=native['native_local_grid'],GT_read=False)
                g=native['geometry'];positions=native['planned_positions'];del native
            if 'native_interval' not in cap:
                U,nq=mass_table(probabilities(cap['records'],[x[-1] for x in cap['logits']]));assert U==ids;cap['native_interval']=final_mbr(ids,nq)['interval']
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
            compact={k:cap[k] for k in ['Source','E','frame_ids','positions8','records','logits','geometry','native_interval','content_features','GT_read']}
            npz.parent.mkdir(parents=True,exist_ok=True);cp.parent.mkdir(parents=True,exist_ok=True);torch.save(compact,cp);np.savez_compressed(npz,**arr)
            after=visual.stats();rec=dict(fingerprint=p['fingerprint'],input_binding=b,prediction_sha256=sha(npz),capture_sha256=sha(cp),worker=identity(),old_Source_E_Global6_bitwise=True,old_prediction_sha256=oldentry['sha256'],reuse_kind=entry['kind'] if entry else 'new_geometry_cached_teacher',complete_capture_reused=bool(entry and entry['kind']=='native2d_features'),Source_recomputed=counts['Source_queries']-source_before,expert_forwards=0,capture_cost=cost,query_wall_seconds=time.perf_counter()-tick,visual_delta={k:after[k]-vc[k] for k in ['hits','misses']},RGB_delta=dict(hits=rgb.hits-rc[0],misses=rgb.misses-rc[1]),GT_read=False,trace=trace)
            atomic(rp,rec);files[key]=dict(path=str(npz.relative_to(dest)),sha256=sha(npz),receipt_path=str(rp.relative_to(dest)),receipt_sha256=sha(rp),capture_path=str(cp.relative_to(dest)),capture_sha256=sha(cp))
            recent.append(rec['query_wall_seconds']);recent=recent[-100:];atomic(dest/'progress.json',dict(completed=i+1,total=len(rows),recent_seconds=float(np.mean(recent)),ETA_seconds=float(np.mean(recent))*(len(rows)-i-1),counts=counts,visual=after,**identity(),GT_read=False))
            print(json.dumps(dict(dataset=ds,completed=i+1,total=len(rows),seconds=rec['query_wall_seconds'],reuse=rec['reuse_kind'])),flush=True)
            del cap,compact,arr,g,sem,frames
        check();atomic(dest/'sealed.json',dict(status='sealed',fingerprint=p['fingerprint'],queries=len(rows),files=files,GT_read=False));atomic(dest/'run.json',dict(status='completed_sealed',seconds=time.perf_counter()-start,counts=counts,**identity()))
    finally:teacher.close();gpu.close()

def controller(resume=False):
    p=check(True);bridge=read(RUN/'bridge.json');assert bridge['status']=='passed'and bridge['checks']==8;proof=read(RUN/'regression.json');assert proof['status']=='passed' and proof['plan_sha256']==sha(RUN/'plan.json') and proof['queries']==p['regression_queries'];lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
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
