"""Frozen RegionKernel dual-domain full evaluation; old queues are read-only evidence."""
import argparse,fcntl,gc,hashlib,json,os,subprocess,sys,time,traceback
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OLD=ROOT.parent/'stvg-mainline';PROVIDER=ROOT.parent/'stvg-opd'
K16=ROOT.parent/'stvg-k16'
RUN=ROOT/'outputs/region_kernel_full_v2';PUBLIC=ROOT/'results/region_kernel_full_v2'

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
    for root,k in [(ROOT,'code'),(K16,'k16_code'),(OLD,'old_paper_code'),(PROVIDER,'provider_sources')]:
        for f,h in p[k].items():assert sha(root/f)==h,(k,f)
    for f,h in p['input_bindings'].items():assert sha(f)==h,f
    if assets:
        for f,h in p['assets'].items():assert sha(PROVIDER/f)==h,f
    return p

def prepare():
    assert not(RUN/'plan.json').exists(),'Never replace frozen plan'
    paper=read(OLD/'outputs/tastvg_paper_v1/plan.json');old=read(PROVIDER/'outputs/boxfixed_full_v1/plan.json')
    assert read(OLD/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
    assert read(PROVIDER/'outputs/boxfixed_full_v1/state.json')['status']=='complete_published'
    assert read(K16/'outputs/k16_full_v1/state.json')['status']=='paused_by_user'
    kp=read(K16/'outputs/k16_full_v1/plan.json');reuse={ds:{} for ds in paper['jobs']};bound={};sem={}
    for file in [OLD/'outputs/tastvg_paper_v1/plan.json',K16/'outputs/k16_full_v1/plan.json',K16/'outputs/maintenance/k16_pause_general_20261023/pause_receipt.json',PROVIDER/'outputs/boxfixed_full_v1/capture_index.json']:
        bound[str(file)]=sha(file)
    source=OLD/'outputs/target_semantic128_collect_v3';ss=read(source/'sealed.json')
    for stage in ['general_region_v1','general_region_recheck_v3']:
        folder=K16/'outputs'/stage;study=read(folder/'plan.json');sealed=read(folder/'sealed.json')
        for file in [folder/'plan.json',folder/'sealed.json']:bound[str(file)]=sha(file)
        for item in study['items']:
            e=sealed['files'][item['alias']]
            if stage=='general_region_v1':
                raw=ss['files'][item['alias']];path=source/raw['capture_path'];capture_sha=raw['capture_sha256']
            else:path=folder/e['capture_path'];capture_sha=e['capture_sha256']
            sem[item['dataset'],item['row']['id']]=dict(path=str(path),sha256=capture_sha,feature_path=str(folder/e['trace_path']),feature_sha256=e['trace_sha256'],reference_prediction=str(folder/e['path']),reference_prediction_sha256=e['sha256'],input_binding=binding(item['row'],paper['provider_runtime_fingerprint']))
    assert len(sem)==320
    ix=read(PROVIDER/'outputs/boxfixed_full_v1/capture_index.json')['files']
    for ds,j in paper['jobs'].items():
        bound[str(PROVIDER/j['manifest'])]=j['manifest_sha256'];sealpath=PROVIDER/'outputs/boxfixed_full_v1'/ds/'sealed.json';bound[str(sealpath)]=sha(sealpath)
        records=[json.loads(s) for s in (PROVIDER/j['manifest']).read_text().splitlines()];assert len(records)==j['queries']
        for r in records:
            key=r['id'];b=binding(r,paper['provider_runtime_fingerprint'])
            if (ds,key) in sem:
                e=sem[ds,key];assert e['input_binding']==b;reuse[ds][key]=dict(kind='complete_features',**e);continue
            oldbase=OLD/'outputs/tastvg_paper_v1'/ds;rp=oldbase/'receipts'/(key+'.json')
            if rp.exists():
                rec=read(rp);assert rec['input_binding']==b;reuse[ds][key]=dict(kind='geometry',path=str(oldbase/'captures'/(key+'.pt')),sha256=rec['capture_sha256'],input_binding=b)
            elif b in ix:
                e=ix[b];reuse[ds][key]=dict(kind='geometry',path=str(PROVIDER/e['path']),sha256=e['sha256'],input_binding=b)
    atomic(RUN/'reuse_index.json',reuse);bound[str(RUN/'reuse_index.json')]=sha(RUN/'reuse_index.json')
    code={str(f.relative_to(ROOT)):sha(f) for f in [*sorted((ROOT/'stvg').glob('*.py')),*sorted((ROOT/'tools').glob('full*.py')),ROOT/'configs/default.json']}
    p=dict(schema='region_kernel_full_v2',method='BoxFixed-RegionKernel',authorization='User explicitly ACCEPTS fixed newHC32/Vid32 tradeoff: HCvsK16 +.1639t/+.0676vpp, Vid -1.0688t/-.5542vpp, Vid8wins17harms4severe vsK161severe. Freeze EXACT RegionKernel and launch NEWfull, spatial unchanged, no more tuning. Both failed qualifications remain FALSE; engineering selection is NOT scientific validation.',validation_passed=False,prior_Source_short_gate_failed_preserved=True,code=code,k16_code=kp['code'],old_paper_code=paper['code'],provider_sources=old['sources'],assets=old['assets'],jobs=paper['jobs'],provider_runtime_fingerprint=paper['provider_runtime_fingerprint'],input_bindings=bound,
      profile=dict(temporal='Global6 original two native224 offsets/sixlayer old hull mean -> samefulltimeBoxFixed target1.5 -> original fullcaption fused motion memory fractional mean -> oppositeoffset center+unitnorm -> nonnegative cosine kernel prototypes/commonmean16 -> ONEq0prior update -> originalphysical MBR',rounds=1,kappa=16,context=1.5,observed=.5,guard=.1,link=.25,seed=20260920),
      expert='same original ReferDINO/eightpositions, old contentbound masks/boxes only; cache miss hardfails and cannot call expert',new_expert_calls=0,new_training=0,no_more_tuning=True,GT_in_inference=False,
      main_arms=['Source','Interpolation','OursS','Global6','Full'],ablation_arms=['Full','NoLocal','Independent','OursS'],mechanism_arms=['Source','SharedInit','OursS'],
      aliases='Source/SharedInit/Global6 must exactly match originalfull. NativeTemporal=OursS. Full is NEWRegionKernel, never alias oldGlobal6. Space ablations fix NEWFull time. No w/oSharedInit.',
      reuse='Exact input/model/evidence-bound completefeature caches320 first, compatible oldpaper or completeindex geometry next. Otherwise originalSource capture with cachedteacher, originalGlobal6/Source/E/positions verified before accepting newRegionKernel. No old cost counted as new. RGB512MiB/actualtensorVisual4GiB, counters perPID/startticks.',
      exposure='Official full and development panels exposed. Target-label development and failed rechecks disclosed. User-directed frozen final evaluation, not unseen-test or statistical equivalence claim.',
      output='All12205officialqueries, no skips. Seal each complete domain before independentCPUfixedlabels. Queryequal means, parentcluster10000seed20261022; thresholds/time/errors/observed+unobservedspace/all signed/severe/extent/distance. No full-result selection.',
      post_full='After both scored: independent fair end-to-end timing16parents/domain3repeats withoutcrossqueryreuse, source/8expert/spatial/regionpool/prototype/DP components; private continuous-frame review. No parameter changes. Publish all signs, no rawmedia/caption/GTcoords/weights/features.',bootstrap='parentcluster10000seed20261022',tta_audit_sha256=sha(OLD/'outputs/tastvg_paper_v1/tta_audit.json'))
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(RUN/'plan.json',p)
    for f in code:
        q=RUN/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/f).read_bytes())
    atomic(RUN/'tta_audit.json',read(OLD/'outputs/tastvg_paper_v1/tta_audit.json'))
    atomic(PUBLIC/'protocol.json',{k:v for k,v in p.items() if k not in ['assets','jobs','old_paper_code','provider_sources','input_bindings']})
    atomic(RUN/'state.json',dict(status='prepared',fingerprint=p['fingerprint'],no_old_job_resumed=True));check(True)
    print(p['fingerprint'])

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
    from stvg.representation import extract
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
    rgb=RGB();visual=VisualCache(4*2**30);files={};recent=[];start=time.perf_counter();counts=dict(feature_reuse=0,geometry_reuse=0,new_geometry=0,Source_queries=0,additional_classifier_forwards=0,new_expert=0)
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
                if entry['kind']=='complete_features':
                    assert sha(entry['feature_path'])==entry['feature_sha256'];oldtrace=torch.load(entry['feature_path'],weights_only=False,map_location='cpu');sem=oldtrace['features']['RegionKernel'];counts['feature_reuse']+=1;del oldtrace
                else:counts['geometry_reuse']+=1
                if sem is None:
                    tc=cap.get('temporal_cache',cap.get('native_hidden'))
                    if tc is None:
                        if model.source is None:model.source=load_source(model.config)
                        frames,ids=rgb.get(r['metadata']);model.source.cuda()
                        if model.parser is None:model.parser=QuerySubjectParser(ROOT/'checkpoints/stanza')
                        rr=dict(input=r['metadata'],frame_ids=ids,parses={'subject':model.parser(r['metadata']['caption'])['subject']})
                        st=time.perf_counter()
                        with visual.context(model.source,frames,ids):batch,recs,base=frozen_native_forward(model.source,frames,rr)
                        assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source']);tc=cache_temporal(model.source,base,recs,ids);del batch,base;model.source.cpu();counts['Source_queries']+=1;cost['Source_seconds']=time.perf_counter()-st
                    else:recs=cap['records']
                    if model.source is None:model.source=load_source(model.config)
                    head=model.source.temp_embed.cuda();logs,_=read_logits(head,[v['hidden'] for v in tc['offsets']],[v['native_logits'] for v in tc['offsets']]);head.cpu()
                    if 'logits' in cap:assert all(torch.equal(a,c) for a,c in zip(logs,cap['logits']))
                    cap.update(records=recs,logits=logs,temporal_cache=tc)
            else:
                if model.source is None:model.source=load_source(model.config)
                frames,ids=rgb.get(r['metadata']);st=time.perf_counter()
                with visual.context(model.source,frames,ids):native=model.capture(frames,ids,r['metadata'])
                assert native['cost']['ReferDINO_query_forwards']==0;counts['Source_queries']+=1;counts['new_geometry']+=1;cost=dict(native['cost'])
                head=model.source.temp_embed.cuda();tc=native['native_hidden'];logs,_=read_logits(head,[v['hidden'] for v in tc['offsets']],[v['native_logits'] for v in tc['offsets']]);head.cpu()
                cap=dict(Source=native['Source'],E=native['E'],frame_ids=ids,positions8=native['planned_positions'],records=native['records'],logits=logs,geometry=native['geometry'].state_dict(),native_interval=native['intervals']['Native'],temporal_cache=tc,GT_read=False)
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
            after=visual.stats();rec=dict(fingerprint=p['fingerprint'],input_binding=b,prediction_sha256=sha(npz),capture_sha256=sha(cp),worker=identity(),old_Source_E_Global6_bitwise=True,old_prediction_sha256=oldentry['sha256'],reuse_kind=entry['kind'] if entry else 'new_geometry_cached_teacher',complete_capture_reused=bool(entry and entry['kind']=='complete_features'),Source_recomputed=counts['Source_queries']-source_before,expert_forwards=0,capture_cost=cost,query_wall_seconds=time.perf_counter()-tick,visual_delta={k:after[k]-vc[k] for k in ['hits','misses']},RGB_delta=dict(hits=rgb.hits-rc[0],misses=rgb.misses-rc[1]),GT_read=False,trace=trace)
            atomic(rp,rec);files[key]=dict(path=str(npz.relative_to(dest)),sha256=sha(npz),receipt_path=str(rp.relative_to(dest)),receipt_sha256=sha(rp),capture_path=str(cp.relative_to(dest)),capture_sha256=sha(cp))
            recent.append(rec['query_wall_seconds']);recent=recent[-100:];atomic(dest/'progress.json',dict(completed=i+1,total=len(rows),recent_seconds=float(np.mean(recent)),ETA_seconds=float(np.mean(recent))*(len(rows)-i-1),counts=counts,visual=after,**identity(),GT_read=False))
            print(json.dumps(dict(dataset=ds,completed=i+1,total=len(rows),seconds=rec['query_wall_seconds'],reuse=rec['reuse_kind'])),flush=True)
            del cap,compact,arr,g,sem,frames
        check();atomic(dest/'sealed.json',dict(status='sealed',fingerprint=p['fingerprint'],queries=len(rows),files=files,GT_read=False));atomic(dest/'run.json',dict(status='completed_sealed',seconds=time.perf_counter()-start,counts=counts,**identity()))
    finally:teacher.close();gpu.close()

def controller(resume=False):
    p=check(True);proof=read(RUN/'regression.json');assert proof['status']=='passed' and proof['plan_sha256']==sha(RUN/'plan.json') and proof['queries']==320;lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
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
