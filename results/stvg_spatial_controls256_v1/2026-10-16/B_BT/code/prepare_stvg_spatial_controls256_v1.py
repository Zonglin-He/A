"""Metadata-only cohort and frozen Source intake for all current controls."""
from pathlib import Path
import collections,copy,gc,hashlib,json,os,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referent_abc002 import BASE as SMALL,read,put,sha,guard
BASE=ROOT/'artifacts/stvg_spatial_controls256_v1';UNIT='stvg-spatial-controls256-source-001.service'
CP={'hc2':'5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83','vidstg':'47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036'}
def setup():
    BASE.mkdir(exist_ok=False);p=ROOT/'artifacts/stvg_final_table1_v1/FULL/FULL_SOURCE_ROSTER_LOCK.json';full=read(p);assert not full['GT_read']and full['selection_uses_GT_or_scores']is False;excluded={(r['dataset'],r['parent_sha256'])for r in read(SMALL/'ROSTER_LOCK.json')['records']};rows=[]
    for ds in ('hc2','vidstg'):
        parents={}
        for r in full['records']:
            if r['dataset']!=ds or(ds,r['parent_sha256'])in excluded:continue
            previous=parents.get(r['parent_sha256'])
            if previous is None or r['query_key_sha256']<previous['query_key_sha256']:parents[r['parent_sha256']]=r
        chosen=sorted(parents.values(),key=lambda r:(r['parent_sha256'],r['query_key_sha256']))[:128];assert len(chosen)==128
        for r in chosen:
            x=copy.deepcopy(r);x.update(source_case_path=str((BASE/'source_cases'/(r['probe_alias']+'.pt')).relative_to(ROOT)),source_checkpoint_sha256=CP[ds]);rows.append(x)
    assert len({(r['dataset'],r['parent_sha256'])for r in rows})==256 and not excluded&{(r['dataset'],r['parent_sha256'])for r in rows};assert len({r['probe_alias']for r in rows})==256
    put(BASE/'ROSTER_LOCK.json',dict(status='locked',count=256,unique_parent_count=256,domain_counts={'hc2':128,'vidstg':128},records=rows,full_registry_sha256=sha(p),excluded_small_roster_sha256=sha(SMALL/'ROSTER_LOCK.json'),membership_before_cache_inventory=True,globally_unseen_claim=False,selection_uses_GT_or_scores=False,GT_read=False))
    # Inventory only AFTER membership is irreversibly locked.
    known={};registries=[]
    for rel in ('artifacts/stvg_soft_ep_coordinate128_v1','artifacts/stvg_lite_event_support_v1/confirmation'):
        b=ROOT/rel;roster=read(b/'ROSTER_LOCK.json');bar=read(b/'SOURCE_CAPTURE_BARRIER.json');assert bar['status']=='complete'and not bar['GT_read'];files={r['case_id']:r for r in bar['files']};registries.append(dict(path=rel,roster_sha256=sha(b/'ROSTER_LOCK.json'),source_barrier_sha256=sha(b/'SOURCE_CAPTURE_BARRIER.json')))
        for r in roster['records']:
            f=files[r['probe_alias']];q=ROOT/f['path']
            if q.exists()and q.with_suffix('.receipt.json').exists():known.setdefault((r['dataset'],r['query_key_sha256']),dict(record=r,file=f,receipt_sha256=sha(q.with_suffix('.receipt.json'))))
    inventory=[]
    for r in rows:
        x=known.get((r['dataset'],r['query_key_sha256']));ok=x is not None
        if ok:
            for k in ('parent_sha256','plan_sha256','subject_sha256','video_sha256','frame_ids','width','height','fps','video_bounds'):assert r[k]==x['record'][k],k
        inventory.append(dict(case_id=r['probe_alias'],dataset=r['dataset'],cached=x if ok else None,GT_read=False))
    put(BASE/'SOURCE_CACHE_INVENTORY.json',dict(status='locked',count=256,records=inventory,registries=registries,cache_candidates=sum(x['cached']is not None for x in inventory),selection_already_fixed=True,GT_read=False))
    exposure=[]
    for reg in registries:
        old=read(ROOT/reg['path']/'ROSTER_LOCK.json')['records'];parents={(r['dataset'],r['parent_sha256'])for r in old};exposure.append(dict(registry=reg['path'],parent_overlap=sum((r['dataset'],r['parent_sha256'])in parents for r in rows),query_overlap=sum(any(o['dataset']==r['dataset']and o['query_key_sha256']==r['query_key_sha256']for o in old)for r in rows)))
    put(BASE/'EXPOSURE_READBACK.json',dict(status='pass',count=256,original16_parent_overlap=0,known_prior_exposure=exposure,global_history_not_exhaustively_unseen=True,GT_read=False))
    arms=['Full_E1','A','B','C','FULL_ORDER','A_ORDER','A_BT','B_BT','B_G10','B_GR11'];put(BASE/'DESIGN_LOCK.json',dict(status='locked',count=256,per_domain128=True,common_cohort=True,unique_spatial_arms=arms,shared_baselines=['Frozen','Temporal'],unique_WHO_inputs=['Full_E1','A','B','C','FULL_ORDER','A_ORDER'],all_current_round_experiments_included=True,one_query_per_parent=True,small16_excluded=True,original_parameters_frozen=True,prompt_versions=1,no_GT_online=True,bootstrap=False,CURRENT_change=False,official_fullset=False,GT_read=False))
    for i in range(16):
        rr=[r for ds in ('hc2','vidstg')for r in [x for x in rows if x['dataset']==ds][i*8:(i+1)*8]];assert len(rr)==16;put(BASE/'shards'/f'{i:02d}'/'ROSTER_LOCK.json',dict(status='locked',count=16,records=rr,execution_shard_only=True,GT_read=False))
    original=read(ROOT/'artifacts/stvg_final_table1_v1/SOURCE_RUNTIME.json');pins={str(q.relative_to(ROOT)):sha(q)for q in [Path(__file__),ROOT/'protocols/stvg_spatial_controls256_v1.md',BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CACHE_INVENTORY.json',BASE/'DESIGN_LOCK.json',SMALL/'ROSTER_LOCK.json',ROOT/'scripts/run_stvg_table1_source_v1.py',ROOT/'scripts/run_decota_paper_main_v1.py']};put(BASE/'SOURCE_RUNTIME.json',dict(status='locked',count=256,pins=pins,original_Source_runtime_sha256=sha(ROOT/'artifacts/stvg_final_table1_v1/SOURCE_RUNTIME.json'),source_checkpoints=original['source_checkpoints'],GT_read=False))
    put(BASE/'STATUS.json',dict(status='prepared',stage='source_capture',count=256,GT_read=False));print(json.dumps(dict(status='locked',count=256,domain_counts={'hc2':128,'vidstg':128},cache_candidates=sum(x['cached']is not None for x in inventory),query_types={ds:dict(collections.Counter(r['query_type']for r in rows if r['dataset']==ds))for ds in ('hc2','vidstg')})))
def verify():
    r=read(BASE/'SOURCE_RUNTIME.json');assert r['status']=='locked'and not r['GT_read']
    for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
def source():
    import torch,numpy as np
    from scripts import run_stvg_table1_source_v1 as adapter
    from scripts.run_decota_paper_main_v1 import gpu,model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.stvg_native_instance_common_v1 import write
    verify();adapter.verify_runtime();sys.addaudithook(guard);lease=gpu();torch.set_num_threads(2);torch.manual_seed(20261016);np.random.seed(20261016);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    inventory={r['case_id']:r for r in read(BASE/'SOURCE_CACHE_INVENTORY.json')['records']};rows=read(BASE/'ROSTER_LOCK.json')['records'];files=[];rebound=new=0;model=None;domain=None;tick=time.perf_counter()
    for rec in rows:
        cid=rec['probe_alias'];dest=ROOT/rec['source_case_path'];receipt=dest.with_suffix('.receipt.json');cache=inventory[cid]['cached'];provenance={};started=time.perf_counter()
        if dest.exists()or receipt.exists():raise RuntimeError('Existing partial capture must be independently reviewed: '+cid)
        if cache:
            f=cache['file'];p=ROOT/f['path'];assert sha(p)==f['sha256']and sha(p.with_suffix('.receipt.json'))==cache['receipt_sha256'];c=torch.load(p,map_location='cpu',weights_only=False);assert adapter.content_sha256(c)==c['source_case_sha256']==f['source_case_sha256']and c['source_checkpoint_sha256']==CP[rec['dataset']]and c['query_key_sha256']==rec['query_key_sha256']and c['frame_ids']==rec['frame_ids']and not c['GT_read']
            oldid=c['case_id'];oldsha=c['source_case_sha256'];body=lambda x:{k:v for k,v in adapter._canonical(x).items()if k not in ('case_id','source_case_sha256')};before=hashlib.sha256(json.dumps(body(c),sort_keys=True).encode()).hexdigest();c['case_id']=cid;c['source_case_sha256']=adapter.content_sha256(c);assert before==hashlib.sha256(json.dumps(body(c),sort_keys=True).encode()).hexdigest();provenance=dict(kind='metadata_only_alias_rebound',original_case_id=oldid,original_source_case_sha256=oldsha,original_file_sha256=f['sha256'],non_id_payload_sha256=before);rebound+=1
        else:
            if domain!=rec['dataset']:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                model=model_for(adapter.SOURCE_MODEL_NAME[rec['dataset']]);domain=rec['dataset']
            c=adapter.capture_one(model,rec,collections.OrderedDict());assert c['source_checkpoint_sha256']==CP[rec['dataset']]and state_hash(model.state_dict())==c['source_model_state_sha256'];provenance=dict(kind='actual_new_frozen_Source_capture');new+=1
        assert c['case_id']==cid and c['query_key_sha256']==rec['query_key_sha256']and adapter.content_sha256(c)==c['source_case_sha256']
        dest.parent.mkdir(parents=True,exist_ok=True);torch.save(c,dest);put(receipt,dict(status='sealed',case_id=cid,query_key_sha256=rec['query_key_sha256'],sha256=sha(dest),bytes=dest.stat().st_size,source_case_sha256=c['source_case_sha256'],source_runtime_sha256=sha(BASE/'SOURCE_RUNTIME.json'),provenance=provenance,GT_read=False));files.append(dict(case_id=cid,path=str(dest.relative_to(ROOT)),sha256=sha(dest),bytes=dest.stat().st_size,source_case_sha256=c['source_case_sha256'],receipt_path=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt),provenance=provenance,wall_seconds=time.perf_counter()-started));write(BASE/'SOURCE_STATUS.json',dict(status='running',stage='source_capture',done=len(files),total=256,new_captures=new,metadata_rebounds=rebound,GT_read=False),mutable=True);print('SOURCE_CAPTURE_PROGRESS',len(files),256,new,rebound,flush=True);del c;gc.collect()
    put(BASE/'SOURCE_CAPTURE_BARRIER.json',dict(status='complete',count=256,files=files,actual_new_captures=new,metadata_only_rebounds=rebound,wall_seconds=time.perf_counter()-tick,GT_read=False));print('SOURCE256_SEALED',new,rebound)
def supervise():
    from scripts import launch_stvg_instance_identity_v1 as res
    res.BASE=BASE;res.UNIT=UNIT;res.STAGE='source256';g=1024**3;v=res.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=res.monitor,args=(stop,),daemon=True).start();out=BASE/'execution/source';out.mkdir(parents=True,exist_ok=False)
    try:
        with(out/'worker.log').open('xb')as log:
            t=time.perf_counter();p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__)),'source'],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=False));rc=p.wait()
        put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0
    finally:stop.set()
if __name__=='__main__':{'setup':setup,'source':source,'supervise':supervise}[sys.argv[1]]()
