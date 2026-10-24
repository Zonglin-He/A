"""Full-set current-MAP structural controls; original raw collector reused read-only."""
import common as shared_common
from common import *
import argparse,gc,traceback,subprocess
from spatial_core import ARMS
SUITE=D;D=SUITE/'spatial';P=P/'spatial'
def require_gate():
 assert read(SUITE/'spatial_design_bridge.json')['status']=='passed'
 assert all(read(R/'outputs/cgstvg_full_v1'/f'{ds}_publication.json')['status']=='complete_published'for ds in COUNTS)
def validate():
 p=shared_common.validate('spatial');return {**p,'budget':dict(max_domain_seconds=259200,min_disk_free_bytes=80*2**30,max_private_bytes=160*2**30)}
def save_torch(path,value):
 import torch
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);os.replace(tmp,path)
def rows(ds):return records(ds)

def predict(ds):
    require_gate();p=validate()
    import torch,numpy as np
    from stvg.runtime import setup,lease,load_source
    from stvg.native import frozen_native_forward
    from stvg.evidence import extract_visual_grids
    from stvg.representation import observe_native_grid,extract
    from stvg.visual_prefix import native_appearance_grid
    from stvg.visual_cache import VisualCache
    from stvg.decode import decode
    from stvg.query import QuerySubjectParser
    from stvg.spatial import SpatialGeometry,evidence_dp
    from spatial_core import derive,replay
    setup();gpu=lease();who=identity();atomic(D/f'{ds}_worker.json',who)
    fullseal=read(RUN/ds/'sealed.json');shared=read(SUITE/'raw_reuse_index.json')[ds];legacy=read(RUN/'reuse_index.json')[ds]
    cfg=read(R/'configs/default.json')['datasets'][ds];runtime=read(RUN/'plan.json')['provider_runtime_fingerprint']
    model=None;parser=None;visual=VisualCache(2*2**30);files={};start=time.perf_counter();model_load_seconds=0.
    counts=dict(queries=0,shared_raw_grid_hits=0,legacy_raw_grid_hits=0,full_Source_queries=0,appearance_prefix_queries=0,new_expert=0)
    try:
        for i,row in enumerate(rows(ds),1):
            assert time.perf_counter()-start<p['budget']['max_domain_seconds'];assert shutil.disk_usage(R).free>p['budget']['min_disk_free_bytes']
            if i%64==1:assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<p['budget']['max_private_bytes']
            key=row['id'];e=fullseal['files'][key];oldcp=RUN/ds/e['capture_path'];oldpred=RUN/ds/e['path'];oldrec=RUN/ds/e['receipt_path']
            assert sha(oldcp)==e['capture_sha256']and sha(oldpred)==e['sha256']and sha(oldrec)==e['receipt_sha256']
            assert read(oldrec)['input_binding']==binding(row,runtime)
            cap=torch.load(oldcp,map_location='cpu',weights_only=False,mmap=True);assert not cap['GT_read']
            dest=D/ds;out=dest/'predictions'/(key+'.npz');compactpath=dest/'compact'/(key+'.pt');rp=dest/'receipts'/(key+'.json')
            if rp.exists():
                rec=read(rp);assert rec['fingerprint']==p['fingerprint']and sha(out)==rec['sha256']and sha(compactpath)==rec['compact_sha256']
                files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));counts['queries']+=1;continue
            assert not out.exists()and not compactpath.exists(),'Preserve incomplete unbound outputs, no blind retry'
            tick=time.perf_counter();scratch=D/'active'/ds/key;rawfile=scratch/'raw.pt';rawmeta=scratch/'raw.json';attempt=scratch/'attempt.json'
            raw=None;origin={};source_seconds=0.;mode=None
            if key in shared:
                proof=shared[key];path=Path(proof['path']);assert sha(path)==proof['sha256']
                raw=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                assert raw['parent_capture_sha256']==e['capture_sha256']and raw['frame_ids']==cap['frame_ids']and not raw['GT_read']
                mode='shared_raw';origin=dict(path=str(path),sha256=proof['sha256']);counts['shared_raw_grid_hits']+=1
            elif rawmeta.exists():
                meta=read(rawmeta);assert sha(rawfile)==meta['sha256']and meta['parent_capture_sha256']==e['capture_sha256']
                raw=torch.load(rawfile,map_location='cpu',weights_only=False,mmap=True);mode='paid_raw_recovered';origin=meta
            else:
                assert not attempt.exists(),'Unresolved paid Source attempt: preserve, no blind rerun'
                # Only exact input-bound raw inputs may be reused. Compact candidate
                # features are not a full postfusion grid and cannot substitute.
                prior=legacy.get(key)
                if prior:
                    assert prior['input_binding']==binding(row,runtime)and sha(prior['path'])==prior['sha256']
                    old=torch.load(prior['path'],map_location='cpu',weights_only=False,mmap=True)
                    if 'spatial_inputs'in old and 'records'in old and ('native_logits'in old or 'logits'in old):
                        logs=old.get('native_logits',[z[-1]for z in old.get('logits',[])])
                        assert np.array_equal(old['Source'],cap['Source'])and old['frame_ids']==cap['frame_ids']
                        assert all(torch.equal(a,b)for a,b in zip(logs,cap['native_logits']))and len(logs)==2
                        spatial,valid,ids,shape,_=extract_visual_grids(old['spatial_inputs'],cap['frame_ids'])
                        raw=dict(frame_ids=ids,spatial_features=spatial,spatial_valid=valid,spatial_grid_shape=shape)
                        if 'native_local_grid'in old:raw['native_local_grid']=old['native_local_grid'];mode='legacy_both_raw'
                        else:mode='legacy_spatial_need_prefix'
                        origin=dict(path=prior['path'],sha256=prior['sha256'])
                    del old
                if raw is None or 'native_local_grid'not in raw:
                    if model is None:
                        load_tick=time.perf_counter();model=load_source(cfg);model_load_seconds=time.perf_counter()-load_tick
                        atomic(D/f'{ds}_model_load.json',dict(worker=who,seconds=model_load_seconds,loads=1,nested_in_first_query_seconds=True))
                    if parser is None:parser=QuerySubjectParser(R/'checkpoints/stanza')
                    frames,ids=decode(row['metadata'],ds);assert ids==cap['frame_ids']
                    scratch.mkdir(parents=True,exist_ok=True);atomic(attempt,dict(worker=who,mode=mode or'need_full_Source',before_Source=True,GT_read=False))
                    st=time.perf_counter();torch.manual_seed(20260920);np.random.seed(20260920)
                    if raw is not None:
                        with visual.context(model,frames,ids):grid=native_appearance_grid(model,frames,ids,row['metadata'])
                        raw['native_local_grid']=grid;counts['appearance_prefix_queries']+=1;mode='appearance_prefix'
                    else:
                        model.cuda()
                        with observe_native_grid(model)as grid,visual.context(model,frames,ids):
                            batch,recs,base=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses={'subject':parser(row['metadata']['caption'])['subject']}))
                        assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
                        assert all(torch.equal(a.cpu(),b)for a,b in zip(base.zero['logits'],cap['native_logits']))
                        assert [r['frame_ids']for r in recs]==[r['frame_ids']for r in cap['records']]
                        spatial,valid,_,shape,_=extract_visual_grids(base.spatial_inputs,ids)
                        raw=dict(frame_ids=ids,native_local_grid=grid,spatial_features=spatial,spatial_valid=valid,spatial_grid_shape=shape)
                        del batch,base;model.cpu();counts['full_Source_queries']+=1;mode='full_Source'
                    source_seconds=time.perf_counter()-st;raw['GT_read']=False;save_torch(rawfile,raw)
                    origin=dict(sha256=sha(rawfile),parent_capture_sha256=e['capture_sha256'],mode=mode,Source_seconds=source_seconds,worker=who,GT_read=False)
                    atomic(rawmeta,origin);del frames;gc.collect();torch.cuda.empty_cache()
                else:counts['legacy_raw_grid_hits']+=1
            assert raw['frame_ids']==cap['frame_ids']
            # Full-control ROI pool must be exact for every materialized raw grid.
            geo=SpatialGeometry.from_cached_geometry(cap['geometry']);aux=evidence_dp(geo.for_interval([cap['frame_ids'][0],cap['frame_ids'][-1]+1]))['final'].numpy()
            pooled,_=extract(raw,aux);assert np.array_equal(pooled,cap['content_features'])
            arrays,compact=derive(cap,raw)
            with np.load(current_MAP(ds,key)[0])as z:
                assert np.array_equal(arrays['FullMAP'],z['FullMAP'])and np.array_equal(arrays['FullMAP_interval'],z['FullMAP_interval'])
            compact.update(parent_capture_sha256=e['capture_sha256'],raw_origin=origin)
            save_torch(compactpath,compact);cold=torch.load(compactpath,map_location='cpu',weights_only=False,mmap=True)
            again=replay(cold);assert all(np.array_equal(arrays[k],again[k])for k in arrays)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.with_suffix('.npz.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
            os.replace(out.with_suffix('.npz.tmp'),out)
            rec=dict(path=str(out.relative_to(D)),sha256=sha(out),compact_path=str(compactpath.relative_to(D)),compact_sha256=sha(compactpath),
                fingerprint=p['fingerprint'],input_binding=binding(row,runtime),parent_capture_sha256=e['capture_sha256'],raw_origin=origin,mode=mode,
                full_pool_and_outputs_bitwise=True,all_compact_replay_exact=True,worker=who,Source_seconds=source_seconds,
                query_seconds=time.perf_counter()-tick,new_expert=0,GT_read=False)
            atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));counts['queries']+=1
            if rawfile.exists():
                assert sha(rawfile)==read(rawmeta)['sha256'];rawfile.unlink();atomic(scratch/'disposed.json',dict(raw_sha256=read(rawmeta)['sha256'],compact_sha256=rec['compact_sha256'],cold_replay_exact=True))
            atomic(D/f'{ds}_progress.json',dict(completed=i,total=COUNTS[ds],counts=counts,seconds=time.perf_counter()-start,worker=who,GT_read=False));print(ds,i,mode,flush=True)
            del cap,raw,arrays,compact,cold,again,geo,pooled;gc.collect()
        assert len(files)==COUNTS[ds]
        assert sum(e['mode']=='shared_raw'for e in files.values())==len(shared)
        assert counts['full_Source_queries']+counts['appearance_prefix_queries']<=COUNTS[ds]-len(shared)
        atomic(D/f'{ds}_sealed.json',dict(queries=len(files),files=files,counts=counts,worker=who,seconds=time.perf_counter()-start,model_load_seconds=model_load_seconds,GT_read=False,fingerprint=p['fingerprint']))
    finally:gpu.close()
def prepare():
 files=[Path(__file__),S/'spatial_core.py',S/'common.py',R/'studies/strict_deletion_v1/run.py',R/'stvg/spatial.py',R/'stvg/evidence.py',R/'stvg/representation.py']
 p=shared_common.freeze('spatial',dict(arms=ARMS,queries=COUNTS,source='Reuse same parent native Source/E/8teacher; raw Source/prefix acquisition ONLY when original grids missing, max11821combined.',NoSharedCorrection='E=Source; reconstruct geometry/auxDP/actualROIpool/q1MAP/finalDP.',PastOnly='Delete future-anchor support from unchanged original bidirectional candidate fits; observed own anchor retained; rebuild unary/links/auxpool/q1/finalDP. This tests past-only availability, not a causal-online model.',OneWayAffinity='Same both anchors; fit a→t affine using only at*seed instead of at*ta.T*seed. All geometry/auxpool/q1/finalDP rebuilt. Source feature grids and all other fitting constraints unchanged.',AuxIndependent='Same full-span original candidatebank/validity/unary; per-frame unaryargmax removes only first auxiliary links/DP, then actualrepool/q1MAP/finalDP.',AuxSourceROI='Use original Source ROI for appearancepool; all downstream q1MAP/finalDP rebuilt.',AuxE_ROI='Use existing E ROI for appearancepool; all downstream q1MAP/finalDP rebuilt.',NoSpatialTrajectory='No candidate propagation/auxpath/finalDP: existing E output and E-ROI repool→q1MAP.',NoTemporalInference='Delete visualposterior revision and current jointMAP temporal branch; nativeperoffsetMAPphysicalhull restored, spatialDP recomputed there. Separate matched q0jointMAP S1T0 isolates no posterior update without readout confound.',factorial='Keep existing shared E. S0T0=Eboxes/q0jointMAP; S1T0=geometryDP/q0jointMAP; S0T1=Eboxes/Epoolq1MAP; S1T1=currentFullMAP. Interaction=(S11-S10)-(S01-S00), paired parent CI; do not assume superadditivity.',fixed='Original Source224FP32/native two-offset grids/same8/k64beta1context1.5/one round; every FullMAP and original repool exactly match parent. Allsets exposed; no outcome selection.'),files)
 atomic(SUITE/'spatial_state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def preflight():
 import torch,numpy as np
 from spatial_core import derive,replay
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);p=validate();proof=[]
 for ds in COUNTS:
  lookup={r['id']:r for r in records(ds)}
  for key,e in list(read(SUITE/'raw_reuse_index.json')[ds].items())[:2]:
   row=lookup[key];c,parent=load_parent(ds,row);assert sha(e['path'])==e['sha256'];raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);a,compact=derive(c,raw);b=replay(compact);assert all(np.array_equal(a[k],b[k])for k in a)
   with np.load(current_MAP(ds,key)[0])as z:assert np.array_equal(a['FullMAP'],z['FullMAP'])and np.array_equal(a['FullMAP_interval'],z['FullMAP_interval'])
   proof.append(dict(dataset=ds,parent=parent,all_arrays_exact=True,FullMAP_exact=True))
 atomic(D/'preflight_receipt.json',dict(status='passed',cases=4,proof=proof,GT_read=False,new_Source=0,new_expert=0,new_GPU=0,fingerprint=p['fingerprint']));atomic(P/'preflight_receipt.json',dict(status='passed',cases=4,all_arrays_coldreplay_exact=True,FullMAP_bitwise=True,GT_read=False,new_Source=0,new_expert=0,new_GPU=0,fingerprint=p['fingerprint']))
def pairs():return [('FullMAP',a)for a in ARMS if a not in ['FullMAP','S1T1']]+[('S1T1','S1T0'),('S0T1','S0T0'),('S1T0','S0T0')]
def score(ds):
 import torch,numpy as np
 from full_score import metric,cluster
 torch.set_num_threads(1);p=validate();seal=read(D/f'{ds}_sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=labels(ds);rr=[];checks=0;maxerr=0.
 for row in records(ds):
  e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['compact_path'])==e['compact_sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];vals={}
  with np.load(D/e['path'])as z:
   r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for arm in ARMS:vals[arm],err=metric(z[arm],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);assert err<2e-10;maxerr=max(maxerr,err);checks+=1
  rr.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals))
 folder=P/ds;res=parent_report(rr,ARMS,pairs(),folder);signed_tails(rr,pairs(),folder);synthetic=[]
 for r in rr:
  v=r['values'];delta={k:(v['S1T1'][k]-v['S1T0'][k]-v['S0T1'][k]+v['S0T0'][k])if all(v[a][k]is not None for a in ['S1T1','S1T0','S0T1','S0T0'])else None for k in v['S1T1']};synthetic.append(dict(parent=r['parent'],values={'Interaction':delta,'Zero':{k:0. if x is not None else None for k,x in delta.items()}}))
 atomic(folder/'interaction.json',dict(definition='(S1T1-S1T0)-(S0T1-S0T0); statistic, not an accuracy arm',paired=cluster(synthetic,'Interaction','Zero'),no_superadditivity_assumed=True))
 atomic(D/ds/'metrics.json',dict(private_rows=rr,report=res));rec=dict(status='scored',queries=len(rr),checks=checks,max_error=maxerr,seal_sha256=sha(D/f'{ds}_sealed.json'),whole_domain_sealed_before_GT=True,fingerprint=p['fingerprint']);atomic(D/f'{ds}_score_receipt.json',rec);atomic(folder/'score_receipt.json',rec)
def audit(ds):
 import torch,numpy as np
 from spatial_core import replay
 torch.set_num_threads(1);p=validate();seal=read(D/f'{ds}_sealed.json');score=read(D/f'{ds}_score_receipt.json');assert score['seal_sha256']==sha(D/f'{ds}_sealed.json');segments={};checks=0
 for row in records(ds):
  e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['compact_path'])==e['compact_sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];c=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);a=replay(c)
  with np.load(D/e['path'])as z:assert all(np.array_equal(a[k],z[k])for k in a)
  checks+=3;wk=f"{e['worker']['pid']}:{e['worker']['startticks']}";s=segments.setdefault(wk,dict(queries=0,query_seconds=0.,Source_seconds=0.,modes={}));s['queries']+=1;s['query_seconds']+=e['query_seconds'];s['Source_seconds']+=e['Source_seconds'];s['modes'][e['mode']]=s['modes'].get(e['mode'],0)+1
 rr=read(D/ds/'metrics.json')['private_rows'];report=read(P/ds/'report.json')
 for arm in ARMS:
  for f,v in report['means'][arm].items():
   x=[r['values'][arm][f]for r in rr if r['values'][arm][f]is not None];assert(v is None and not x)or abs(v-float(np.mean(x)))<1e-12
 rec=dict(status='verified_whole_domain',queries=COUNTS[ds],SHA_checks=checks,all_arrays_coldreplay_exact=True,all_means_recomputed=True,worker_segments=segments,score_sha256=sha(D/f'{ds}_score_receipt.json'),scope='Actual paidSource/prefix segments plus cachedCPUgeometry/repool cost; oldacquisition notzero deployment, nested times not added.',new_GT_scoring=0,new_expert=0);atomic(D/f'{ds}_reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec)
def domain(ds):
 who=identity();validate()
 try:
  for command,suffix in [('predict','sealed'),('score','score_receipt'),('audit','reporting_receipt')]:
   assert not(D/f'{ds}_{suffix}.json').exists(),'No blind restart';atomic(D/f'{ds}_state.json',dict(status=command,worker=who));log=(D/f'{ds}_{command}.log').open('ab');c=subprocess.Popen([sys.executable,'-B',__file__,command,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);atomic(D/f'{ds}_{command}_child.json',identity(c.pid));rc=c.wait();log.close();atomic(D/f'{ds}_{command}_exit.json',dict(code=rc));assert rc==0,'Preserve all paidraw and outputs; no retry/skip'
  atomic(D/f'{ds}_state.json',dict(status='complete_scored_audited',worker=who))
 except BaseException:atomic(D/f'{ds}_state.json',dict(status='failed_preserved',worker=who,error=traceback.format_exc()));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight','predict','score','audit','domain']);a.add_argument('--dataset',choices=list(COUNTS));z=a.parse_args();dict(prepare=prepare,preflight=preflight,predict=lambda:predict(z.dataset),score=lambda:score(z.dataset),audit=lambda:audit(z.dataset),domain=lambda:domain(z.dataset))[z.command]()
