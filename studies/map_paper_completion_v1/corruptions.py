"""Genuine controlled-view currentMAP recomputation, source checkpoint before teacher."""
from common import *
import argparse,gc,traceback,inspect,textwrap,copy
G=D/'corruptions';TRANSFORMS=['Blur','JPEG','Dim']
def prepare():
 files=[Path(__file__),S/'common.py',S/'live_map.py',R/'studies/paper_review_suite_v1/shifts.py',R/'stvg/predictor.py',R/'stvg/teacher_bridge.py']
 p=freeze('corruptions',dict(cohort='Existing confirm128queries/128parents/domain, identical frozen selection; bothdomains all768 queryviews beforeGT.',transforms=TRANSFORMS,definitions='Original shifts.transform: Blur5x5sigma1.5, JPEGquality30subsampling2, Dim0.5 uint8.',main='NativeSource/currentFullMAP actualsameRGB/native224FP32/same8teacher; q1jointMAP then realnewintervalDP.',reuse='Old Source/FullMBR finalarrays NOT reused as new Full; old teacher payloads reused only exactRGB/contextSHA. New nativeSource must bitwise old sameviewSource, and same upstream q1/geometry reproduce oldMBR full only as hidden compatibility check, no MBR comparison report.',checkpoint='Every new paidSource raw state persisted withSHA before teacher. Old teacher cache read-only; newteacher misses writtenonlynewnamespace. Fullcompact coldreplay and committedpred/receipt before onlyowncompletedrawscratchdisposal.',evaluation='Pertransform source-native/currentMAP vs clean-source-native/currentMAP on identicalconfirmqueries; independentofficialscore afterall768 seal, allsigned/tails/timegroupsCI/cost. No corruption-specific tuning, no unsupported naturalrobustness claim.'),files);atomic(G/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def klass():
 import stvg.predictor as base
 from map_core import map_interval
 code=textwrap.dedent(inspect.getsource(base.BoxFixedPredictor.capture));assert code.count("native_interval=final_mbr(U,q)['interval']")==1;code=code.replace("native_interval=final_mbr(U,q)['interval']",'native_interval=map_interval(U,q)')
 needle='    if teacher_override is not None:';assert code.count(needle)==1
 callback="    self.acquisition_checkpoint(dict(Source=src.numpy(),frame_ids=U,planned_positions=positions,spatial_inputs=spatial,native_logits=logits,native_local_grid=native_grid,records=recs,posterior={'Native':q},intervals={'Native':native_interval},Source_native_interval=native['physical_interval'],cost=copy.deepcopy(costs),GT_read=False))\n"
 code=code.replace(needle,callback+needle);scope={**vars(base),'map_interval':map_interval,'copy':copy};exec(compile(code,'<current_MAP_capture_paid_Source_checkpoint>','exec'),scope);return type('CheckpointedCurrentMAP',(base.BoxFixedPredictor,),{'capture':scope['capture']})
class Teacher:
 def __init__(self,fingerprint):
  from stvg.teacher_bridge import ResidentTeacher
  self.fingerprint=fingerprint;self.live=ResidentTeacher(cache_dir=G/'teacher_cache',fingerprint=fingerprint);self.last=None
 def __call__(self,frames,caption,**kw):
  from stvg.teacher_bridge import evidence_key
  import torch
  key,context=evidence_key(frames,caption,fingerprint=self.fingerprint,**kw);old=OLD/'shift_teacher_cache';rp=old/(key+'.json');pp=old/(key+'.pt')
  if rp.exists():
   rec=read(rp);assert rec['key']==key and rec['context']==context and sha(pp)==rec['sha256'];z=torch.load(pp,map_location='cpu',weights_only=False);z=copy.deepcopy(z);z['evidence_original_cost']=copy.deepcopy(z['cost']);z['cost']={k:0. for k in z['cost']};z.update(cache_hit=True,cache_key=key,model_load_seconds=0.,process_wall_seconds=0.,query_forwards=0,peak_GPU_bytes=0);self.last=dict(path=str(pp),sha256=rec['sha256'],receipt=str(rp),receipt_sha256=sha(rp),old_acquisition_reused=True,original_cost=z['evidence_original_cost']);return z
  z=self.live(frames,caption,**kw);rp=G/'teacher_cache'/(key+'.json');pp=G/'teacher_cache'/(key+'.pt');self.last=dict(path=str(pp),sha256=sha(pp),receipt=str(rp),receipt_sha256=sha(rp),old_acquisition_reused=False,original_cost=z.get('evidence_original_cost',z['cost']));return z
 def close(self):self.live.close()
def save(path,value):
 import torch
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)
def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.query import QuerySubjectParser
 from stvg.readout import prediction
 from stvg.temporal import final_mbr
 from stvg.spatial import evidence_dp
 from studies.paper_review_suite_v1.shifts import transform
 from live_map import finish
 p=validate('corruptions');assert not(G/'worker.json').exists(),'No blind modelwork restart';setup();gpu=lease();who=identity();atomic(G/'worker.json',who);Klass=klass();selection=read(D/'selection.json');oldseal=read(OLD/'shifts_confirm_sealed.json');fingerprint=read(OLD/'shifts_plan.json')['runtime_fingerprint'];files={};counts=dict(queryviews=0,Source=0,new_expert=0,old_teacher_hits=0);start=time.perf_counter()
 try:
  for ds,groups in selection.items():
   files[ds]={};model=Klass(ds);model.source=load_source(model.config);model.source.cpu();teacher=Teacher(fingerprint);model.teacher_service=teacher;parser=QuerySubjectParser(R/'checkpoints/stanza')
   try:
    for row in groups['confirm']:
     frames,ids=decode(row['metadata'],ds);originalsha=hashlib.sha256(frames.tobytes()).hexdigest();parses=dict(subject=parser(row['metadata']['caption'])['subject']);files[ds][row['id']]={}
     for name in TRANSFORMS:
      assert time.perf_counter()-start<259200;assert __import__('shutil').disk_usage(R).free>80*2**30;key=row['id']+'.'+name;dest=G/ds;out=dest/'predictions'/(key+'.npz');cp=dest/'compact'/(key+'.pt');rp=dest/'receipts'/(key+'.json');scratch=G/'active'/ds/key;raw=scratch/'Source.pt';meta=scratch/'Source.json';attempt=scratch/'attempt.json';assert not attempt.exists()and not out.exists()and not cp.exists(),'Preserve existingpaidattempt, no blindrepeat';atomic(attempt,dict(worker=who,before_Source=True,GT_read=False));view=transform(frames,name);rgbsha=hashlib.sha256(view.tobytes()).hexdigest();old=oldseal['files'][ds][row['id']][name];assert originalsha==old['original_RGB_sha256']and rgbsha==old['transformed_RGB_sha256'];tick=time.perf_counter()
      def checkpoint(z):
       save(raw,z);atomic(meta,dict(path=str(raw),sha256=sha(raw),transformed_RGB_sha256=rgbsha,worker=who,Source_completed=True,GT_read=False));counts['Source']+=1
      model.acquisition_checkpoint=checkpoint;torch.manual_seed(20260920);np.random.seed(20260920);cap=model.capture(view,ids,row['metadata'],parses=parses,collect_mask_diagnostics=True);result=finish(cap);native=prediction(cap['native_logits'],torch.as_tensor(cap['Source']),cap['records'],ids)['physical_interval']
      assert sha(OLD/old['path'])==old['sha256']
      with np.load(OLD/old['path'])as z:
       assert np.array_equal(cap['Source'],z['Source'])and np.array_equal(ids,z['frame_ids'])and np.array_equal(cap['planned_positions'],z['positions8']);iv=final_mbr(ids,result['posterior'])['interval'];hidden=evidence_dp(cap['geometry'].for_interval(iv))['final'].numpy();assert np.array_equal(iv,z['Full_interval'])and np.array_equal(hidden,z['Full']),'Original sameview pipeline compatibility failed'
      compact=dict(Source=np.asarray(cap['Source']),Source_interval=native,frame_ids=ids,positions8=cap['planned_positions'],geometry=cap['geometry'].state_dict(),content_features=result['content_features'],native_prior=cap['posterior']['Native'],GT_read=False);save(cp,compact);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);again=replay(cold);arrays=dict(Source=compact['Source'],Source_interval=np.asarray(native),FullMAP=result['boxes'],FullMAP_interval=np.asarray(result['interval']),frame_ids=np.asarray(ids),positions8=np.asarray(cap['planned_positions']));assert all(np.array_equal(arrays[k],again[k])for k in arrays);out.parent.mkdir(parents=True,exist_ok=True)
      with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
      os.replace(out.with_suffix('.tmp'),out)
      with np.load(out)as z:assert all(np.array_equal(arrays[k],z[k])for k in arrays)
      e=dict(path=str(out.relative_to(D)),sha256=sha(out),compact_path=str(cp.relative_to(D)),compact_sha256=sha(cp),raw_Source_original_sha256=sha(raw),teacher=teacher.last,worker=who,transformed_RGB_sha256=rgbsha,original_RGB_sha256=originalsha,Source_cost=cap['cost'],finish_cost=result['cost'],query_seconds=time.perf_counter()-tick,GT_read=False,all_arrays_coldreplay_exact=True);atomic(rp,e);files[ds][row['id']][name]=dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));counts['queryviews']+=1;counts['new_expert']+=cap['cost']['ReferDINO_query_forwards'];counts['old_teacher_hits']+=int(teacher.last['old_acquisition_reused']);assert sha(raw)==read(meta)['sha256'];raw.unlink();atomic(scratch/'disposed.json',dict(raw_sha256=e['raw_Source_original_sha256'],committed_receipt_sha256=sha(rp),exact_replay=True));atomic(G/'progress.json',dict(counts=counts,total=768,worker=who,GT_read=False));print(ds,row['id'],name,counts,flush=True);del view,cap,result,arrays,compact,cold,again,hidden;gc.collect();torch.cuda.empty_cache()
     del frames
   finally:teacher.close();del model;gc.collect();torch.cuda.empty_cache()
  assert counts['queryviews']==768;atomic(G/'sealed.json',dict(queries=256,queryviews=768,files=files,counts=counts,GT_read=False,worker=who,seconds=time.perf_counter()-start,fingerprint=p['fingerprint']))
 except BaseException:atomic(G/'state.json',dict(status='failed_preserved',error=traceback.format_exc(),worker=who));raise
 finally:gpu.close()
def replay(c):
 import numpy as np
 from stvg.prototype import one_step
 from stvg.spatial import SpatialGeometry,evidence_dp
 from map_core import map_interval
 q,_=one_step(c['content_features'],c['native_prior']);iv=map_interval(c['frame_ids'],q);g=SpatialGeometry.from_cached_geometry(c['geometry']);return dict(Source=c['Source'],Source_interval=np.asarray(c['Source_interval']),FullMAP=evidence_dp(g.for_interval(iv))['final'].numpy(),FullMAP_interval=np.asarray(iv),frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']))
def score():
 import torch,numpy as np
 from full_score import metric
 torch.set_num_threads(1);validate('corruptions');seal=read(G/'sealed.json');assert seal['queryviews']==768 and not seal['GT_read'];selection=read(D/'selection.json');checks=0;maxerr=0.
 for ds,groups in selection.items():
  truth=labels(ds);clean={r['id']:r for r in read(MAP/'TA'/ds/'metrics.json')['private_rows']}
  for name in TRANSFORMS:
   rr=[]
   for row in groups['confirm']:
    e=seal['files'][ds][row['id']][name];assert sha(D/e['path'])==e['sha256'];g=truth[row['id']];vals={}
    with np.load(D/e['path'])as z:
     r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
     for arm in ['Source','FullMAP']:vals[arm],err=metric(z[arm],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);assert err<2e-10;maxerr=max(maxerr,err);checks+=1
    vals.update(CleanSource=clean[row['id']]['values']['Source'],CleanFullMAP=clean[row['id']]['values']['FullMAP']);rr.append(dict(id=row['id'],parent=str(g['parent']),duration_group=clean[row['id']]['duration_group'],values=vals))
   folder=P/'corruptions'/ds/name;pairs=[('FullMAP','Source'),('Source','CleanSource'),('FullMAP','CleanFullMAP')];res=parent_report(rr,['Source','FullMAP','CleanSource','CleanFullMAP'],pairs,folder);signed_tails(rr,pairs,folder);atomic(G/ds/(name+'_metrics.json'),dict(private_rows=rr,report=res))
 atomic(G/'score_receipt.json',dict(status='scored',checks=checks,max_error=maxerr,whole768sealed_before_GT=True,seal_sha256=sha(G/'sealed.json')))
def audit():
 import torch,numpy as np
 torch.set_num_threads(1);validate('corruptions');seal=read(G/'sealed.json');score=read(G/'score_receipt.json');assert score['seal_sha256']==sha(G/'sealed.json');checks=0
 for ds,rows in seal['files'].items():
  for key,views in rows.items():
   for name,e in views.items():
    for f,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(D/e[f])==e[h];checks+=1
    t=e['teacher'];assert sha(t['path'])==t['sha256']and sha(t['receipt'])==t['receipt_sha256'];c=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);a=replay(c)
    with np.load(D/e['path'])as z:assert all(np.array_equal(a[k],z[k])for k in a)
 for ds in COUNTS:
  for name in TRANSFORMS:
   rr=read(G/ds/(name+'_metrics.json'))['private_rows'];r=read(P/'corruptions'/ds/name/'report.json')
   for arm,fields in r['means'].items():
    for f,v in fields.items():
     vv=[x['values'][arm][f]for x in rr if x['values'][arm][f]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 result=dict(status='verified_complete_corruptions_currentMAP',queries=256,queryviews=768,SHA_checks=checks,all_compact_arrays_exact=True,all_means_recomputed=True,score_sha256=sha(G/'score_receipt.json'),actual_counts=seal['counts'],cost_scope='ActualSource/teacher/cachedacquisition costs in own perview receipts. Oldteacherhits inherit nonzero provenance; no cache-based deployment claim.',new_GT_scoring=0);atomic(G/'reporting_receipt.json',result);atomic(P/'corruptions/reporting_receipt.json',result);atomic(G/'state.json',dict(status='complete_scored_audited'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','predict','score','audit']);z=a.parse_args();dict(prepare=prepare,predict=predict,score=score,audit=audit)[z.command]()
