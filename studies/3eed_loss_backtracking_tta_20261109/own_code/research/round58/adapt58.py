"""R57 targets and correspondence fixed; matched original/backtracking CPU fits."""
import argparse,time,torch,collections,sys
from common58 import *
import numerics58 as n

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);out=ROOT/p['prediction_root'];assert not out.exists();out.mkdir(parents=True)
 for folder in ['samples_PRIVATE','fitted_PRIVATE','cases_PRIVATE']:(out/folder).mkdir()
 torch.set_num_threads(4);torch.manual_seed(0);torch.set_float32_matmul_precision('highest');rows=read(ROOT/p['manifest']);initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');source=n.head(initial).requires_grad_(False);source_hash=n.phash(source);seals={key:read(ROOT/p[key]/'SEALED.json') for key in ['AB_root','capture_root','association_root','r57_prediction_root','r56_prediction_root']};cache={};results={};records={};baselines={};old_outputs={};counts=collections.Counter();start=time.perf_counter();bridge={}
 def member(key,rel):assert sha(ROOT/p[key]/rel)==seals[key]['files'][rel];return load(ROOT/p[key]/rel)
 def snapshot(dest,m,arm):
  for name,value in n.state(m).items():dest[arm+'__'+name]=value
 def save_endpoint(sid,arm,m,d,fitted):
  with torch.no_grad():results[sid][arm]=n.ensemble(m,d).numpy().copy()
  snapshot(fitted,m,arm)
 def update_counts(rec,rule):
  counts[rule+'_proposals']+=1;counts[rule+'_accepted']+=int(rec['accepted']);counts[rule+'_trial_evaluations']+=rec['trial_count'];counts[rule+'_loss_increased']+=int(rec['KL_after_same_detached_target']>rec['KL_before']+1e-12)
 try:
  for row in rows:
   sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';v=member('AB_root',rel);c=member('capture_root',rel);g=member('association_root',rel);r57=member('r57_prediction_root',rel);r56=member('r56_prediction_root',rel);d=n.prepared(v,c,g);cache[sid]=d;assert np.array_equal(r57['source_boxes'],r56['source_boxes']) and np.array_equal(v['A_boxes'],r57['source_boxes']) and np.array_equal(r57['source_scores'],r56['source_scores']);base={str(k):value.copy() for k,value in zip(r57['arm_names'],r57['scores'])};old_outputs[sid]=base
   with torch.no_grad():cpu=n.ensemble(source,d).numpy().copy()
   results[sid]={'source':base['source'],'frozen_ABC':base['frozen_ABC'],'cpu_frozen_ABC':cpu,'r56_one':r56['scores'][list(r56['arm_names']).index('tta1')]};baselines[sid]={'boxes':v['A_boxes'],'scores':v['A_scores']};records[sid]={'platform':row['platform'],'episodic':{},'online':{}};fitted={}
   for rule in ['original','backtrack']:
    for kind in ['fixed','ema']:
     s=n.head(initial).requires_grad_(True);t=n.head(initial).requires_grad_(False);opt=n.optimizer(s);trace=[]
     for k in range(1,17):
      rec=n.step(s,t,opt,d,kind,rule);update_counts(rec,rule);trace.append(rec)
      if k in [1,3,16]:
       save_endpoint(sid,f'{rule}_epi_{kind}_student_{k}',s,d,fitted)
       if kind=='ema':save_endpoint(sid,f'{rule}_epi_ema_teacher_{k}',t,d,fitted)
     records[sid]['episodic'][rule+'_'+kind]={'trace':trace,'initial_source_reset':True,'student_final_hash':n.phash(s),'teacher_final_hash':n.phash(t),'optimizer_final_hash':n.ohash(opt)};del s,t,opt
   np.savez_compressed(out/'fitted_PRIVATE'/(sid+'_episodic.npz'),**fitted)
   if len(cache)%16==0:print('EPISODIC58',len(cache),336,'proposals',sum(counts[k+'_proposals'] for k in ['original','backtrack']),flush=True)
  orders=read(ROOT/p['r57_prediction_root']/'ONLINE_ORDERS_PRIVATE.json');dump(out/'ONLINE_ORDERS_PRIVATE.json',orders)
  for rule in ['original','backtrack']:
   for kind in ['fixed','ema']:
    for pl,stream in orders.items():
     assert len(stream)==168;s=n.head(initial).requires_grad_(True);t=n.head(initial).requires_grad_(False);opt=n.optimizer(s)
     for index,sid in enumerate(stream):
      d=cache[sid];entry={'stream_index':index,'student_pre_hash':n.phash(s),'teacher_pre_hash':n.phash(t),'optimizer_pre_hash':n.ohash(opt)};fitted={}
      for timing in ['pre','post']:
       if timing=='post':rec=n.step(s,t,opt,d,kind,rule);update_counts(rec,rule);entry.update(rec)
       save_endpoint(sid,f'{rule}_online_{kind}_student_{timing}',s,d,fitted)
       if kind=='ema':save_endpoint(sid,f'{rule}_online_ema_teacher_{timing}',t,d,fitted)
      entry.update(student_post_hash=n.phash(s),teacher_post_hash=n.phash(t),optimizer_post_hash=n.ohash(opt));records[sid]['online'][rule+'_'+kind]=entry;np.savez_compressed(out/'fitted_PRIVATE'/(sid+'_online_'+rule+'_'+kind+'.npz'),**fitted)
     del s,t,opt
    print('ONLINE58',rule,kind,dict(counts),flush=True)
  statuses=[]
  for row in rows:
   sid=row['id'];assert set(results[sid])==set(ARMS);ss=np.stack([results[sid][x] for x in ARMS]);q=ss.argmax(1).astype(np.int64);boxes=baselines[sid]['boxes'];np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS),scores=ss,queries=q,source_boxes=boxes,source_scores=baselines[sid]['scores'],selected_original_boxes=boxes[q]);dump(out/'cases_PRIVATE'/(sid+'.json'),records[sid]);status={}
   for arm in ARMS:
    unchanged=arm in ['source','frozen_ABC','cpu_frozen_ABC']
    if arm.startswith('backtrack_'):
     if '_epi_' in arm:
      parts=arm.split('_');kind=parts[2];step=int(parts[-1]);unchanged=not any(x['accepted'] for x in records[sid]['episodic']['backtrack_'+kind]['trace'][:step])
     else:
      parts=arm.split('_');kind=parts[2];timing=parts[-1];entry=records[sid]['online']['backtrack_'+kind];hh=entry['student_'+timing+'_hash'] if parts[3]=='student' else entry['teacher_'+timing+'_hash'];unchanged=hh==source_hash
    status[arm]={'unknown':False,'fallback':bool(arm.startswith('backtrack_') and unchanged),'no_parameter_update':bool(unchanged)}
    if arm.startswith('original_'):
     old=arm[len('original_'):]
     if old.startswith('online_'):old=old.replace('online_fixed_','online_fixed_AB_').replace('online_ema_','online_ema_AB_')
     error=float(abs(results[sid][arm]-old_outputs[sid][old]).max());bb=bridge.setdefault(arm,{'n':0,'max_absolute_score_error':0.,'Top1_disagreements':0});bb['n']+=1;bb['max_absolute_score_error']=max(bb['max_absolute_score_error'],error);bb['Top1_disagreements']+=int(results[sid][arm].argmax()!=old_outputs[sid][old].argmax())
   statuses.append({'id':sid,'arm_statuses':status})
  assert counts['original_proposals']==counts['backtrack_proposals']==11424 and counts['backtrack_loss_increased']==0 and n.phash(source)==source_hash
  dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'CPU_R57_BRIDGE.json',bridge);costs={'n':336,'device':'cpu','threads':4,'proposal_attempts_total':22848,'counts':dict(counts),'source_head_unchanged':True,'new_full_model_forwards':0,'seconds':time.perf_counter()-start,'max_RSS_bytes':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024};dump(out/'COSTS.json',costs);seal(out,'R58_MATCHED_CPU_ADAM_AND_MONOTONE_BACKTRACKING_PREDICTIONS_NO_GT',n=336,arm_names=ARMS,plan_sha256=sha(a.plan),**{'proposal_attempts_total':22848});print(costs,flush=True)
 except BaseException as e:dump(out/'FAILED.json',{'type':type(e).__name__,'error':str(e),'completed_inputs':len(cache),'counts':dict(counts),'task_labels_accessed':False});raise
if __name__=='__main__':main()
