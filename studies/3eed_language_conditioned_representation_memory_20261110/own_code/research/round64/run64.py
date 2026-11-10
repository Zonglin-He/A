"""Three complete read-before-write streams and matched old R58 baselines."""
import argparse,sys,time,torch
from common64 import *
import memory64 as mem
sys.path.insert(0,str(ROOT/'research/round62'));from compute62 import score
sys.path.insert(0,str(ROOT/'research/round58'));import numerics58 as r58
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);rows=read(ROOT/p['manifest']);by={r['id']:r for r in rows};index={r['id']:i for i,r in enumerate(rows)};orders=read(ROOT/p['orders']);out=ROOT/('research/round64/engineering_attempt001' if a.engineering else p['prediction_root']);assert not out.exists();out.mkdir(parents=True)
 for f in ['samples_PRIVATE','cases_PRIVATE']:(out/f).mkdir()
 prep=ROOT/p['prepared_root'];ps=read(prep/'SEALED.json');data={};results={r['id']:{} for r in rows};records={r['id']:{} for r in rows};initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');start=time.perf_counter();counts={'memory_main_steps':0,'R58_new_proposals':0,'R58_original_reused_inputs':0,'write_proposals':0};arms=[o+'__'+a for o in p['order_names'] for a in p['per_order_arms']]
 for row in rows:
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(prep/rel)==ps['files'][rel];data[sid]=load(prep/rel)
 for order,streams in orders.items():
  for pl,stream0 in streams.items():
   stream=stream0[:2] if a.engineering else stream0;pool=[]
   if order!='original' and not a.engineering:s=r58.head(initial).requires_grad_(True);t=r58.head(initial).requires_grad_(False);opt=r58.optimizer(s)
   for pos,sid in enumerate(stream):
    z=data[sid];rec={'stream_index':pos,'pool_before':len(pool),'memory':{}};result={'source':z['source_prob'],'frozen_ABC':z['ABC'],'r59_direct_all':z['R59'],'reset_each_input':z['R59'].copy()}
    for kind in ['memory','shuffled_values','no_explicit_text']:
     raw={};rr={}
     for v in 'ABC':
      delta,trace=mem.correction(pool,z,v,p['memory'],kind,order+'|'+pl+'|'+str(pos),pos);raw[v]=z[v+'_fused_raw'].copy();used=np.linalg.norm(delta,axis=1)>0;raw[v][used]+=p['memory']['alpha']*np.linalg.norm(raw[v][used],axis=1)[:,None]*delta[used];rr[v]=trace
     result[kind]=score(raw,z,z);rec['memory'][kind]=rr
    # All prediction methods finish before the shared frozen-source write.
    items=mem.proposals(z,p['memory'],index[sid],pos);replaced,evicted=mem.insert(pool,items,p['memory']);counts['write_proposals']+=len(items);rec.update(write_proposals=len(items),replacements=replaced,evictions=evicted,pool_after=len(pool));assert len(pool)<=p['memory']['capacity'];counts['memory_main_steps']+=1
    if not a.engineering:
     if order=='original':
      rel='samples_PRIVATE/'+sid+'.npz';old=load(ROOT/p['r58_root']/rel);counts['R58_original_reused_inputs']+=1
      for who in ['student','teacher']:
       for timing in ['pre','post']:result['r58_'+who+'_'+timing]=old['scores'][list(old['arm_names']).index('backtrack_online_ema_'+who+'_'+timing)]
     else:
      ab=load(ROOT/p['AB_root']/'samples_PRIVATE'/(sid+'.npz'));c=load(ROOT/p['C_root']/'samples_PRIVATE'/(sid+'.npz'));g=load(ROOT/p['association_root']/'samples_PRIVATE'/(sid+'.npz'));d=r58.prepared(ab,c,g)
      with torch.no_grad():result['r58_student_pre']=r58.ensemble(s,d).numpy().copy();result['r58_teacher_pre']=r58.ensemble(t,d).numpy().copy()
      rec['R58_step']=r58.step(s,t,opt,d,'ema','backtrack');counts['R58_new_proposals']+=1
      with torch.no_grad():result['r58_student_post']=r58.ensemble(s,d).numpy().copy();result['r58_teacher_post']=r58.ensemble(t,d).numpy().copy()
    results[sid].update({order+'__'+k:v for k,v in result.items()});records[sid][order]=rec
    if (pos+1)%32==0:print('R64_STREAM',order,pl,pos+1,len(stream),len(pool),flush=True)
   if order!='original' and not a.engineering:del s,t,opt
  if a.engineering:break
 selected=[r for r in rows if results[r['id']]];actual_arms=list(results[selected[0]['id']]) if a.engineering else arms;statuses=[]
 for r in selected:
  sid=r['id'];z=data[sid];scores=np.stack([results[sid][arm] for arm in actual_arms]);q=scores.argmax(1);assert np.isfinite(scores).all();np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(actual_arms),scores=scores,queries=q,source_boxes=z['source_boxes'],source_scores=z['source_scores'],selected_original_boxes=z['source_boxes'][q]);dump(out/'cases_PRIVATE'/(sid+'.json'),records[sid]);statuses.append({'id':sid,'arm_statuses':{arm:{'unknown':False,'fallback':bool(arm.split('__')[1] in ['memory','shuffled_values','no_explicit_text'] and not any(x['matched_candidates'] for x in records[sid][arm.split('__')[0]]['memory'][arm.split('__')[1]].values())),'no_parameter_update':not arm.split('__')[1].startswith('r58_')} for arm in actual_arms}})
 assert a.engineering or set(actual_arms)==set(arms);dump(out/'COSTS.json',{'n':len(selected),'orders':1 if a.engineering else 3,'arms':len(actual_arms),'counts':counts,'main_gradient_steps':0,'new_full_source_forwards':0,'GPU_used':False,'seconds':time.perf_counter()-start,'max_RSS_bytes':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024});dump(out/'READOUT_STATUS_PRIVATE.json',statuses);seal(out,'R64_MEMORY_AND_MATCHED_ORDER_BASELINES_SEALED_BEFORE_SCORE',n=len(selected),arm_names=actual_arms);print(read(out/'COSTS.json'))
if __name__=='__main__':main()
