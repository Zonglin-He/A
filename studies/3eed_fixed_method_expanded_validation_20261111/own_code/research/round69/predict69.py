"""All fixed methods on canonical A256, same observation and original transport."""
import sys,time,torch
from common69 import *
from method69 import G,N,cpu_head,readouts,direct,set_direct
H=ROOT/'research/round69';p=read(H/'experiment/PREDICT_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);out=ROOT/p['prediction_root'];assert not out.exists();out.mkdir(parents=True)
for name in ['samples_PRIVATE','association_PRIVATE','cases_PRIVATE']:(out/name).mkdir()
initial=load(ROOT/p['initial_head']);source=cpu_head(initial);torch.set_grad_enabled(False);torch.set_num_threads(4);start=time.perf_counter();statuses=[]
for i,r in enumerate(rows):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';ab=load(ROOT/p['AB_capture_root']/rel);c=load(ROOT/p['C_random_root']/rel);e=load(ROOT/p['evidence_root']/rel);state=load(ROOT/p['fit_root']/rel);fitrec=read(ROOT/p['fit_root']/'cases_PRIVATE'/(sid+'.json'));path=Path(r['pointpath']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,np.float32).reshape(-1,4);indices=np.unique(np.r_[ab['A_point_indices'],ab['B_point_indices'],c['point_indices']]);x=np.array(raw[indices,:3],np.float32,copy=True)
 if r['platform']=='drone':x[:,2]+=1.8
 za={k[2:]:v for k,v in ab.items() if k.startswith('A_')};zb={k[2:]:v for k,v in ab.items() if k.startswith('B_')};base=G.native(za);g={'base':base,'union_raw_indices':indices}
 for v,zv in [('B',zb),('C',c)]:
  af,_,_=G.affinity(za['boxes'].astype(float),zv['boxes'].astype(float),x.astype(float));W=af.astype(float)/(1+af.astype(float).sum(0,keepdims=True));g[v+'_affinity']=af;g[v+'_W']=W;g[v+'_fixed_mass']=np.array(float((W@G.native(zv)).sum()))
 hs={'A':za['h'],'B':zb['h'],'C':c['h']};z={v+'_'+k:zv[k] for v,zv in [('A',za),('B',zb),('C',c)] for k in ['h','tokens','token_mask']};result={'source':za['scores']};result['frozen_ABC'],result['R59'],_=readouts(source,hs,z,g);adapted=cpu_head(state);_,result['R66_R59'],_=readouts(adapted,hs,z,g);result['R66_E_direct']=direct(result['R59'],e);result['R66_set_direct']=set_direct(result['R59'],e);d=N.prepared(ab,c,g);result['C_source']=N.distribution(source,d,'C').numpy();result['C_R66']=N.distribution(adapted,d,'C').numpy()
 for repeat in range(3):
  hz={v:c['s'+str(repeat)+'_'+v+'_h'] for v in 'ABC'};zz={v+'_tokens':c['s'+str(repeat)+'_'+v+'_tokens'] for v in 'ABC'};zz.update({v+'_token_mask':z[v+'_token_mask'] for v in 'ABC'});_,result['random_s'+str(repeat)],_=readouts(source,hz,zz,g)
 scores=np.stack([result[a] for a in ARMS]);assert np.isfinite(scores).all();q=scores.argmax(1);np.savez_compressed(out/rel,arm_names=np.array(ARMS),scores=scores,queries=q,source_boxes=za['boxes'],source_scores=za['scores'],selected_original_boxes=za['boxes'][q]);np.savez_compressed(out/'association_PRIVATE'/(sid+'.npz'),**g);statuses.append({'id':sid,'arm_statuses':{a:{'unknown':False,'fallback':bool(a in ['R66_E_direct','R66_set_direct'] and not e['tested_query_mask'].any()),'no_parameter_update':not(a in ['R66_R59','C_R66'] and fitrec['step_L2']>0)} for a in ARMS}});dump(out/'cases_PRIVATE'/(sid+'.json'),{'eligible':bool(e['update_allowed']),'accepted':fitrec['accepted'],'step_L2':fitrec['step_L2'],'C_B_known_mass':[float(g[v+'_fixed_mass']) for v in 'BC']})
 if(i+1)%64==0:print('PREDICT69',i+1,len(rows),flush=True)
dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COSTS.json',{'n':len(rows),'arms':len(ARMS),'seconds':time.perf_counter()-start});seal(out,'R69_ALL_FIXED_METHOD_AND_DIAGNOSTIC_PREDICTIONS_SEALED_BEFORE_LABELS',n=len(rows),arm_names=ARMS);print({'n':len(rows),'arms':len(ARMS)})
