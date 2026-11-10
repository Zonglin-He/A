"""Real-cache internal representation targets, terminal-layer GD and ridge edits."""
import argparse,time,torch,resource
from common59 import *
import math59 as m

def one(row,p,initial,*,engineering=False):
 sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';caches={}
 for key in ['AB_root','capture_root','association_root','r57_prediction_root','r56_prediction_root','r58_prediction_root']:
  root=ROOT/p[key];s=read(root/'SEALED.json');assert sha(root/rel)==s['files'][rel];caches[key]=load(root/rel)
 v=caches['AB_root'];c=caches['capture_root'];g=caches['association_root'];r57=caches['r57_prediction_root'];r56=caches['r56_prediction_root'];r58=caches['r58_prediction_root'];model=m.head(initial);fp=m.fingerprint(model);prefix0={k:z.numpy().copy() for k,z in model.state_dict().items() if not k.startswith('4.')};M=np.column_stack([initial['4.weight'],initial['4.bias']]).astype(float);f=m.features(model,v,c);raw,targets,confidence,Hw,E,G,B=m.target_and_system(f,g,M);delta,diag=m.solve(G,B,M);Wedit=M+delta;assert diag['relative_edit_L2']<=.05+1e-12
 result={}
 def inherited(cache,name):return cache['scores'][list(cache['arm_names']).index(name)].copy()
 result['source']=inherited(r57,'source');result['frozen_ABC']=inherited(r57,'frozen_ABC');result['r56_one']=inherited(r56,'tta1');result['r58_online_ema_teacher']=inherited(r58,'backtrack_online_ema_teacher_post');result['r58_online_ema_student']=inherited(r58,'backtrack_online_ema_student_post');result['zero_edit']=m.readout(model,f,g);bridge=float(abs(result['zero_edit']-result['frozen_ABC']).max());assert bridge<3e-5 and int(result['zero_edit'].argmax())==int(result['frozen_ABC'].argmax())
 with torch.no_grad():raw32={z:model[4](torch.as_tensor(f[z+'_X'])[None])[0].numpy() for z in 'ABC'}
 directC={z:raw32[z].copy() for z in 'ABC'};directC['C']=(1-confidence['C'][:,None])*raw32['C']+confidence['C'][:,None]*targets['C'];result['direct_C']=m.score_from_raw(directC,f,g);directAll={z:(1-confidence[z][:,None])*raw32[z]+confidence[z][:,None]*targets[z] for z in 'ABC'};result['direct_all']=m.score_from_raw(directAll,f,g)
 gd,logs=m.gradient_trajectory(M,Hw,E,diag['lambda_effective'],diag['step_size']);fitted={}
 for step,W in gd.items():
  arm='grad'+str(step);m.install(model,W);result[arm]=m.readout(model,f,g);fitted[arm+'_M64']=W;assert np.linalg.norm(W-M)<=diag['edit_cap_L2']+1e-8
 m.install(model,Wedit);result['ridge_edit']=m.readout(model,f,g);fitted['ridge_edit_M64']=Wedit;fitted['ridge_edit_M32']=np.column_stack([model[4].weight.detach().numpy(),model[4].bias.detach().numpy()]);changed=m.fingerprint(model)!=fp;m.install(model,M);assert m.fingerprint(model)==fp
 result['functional_edit']=m.virtual_readout(fitted['ridge_edit_M32'],f,g);assert m.fingerprint(model)==fp and np.array_equal(result['functional_edit'],result['ridge_edit'])
 valid=np.flatnonzero(confidence['C']>0);perm=np.arange(256);rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('R59_shuffle_v1|'+sid).encode()).digest()[:8],'little'));perm[valid]=rng.permutation(valid);ts=targets['C'].copy();ts[valid]=np.linalg.norm(raw['C'][valid],axis=1)[:,None]*m.unit(targets['C'][perm[valid]]);ws=np.sqrt(confidence['C']/max(float(confidence['C'].sum()),1e-12))[:,None];Es=(ts-raw['C'])*ws;ds=np.linalg.solve(G+diag['lambda_effective']*np.eye(289),(Es.T@Hw).T).T;dn=np.linalg.norm(ds);ds=ds*(np.linalg.norm(delta)/dn) if dn>1e-20 else np.zeros_like(ds);m.install(model,M+ds);result['shuffled_edit']=m.readout(model,f,g);fitted['shuffled_edit_M64']=M+ds;m.install(model,M);assert m.fingerprint(model)==fp and all(np.array_equal(model.state_dict()[k].numpy(),z) for k,z in prefix0.items())
 assert set(result)==set(ARMS);q=np.array([result[k].argmax() for k in ARMS],np.int64);scores=np.stack([result[k] for k in ARMS]);assert np.isfinite(scores).all();assert np.array_equal(v['A_boxes'],r57['source_boxes']) and np.array_equal(v['A_boxes'],r58['source_boxes']) and np.array_equal(v['A_boxes'],r56['source_boxes'])
 data={**f,'M0':M,'target_A':targets['A'],'target_B':targets['B'],'target_C':targets['C'],'confidence_A':confidence['A'],'confidence_B':confidence['B'],'confidence_C':confidence['C'],'Hw':Hw,'E':E,'G':G,'B':B,'shuffle_permutation':perm,'source_boxes':v['A_boxes'],'source_scores':v['A_scores']}
 stats={'platform':row['platform'],**diag,'gradient_checkpoints':logs,'source_bridge_max_absolute_error':bridge,'source_restored_exact':True,'earlier_layers_unchanged':True,'ridge_parameters_changed':changed,'edited_parameter_count':18496,'C_supported_candidates':int((confidence['C']>0).sum()),'C_geometric_weight_sum':float(confidence['C'].sum()),'weighted_target_residual_L2':float(np.linalg.norm(E)),'objective_source':m.objective(np.zeros_like(M),Hw,E,diag['lambda_effective']),'objective_ridge':m.objective(delta,Hw,E,diag['lambda_effective']),'grad256_distance_to_ridge_L2':float(np.linalg.norm(gd[256]-Wedit)),'functional_vs_weight_readout_bitexact':True,'all_boxes_fixed':True,'shuffled_edit_norm_matched':bool(abs(np.linalg.norm(ds)-np.linalg.norm(delta))<1e-10),'no_GT_used':True}
 return {'arm_names':np.array(ARMS),'scores':scores,'queries':q,'source_boxes':v['A_boxes'],'source_scores':v['A_scores'],'selected_original_boxes':v['A_boxes'][q]},data,fitted,stats

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);torch.manual_seed(0);rows=read(ROOT/p['manifest']);initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');start=time.perf_counter()
 if a.engineering:
  rows=[min([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256(('R59_engineering|'+r['id']).encode()).hexdigest()) for pl in ['drone','quad']];out=ROOT/'research/round59/engineering_attempt001'
 else:out=ROOT/p['prediction_root'];assert read(ROOT/'research/round59/engineering_attempt001/RESULT.json')['state'].startswith('PASS_')
 assert not out.exists();out.mkdir(parents=True)
 for name in ['samples_PRIVATE','targets_PRIVATE','fitted_PRIVATE','cases_PRIVATE']:(out/name).mkdir()
 statuses=[];stats=[]
 try:
  for k,row in enumerate(rows,1):
   z,d,f,s=one(row,p,initial,engineering=a.engineering);sid=row['id'];np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),**z);np.savez_compressed(out/'targets_PRIVATE'/(sid+'.npz'),**d);np.savez_compressed(out/'fitted_PRIVATE'/(sid+'.npz'),**f);dump(out/'cases_PRIVATE'/(sid+'.json'),s);stats.append(s);statuses.append({'id':sid,'arm_statuses':{arm:{'unknown':s['C_supported_candidates']==0,'fallback':s['C_supported_candidates']==0 and arm not in ['source','frozen_ABC','r56_one','r58_online_ema_teacher','r58_online_ema_student','zero_edit'],'no_parameter_update':arm in ['source','frozen_ABC','zero_edit','direct_C','direct_all','functional_edit'] or (arm in ['ridge_edit','shuffled_edit'] and not s['ridge_parameters_changed'])} for arm in ARMS}})
   if k%16==0:print('R59_REAL_CACHE_EDIT',k,len(rows),flush=True)
  costs={'n':len(rows),'new_full_model_forwards':0,'SGD_steps':256*len(rows),'editable_parameter_count':18496,'seconds':time.perf_counter()-start,'max_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_used':False};dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COSTS.json',costs)
  if a.engineering:dump(out/'RESULT.json',{'state':'PASS_R59_REAL_CACHE_SOURCE_BRIDGE_LINEAR_EDIT_AND_GRADIENT_ENGINEERING','n':2,'seconds':costs['seconds'],'records':stats})
  seal(out,'R59_FROZEN_INTERNAL_TARGET_LINEAR_EDITS_NO_TASK_LABELS',n=len(rows),arm_names=ARMS,plan_sha256=sha(a.plan));print(costs,flush=True)
 except BaseException as error:dump(out/'FAILED.json',{'type':type(error).__name__,'error':str(error),'completed':len(stats),'GT_access':False});raise
if __name__=='__main__':main()
