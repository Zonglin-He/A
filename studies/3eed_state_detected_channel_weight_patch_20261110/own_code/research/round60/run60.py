"""Actual source-reset weight-column patches on real cached head inputs."""
import argparse,sys,time,resource,torch
from common60 import *
from detect60 import get_inputs
import mechanism60 as m
sys.path.insert(0,str(ROOT/'research/round59'));from math59 import head,fingerprint

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);torch.set_grad_enabled(False);rows=read(ROOT/p['manifest']);d=ROOT/p['detection_root'];ds=read(d/'SEALED.json');assert ds['n']==336
 for rel,h in ds['files'].items():assert sha(d/rel)==h
 fixed=load(d/'FIXED_PATCHES_PRIVATE.npz');donors=read(d/'PATCH_DONORS_PRIVATE.json');initial=load(ROOT/p['source_parameters']);model=head(initial);original=fingerprint(model);W=initial['4.weight'].astype(float)
 out=ROOT/p['prediction_root'];assert not out.exists();out.mkdir(parents=True)
 for folder in ['samples_PRIVATE','patches_PRIVATE','cases_PRIVATE']:(out/folder).mkdir()
 start=time.perf_counter();statuses=[];maxeq=0.;changed={k:0 for k in ARMS}
 try:
  for i,row in enumerate(rows):
   f,g=get_inputs(row,p);assert np.array_equal(f['M0'],np.column_stack([initial['4.weight'],initial['4.bias']]).astype(float));root=ROOT/p['feature_root'];s=read(root/'SEALED.json');rel='samples_PRIVATE/'+row['id']+'.npz';assert sha(root/rel)==s['files'][rel];old=load(root/rel)
   result={arm:old['scores'][list(old['arm_names']).index('direct_all' if arm=='r59_direct_all' else arm)].copy() for arm in ARMS[:6]};base,basestats=m.replay(model,f,g,np.zeros(288));assert abs(base-result['frozen_ABC']).max()<3e-5 and base.argmax()==result['frozen_ABC'].argmax()
   det=load(d/'samples_PRIVATE'/(row['id']+'.npz'));other=load(d/'samples_PRIVATE'/(rows[donors[i]]['id']+'.npz'));patches={};stats={}
   for z in MECHANISMS:
    rng=np.random.default_rng(int(hashlib.sha256(('R60_random_'+z+'|'+row['id']).encode()).hexdigest()[:16],16));dynamic=det[z+'_gate'];variants={'dynamic':dynamic,'fixed':fixed[z],'shuffled':other[z+'_gate'],'random':m.random_norm_matched(dynamic,W,rng),'activation':dynamic}
    for kind,gate in variants.items():
     arm=z+'_'+kind;result[arm],st=m.replay(model,f,g,gate,activation=kind=='activation');patches[arm]=gate;stats[arm]=st;changed[arm]+=st['actual_weight_changed'];assert st['edited_columns']<=8 and st['max_column_attenuation']<=.5+1e-12;assert fingerprint(model)==original
    eq=float(abs(result[z+'_dynamic']-result[z+'_activation']).max());assert eq<2e-5;assert result[z+'_dynamic'].argmax()==result[z+'_activation'].argmax();maxeq=max(maxeq,eq)
   scores=np.stack([result[k] for k in ARMS]);queries=scores.argmax(1);assert np.isfinite(scores).all() and scores.shape==(16,256)
   data={'arm_names':np.array(ARMS),'scores':scores,'queries':queries,'source_boxes':old['source_boxes'],'source_scores':old['source_scores'],'selected_original_boxes':old['source_boxes'][queries]};np.savez_compressed(out/'samples_PRIVATE'/(row['id']+'.npz'),**data);np.savez_compressed(out/'patches_PRIVATE'/(row['id']+'.npz'),**patches)
   dump(out/'cases_PRIVATE'/(row['id']+'.json'),{'source':basestats,'arms':stats,'no_task_labels_used':True,'no_pseudo_labels_fitted':True,'optimizer_calls':0,'source_restored_exact':True,'detector_support_mass':float(det['support_mass']),'supported_anchors':int(det['supported_anchor_count'])})
   statuses.append({'id':row['id'],'arm_statuses':{arm:{'unknown':float(det['support_mass'])<=1e-12 if arm.startswith('noise_') else False,'fallback':stats.get(arm,{}).get('edited_columns',1)==0,'no_parameter_update':not stats.get(arm,{}).get('actual_weight_changed',False)} for arm in ARMS}})
   if (i+1)%32==0:print('R60_PATCH_REPLAY',i+1,len(rows),flush=True)
  cost={'n':len(rows),'new_full_model_forwards':0,'new_head_replays':len(rows)*11,'gradient_steps':0,'solver_calls':0,'pseudo_label_fits':0,'external_model_calls':0,'GPU_used':False,'seconds':time.perf_counter()-start,'max_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'actual_weight_changed_cases':changed,'activation_weight_control_max_score_difference':maxeq};dump(out/'COSTS.json',cost);dump(out/'READOUT_STATUS_PRIVATE.json',statuses);seal(out,'R60_DETECT_PATCH_REPLAY_PREDICTIONS_SEALED_BEFORE_TASK_SCORE',n=len(rows),arm_names=ARMS,plan_sha256=sha(a.plan),detection_seal_sha256=sha(d/'SEALED.json'));print(cost,flush=True)
 except BaseException as error:dump(out/'FAILED.json',{'type':type(error).__name__,'error':str(error),'completed':len(statuses)});raise
if __name__=='__main__':main()
