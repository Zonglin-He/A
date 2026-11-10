"""Sequential combo and layer-position stages; full native downstream replay."""
import argparse,sys,time,torch
from common62 import *
import compute62 as m
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--mode',choices=['combo','layers'],required=True);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();p=read(a.plan);pin(p);m.gpu_strict();rows=read(ROOT/p['manifest']);rows=engineering_rows(rows) if a.engineering else rows;cap=ROOT/('research/round62/engineering_capture_attempt001' if a.engineering else p['capture_root']);cs=read(cap/'SEALED.json');out=ROOT/('research/round62/engineering_combo_attempt001' if a.engineering else p[a.mode+'_root']);assert not out.exists();out.mkdir(parents=True)
 for d in ['samples_PRIVATE','states_PRIVATE','cases_PRIVATE']:(out/d).mkdir()
 if a.mode=='layers':assert read(ROOT/'research/round62/score_combo336_attempt001/SCORING_RECEIPT.json')['n']==336
 model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');head=m.cpu_head(initial);arms=COMBO_ARMS if a.mode=='combo' else LAYER_ARMS;start=time.perf_counter();status=[];max_h=0.;max_box=0.;max_abc=0.;max_r59=0.
 try:
  for k,row in enumerate(rows):
   sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel];z=load(cap/rel);G=ROOT/p['association_root'];assert sha(G/rel)==read(G/'SEALED.json')['files'][rel];g=load(G/rel);result={};states={};details={}
   if a.mode=='combo':
    root=ROOT/p['r59_root'];assert sha(root/rel)==read(root/'SEALED.json')['files'][rel];old=load(root/rel);r61=ROOT/p['r61_root'];assert sha(r61/rel)==read(r61/'SEALED.json')['files'][rel];prior=load(r61/rel)
    for arm,key in [('source','source'),('frozen_ABC','frozen_ABC'),('r59_direct_all','direct_all')]:result[arm]=old['scores'][list(old['arm_names']).index(key)]
    result['r61_last15']=prior['scores'][list(prior['arm_names']).index('dec_language_boost')];hs={};bb={}
    for v in 'ABC':
     h,hist,_=m.decoder_chain(model,z,v);hs[v]=h;bb[v]=hist[-1];max_h=max(max_h,float(abs(h-z[v+'_h']).max()));max_box=max(max_box,float(abs(bb[v]-z[v+'_boxes']).max()));assert np.array_equal(bb[v],z[v+'_boxes']) and np.array_equal(h,z[v+'_h'])
    result['zero_ABC'],result['zero_R59'],_=m.readouts(head,hs,z,g);e=float(abs(result['zero_ABC']-result['frozen_ABC']).max());e59=float(abs(result['zero_R59']-result['r59_direct_all']).max());max_abc=max(max_abc,e);max_r59=max(max_r59,e59);assert e<p['source_bridge_score_abs_tolerance'] and e59<p['source_bridge_score_abs_tolerance'];assert result['zero_ABC'].argmax()==result['frozen_ABC'].argmax() and result['zero_R59'].argmax()==result['r59_direct_all'].argmax();layers=[5]
   else:
    root=ROOT/p['combo_root'];old=load(root/rel);assert sha(root/rel)==read(root/'SEALED.json')['files'][rel];result={a:old['scores'][list(old['arm_names']).index(a)] for a in COMBO_ARMS};layers=list(range(5))
   for layer in layers:
    hs={};bb={}
    for v in 'ABC':
     h,hist,_=m.decoder_chain(model,z,v,layer);hs[v]=h;bb[v]=hist[-1];states['layer'+str(layer)+'_'+v+'_h']=h;states['layer'+str(layer)+'_'+v+'_internal_boxes']=bb[v]
    native,fused,extra=m.readouts(head,hs,z,g);result['native_l'+str(layer)]=native
    if layer==5:
     result['R59_l5']=fused;assert native.argmax()==result['r61_last15'].argmax() and abs(native-result['r61_last15']).max()<p['source_bridge_score_abs_tolerance']
    details['layer'+str(layer)]={'identity':m.lineage(z,bb,g),'source_point_anchor_lineage_and_slot_order_fixed':True,'no_GT_in_control':True}
   scores=np.stack([result[a] for a in arms]);q=scores.argmax(1);assert np.isfinite(scores).all();np.savez_compressed(out/rel,arm_names=np.array(arms),scores=scores,queries=q,source_boxes=z['A_boxes'],source_scores=z['A_scores'],selected_original_boxes=z['A_boxes'][q]);np.savez_compressed(out/'states_PRIVATE'/(sid+'.npz'),**states);dump(out/'cases_PRIVATE'/(sid+'.json'),details);status.append({'id':sid,'arm_statuses':{arm:{'unknown':False,'fallback':False,'no_parameter_update':not arm.startswith(('native_l','R59_l'))} for arm in arms}})
   if (k+1)%16==0:print('R62_'+a.mode,k+1,len(rows),flush=True)
  assert fingerprint(model)==fp;cost={'n':len(rows),'arms':len(arms),'source_restored_exact':True,'source_h_bridge_max_error':max_h,'source_box_bridge_max_error':max_box,'source_ABC_bridge_max_error':max_abc,'source_R59_bridge_max_error':max_r59,'new_full_source_forwards':0,'new_gradient_or_fit_steps':0,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COSTS.json',cost);dump(out/'READOUT_STATUS_PRIVATE.json',status)
  if a.engineering:dump(out/'RESULT.json',{'state':'PASS_R62_PREFIX0_NATIVE_CHAIN_AND_RECOMPUTED_R59_COMBO_ENGINEERING','n':len(rows),'cost':cost})
  seal(out,'R62_'+a.mode.upper()+'_PREDICTIONS_SEALED_NO_GT',n=len(rows),arm_names=arms,plan_sha256=sha(a.plan));print(cost,flush=True)
 except BaseException as error:dump(out/'FAILED.json',{'type':type(error).__name__,'error':str(error),'completed':len(status),'automatic_retry':False});raise
if __name__=='__main__':main()
