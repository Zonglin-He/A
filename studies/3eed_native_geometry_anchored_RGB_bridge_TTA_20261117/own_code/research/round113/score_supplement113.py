"""CPU audit then postseal scoring of only the missing natural-step controls."""
from core113 import *
sys.path.insert(0,str(R113/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
def main():
 D0=B113/'supplement_wrong_natural_attempt001';mat=jread(D0/'MATRIX_SEALED.json');assert mat['n']==768;out=H113/'supplement_scores_attempt001';assert not out.exists();rows=[];heads=0;mx=0.;ties=0
 # Complete independent CPU readout audit before any new task GT access.
 for name,h in mat['directions'].items():
  src,_,tgt=name.split('_');D=D0/name;assert sha(D/'SEALED.json')==h
  for p,v in jread(D/'SEALED.json')['files'].items():assert sha(D/p)==v
  sh=load(B113/(src+'_SOURCE_READOUT_PRIVATE.npz'));head={k[6:]:v for k,v in sh.items() if k.startswith('match.')}
  for r in jread(R113/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
   sid=r['id'];pred=load(D/'samples_PRIVATE'/(sid+'.npz'));pmap=dict(zip(pred['arm_names'],pred['scores']));old=load(B113/name/'samples_PRIVATE'/(sid+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']));meta=jread(D/'records_PRIVATE'/(sid+'.json'));assert np.array_equal(pred['source_boxes'],old['source_boxes']) and np.array_equal(pred['queries'],pred['scores'].argmax(1));assert all(np.array_equal(pmap[k],oldp[k]) for k in mat['arms'] if k.startswith('anchor_bridge_natural_raw_'))
   if meta['protected_alias_fallback']:assert all(np.array_equal(pmap[k],oldp['prepared_frozen']) for k in mat['new_arms']);continue
   ff=load(B113/name/'features_PRIVATE'/(sid+'.npz'));new=load(D/'features_PRIVATE'/(sid+'.npz'));assert meta['old_natural_correct_max_error']<1e-7
   for arm in mat['new_arms']:
    view=arm.split('_')[-1];prefix='frozen_anchor_'+view+'_';a={k[len(prefix):]:v for k,v in ff.items() if k.startswith(prefix)};p=cpu112(a['h'][0],a['u'][0],a['mask'],new[arm+'_late'],a['r'],head);err=float(abs(p-pmap[arm]).max());mx=max(mx,err);assert err<5e-6
    if int(p.argmax())!=int(pmap[arm].argmax()):assert pmap[arm].max()-pmap[arm][int(p.argmax())]<=2*err+1e-12;ties+=1
    heads+=1
 assert heads==4344;jdump(H113/'SUPPLEMENT_NATIVE_AUDIT.json',{'state':'PASS_SOURCE_SEALS_OLD_CORRECT_PARITY_NEW6_CPU_READOUTS_ALL_ALIASES','n':768,'new_CPU_readouts':heads,'max_CPU_probability_error':mx,'CPU_near_ties':ties,'GT_used':False,'FINAL_access':False})
 out.mkdir();methods=[];allrows=[];maxerr=0.
 for name in mat['directions']:
  src,_,tgt=name.split('_');oldrows={r['id_PRIVATE']:r for r in jread(H113/'scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'))};golds=jread(R113/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');rr=[]
  for r in jread(R113/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
   pred=load(D0/name/'samples_PRIVATE'/(r['id']+'.npz'));boxes=np.c_[pred['source_boxes'],np.zeros(256)].astype(float);boxes[:,3:6]=np.maximum(boxes[:,3:6],1e-6);gt=np.array(golds[r['anno_index']]['source_GT7_float32']);ys=many_iou(boxes,gt);values={k:v for k,v in oldrows[r['id']]['arms'].items() if k in ['neither_correct_fusedABC','anchor_frozen_feature3_fusedABC','anchor_frozen_mean3_fusedABC','anchor_noLate_fusedABC','anchor_noLate_feature3_fusedABC','anchor_frozen_C']}
   for arm,p in zip(pred['arm_names'],pred['scores']):
    q=int(p.argmax());val=float(ys[q]);values[str(arm)]={'IoU':val,'query_PRIVATE':q}
    if arm in mat['new_arms']:
     c,s=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]]);v=oriented_iou(boxes[q,:3],boxes[q,3:6],np.eye(3),gt[:3],gt[3:6],rot);maxerr=max(maxerr,abs(val-v));assert abs(val-v)<1e-6
    else:assert val==oldrows[r['id']]['arms'][str(arm)]['IoU']
   rr.append({'source':src,'target':tgt,'id_PRIVATE':r['id'],'scene_PRIVATE':r['scene'],'arms':values})
  jdump(out/(name+'_ALL_ROWS_PRIVATE.json'),rr);allrows+=rr
  for arm in rr[0]['arms']:
   y=[r['arms'][arm]['IoU'] for r in rr];methods.append({'source':src,'target':tgt,'arm':arm,'n':128,'mIoU_percent':float(np.mean(y)*100),'Acc25_percent':float(np.mean(np.array(y)>.25)*100),'Acc50_percent':float(np.mean(np.array(y)>.5)*100)})
 agg=[]
 for group in ['all_six_equal','waymo','drone','quad']:
  for arm in allrows[0]['arms']:
   ss=[x for x in methods if x['arm']==arm and (group=='all_six_equal' or x['target']==group)];agg.append({'group':group,'arm':arm,**{k:float(np.mean([x[k] for x in ss])) for k in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 aa={r['arm']:r['mIoU_percent'] for r in agg if r['group']=='all_six_equal'};method='anchor_bridge_natural_raw_fusedABC';controls=['neither_correct_fusedABC','anchor_frozen_feature3_fusedABC','anchor_frozen_mean3_fusedABC','anchor_noLate_fusedABC','anchor_noLate_feature3_fusedABC','wrong_natural_raw_fusedABC','wrong_natural_sameL2_raw_fusedABC'];decision={'original_R113_primary_still_failed_unchanged':True,'development_secondary':method,'mIoU_percent':aa[method],'gains_pp':{k:aa[method]-aa[k] for k in controls},'passes_missing_wrong_controls_and_sameinfo_zeros':all(aa[method]>aa[k] for k in controls),'not_independent_confirmation':True,'action':'fixed_expansion_if_gate_passes_else_do_not_adopt_secondary','FINAL_access':False}
 for arm in aa:
  vals=[r['arms'][arm]['IoU'] for r in allrows];assert abs(100*sum(vals)/len(vals)-aa[arm])<1e-9
 jdump(out/'ALL_METHOD_ROWS.json',methods);jdump(out/'AGGREGATE_ROWS.json',agg);jdump(H113/'SUPPLEMENT_DECISION.json',decision);jdump(H113/'SUPPLEMENT_SCALAR_AUDIT.json',{'state':'PASS_ALL_NEW_CONTROL_TASK_VALUES_INDEPENDENT_GEOMETRY_AND_AGGREGATES','n':768,'new_readouts':4608,'old_correct_values_reused':2304,'max_geometry_error':maxerr});print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__':main()
