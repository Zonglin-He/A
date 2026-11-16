"""Only cached source candidate IoUs; source-GT fits never counted as TTA."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round124';B=R/'crossplatform/round124';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def metric(rows,k):
 y=np.array([r['arms'][k]['IoU'] for r in rows]);return {'n':len(y),'mIoU_percent':float(y.mean()*100),'Acc25_percent':float((y>.25).mean()*100),'Acc50_percent':float((y>.5).mean()*100)}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['n']==384 and not O.exists();mat=rd(B/'MATRIX_SEALED.json');rows=[];cases=[];process=[]
 for e in rd(R/'research/round121/SOURCE_CACHE_REGISTRY_PRIVATE.json'):
  src=e['source'];sid=e['row']['id'];D=B/'formal'/src;a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));rec=rd(D/'records_PRIVATE'/(sid+'.json'));y=np.load(R/e['files']['label']['path'])['ious'];arms={str(k):{'IoU':float(y[q]),'query_PRIVATE':int(q)} for k,q in zip(a['arm_names'],a['queries'])};rows.append({'source':src,'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'frame_group':e['source_frame_group'],'eligible':rec['eligible'],'oracle_IoU':float(y.max()),'seconds':rec['seconds'],'arms':arms})
  for arm,x in rec['endpoints'].items():
   z=rec['endpoints']['zero'];process.append({'source':src,'id_PRIVATE':sid,'arm':arm,'eligible':rec['eligible'],'parameter_L2':x['parameter_L2'],'GT_loss':x['GT_loss'],'GT_loss_change':x['GT_loss']-z['GT_loss'],'hidden_change_L2':x['h_change_L2'],'changed_tensors':x['changed_tensors']})
  for arm in ['oracle1','oracle3','random1','random3']:
   d=arms[arm]['IoU']-arms['zero']['IoU']
   if abs(d)>1e-12:cases.append({'source':src,'id_PRIVATE':sid,'arm':arm,'zero_IoU':arms['zero']['IoU'],'new_IoU':arms[arm]['IoU'],'delta_IoU':d,'GT_supervised_diagnostic':arm.startswith('oracle'),'identity':'UNKNOWN_without_private_media_review'})
 dump(O/'ALL_ROWS_PRIVATE.json',rows);aggregate=[];scenes=[];pairs=[];concentration=[]
 for group in ['all','group0','group1']:
  rr=rows if group=='all' else [r for r in rows if r['frame_group']==int(group[-1])]
  for src in ['all_three_equal','waymo','drone','quad']:
   ss=rr if src=='all_three_equal' else [r for r in rr if r['source']==src]
   for k in mat['arms']:
    mm=metric(ss,k) if src!='all_three_equal' else {'n':len(ss),**{t:float(np.mean([metric([r for r in rr if r['source']==s],k)[t] for s in ['waymo','drone','quad']])) for t in ['mIoU_percent','Acc25_percent','Acc50_percent']}};aggregate.append({'frame_group':group,'source':src,'arm':k,**mm})
 for src in ['waymo','drone','quad']:
  rr=[r for r in rows if r['source']==src]
  for sc in sorted({r['scene_PRIVATE'] for r in rr}):
   for arm in mat['arms']:scenes.append({'source':src,'scene_PRIVATE':sc,'arm':arm,**metric([r for r in rr if r['scene_PRIVATE']==sc],arm)})
  for arm in ['oracle1','oracle3','random1','random3']:
   dd=np.array([r['arms'][arm]['IoU']-r['arms']['zero']['IoU'] for r in rr]);pairs.append({'source':src,'method':arm,'control':'zero','n':len(rr),'gain_pp':float(dd.mean()*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum()),'query_changes':sum(r['arms'][arm]['query_PRIVATE']!=r['arms']['zero']['query_PRIVATE'] for r in rr)})
 for arm in mat['arms']:
  dd=np.array([r['arms'][arm]['IoU']-r['arms']['zero']['IoU'] for r in rows]);concentration.append({'arm':arm,'n':384,'gain_pp':float(dd.mean()*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum()),'largest_positive_single_input_pp':float(max(0,dd.max())*100/384),'largest_negative_single_input_pp':float(min(0,dd.min())*100/384),'positive_mass_pp':float(dd[dd>0].sum()*100/384),'negative_mass_pp':float(dd[dd<0].sum()*100/384)})
 for name,x in [('AGGREGATE_ROWS',aggregate),('ALL_SCENE_ROWS_PRIVATE',scenes),('ALL_PAIRED_ROWS',pairs)]:dump(O/(name+'.json'),x)
 dump(H/'CONCENTRATION.json',concentration);dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases);dump(H/'PROCESS_ROWS_PRIVATE.json',process);summary={}
 for src in ['all_three_equal','waymo','drone','quad']:
  pp=process if src=='all_three_equal' else [r for r in process if r['source']==src];summary[src]={}
  for arm in ['oracle1','oracle3','random1','random3']:
   part=[r for r in pp if r['arm']==arm and r['eligible']];summary[src][arm]={'n_eligible':len(part),'GT_loss_falls':sum(r['GT_loss_change']<0 for r in part),'mean_GT_loss_change':float(np.mean([r['GT_loss_change'] for r in part])),'mean_parameter_L2':float(np.mean([r['parameter_L2'] for r in part])),'mean_hidden_change_L2':float(np.mean([r['hidden_change_L2'] for r in part]))}
 dump(H/'CAPACITY_SUMMARY.json',{'source_only':True,'source_GT_is_NOT_TTA':True,'no_deployed_weights':True,'no_target_followup_selected':True,'groups':summary});print(concentration,flush=True)
if __name__=='__main__':main()
