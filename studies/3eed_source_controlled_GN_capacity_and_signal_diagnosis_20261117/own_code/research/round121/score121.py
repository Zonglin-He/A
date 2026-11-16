"""Source diagnostic scalar assessment; cached source IoUs only, no fresh GT geometry."""
from pathlib import Path
import json,numpy as np,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round121';B=R/'crossplatform/round121';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();O=H/'scores_attempt001'
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def metric(rr,arm):
 v=np.array([r['arms'][arm]['IoU'] for r in rr]);return {'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100)}
def main():
 assert not O.exists();assert rd(H/'NATIVE_AUDIT.json')['n']==384;mat=rd(B/'MATRIX_SEALED.json');rows=[]
 for e in rd(H/'SOURCE_CACHE_REGISTRY_PRIVATE.json'):
  src=e['source'];sid=e['row']['id'];D=B/'formal'/src;a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));y=np.load(R/e['files']['label']['path'])['ious'];rec=rd(D/'records_PRIVATE'/(sid+'.json'));ar={str(k):{'IoU':float(y[int(q)]),'query_PRIVATE':int(q),'Acc25':bool(y[int(q)]>.25),'Acc50':bool(y[int(q)]>.5)} for k,q in zip(a['arm_names'],a['queries'])};rows.append({'source':src,'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'frame_group':e['source_frame_group'],'oracle_IoU':float(y.max()),'GT_eligible':rec['GT_eligible'],'seconds':rec['seconds'],'arms':ar})
 assert len(rows)==384;dump(O/'ALL_ROWS_PRIVATE.json',rows);agg=[];scenes=[];pairs=[]
 for group in ['all','group0','group1']:
  rr=rows if group=='all' else [r for r in rows if r['frame_group']==int(group[-1])]
  for src in ['all_three_equal','waymo','drone','quad']:
   ss=rr if src=='all_three_equal' else [r for r in rr if r['source']==src]
   for k in mat['arms']:
    if src=='all_three_equal':mm={t:float(np.mean([metric([r for r in rr if r['source']==s],k)[t] for s in ['waymo','drone','quad']])) for t in ['mIoU_percent','Acc25_percent','Acc50_percent']};mm['n']=len(ss)
    else:mm=metric(ss,k)
    agg.append({'frame_group':group,'source':src,'arm':k,**mm})
 for src in ['waymo','drone','quad']:
  rr=[r for r in rows if r['source']==src]
  for sc in sorted({r['scene_PRIVATE'] for r in rr}):
   ss=[r for r in rr if r['scene_PRIVATE']==sc]
   for k in mat['arms']:scenes.append({'source':src,'scene_PRIVATE':sc,'arm':k,**metric(ss,k)})
  for k in mat['arms']:
   for ctrl in ['context_zero','random','wrong_'+k] if k in ['P','RGB','D'] else ['context_zero']:
    dd=np.array([r['arms'][k]['IoU']-r['arms'][ctrl]['IoU'] for r in rr]);pairs.append({'source':src,'method':k,'control':ctrl,'n':len(rr),'gain_pp':float(dd.mean()*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum()),'query_changes':sum(r['arms'][k]['query_PRIVATE']!=r['arms'][ctrl]['query_PRIVATE'] for r in rr)})
 dump(O/'AGGREGATE_ROWS.json',agg);dump(O/'ALL_SCENE_ROWS_PRIVATE.json',scenes);dump(O/'ALL_PAIRED_ROWS.json',pairs)
 def m(group,arm):return next(x['mIoU_percent'] for x in agg if x['frame_group']==group and x['source']=='all_three_equal' and x['arm']==arm)
 picked=max(['P','RGB','D'],key=lambda k:m('group0',k));checks={'group1_above_zero':m('group1',picked)>m('group1','context_zero'),'group1_above_wrong':m('group1',picked)>m('group1','wrong_'+picked),'group1_above_random':m('group1',picked)>m('group1','random')};summary={'picked_on_predeclared_source_group0':picked,'qualification_checks':checks,'qualified':all(checks.values()),'source_group0_scores':{k:m('group0',k) for k in ['context_zero','P','RGB','D','random','wrong_P','wrong_RGB','wrong_D']},'source_group1_scores':{k:m('group1',k) for k in ['context_zero','P','RGB','D','random','wrong_P','wrong_RGB','wrong_D']},'source_oracle_is_NOT_TTA':True,'new_source_training':False,'target_data_read':False,'FINAL_access':False};dump(H/'QUALIFICATION.json',summary)
 impact=[]
 for k in mat['arms']:
  dd=np.array([r['arms'][k]['IoU']-r['arms']['context_zero']['IoU'] for r in rows]);sortedpos=sorted(dd[dd>0],reverse=True);impact.append({'arm':k,'n':384,'gain_pp':float(dd.mean()*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'unchanged':int((abs(dd)<=1e-12).sum()),'largest_positive_contribution_pp':float(dd.max()*100/384),'largest_negative_contribution_pp':float(dd.min()*100/384),'positive_mass_pp':float(dd[dd>0].sum()*100/384),'negative_mass_pp':float(dd[dd<0].sum()*100/384)})
 dump(H/'CONCENTRATION.json',impact)
 physical=[];angles=[];cases=[]
 for e,r in zip(rd(H/'SOURCE_CACHE_REGISTRY_PRIVATE.json'),rows):
  rec=rd(B/'formal'/e['source']/'records_PRIVATE'/(e['row']['id']+'.json'));z=rec['endpoints']['context_zero']
  for k,v in rec['endpoints'].items():physical.append({'source':r['source'],'id_PRIVATE':r['id_PRIVATE'],'arm':k,'actual_patch_L2':v['actual_writeback_L2'],'oracle_loss':v['oracle_loss'],'oracle_loss_delta':v['oracle_loss']-z['oracle_loss'],**{t+'_loss':v['aux_losses'][t] for t in ['P','RGB','D']},'changed_tensors':v['changed_tensors']})
  angles.append({'source':r['source'],'id_PRIVATE':r['id_PRIVATE'],'GT_eligible':r['GT_eligible'],**rec['gradient_oracle_cosine']})
  for k in mat['arms']:
   delta=r['arms'][k]['IoU']-r['arms']['context_zero']['IoU']
   if abs(delta)>1e-12:cases.append({'source':r['source'],'id_PRIVATE':r['id_PRIVATE'],'arm':k,'zero_IoU':r['arms']['context_zero']['IoU'],'new_IoU':r['arms'][k]['IoU'],'delta_IoU':delta,'mechanism_identity':'UNKNOWN_NO_PRIVATE_MEDIA_REVIEW','source_GT_diagnostic':k.startswith('oracle')})
 dump(H/'PHYSICAL_ROWS_PRIVATE.json',physical);dump(H/'ANGLE_ROWS_PRIVATE.json',angles);dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases);print(json.dumps(summary),flush=True);print([(x['arm'],round(x['gain_pp'],6),x['better'],x['worse']) for x in impact],flush=True)
if __name__=='__main__':main()
