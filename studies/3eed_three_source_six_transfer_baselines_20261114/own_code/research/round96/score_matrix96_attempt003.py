"""Post-seal frozen six-direction scoring; never used to select training epochs."""
from pathlib import Path
import sys,json,numpy as np,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou;from obb_iou import oriented_iou
rd=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 seal=rd(B/'frozen_matrix_attempt003/MATRIX_SEALED.json');assert seal['state'].startswith('SIX_');out=H/'frozen_scores_attempt003';out.mkdir();allrows=[];summary=[];audit=[];maxerr=0.;nchecks=0
 for direction in seal['directions']:
  src,tgt=direction['source'],direction['target'];p=Path(direction['prediction_directory']);se=rd(p/'sealed.json');assert sha(p/'sealed.json')==direction['prediction_seal_sha256'];idx=rd(H/'experiment'/('OFFICIAL_'+tgt.upper()+'_VAL_ORDER_PRIVATE.json'));assert len(idx)==se['n'];rows=[]
  for i,r in enumerate(idx):
   f=p/(str(i).zfill(6)+'.npz');assert sha(f)==se['files'][f.name];z=np.load(f,allow_pickle=False);assert z['anno_index']==i;gt=np.asarray(r['source_GT7_float32'],np.float64);parts={}
   for h,hn in enumerate(z['head_names']):
    boxes=z['boxes'][h].astype(np.float64);pred7=np.c_[boxes,np.zeros(256)];ys=many_iou(pred7,gt);c,s=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    for method in ['text_logodds','text_bbf','text_bbs','official_bbf','official_bbs']:
     v=z[method][h];order=np.argsort(-v,kind='stable');q=int(order[0]);k=str(hn)+method;parts[k]={'IoU':float(ys[q]),'top5_IoU':float(ys[order[:5]].max()),'top10_IoU':float(ys[order[:10]].max()),'oracle_IoU_GT_diagnostic_only':float(ys.max())}
     if h==0 and method=='text_logodds':
      check=oriented_iou(pred7[q,:3],pred7[q,3:6],np.eye(3),gt[:3],gt[3:6],rot);maxerr=max(maxerr,abs(check-ys[q]));nchecks+=1;assert abs(check-ys[q])<1e-6
   rows.append({'row':'ROW'+str(i+1).zfill(6),'source':src,'target':tgt,'scene_PRIVATE':r['scene'],'arms':parts});allrows.append(rows[-1])
  for k in rows[0]['arms']:
   v=np.array([r['arms'][k]['IoU'] for r in rows]);summary.append({'source':src,'target':tgt,'arm':k,'n':len(v),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'Top5_Acc25_percent':float(np.mean([r['arms'][k]['top5_IoU']>.25 for r in rows])*100),'Top10_Acc25_percent':float(np.mean([r['arms'][k]['top10_IoU']>.25 for r in rows])*100),'oracle_Acc25_GT_diagnostic_percent':float(np.mean([r['arms'][k]['oracle_IoU_GT_diagnostic_only']>.25 for r in rows])*100),'GT_class_span_assisted':k.split('_',1)[1].startswith('official'),'TTA':False})
  dump(out/(src+'_to_'+tgt+'_ALL_ROWS_PRIVATE.json'),rows);print('SCORE96',src,tgt,len(rows),flush=True)
 dump(out/'ALL_METHOD_ROWS.json',summary);dump(out/'SCORING_RECEIPT.json',{'state':'PASS_R96_POSTSEAL_FROZEN_MATRIX_AND_INDEPENDENT_SELECTED_3D_IOU','n_directions':6,'unique_target_inputs':sum(len(rd(H/'experiment'/('OFFICIAL_'+t.upper()+'_VAL_ORDER_PRIVATE.json'))) for t in ['waymo','drone','quad']),'direction_task_inputs':len(allrows),'independent_general_3D_selected_checks':nchecks,'max_abs_yaw_vs_general_3D_error':maxerr,'primary':'last_text_logodds','GT_used_for_model_selection':False,'FINAL_scored':False,'TTA':False});dump(out/'SEALED.json',{'files':{p.name:sha(p) for p in out.glob('*.json')}})
 # Compact report is actual frozen baselines, not adaptation results.
 L=['# R96 — 三个独立source与六个跨平台冻结基线','','主checkpoint固定epoch100；未做目标域TTA。训练采用官方100epoch/FP32/B24/AdamW配置；原Waymo-only直接复用，Drone与Quad各自从通用RoBERTa和随机grounding模块训练。','', '验证清单在loader读取前排除原FINAL384帧，实际Waymo3782/Drone2792/Quad2736；因此该表不是包含FINAL的官方全量VAL分数。','', '| source → target | n | mIoU % | Acc25 % | Acc50 % |','|---|---:|---:|---:|---:|']
 for s in summary:
  if s['arm']=='last_text_logodds':L.append(f'| {s["source"]} → {s["target"]} | {s["n"]} | {s["mIoU_percent"]:.6f} | {s["Acc25_percent"]:.6f} | {s["Acc50_percent"]:.6f} |')
 L+=['','纯文本主体词readout沿用已注册的first-noun词表，无目标GT参与选框。官方类别跨度辅助BBF/BBS仅作分开列出的诊断，不与部署式指标混称。原256框由各自独立source生成，不跨模型硬对齐query编号。GT与native训练预处理一致，Dronez+1.8、GT9D取前7D、Waymo前视坐标；这些官方表示限制需与未来完整旋转几何评价分开。','', '这轮仅建立跨平台TTA的source与冻结参照，不宣称适应增益。R91的Waymo准备模块没有载入Drone/Quad-only模型。',''];(H/'REPORT.md').write_text('\n'.join(L));print({'complete':True,'directions':6,'TTA':False})
if __name__=='__main__':main()
