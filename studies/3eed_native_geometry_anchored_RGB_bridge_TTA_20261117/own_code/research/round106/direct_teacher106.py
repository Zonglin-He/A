"""Complete same-information controls from the targets ALREADY sealed before GT.

No new model, coefficient search or GT-dependent selection. Original 34-arm
matrix omitted explicit direct q; retain it and add this transparent supplement.
"""
from pathlib import Path
import json,hashlib,numpy as np,sys
R=Path(__file__).resolve().parents[2];H=R/'research/round106';B=R/'crossplatform/round106';O=B/'direct_teacher_supplement_PRIVATE';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
def dump(p,z):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def main():
 matrix=rd(B/'MATRIX_SEALED.json');O.mkdir(exist_ok=False);arms=['teacher_direct_all','teacher_direct_matched','teacher_direct_matched_C'];seals={}
 # q, pA/pC and fallback predictions already existed before original task scoring.
 for name,h in matrix['directions'].items():
  root=B/name;assert sha(root/'SEALED.json')==h;D=O/name;D.mkdir();src,_,tgt=name.split('_')
  for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
   z=np.load(root/'samples_PRIVATE'/(r['id']+'.npz'));pp=dict(zip(z['arm_names'],z['scores']));rec=rd(root/'records_PRIVATE'/(r['id']+'.json'));fit=rec['fits'].get('real');q=np.array(fit['target']) if fit else pp['prepared_frozen'];enabled=rec['families'].get('real',{}).get('updated_output_enabled',False);matched=q if enabled else pp['real_frozen_fusedABC'];fc=(q+pp['real_frozen_C'])/2 if enabled else pp['real_frozen_ensemble'];scores=np.stack([q,matched,fc]);assert np.isfinite(scores).all() and np.allclose(scores.sum(1),1);np.savez_compressed(D/(r['id']+'.npz'),arm_names=np.array(arms),scores=scores,source_boxes=z['source_boxes'],anno_index=z['anno_index'])
  dump(D/'SEALED.json',{'files':{f.name:sha(f) for f in sorted(D.glob('*.npz'))},'origin_original_seal_sha256':h});seals[name]=sha(D/'SEALED.json')
 dump(O/'MATRIX_SEALED.json',{'n':768,'arms':arms,'directions':seals,'original_targets_sealed_before_any_GT':True,'no_new_neural_predictions':True,'original_matrix_not_overwritten':True})
 # Only now enter post-seal supplementary target scoring.
 results=[];tasks=[]
 for name,h in seals.items():
  src,_,tgt=name.split('_');assert sha(O/name/'SEALED.json')==h;rows=rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');golds=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');values={a:[] for a in arms};oldrows={x['id_PRIVATE']:x for x in rd(H/'scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'))}
  for r in rows:
   f=O/name/(r['id']+'.npz');z=np.load(f);boxes=z['source_boxes'].astype(float);boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);ys=many_iou(np.c_[boxes,np.zeros(256)],np.array(golds[r['anno_index']]['source_GT7_float32']));v={a:float(ys[p.argmax()]) for a,p in zip(arms,z['scores'])};tasks.append({'source':src,'target':tgt,'id_PRIVATE':r['id'],'arms':v});row=oldrows[r['id']]
   for a in arms:values[a].append(v[a])
  for a in arms:
   x=np.array(values[a]);base=np.array([oldrows[r['id']]['arms']['real_TTA_fusedABC']['IoU'] for r in rows]);results.append({'source':src,'target':tgt,'arm':a,'n':len(x),'mIoU_percent':float(x.mean()*100),'Acc25_percent':float((x>.25).mean()*100),'Acc50_percent':float((x>.5).mean()*100),'TTA_minus_direct_pp':float((base-x).mean()*100),'TTA_better':int((base>x+1e-12).sum()),'TTA_worse':int((base<x-1e-12).sum()),'TTA_same':int((abs(base-x)<=1e-12).sum())})
 aggregate=[{'arm':a,**{k:float(np.mean([x[k] for x in results if x['arm']==a])) for k in ['mIoU_percent','Acc25_percent','Acc50_percent','TTA_minus_direct_pp']}} for a in arms];dump(H/'DIRECT_TEACHER_SUMMARY.json',{'state':'PASS_SEALED_TEACHER_SAME_INFORMATION_CONTROL_SUPPLEMENT','n':768,'readouts':2304,'aggregate':aggregate,'directions':results,'original_matrix34arms_retained':True,'GT_not_used_to_construct_readouts':True,'no_new_model':True});dump(H/'DIRECT_TEACHER_ROWS_PRIVATE.json',tasks);p=H/'REPORT.md';p.write_text(p.read_text()+'\n## 补齐直接采用监督分布的同信息对照\n\n正式34臂矩阵没有单列直接q读出。为完整评价“把监督写入参数是否有额外收益”，这里保留原矩阵，并从当时已封存的q、A/C预测构造固定补充读出；没有新模型调用或按GT选择权重。direct_all在B可用时直接采用q；direct_matched只在与正式TTA相同的B/C可用条件采用q，否则使用原冻结融合；direct_matched_C在相同条件将q与冻结C等权平均，否则使用原冻结ensemble。补充预测再次封存后才评分，属于已暴露开发集诊断，不是独立确认。\n\n'+json.dumps(aggregate,indent=2)+'\n');print(json.dumps(aggregate,indent=2))
if __name__=='__main__':main()
