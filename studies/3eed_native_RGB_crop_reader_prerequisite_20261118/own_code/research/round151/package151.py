"""Independent scalar checks and public export of native RGB crop controls."""
from common151 import *
import numpy as np,csv,io,re,collections,shutil
PREFIX='studies/3eed_native_RGB_crop_reader_prerequisite_20261118'
SPLITS=[]
def csvout(p,rows):
 if not rows:return
 cols=list(dict.fromkeys(k for r in rows for k in r));buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=cols,lineterminator='\n');w.writeheader();head=buf.getvalue();parts=[];counts=[];ss=head;nr=0
 for r in rows:
  b=io.StringIO();csv.DictWriter(b,fieldnames=cols,lineterminator='\n').writerow(r);line=b.getvalue()
  if nr and len((ss+line).encode())>400000:parts.append(ss);counts.append(nr);ss=head;nr=0
  ss+=line;nr+=1
 parts.append(ss);counts.append(nr)
 if len(parts)==1:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(parts[0])
 else:
  d=p.with_suffix('');d.mkdir(parents=True,exist_ok=True)
  for i,s in enumerate(parts,1):(d/('part'+str(i).zfill(3)+'.csv')).write_text(s)
  assert sum(len(list(csv.DictReader(io.StringIO(s)))) for s in parts)==len(rows);SPLITS.append(dict(table=p.name,part_rows=counts,total_rows=len(rows),row_order_preserved=True))
def table(rows,cols):return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(str(r.get(k,'')) for k in cols)+' |' for r in rows)
def main():
 check();assert rd(H/'FROZEN_AUDIT.json')['status']=='PASS';dec=rd(H/'DECISION.json');assert not dec['proceed_sourcecal_and_TTA'],'Finish the registered source/TTA phase before packaging a positive prerequisite.'
 rows=rd(H/'ALL_ROWS_PRIVATE.json');mm=rd(H/'METHOD_ROWS.json');aa=rd(H/'AGGREGATE_ROWS.json');nv=0;mx=0.
 controls={'high_local':['norm_joint_zero','region_swap_local','wrong_local','duplicate_local','low_local','norm_TTA','Frozen_A'],'high_parent_mean':['norm_TTA','high_zero_mean','high_local','region_swap_parent_mean','wrong_parent_mean','duplicate_parent_mean','low_parent_mean','Frozen_A'],'low_local':['norm_joint_zero','high_local','duplicate_local','norm_TTA','Frozen_A']}
 for r in rows:
  name=r['source']+'_to_'+r['target'];sid=r['id_PRIVATE'];y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];a=np.load(B/'frozen'/name/'samples_PRIVATE'/(sid+'.npz'));old=np.load(R/'crossplatform/round149/method'/name/'samples_PRIVATE'/(sid+'.npz'));p150=np.load(R/'crossplatform/round150/frozen'/name/(sid+'.npz'));qs=dict(zip(old['arm_names'],old['queries']));qs.update(dict(zip(p150['arm_names'],p150['queries'])));qs.update(dict(zip(a['arm_names'],a['queries'])));assert np.array_equal(a['queries'],a['scores'].argmax(1))
  assert set(qs)==set(r['IoUs'])
  for arm,q in qs.items():assert r['IoUs'][arm]==float(y[q]);nv+=1
 for m in mm:
  rr=[r for r in rows if (r['source'],r['target'])==(m['source'],m['target'])];w=np.array([r['sampling_weight'] if m['aggregation']=='design_weighted' else 1 for r in rr]);y=np.array([r['IoUs'][m['arm']] for r in rr]);v=np.array([np.average(z,weights=w)*100 for z in [y,y>.25,y>.5]]);err=float(abs(v-np.array([m[k] for k in ['mIoU','Acc25','Acc50']])).max());mx=max(mx,err);assert err<1e-10
 for a in aa:
  rr=[m for m in mm if (m['aggregation'],m['arm'])==(a['aggregation'],a['arm'])];assert len(rr)==6
  for k in ['mIoU','Acc25','Acc50']:assert abs(np.mean([r[k] for r in rr])-a[k])<1e-10
 den={(s,t):sum(r['sampling_weight'] for r in rows if (r['source'],r['target'])==(s,t)) for s,t in {(r['source'],r['target']) for r in rows}};contr=[]
 for method,cs in controls.items():
  for ctrl in cs:
   dy=np.array([r['IoUs'][method]-r['IoUs'][ctrl] for r in rows]);cc=np.array([d*r['sampling_weight']/den[r['source'],r['target']]*100/6 for d,r in zip(dy,rows)]);ch=[r for r,d in zip(rows,dy) if abs(d)>1e-12];contr.append(dict(method=method,control=ctrl,better=int((dy>1e-12).sum()),worse=int((dy< -1e-12).sum()),weighted_gain_pp=float(cc.sum()),largest_positive_pp=float(cc.max(initial=0)),largest_negative_pp=float(cc.min(initial=0)),changed_content_groups=len({r['content_group_PRIVATE'] for r in ch}),changed_frames=len({r['frame_PRIVATE'] for r in ch}),changed_scene_groups=len({(r['target'],r['scene_PRIVATE']) for r in ch})))
 scalar=dict(status='PASS',n=len(rows),arms=len(rows[0]['IoUs']),task_values=nv,max_error=mx,all_historical_and_new_negative_cases_kept=True);dump(H/'SCALAR_AUDIT.json',scalar);dump(H/'CONTRIBUTION_ROWS.json',contr)
 prep=rd(H/'PREPARATION.json');summary=dict(decision=dec,preparation=prep,native_audit=rd(H/'FROZEN_AUDIT.json'),engineering_audit=rd(H/'FROZEN_ENGINEERING_AUDIT.json'),scalar_audit=scalar,source_calibration_run=False,TTA_run=False,FINAL_access=False,exposed_development_not_independent=True);dump(H/'SUMMARY.json',summary)
 method='''Original full A and B images remain present. Up to three local banks are selected from the original supported candidate groups, ranked by the fixed half-mean of frozen A-image and B-image predictions. Support Jaccard must be <=0.5 relative to previously selected regions. At least four distinct visible points are required in one camera; choose the camera with more supported points, breaking ties toward A. This selects uncertain predicted regions, not certified instances, and does not read task GT.

Each ROI comes from those projected points with 1.5 margin, 3:2 aspect ratio, raw minimum 48x32, shifted inside the image. High-resolution evidence is cropped from the original licensed RGB frame and resized to 384x256. The low-resolution control uses exactly the same floating ROI on the existing downsized full-frame image, then resizes to the same bank size. Their point/pixel graphs are identical. Local 3x3 point-to-image and depth-gated eight-point image-to-point associations are independently recomputed after the crop affine. Physical coordinates, camera visibility/depth, 1024 fused point features, normalized positional references, original A256 and trained weights remain unchanged.

Every bank passes through the same trained three-convolution RGB encoder with independent per-bank GroupNorm, then the existing native bidirectional interaction and full language/competition reader. No new image encoder or source training is added. Crop banks are an input distribution change; trained scale invariance is not assumed.

Controls: (1) low-resolution crops isolate fresh detail from magnification/context and new graph support; (2) within-crop point-binding permutation preserves images and visibility but may retain the same object evidence; (3) duplicate corresponding full frames match bank count but differ in graph extent; (4) cyclic crop-image exchange across selected regions preserves the entire crop-image multiset and original high-crop graphs while breaking region/content binding. Original joint zero-update, both single-image readers, original R145 TTA and all 101 parent arms remain available. Fixed half-means with the parent TTA are ensemble controls, not new parameter updates or independent replications.'''
 audit='''Producer, crop selection, independent geometry auditor, formal auditor and task scorer were pinned before the first network forward. Before any model or task score, a control amendment added the stronger region-image swap while retaining the weaker within-region point permutation; the old preparation/profile attempts are retained privately and the independent preparation audit was rerun. It is not a post-score gate change.

Engineering executes 12 inputs: 30 new reader forwards plus six original-joint bridge forwards. Independent CPU checks regenerate actual CNN inputs from raw media, the CNN outputs and all geometry, and reconstruct the full learned matching/competition readout. Actual RGB pre-hooks verify every normalized tensor in the native CNN call order; point-adapter pre-hooks verify bit-identical frozen point features. All prepared and source states are checked unchanged. Formal execution streams chunks of eight, seals new features/graphs/predictions, independently reconstructs the native head and crop geometry on CPU, and only after PASS deletes its own new TEMP feature/graph files. Persistent predictions, recipe provenance, pixel/input hashes and audit receipts remain. Full transformer forward and backprop are not independently regenerated on CPU. Formal CNN tensors are provenance-checked in the actual forward; engineering CNN outputs are independently regenerated.

All 1798 inputs, 764 with original B and 1034 without B, remain in every denominator. No-B inputs reproduce complete Frozen-A. Predictions are sealed before task scoring; post-seal GT is for evaluation only. All positive and negative results and recording/frame/content clusters are retained. IoU changes do not establish object identity. This is repeated exposed-development analysis, not independent confirmation, and no single-scene population CI is claimed. FINAL is untouched.

Resource cap is 0.4 GiB of new model payload, with 12 GiB free floor plus 1 GiB safety and streamed audit/deletion of only new temporary files. No historical experiments, data, weights or checkpoints are deleted. Runtime tables include CPU audit and cached-pipeline overhead; they are not cold end-to-end deployment latency.'''
 key=['Frozen_A','norm_joint_zero','norm_Aimage','norm_Bimage','norm_TTA','high_local','low_local','wrong_local','region_swap_local','duplicate_local','high_zero_mean','high_parent_mean','low_parent_mean','wrong_parent_mean','region_swap_parent_mean','duplicate_parent_mean']
 doc=['# R151 — Native original-RGB crop reader prerequisite','','## Decision','','**'+dec['disposition']+'**. The predeclared frozen-evidence gate fails. No new source calibration or parameter adaptation is run. Keep the prior R145 development candidate (13.819603% versus intact Frozen-A 13.394463%, +0.425140 pp); the requested >=2 pp improvement is NOT achieved.','','## Input mechanism and controls','',method,'','## Results','',table([a for a in aa if a['arm'] in key],['aggregation','arm','mIoU','Acc25','Acc50']),'',json.dumps(dec,indent=2),'','## Positive and negative effects','',table(contr,['method','control','better','worse','weighted_gain_pp','largest_positive_pp','largest_negative_pp','changed_content_groups','changed_frames','changed_scene_groups']),'','The primary gate requires high_local to beat original joint zero, duplicate frames and region-image swap in both raw and design-weighted six-direction means; it does not require all directions to rise. Low resolution and within-crop point permutation are fully reported diagnostics. Parent-TTA ensemble retention is separately gated against parent TTA, high zero-mean, pure high, and low/swap/duplicate parent means. No per-input, per-platform or per-direction GT choice is made. A failed crop recipe does not exclude all higher-resolution, RGB or TTA methods.','','## Measurement, resources and limitations','','The first package-only scalar checker used a wrong comprehension variable and raised before export. It was repaired without changing any predictions, scoring definitions or results; the failed code/log are retained privately.','',audit,'',json.dumps(summary,indent=2)]
 (H/'REPORT.md').write_text('\n'.join(doc)+'\n');O=H/'publication/public_export_attempt001';O.mkdir(parents=True,exist_ok=False)
 fm={s:f'FRAME{i+1:04d}' for i,s in enumerate(sorted({r['frame_PRIVATE'] for r in rows}))};cm={s:f'CONTENT{i+1:04d}' for i,s in enumerate(sorted({r['content_group_PRIVATE'] for r in rows}))};sc={t:sorted({r['scene_PRIVATE'] for r in rows if r['target']==t}) for t in ['waymo','drone','quad']};changed=[]
 for name in sorted({r['source']+'_to_'+r['target'] for r in rows}):
  ids={r['id_PRIVATE']:i+1 for i,r in enumerate(rd(R/'research/round135/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json')))};pub=[]
  for r in [r for r in rows if r['source']+'_to_'+r['target']==name]:
   basic=dict(source=r['source'],target=r['target'],row=f'ROW{ids[r["id_PRIVATE"]]:04d}',frame_cluster=fm[r['frame_PRIVATE']],content_cluster=cm[r['content_group_PRIVATE']],scene_cluster=f'CLUSTER{sc[r["target"]].index(r["scene_PRIVATE"])+1:03d}',has_B=r['has_B'],sampling_weight=r['sampling_weight']);pub.append(basic|{k+'__IoU':v for k,v in r['IoUs'].items()})
   for method,cs in controls.items():
    for ctrl in cs:
     d=r['IoUs'][method]-r['IoUs'][ctrl]
     if abs(d)>1e-12:changed.append(basic|dict(method=method,control=ctrl,method_IoU=r['IoUs'][method],control_IoU=r['IoUs'][ctrl],delta_IoU=d,query_changed=r['queries_PRIVATE'][method]!=r['queries_PRIVATE'][ctrl])|r['components'])
  csvout(O/'tasks'/(name+'.csv'),pub)
 csvout(O/'results/ALL_CHANGED_CASES.csv',changed)
 for fn in ['METHOD_ROWS','AGGREGATE_ROWS','PAIRED_ROWS','CONTRIBUTION_ROWS']:csvout(O/'results'/(fn+'.csv'),rd(H/(fn+'.json')))
 csvout(O/'results/SCENE_ROWS.csv',[{**{k:v for k,v in r.items() if not k.endswith('_PRIVATE')},'scene_cluster':f'CLUSTER{sc[r["target"]].index(r["scene_PRIVATE"])+1:03d}'} for r in rd(H/'SCENE_ROWS_PRIVATE.json')])
 jobs=[]
 for p in sorted((H/'job_receipts').glob('*.json')):
  j=rd(p);jobs.append(dict(resource=j['resource'],program=Path(j['command'][1]).name,seconds=(datetime.datetime.fromisoformat(j['finished_utc'])-datetime.datetime.fromisoformat(j['started_utc'])).total_seconds(),returncode=j['returncode']))
 csvout(O/'results/JOB_COSTS.csv',jobs)
 resource=dict(status='PASS',measured_utc=utc(),payload_bytes=sum(p.stat().st_size for p in B.rglob('*') if p.is_file()),free_bytes=shutil.disk_usage(R).free,cap_GiB=.4,free_floor_GiB=12,safety_GiB=1,pre_model_peak_estimate=rd(H/'RESOURCE_PROFILE.json')['conservative_peak_bytes'],actual_runtime_storage_checked_every_chunk=True,post_score_observation_not_measured_peak=True);assert resource['payload_bytes']<=.4*1024**3 and resource['free_bytes']>=12*1024**3;dump(H/'RESOURCE_AUDIT.json',resource)
 for fn in ['SUMMARY','DECISION','PREPARATION','PREPARATION_AUDIT','RESOURCE_PROFILE','RESOURCE_AUDIT','FROZEN_AUDIT','FROZEN_ENGINEERING_AUDIT','SCALAR_AUDIT']:dump(O/(fn+'.json'),rd(H/(fn+'.json')))
 for p in (H/'experiment').glob('*.json'):
  if p.name!='DEPENDENCIES_LOCK.json':dump(O/'config'/p.name,rd(p))
 dump(O/'config/PRIVATE_DEPENDENCIES_RECEIPT.json',dict(sha256=sha(H/'experiment/DEPENDENCIES_LOCK.json'),private_paths_not_exported=True))
 for p in list(H.glob('*.py'))+[R/'tools/experiment_guard_round151.py']:
  q=O/'own_code'/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(p.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 (O/'REPORT.md').write_text((H/'REPORT.md').read_text());zh=['# R151：原生 RGB 原图局部读取','','结论：原图裁剪没有通过预设的总体证据条件，本轮停止该裁剪配方，不继续标定或拟合新的教师目标。保持 R145 为开发参照；2pp 目标未实现。','','保留完整 A/B 场景，仅增加最多三个候选局部区域。区域来自固定 A/B 半均值排序与可见点支持，不使用 GT。原图裁剪与缩小图裁剪使用完全相同区域和几何图，分别检查真实细节与单纯放大。重复整图控制额外分支数量；区域内点置换是细对应诊断，跨区域交换图像是更强的内容绑定对照。两者都保留。','','三个区域中有2291个来自A、1个来自B，不能将其称为取得了新的B视角。原图细节来自已有授权观测，不是新增相机。原1024点特征、位置、256框、主干及源域训练权重全部固定。','','## 六方向设计加权结果','',table([a for a in aa if a['aggregation']=='design_weighted' and a['arm'] in key],['arm','mIoU','Acc25','Acc50']),'','## 预设裁决','',json.dumps(dec,indent=2),'','先封存预测、逐批独立重建几何与完整匹配读出，再读取GT评分。实际CNN输入与点特征通过前向hook核验，工程阶段另用CPU重算CNN。正式阶段只在独立审计通过后删除本轮新增临时特征，保留预测、配方和哈希回执。全部1798条、所有正负例、场景和内容簇均保留，112个历史及新增读出完整公开。冻结组合不是新的参数TTA，也不是独立复验。','','这一结果只否定当前候选驱动裁剪配方的整体收益，不等于所有RGB细节或三模态适应均无效。候选前验可能漏掉正确区域，局部放大也可能改变源训练尺度和上下文。不得依靠事后挑选平台或案例改写总体结论。FINAL未访问。'];(O/'REPORT_ZH.md').write_text('\n'.join(zh)+'\n')
 (O/'REPRODUCTION.md').write_text('# Reproduction\nPrivate source states, original licensed RGB, calibration, point features, recipes and parent R145/R149/R150 outputs are required and excluded. prepare151 -> audit_geometry151 -> profile151 -> frozen151 --engineering -> audit_frozen151 --engineering -> frozen151 -> audit_frozen151 -> score_frozen151 -> package151. Every model and CPU task scoring uses experiment_guard_round151 and the fixed 09:00 Beijing Nov18 cutoff. No new source calibration or TTA after failed frozen gate.\n')
 dump(O/'TRANSPORT_SPLITS.json',dict(status='PASS',tables=SPLITS,all_rows_retained=True));tokens=set()
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  for ob in [e['row'],e['A_sensor']]+([e['pair_PRIVATE']['B']] if e['has_B'] else []):
   for k in ['id','caption','scene','imagepath','pointpath','metapath']:
    v=ob.get(k)
    if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 for p in O.rglob('*'):
  if p.is_file():
   t=p.read_text();assert 'LOCAL_HOME/' not in t and not any(x in t for x in tokens),p;assert not re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',t)
 dump(H/'PUBLIC_EXPORT_AUDIT.json',dict(status='PASS',private_literals_scanned=len(tokens),raw_media_words_coords_IDs_features_weights_keys_exported=False));dump(O/'MANIFEST.json',dict(prefix=PREFIX,files=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()]));print('PACKAGE151',scalar,dec['disposition'],flush=True)
if __name__=='__main__':main()
