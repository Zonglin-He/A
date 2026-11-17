"""Independent scalar checks and public export of native RGB crop controls."""
from common152 import *
import numpy as np,csv,io,re,collections,shutil
PREFIX='studies/3eed_fullframe_native_RGB_resolution_prerequisite_20261118'
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
 controls={'full_high':['norm_joint_zero','full_low','full_wrong','norm_TTA','Frozen_A'],'full_high_parent_mean':['norm_TTA','full_high_zero_mean','full_high','full_low_parent_mean','full_wrong_parent_mean','Frozen_A'],'full_low':['norm_joint_zero','full_high','full_wrong','norm_TTA','Frozen_A']}
 for r in rows:
  name=r['source']+'_to_'+r['target'];sid=r['id_PRIVATE'];y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];a=np.load(B/'frozen'/name/'samples_PRIVATE'/(sid+'.npz'));old=np.load(R/'crossplatform/round149/method'/name/'samples_PRIVATE'/(sid+'.npz'));p150=np.load(R/'crossplatform/round150/frozen'/name/(sid+'.npz'));qs=dict(zip(old['arm_names'],old['queries']));qs.update(dict(zip(p150['arm_names'],p150['queries'])));p151=np.load(R/'crossplatform/round151/frozen'/name/'samples_PRIVATE'/(sid+'.npz'));qs.update(dict(zip(p151['arm_names'],p151['queries'])));qs.update(dict(zip(a['arm_names'],a['queries'])));assert np.array_equal(a['queries'],a['scores'].argmax(1))
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
 prep=rd(H/'PREPARATION_AUDIT.json');summary=dict(decision=dec,preparation=prep,native_audit=rd(H/'FROZEN_AUDIT.json'),engineering_audit=rd(H/'FROZEN_ENGINEERING_AUDIT.json'),scalar_audit=scalar,source_calibration_run=False,TTA_run=False,FINAL_access=False,exposed_development_not_independent=True);dump(H/'SUMMARY.json',summary)
 method='''Full original A and registered B images are both resized to 768x512, producing 96x64=6144 native CNN tokens per bank and 12288 total. There is no candidate-dependent crop selection. Original physical visibility, point positions, fused 1024 point features, normalized position references, original A256 and all source-prepared weights remain fixed. The native model is fully convolutional on images; no image encoder weights, extra semantic model, source training or new architecture are added.

The low-detail control first downsizes each original frame to 384x256, then upsizes it to 768x512. It uses exactly the same high-grid graphs as the fresh-pixel reader. This separates genuine additional pixel information from image lattice, spatial neighborhood, normalization and token-budget changes. All original media are at least as large as the high-resolution input.

Point-to-image edges use the original 3x3 grid-neighborhood rule at the new 96x64 lattice. Image-to-point edges use up to eight projected points with the same distance and depth discontinuity criteria. Offsets are derived from each actual image token count, not the old hardcoded1536. The unchanged CNN is evaluated separately per image with GroupNorm, and outputs are concatenated in exactly the same order as graph offsets. No 32x48 image-mask auxiliary task is used. The physical extent of a 3x3 kernel is smaller than before; this is part of the input/grid change and is controlled by the low-detail arm, not claimed to be pure photometric enhancement.

The wrong-binding control deterministically permutes visible points within exact forward-degree buckets independently per camera and inversely remaps image-to-point edges. It preserves images, visibility, lattice and the exact number of forward edges for every point. This changes content/location binding, not just synchronous K/V ordering. Cases with no effective permutation are retained. Original joint zero, R145 TTA and all112 historical arms remain. Fixed half-means with zero or old TTA are ensemble controls, not new parameter adaptation or independent confirmation.'''

 audit='''All variable-grid geometry and actual raw-pixel provenance are independently checked before any model, across the complete1798 input registry. Producer, independent graph/geometry auditor, CNN engineering auditor, streamed full matching/competition auditor and task scorer are pinned before the first forward. Engineering covers12 inputs,18 new full-reader forwards plus6 original-zero bridge forwards. Actual CNN pre-hooks check every normalized 512x768 RGB tensor and point-adapter pre-hooks check exact frozen F1024. Independent CPU reconstruction regenerates engineering CNN outputs, physical graphs and complete final learned readout. Source and prepared weights stay unchanged.

Formal execution seals chunks of two cases, independently audits native full matching/competition and variable-grid geometry, then deletes only its own new TEMP features/graphs after PASS. Predictions, raw-pixel/input hashes, graph recipes, persistent records and audit receipts remain. Full transformer forward is not independently regenerated on CPU; no backprop occurs in this prerequisite. Formal actual pixel tensors are provenance-checked by hooks, while engineering CNN outputs are separately recomputed on CPU.

All1798 inputs,764 eligible B and1034 no-B completeFrozen-A fallbacks, remain in every denominator. Predictions are sealed and audited before taskGT evaluation. Shared recording/frame/content clusters and all positive/negative results are retained; IoU alone is not an instance-identity annotation. This is exposed development, not new independent validation; no single-scene population CI or per-input/per-direction GT selection is claimed. FINAL is untouched.

Resource profile includes retained engineering CNN arrays, all new predictions and the formal streamed chunk. New model payload cap0.18GiB, free floor12GiB plus1GiB safety. No old data, weights or experiment artifacts are deleted. Cached-pipeline runtime plus CPU audit is not a cold deployment latency benchmark. The fixed09:00 Beijing Nov18 deadline remains in effect.'''

 key=['Frozen_A','norm_joint_zero','norm_Aimage','norm_Bimage','norm_TTA','full_high','full_low','full_wrong','full_high_zero_mean','full_high_parent_mean','full_low_parent_mean','full_wrong_parent_mean']
 doc=['# R152 — Full-frame native RGB resolution prerequisite','','## Decision','','**'+dec['disposition']+'**. The predeclared full-frame fresh-detail gate fails. No new source calibration or parameter adaptation is run. Keep the prior R145 development candidate (13.819603% versus intact Frozen-A 13.394463%, +0.425140 pp); the requested >=2 pp improvement is NOT achieved.','','## Input mechanism and controls','',method,'','## Results','',table([a for a in aa if a['arm'] in key],['aggregation','arm','mIoU','Acc25','Acc50']),'',json.dumps(dec,indent=2),'','## Positive and negative effects','',table(contr,['method','control','better','worse','weighted_gain_pp','largest_positive_pp','largest_negative_pp','changed_content_groups','changed_frames','changed_scene_groups']),'','The primary gate requires full_high to beat original joint zero, full_low and full_wrong in both raw and design-weighted six-direction means; it does not require all directions to rise. Low detail and full-frame point-binding permutation are matched controls. Parent-TTA ensemble retention is separately gated against parent TTA, high zero-mean, pure high, low and wrong parent means. No per-input, per-platform or per-direction GT choice is made. A failed fixed2x full-frame recipe does not exclude all RGB or TTA methods.','','## Measurement, resources and limitations','','The scalar checker independently recomputes all saved task values.','',audit,'',json.dumps(summary,indent=2)]
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
 resource=dict(status='PASS',measured_utc=utc(),payload_bytes=sum(p.stat().st_size for p in B.rglob('*') if p.is_file()),free_bytes=shutil.disk_usage(R).free,cap_GiB=.18,free_floor_GiB=12,safety_GiB=1,pre_model_peak_estimate=rd(H/'RESOURCE_PROFILE.json')['conservative_peak_bytes'],actual_runtime_storage_checked_every_chunk=True,post_score_observation_not_measured_peak=True);assert resource['payload_bytes']<=.18*1024**3 and resource['free_bytes']>=12*1024**3;dump(H/'RESOURCE_AUDIT.json',resource)
 for fn in ['SUMMARY','DECISION','PREPARATION_AUDIT','RESOURCE_PROFILE','RESOURCE_AUDIT','FROZEN_AUDIT','FROZEN_ENGINEERING_AUDIT','SCALAR_AUDIT']:dump(O/(fn+'.json'),rd(H/(fn+'.json')))
 for p in (H/'experiment').glob('*.json'):
  if p.name!='DEPENDENCIES_LOCK.json':dump(O/'config'/p.name,rd(p))
 dump(O/'config/PRIVATE_DEPENDENCIES_RECEIPT.json',dict(sha256=sha(H/'experiment/DEPENDENCIES_LOCK.json'),private_paths_not_exported=True))
 for p in list(H.glob('*.py'))+[R/'tools/experiment_guard_round152.py']:
  q=O/'own_code'/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(p.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 (O/'REPORT.md').write_text((H/'REPORT.md').read_text());zh=['# R152：完整场景原生 RGB 分辨率对照','','结论：完整场景2倍分辨率没有通过预设证据条件，停止这一配方，不继续源域数值标定或参数拟合。保持R145为开发参照，2pp目标未实现。','','本轮保留完整A/B场景，从原图得到768×512输入；低细节对照先缩为384×256再上采样，使用完全相同的96×64特征格和几何关联。错配在可见点的同连接数分桶内置换点—像素关系，同时保留反向映射、图像内容和可见性。没有候选区域裁剪，没有新增视觉模型。','','所有1024点特征、位置参照、原256框、主体/参照物接口及源域训练权重保持固定。不能只改图像而沿用旧1536像素索引；本轮关联图已按6144格重新生成，并独立重建验证。原3×3格邻域对应的物理范围会变化，低细节同格对照用于区分这项影响与真实新增像素。','','## 六方向设计加权结果','',table([a for a in aa if a['aggregation']=='design_weighted' and a['arm'] in key],['arm','mIoU','Acc25','Acc50']),'','## 预设裁决','',json.dumps(dec,indent=2),'','先封存预测并完成独立原生读出审计，再读取GT评分。工程阶段实际CNN输入、输出另用CPU核对；正式阶段逐批核对几何、点输入和完整匹配/竞争读出。所有1798条、1034条无B完整Frozen回退和全部正负例保留，119个历史及新增读出完整报告。冻结集成不是新的参数TTA。FINAL未访问。'];(O/'REPORT_ZH.md').write_text('\n'.join(zh)+'\n')
 (O/'REPRODUCTION.md').write_text('# Reproduction\nPrivate source states, original licensed RGB, calibration, point features and parent R145/R149/R150/R151 outputs are required and excluded. prepare152 (independent geometry included) -> profile152 -> frozen152 --engineering -> audit_frozen152 --engineering -> frozen152 -> audit_frozen152 -> score_frozen152 -> package152. Every model and CPU task scoring uses experiment_guard_round152 and the fixed 09:00 Beijing Nov18 cutoff. No new source calibration or TTA after failed full-frame gate.\n')
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
 dump(H/'PUBLIC_EXPORT_AUDIT.json',dict(status='PASS',private_literals_scanned=len(tokens),raw_media_words_coords_IDs_features_weights_keys_exported=False));dump(O/'MANIFEST.json',dict(prefix=PREFIX,files=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()]));print('PACKAGE152',scalar,dec['disposition'],flush=True)
if __name__=='__main__':main()
