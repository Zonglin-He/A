"""Saved-scalar audit and private-data-free public report/export."""
from common144 import *
import numpy as np,csv,io,re,shutil
PREFIX='studies/3eed_source_reference_positional_conditioning_20261118'
SPLITS=[]
def csvout(p,rows):
 if not rows:return
 cols=list(dict.fromkeys(k for r in rows for k in r));buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=cols,lineterminator='\n');w.writeheader();header=buf.getvalue();parts=[];ss=header;nr=0;ns=[]
 for r in rows:
  b=io.StringIO();ww=csv.DictWriter(b,fieldnames=cols,lineterminator='\n');ww.writerow(r);line=b.getvalue()
  if nr and len((ss+line).encode())>400000:parts.append(ss);ns.append(nr);ss=header;nr=0
  ss+=line;nr+=1
 parts.append(ss);ns.append(nr)
 if len(parts)==1:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(parts[0])
 else:
  d=p.with_suffix('');d.mkdir(parents=True,exist_ok=True)
  for i,s in enumerate(parts,1):(d/('part'+str(i).zfill(3)+'.csv')).write_text(s)
  assert sum(len(list(csv.DictReader(io.StringIO(s)))) for s in parts)==len(rows);SPLITS.append({'table':p.name,'part_rows':ns,'total_rows':len(rows),'row_order_preserved':True})
def table(rr,cols):return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(str(r.get(k,'')) for k in cols)+' |' for r in rr)
def main():
 check();assert rd(H/'AUDIT.json')['status']=='PASS';rows=rd(H/'ALL_ROWS_PRIVATE.json');mm=rd(H/'METHOD_ROWS.json');aa=rd(H/'AGGREGATE_ROWS.json');dec=rd(H/'DECISION.json');nv=0;mx=0.
 for r in rows:
  name=r['source']+'_to_'+r['target'];p=np.load(B/'formal'/name/'samples_PRIVATE'/(r['id_PRIVATE']+'.npz'));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+r['id_PRIVATE']+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];assert np.array_equal(p['queries'],p['scores'].argmax(1))
  for arm,q in zip(p['arm_names'],p['queries']):assert r['IoUs'][arm]==float(y[q]);nv+=1
 for m in mm:
  rr=[r for r in rows if (r['source'],r['target'])==(m['source'],m['target'])];w=np.array([r['sampling_weight'] if m['aggregation']=='design_weighted' else 1 for r in rr]);y=np.array([r['IoUs'][m['arm']] for r in rr]);v=np.array([np.average(z,weights=w)*100 for z in [y,y>.25,y>.5]]);err=float(abs(v-np.array([m[k] for k in ['mIoU','Acc25','Acc50']])).max());mx=max(mx,err);assert err<1e-10
 for a in aa:
  m=[m for m in mm if (m['aggregation'],m['arm'])==(a['aggregation'],a['arm'])];assert len(m)==6
  for k in ['mIoU','Acc25','Acc50']:assert abs(np.mean([r[k] for r in m])-a[k])<1e-10
 scalar={'status':'PASS','n':len(rows),'task_values':nv,'max_error':mx,'all_negative_noB_ineffective_rows_kept':True};dump(H/'SCALAR_AUDIT.json',scalar);meta=rd(H/'POSITION_ROWS_PRIVATE.json');ms=[]
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  dd=[r['delta'] for r in meta if (r['source'],r['target'])==(src,tgt) and r['has_B']];ms.append({'source':src,'target':tgt,'B':len(dd),'delta_mean':float(np.mean(dd)),'delta_q10':float(np.quantile(dd,.1)),'delta_median':float(np.median(dd)),'delta_q90':float(np.quantile(dd,.9))})
 dump(H/'DELTA_SUMMARY.json',ms);contr=[]
 for ctrl in ['Frozen_A','F_RGBB_only','opposite_reference','half_zero_aligned','image_B_TTA']:
  den={(s,t):sum(r['sampling_weight'] for r in rows if (r['source'],r['target'])==(s,t)) for s,t in {(r['source'],r['target']) for r in rows}};dy=np.array([r['IoUs']['aligned_reference']-r['IoUs'][ctrl] for r in rows]);cc=np.array([d*r['sampling_weight']/den[r['source'],r['target']]*100/6 for d,r in zip(dy,rows)]);ch=[r for r,d in zip(rows,dy) if abs(d)>1e-12];contr.append({'control':ctrl,'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'weighted_gain_pp':float(cc.sum()),'largest_positive_pp':float(cc.max(initial=0)),'largest_negative_pp':float(cc.min(initial=0)),'changed_content_groups':len({r['content_group_PRIVATE'] for r in ch}),'changed_frames':len({r['frame_PRIVATE'] for r in ch}),'changed_scene_groups':len({(r['target'],r['scene_PRIVATE']) for r in ch})})
 dump(H/'CONTRIBUTION_ROWS.json',contr);runtime={k:v for k,v in rd(B/'formal/SEALED.json').items() if k!='files'};summary={'decision':dec,'preparation':rd(H/'PREPARATION.json'),'engineering':rd(H/'ENGINEERING_AUDIT.json'),'native_audit':rd(H/'AUDIT.json'),'scalar_audit':scalar,'runtime':runtime,'delta_summary':ms,'parameter_TTA_claim':False,'FINAL_access':False};dump(H/'SUMMARY.json',summary)
 doc=['# R144 — Source-reference positional conditioning prerequisite','','## Decision','','**'+dec['disposition']+'**. This is a fixed, source-calibrated input-state prerequisite, not a claim of a new parameter-TTA gain. R142 remains an unchanged development reference. All1798 previously exposed records, sixdirections and allnegative/noB cases remain.','','## Mechanism and fixed scope','','The checkpoint-compatible archived runtime learns absolute positions in point visual self-attention, vision-to-language queries, and decoder query self-attention. Language-to-vision keys do not directly receive pos_feats. Source-specific absolute-origin priors are a hypothesis; no camera-coordinate/preprocessing error is asserted. The current public official bdetr uses a different backbone backend, so the actual checkpoint-compatible runtime is the computational authority. [Official positional modules](https://github.com/worldbench/3EED/blob/main/models/encoder_decoder_layers.py) support the entry-point observation, not the efficacy of our estimator.','','One fixed source-only formula: Hsource is the median rawsample lower10percentile height over302deduplicatedsource-owninputs; delta=Hsource-q10(exactoriginalA16384sample). Labels do not enter source statistics, target quantiles or the sign/scale. Lower10percentile is NOT certifiedground/sensorheight; terrain, scenecomposition andFOV are competing explanations. No quantile/offset/strength/axis search.','','On the original764B-available records only, recompute pointpos_embed atz+delta and translatecenterZ in allsix cachedqueryposition arrays. Keep physicalxyz, PointNet features, registeredFpoint1024, A/Bcamera graphs, originalroute/support andrelativegeometricpriors, sourceparameters, queryidentities andoriginalA256outputboxes fixed. Recompute complete native matching andlivecompetition. This isolates positional conditioning, not a fully translated pointcloud/backbonepass. Frozenrouting is an intentionally held prior for allarms. All1034noB remain exactcompleteFrozenA, including the newcontrols. No sourceweighttraining, expert, architecture, loss or parameter update.','','Controls are same-magnitude opposite-reference adjustment and fixedone-half originaljointzero/aligned probabilitymean, plus allR142 zero/direct/wrong/TTA/pureB/canonicalmeans byte-reused. A positive inputstate method would be an additional frozen baseline; parameteradaptation must later exceed that baseline before an adaptation contribution. No matching to an unexecuted globallytranslated PointNet is claimed.','','## Whole-cohort results','',table(aa,['aggregation','arm','mIoU','Acc25','Acc50']),'',json.dumps(dec,indent=2),'','The predeclared retention rule requires aligned > ownzero/opposite/fixedhalfmean inbothraw anddesignweightedmeans. R142 comparisons are also shown. This is exposedtarget development, not an independent test, and no per-direction bestoutput is selected. A negative recipe ends without an offset/quantilegrid; it only weakens this tested estimator/intervention, not every possible coordinate adaptation.','','## Input reference distribution','',table(ms,['source','target','B','delta_mean','delta_q10','delta_median','delta_q90']),'','Signed reference changes differ because each source has its own frozen statistic. These are aggregated diagnostics, not object coordinates or inferred identity labels.','','## All gains and harms','',table(contr,['control','better','worse','weighted_gain_pp','largest_positive_pp','largest_negative_pp','changed_content_groups','changed_frames','changed_scene_groups']),'','IoU changes need not imply a different physicalobject. Frames/content groups can be shared by multiple descriptions/source directions. Full sixdirection/scenegroup and changedrecord tables remain, without deleting large harms or claiming a singleDrone scene yields a populationconfidenceinterval.','','## Audit and resources','',json.dumps(summary,indent=2),'','Predictions sealed before taskscore. Engineering zero correction reproduces complete source readout, sourcefingerprints remain unchanged. CPU independently recomputes the actual positional MLP including evalBN, allsixcenterZ changes withunchangedXY/sizes, and fullfinalmatching/competition probabilities from saved nativefeatures. It checks originalA256/ref/noB/halfmean exactly. Source rawsample coordinates were verified against originalseed coordinates during preparation; this preparatory check is not an independent fullbackbone replay. WholeTransformer isnot independently regenerated onCPU. Every savedIoU/aggregate isrecalculated. Allruns use uniqueGPU/deadlineguard; no newexperiment after09:00BeijingNov18. Rawmedia/descriptions/coordinates/IDs/features/modelweights/secrets areprivate; noFINAL.']
 (H/'REPORT.md').write_text('\n'.join(doc)+'\n');O=H/'publication/public_export_attempt001';O.mkdir(parents=True,exist_ok=False);fm={s:f'FRAME{i+1:04d}' for i,s in enumerate(sorted({r['frame_PRIVATE'] for r in rows}))};cm={s:f'CONTENT{i+1:04d}' for i,s in enumerate(sorted({r['content_group_PRIVATE'] for r in rows}))};sc={t:sorted({r['scene_PRIVATE'] for r in rows if r['target']==t}) for t in ['waymo','drone','quad']};changed=[]
 for name in sorted({r['source']+'_to_'+r['target'] for r in rows}):
  old=rd(R/'research/round135/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'));ids={r['id_PRIVATE']:i+1 for i,r in enumerate(old)};pub=[]
  for r in [r for r in rows if r['source']+'_to_'+r['target']==name]:
   basic={'source':r['source'],'target':r['target'],'row':f'ROW{ids[r["id_PRIVATE"]]:04d}','frame_cluster':fm[r['frame_PRIVATE']],'content_cluster':cm[r['content_group_PRIVATE']],'scene_cluster':f'CLUSTER{sc[r["target"]].index(r["scene_PRIVATE"])+1:03d}','has_B':r['has_B'],'sampling_weight':r['sampling_weight']};pub.append(basic|{k+'__IoU':v for k,v in r['IoUs'].items()})
   for c in ['Frozen_A','F_RGBB_only','opposite_reference','half_zero_aligned','image_B_TTA']:
    d=r['IoUs']['aligned_reference']-r['IoUs'][c]
    if abs(d)>1e-12:changed.append(basic|{'control':c,'aligned_IoU':r['IoUs']['aligned_reference'],'control_IoU':r['IoUs'][c],'delta_IoU':d,'query_changed':r['queries_PRIVATE']['aligned_reference']!=r['queries_PRIVATE'][c]})
  csvout(O/'tasks'/(name+'.csv'),pub)
 csvout(O/'results/ALL_CHANGED_CASES.csv',changed)
 for fn in ['METHOD_ROWS','AGGREGATE_ROWS','PAIRED_ROWS','DELTA_SUMMARY','CONTRIBUTION_ROWS']:csvout(O/'results'/(fn+'.csv'),rd(H/(fn+'.json')))
 csvout(O/'results/SCENE_ROWS.csv',[{**{k:v for k,v in r.items() if not k.endswith('_PRIVATE')},'scene_cluster':f'CLUSTER{sc[r["target"]].index(r["scene_PRIVATE"])+1:03d}'} for r in rd(H/'SCENE_ROWS_PRIVATE.json')]);jobs=[]
 for p in sorted((H/'job_receipts').glob('*.json')):
  j=rd(p);jobs.append({'resource':j['resource'],'program':Path(j['command'][1]).name,'flags':' '.join(j['command'][2:]),'seconds':(datetime.datetime.fromisoformat(j['finished_utc'])-datetime.datetime.fromisoformat(j['started_utc'])).total_seconds(),'returncode':j['returncode']})
 csvout(O/'results/JOB_COSTS.csv',jobs)
 for fn in ['SUMMARY','DECISION','PREPARATION','ENGINEERING_AUDIT','AUDIT','SCALAR_AUDIT','DELTA_SUMMARY']:dump(O/(fn+'.json'),rd(H/(fn+'.json')))
 for p in (H/'experiment').glob('*.json'):
  if p.name!='DEPENDENCIES_LOCK.json':dump(O/'config'/p.name,rd(p))
 dump(O/'config/PRIVATE_DEPENDENCIES_RECEIPT.json',{'sha256':sha(H/'experiment/DEPENDENCIES_LOCK.json'),'privatepaths_not_exported':True})
 for p in list(H.glob('*.py'))+[R/'tools/experiment_guard_round144.py']:
  q=O/'own_code'/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(p.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 (O/'REPORT.md').write_text((H/'REPORT.md').read_text());(O/'REPRODUCTION.md').write_text('# Reproduction\nRequireslicensedprivateR135/R142prefixes/sourceweights/physicalgraphs/sourceR140metadata; notredistributed. prepare144 ->run144 engineering ->audit144 engineering ->run144 formal ->audit144 ->score144 ->package144. No parametersupdated ornewtraining. OriginalA256/nobox/noBcompleteFrozenA retained. ExistingR142refs byte-reused, notrerun. Quantileformula isfixedsource-only; targetGTonlypostseal. CSVchunksunderstemfolders preserveallrows/order. NoFINAL. Archiveddeadlineguardneedsfreshauthorizationaftercutoff.\n');dump(O/'TRANSPORT_SPLITS.json',{'status':'PASS','tables':SPLITS,'all_rows_retained':True});tokens=set()
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  for ob in [e['row'],e['A_sensor']]+([e['pair_PRIVATE']['B']] if e['has_B'] else []):
   for k in ['id','caption','scene','imagepath','pointpath','metapath']:
    v=ob.get(k)
    if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  for k in ['id','caption','scene','imagepath','pointpath','metapath']:
   v=e['entry']['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
 for p in O.rglob('*'):
  if p.is_file():
   t=p.read_text();assert p.suffix in ['.json','.md','.csv','.py'] and 'LOCAL_HOME/' not in t and not any(x in t for x in tokens),p;assert not re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',t)
 dump(H/'PUBLIC_EXPORT_AUDIT.json',{'status':'PASS','private_literals_scanned':len(tokens),'rawmedia_descriptions_coordinates_IDs_features_weights_keys_exported':False});dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file()]});print({'decision':dec,'scalar':scalar,'runtime':runtime},flush=True)
if __name__=='__main__':main()
