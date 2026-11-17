"""Full-cohort independent scalar audit and public-safe evidence export."""
from common147 import *
import numpy as np,csv,io,re,collections
PREFIX='studies/3eed_late_RGB_without_point_feature_mixing_20261118'
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
 check();assert rd(H/'IMAGE_AUDIT.json')['status']=='PASS';dec=rd(H/'IMAGE_DECISION.json');assert not dec['endpoints']['inc_pa_zero']['retained_development_candidate'];rows=rd(H/'IMAGE_ALL_ROWS_PRIVATE.json');mm=rd(H/'IMAGE_METHOD_ROWS.json');aa=rd(H/'IMAGE_AGGREGATE_ROWS.json');nv=0;mx=0.;changed=[];cs=['norm_TTA','inc_zero','inc_pa_Aimage','inc_pa_wrong_joint','inc_pa_Bimage','inc_pa_FA_FB_mean','inc_pa_FAB_FB_mean','Frozen_A'];con=[];den={(s,t):sum(r['sampling_weight'] for r in rows if (r['source'],r['target'])==(s,t)) for s,t in {(r['source'],r['target']) for r in rows}}
 for r in rows:
  name=r['source']+'_to_'+r['target'];sid=r['id_PRIVATE'];a=np.load(B/'image'/name/'samples_PRIVATE'/(sid+'.npz'));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];assert np.array_equal(a['queries'],a['scores'].argmax(1))
  for arm,q in zip(a['arm_names'],a['queries']):assert r['IoUs'][arm]==float(y[q]);nv+=1
 for m in mm:
  rr=[r for r in rows if (r['source'],r['target'])==(m['source'],m['target'])];w=np.array([r['sampling_weight'] if m['aggregation']=='design_weighted' else 1 for r in rr]);y=np.array([r['IoUs'][m['arm']] for r in rr]);v=np.array([np.average(z,weights=w)*100 for z in [y,y>.25,y>.5]]);err=float(abs(v-np.array([m[k] for k in ['mIoU','Acc25','Acc50']])).max());mx=max(mx,err);assert err<1e-10
 for a in aa:
  rr=[m for m in mm if (m['aggregation'],m['arm'])==(a['aggregation'],a['arm'])];assert len(rr)==6
  for k in ['mIoU','Acc25','Acc50']:assert abs(np.mean([r[k] for r in rr])-a[k])<1e-10
 for ctrl in cs:
  dy=np.array([r['IoUs']['inc_pa_zero']-r['IoUs'][ctrl] for r in rows]);cc=np.array([d*r['sampling_weight']/den[r['source'],r['target']]*100/6 for d,r in zip(dy,rows)]);ch=[r for r,d in zip(rows,dy) if abs(d)>1e-12];con.append({'method':'inc_pa_zero','control':ctrl,'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'weighted_gain_pp':float(cc.sum()),'largest_positive_pp':float(cc.max(initial=0)),'largest_negative_pp':float(cc.min(initial=0)),'positive_sum_pp':float(cc[cc>0].sum()),'negative_sum_pp':float(cc[cc<0].sum()),'changed_content_groups':len({r['content_group_PRIVATE'] for r in ch}),'changed_frames':len({r['frame_PRIVATE'] for r in ch}),'changed_scene_groups':len({(r['target'],r['scene_PRIVATE']) for r in ch})})
 scalar={'status':'PASS','n':len(rows),'task_values':nv,'max_error':mx,'all_negative_unknown_noB_retained':True};dump(H/'SCALAR_AUDIT.json',scalar);dump(H/'CONTRIBUTION_ROWS.json',con);runtime={k:v for k,v in rd(B/'image/SEALED.json').items() if k!='files'};summary={'phase':'frozen_evidence_entry_only','decision':dec,'TTA_run':False,'source_calibration_run':False,'native_audit':rd(H/'IMAGE_AUDIT.json'),'scalar_audit':scalar,'runtime':runtime,'source_weight_training':False,'FINAL_access':False};dump(H/'SUMMARY.json',summary)
 doc=['# R147 — Late RGB without new point-feature averaging','','## Decision','','**'+dec['disposition']+'**. This was the predeclared frozen-entry prerequisite, not parameterTTA. It failed, so no source-reader recalibration or newUpupdate wasrun; no loss/step/threshold grid rescues thisentry. RetaincoherentR145developmentcandidate, not perrecord/directionbestoutput. Parent>=2ppgoal remainsseparateandnotcountedfromGTdiagnostic/weakerreset baseline.','','## Measured failure and controlled change','','R146fixednew166path had rawA/Bpointfeaturemean-.140899pp, addedRGB+.012761, positionreference+.060488 andUpupdate-.045693pp. This is a specificpath decomposition, notproofallBpoints/registrationarebad. Original764B pointmixhadpositiveFrozenFvsA, so itisnotgloballyremoved. EarlierR105/R141/R142RGBexperiments areacknowledged; RGBB itself isnotnew.','','R147 changesonlythe166previouslymissing-B inputs: useoriginalA1024pointfeatures insteadA/Bmean. OriginalBcamera/image/physicalprojectiongraphs, sparsevisibility/supportqualification, sameHsource-q10positionreference, sourcepreparedCNN/bidirectionalgrounder/fullmatch/livecompetition, originalA256/boxcoordinates/sixqueryidentities unchanged. ActualPointAdapterprehook recordsinputhasheseachnewforward andindependentCPU checksagainstexactAfeatures. No newBPointNet isneeded orclaimed; rawB isstillusedforcalibration/visibilitygeometry, notpointfeatures.','','All1798exposed ordinaryinputs/sixdirections retainoriginalweightsandframe/content/sceneclusters. All49R146endpointrefs copiedfromsealedparentfiles andauditedbitexact. Newpa_* globallyfrozenarms useR145correspondingfrozenreadouts onunchanged1632; inc_pa_* useold1632R145norm_TTA andonlynew166specifiedfrozenreadout. These incrementalcontrols are NOT globallyno-update systems. Remaining868noBstaycompleteFrozenA. No targetGTchoiceperinput/platform/scene.','','## Frozen-entry results','',table(aa,['aggregation','arm','mIoU','Acc25','Acc50']),'',json.dumps(dec,indent=2),'','Registeredinc_pa_zeroprereq mustexceed coherentparentR145, FpointlateBzeroR146, samepositionAimage andwrongBbinding inbothraw/designweightedmeans. PureB/twofixedhalfmeans alsofullyreported, notselectedpercase. Passingthisstagewouldonlyauthorize sourceApointreader-matchedcalibration andsameUpTTA; failedstageisnotdirectevidencethatallpossibleTTAcannotwork.','','## Gains and harms','',table(con,['method','control','better','worse','weighted_gain_pp','largest_positive_pp','largest_negative_pp','changed_content_groups','changed_frames','changed_scene_groups']),'','ThesearecandidateboxIoU changes, notindependentlyverifiedphysicalobjectidentity. Allchangedpositiveandnegativecases kept; sharedframes/descriptions/source directions arenotindependentrecordings. No singleDrone scene populationCI or removinglargestharm.','','## Measurement and cost','',json.dumps(summary,indent=2),'','Engineering18inputs(6new/6oldB/6noB), sourceweightsfingerprints/completeAbridge; nativeCPU independentlyrebuildsBprojection/permutation/visibilitygraphs, positionMLP/sixpositions, finalnativehead/livecompetition fromsavedfeatures; actualpointAdapterinputhashesoriginalA andpositionsR146exact. FullTransformer/PointNet areNOTindependentlyCPUrecomputed. NoUpchanges/newsourceweighttraining/expert/architecture/FINAL. Predictionsseal beforetaskGTscore; allscalarsindependentlyrecomputed. Incrementalresearchcosts reuseoldbackbones/registration/prefixes/parentoutputs andarenotcoldstartlatency. UniqueGPU/hard09:00BeijingNov18guard, scoped2GiBpayload/12GiBfreefloor/1GiBsafety, noolddata/results/weightsdeleted. SourceFpointκwasNOTsilentlyreusedforanApointTTAreader: no newC/teacher/updatetargetwasconstructed afterfailedfrozenprerequisite.']
 (H/'REPORT.md').write_text('\n'.join(doc)+'\n');O=H/'publication/public_export_attempt001';O.mkdir(parents=True,exist_ok=False);fm={s:f'FRAME{i+1:04d}' for i,s in enumerate(sorted({r['frame_PRIVATE'] for r in rows}))};cm={s:f'CONTENT{i+1:04d}' for i,s in enumerate(sorted({r['content_group_PRIVATE'] for r in rows}))};sc={t:sorted({r['scene_PRIVATE'] for r in rows if r['target']==t}) for t in ['waymo','drone','quad']}
 for name in sorted({r['source']+'_to_'+r['target'] for r in rows}):
  old=rd(R/'research/round135/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'));ids={r['id_PRIVATE']:i+1 for i,r in enumerate(old)};pub=[]
  for r in [r for r in rows if r['source']+'_to_'+r['target']==name]:
   basic={'source':r['source'],'target':r['target'],'row':f'ROW{ids[r["id_PRIVATE"]]:04d}','frame_cluster':fm[r['frame_PRIVATE']],'content_cluster':cm[r['content_group_PRIVATE']],'scene_cluster':f'CLUSTER{sc[r["target"]].index(r["scene_PRIVATE"])+1:03d}','has_B':r['has_B'],'original_has_B':r['original_has_B'],'backfilled':r['backfilled'],'sampling_weight':r['sampling_weight']};pub.append(basic|{k+'__IoU':v for k,v in r['IoUs'].items()})
   for ctrl in cs:
    d=r['IoUs']['inc_pa_zero']-r['IoUs'][ctrl]
    if abs(d)>1e-12:changed.append(basic|{'method':'inc_pa_zero','control':ctrl,'method_IoU':r['IoUs']['inc_pa_zero'],'control_IoU':r['IoUs'][ctrl],'delta_IoU':d,'query_changed':r['queries_PRIVATE']['inc_pa_zero']!=r['queries_PRIVATE'][ctrl]})
  csvout(O/'tasks'/(name+'.csv'),pub)
 csvout(O/'results/ALL_CHANGED_CASES.csv',changed)
 for fn in ['IMAGE_METHOD_ROWS','IMAGE_AGGREGATE_ROWS','IMAGE_PAIRED_ROWS','CONTRIBUTION_ROWS']:csvout(O/'results'/(fn+'.csv'),rd(H/(fn+'.json')))
 csvout(O/'results/SCENE_ROWS.csv',[{**{k:v for k,v in r.items() if not k.endswith('_PRIVATE')},'scene_cluster':f'CLUSTER{sc[r["target"]].index(r["scene_PRIVATE"])+1:03d}'} for r in rd(H/'IMAGE_SCENE_ROWS_PRIVATE.json')]);jobs=[]
 for p in sorted((H/'job_receipts').glob('*.json')):
  j=rd(p);jobs.append({'resource':j['resource'],'program':Path(j['command'][1]).name,'flags':' '.join(j['command'][2:]),'seconds':(datetime.datetime.fromisoformat(j['finished_utc'])-datetime.datetime.fromisoformat(j['started_utc'])).total_seconds(),'returncode':j['returncode']})
 csvout(O/'results/JOB_COSTS.csv',jobs)
 for fn in ['SUMMARY','IMAGE_DECISION','IMAGE_ENGINEERING_AUDIT','IMAGE_AUDIT','SCALAR_AUDIT']:dump(O/(fn+'.json'),rd(H/(fn+'.json')))
 for p in (H/'experiment').glob('*.json'):
  if p.name!='DEPENDENCIES_LOCK.json':dump(O/'config'/p.name,rd(p))
 dump(O/'config/PRIVATE_DEPENDENCIES_RECEIPT.json',{'sha256':sha(H/'experiment/DEPENDENCIES_LOCK.json'),'private_paths_not_exported':True})
 for p in list(H.glob('*.py'))+[R/'tools/experiment_guard_round147.py']:
  q=O/'own_code'/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(p.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 (O/'REPORT.md').write_text((H/'REPORT.md').read_text());(O/'REPRODUCTION.md').write_text('# Reproduction\nLicensedprivateR131/R135/R145/R146inputs/sourcepreparedstates/media/calibration andparentsealsrequired, notredistributed. image147 engineering ->audit_image147 engineering -> image147 formal ->audit_image147 formal ->score_image147 ->registereddecisionbranch. Frozenprereqfailed: no sourcecalibration/TTA. package_image147 writesallnegative/scalar/clusterresults. No newexperiment afterarchived09:00deadline withoutnewauthorization.\n');dump(O/'TRANSPORT_SPLITS.json',{'status':'PASS','tables':SPLITS,'all_rows_retained':True});tokens=set()
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  for ob in [e['row'],e['A_sensor']]+([e['pair_PRIVATE']['B']] if e['has_B'] else []):
   for k in ['id','caption','scene','imagepath','pointpath','metapath']:
    v=ob.get(k)
    if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 for p in O.rglob('*'):
  if p.is_file():
   t=p.read_text();assert p.suffix in ['.json','.md','.csv','.py'] and 'LOCAL_HOME/' not in t and not any(x in t for x in tokens),p;assert not re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',t)
 dump(H/'PUBLIC_EXPORT_AUDIT.json',{'status':'PASS','private_literals_scanned':len(tokens),'raw_media_words_coords_IDs_features_weights_secrets_exported':False});dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file()]});print(summary,flush=True)
if __name__=='__main__':main()
