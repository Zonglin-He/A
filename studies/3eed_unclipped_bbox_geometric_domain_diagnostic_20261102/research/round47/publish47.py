"""Explicit anonymous-scalar/flag export; no private geometry, answer or identity."""
import json,csv,io,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parents[2];H=R/'research/round47';E=H/'experiment';I=H/'independent_audit';B=R/'crossplatform/round47/predictions208_attempt001';PREFIX='studies/3eed_unclipped_bbox_geometric_domain_diagnostic_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def safe(s):return s.replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ['files']}
 if isinstance(x,list):return [clean(v) for v in x]
 return safe(x) if isinstance(x,str) else x

def main():
 assert read(I/'AUDIT_RESULT.json')['state']=='PASS_R47_INDEPENDENT_CACHED_DOMAIN_CONTRACT' and read(I/'SCORING_RECEIPT.json')['task_values']==6032
 P=H/'publication';P.mkdir(exist_ok=True);O=P/'public_export_attempt001';O.mkdir(exist_ok=False)
 def cp(p,d):q=O/d;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(safe(Path(p).read_text()))
 def cpj(p,d):dump(O/d,clean(read(p)))
 cp(E/'REPORT.md','REPORT.md');cpj(E/'DECISION.json','DECISION.json')
 for p in sorted((R/'crossplatform/round47').glob('*.py')):cp(p,p.relative_to(R))
 for p in [H/'finalize47.py',Path(__file__),I/'audit47.py',I/'score47.py',I/'audit47_ATTEMPT001_PRESERVED.py']:cp(p,p.relative_to(R))
 for n in ['PLAN_LOCK','IMPLEMENTATION_LOCK','ROOT_EXECUTION_AUTHORIZATION','ROOT_STATIC_REVIEW','POSTSEAL_EVALUATION_AUTHORIZATION','CPU_TECH_CHECKS']:cpj(E/(n+'.json'),Path('protocol')/(n+'.json'))
 for n in ['IMPLEMENTATION_LOCK','IMPLEMENTATION_LOCK_ATTEMPT001','TECHNICAL_AUDIT_REPAIR']:cpj(I/(n+'.json'),Path('audit')/(n+'.json'))
 for n in ['RUN_AUDIT_ATTEMPT001','RUN_AUDIT_ATTEMPT002','RUN_SCORE_ATTEMPT001']:cp(I/(n+'.log'),Path('audit')/(n+'.log'))
 cp(E/'SOFTWARE_CHECKS_ATTEMPT001.log','audit/SOFTWARE_CHECKS_ATTEMPT001.log');cp(E/'PREDICT_ATTEMPT001.log','audit/PREDICT_ATTEMPT001.log')
 for n in ['AUDIT_RESULT','SCORING_RECEIPT','TASK_SUMMARY','DOMAIN_CONTRASTS','COVERAGE_UNKNOWN_SUMMARY']:cpj(I/(n+'.json'),Path('results')/(n+'.json'))
 rows=read(I/'ALL208_TASK_ROWS_PRIVATE.json');assert len(rows)==208;aliases={r['alias']:f"row{i+1:04d}" for i,r in enumerate(rows)};scenes={p:sorted({r['scene_PRIVATE'] for r in rows if r['platform']==p}) for p in ['drone','quad']};scalars=[];flags=[]
 for r in rows:
  alias=aliases[r['alias']];scene=r['platform']+'_scene'+str(scenes[r['platform']].index(r['scene_PRIVATE'])+1)
  for arm,v in r['IoU'].items():scalars.append({'row_alias':alias,'platform':r['platform'],'scene_alias':scene,'arm':arm,'IoU':v,'query_changed_vs_source':r['queries'][arm]!=r['queries']['source']})
  for arm,z in r['readouts'].items():flags.append({'row_alias':alias,'platform':r['platform'],'arm':arm,**{k:z[k] for k in ['mapped','status','group_size','query_changed','old_subset_contains_selected_mask','full_group_nonempty']}})
 assert len(scalars)==6032 and len(flags)==1664
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(scalars[0]),lineterminator='\n');w.writeheader();w.writerows(scalars);(O/'results/ALL6032_ANONYMOUS_TASK_SCALARS.csv').write_text(buf.getvalue());dump(O/'results/ALL1664_READOUT_FLAGS.json',flags)
 neg=[{'row_alias':aliases[r['alias']],**{k:r[k] for k in ['platform','arm','reference','delta_pp']}} for r in read(I/'ALL_ORDERED_NEGATIVE_COMPARISONS.json')];assert len(neg)==16135;dump(O/'results/ALL16135_ORDERED_NEGATIVE_COMPARISONS.json',neg)
 raw=[{'row_alias':aliases[r['alias']],**{k:r[k] for k in ['platform','capacity','raw_unchanged','strict_status','outside_only_recovered','values_unchanged']}} for r in read(I/'ALL416_RAW_BINDING_SCALARS.json')];assert len(raw)==416;dump(O/'results/ALL416_UNCHANGED_RAW_DOMAIN_FLAGS.json',raw)
 scene_rows=[{'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(scenes[r['platform']].index(r['scene_PRIVATE'])+1),'n':r['n'],'mIoU_percent':r['mIoU_percent'],'descriptive_only_few_exposed_scenes':True} for r in read(I/'SCENE_SUMMARIES_PRIVATE.json')];dump(O/'results/SCENE_SUMMARIES.json',scene_rows)
 seal=read(B/'SEALED.json');dump(O/'protocol/PREDICTION_SEAL_METADATA.json',clean(seal)|{'original_container_sha256':sha(B/'SEALED.json'),'private_filemap_excluded':True})
 cp(R/'research/NIGHT_20261101/reviews/BOUNDS_DOMAIN_INDEPENDENT_REVIEW.md','history/PREEXECUTION_BOUNDS_DOMAIN_REVIEW.md')
 dump(O/'SOURCES.json',{'prior3Bbbox':'studies/3eed_native_bbox_capacity_interface_factorial_20261102','prior7Bbbox_and_strict_results':'studies/3eed_native_bbox_capacity_interface_factorial_20261102','official_geometry':'https://docs.pytorch.org/vision/main/generated/torchvision.ops.box_iou.html','model_calls':0,'licensed_private_assets_required_to_reproduce':True,'prior_helpers_referenced_not_redistributed':True,'no_model_or_thirdparty_snapshot':True})
 (O/'README.md').write_text('''# R47 — cached bbox geometry-domain diagnostic

STOP the fixed domain configuration; no new TTA or 7B final admission.
The predeclared 7B unclipped rectangle-IoU primary is20.285883%, below
strict20.465032% and legacy24.727373%. Center secondary21.548940% improves
strict by+.997740pp but is not reselected as primary. All A208 rows are exposed
few-scene development data, not independent confirmation. No FINAL/U/T access.
All29arms/6032taskIoUs,16135orderednegative comparisons,1664readoutflags and
416raw-invariance/domain flags are exported. Negative comparisons repeat the
same observations and are not independent statistical evidence.
Only strict outside-range rejection is relaxed for finite positive xyxy;
3B2/7B115 recoveries keep original coordinates, units, area, schema and grid.
Original21strict arms/4368taskvalues exact. Unclipped intersection uses all
existing masks;59empty-maxgroup events do not skip to second best. Center
must remain in-image and hit one mask. No clipping, reordering, unit switch,
new threshold, prompt, model call or optimization. Geometry-domain admissibility
does not certify original prompt compliance, semantic identity or coordinate units.
First audit failure was a missing diagnostic exception-name string in the
independent helper; status/value agreed. Original code/lock/log retained;
only that absent diagnostic comparison was removed, then audit passed and
one independent post-seal task score ran. No prediction or method changed.
This export contains our code/protocol and anonymous scalar/flag allowlists.
It excludes original UIDs/queries/mask IDs, raw answers/boxes/coordinates,
images/point clouds/captions/token tensors/features/weights and thirdparty
snapshots. Rows are locally anonymous row0001..0208; no private linking map.
''')
 # Content scan is post-analysis metadata validation, not new model/task evaluation.
 forbidden={str(r[k]) for r in rows for k in ['id_PRIVATE','scene_PRIVATE','alias'] if r.get(k)}
 hist=read(R/'research/round27/input/ALL528_PRIVATE.json')[:208]
 forbidden|={str(r[k]) for r in hist for k in ['id','caption','imagepath','pointpath','metapath','scene'] if r.get(k)}
 findings=[]
 for p in O.rglob('*'):
  if not p.is_file():continue
  t=p.read_text()
  for f in forbidden:
   if f in t:findings.append({'file':p.relative_to(O).as_posix(),'kind':'private_token','token_sha256':hashlib.sha256(f.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',t):findings.append({'file':p.relative_to(O).as_posix(),'kind':'home_path'})
 assert not findings,findings[:8]
 blocked={'id','id_PRIVATE','scene','scene_PRIVATE','alias','caption','imagepath','pointpath','metapath','query','queries','mask_index','mask_IoU','value','point','bbox_original_xyxy','bbox_resized_unchanged','raw_answer_PRIVATE','raw','generated_token_ids','source_boxes','source_scores','masks_bool','pixel_values','input_ids','image_grid_thw'};bad=[]
 def walk(x,p):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in blocked:bad.append({'file':p,'key':k})
    walk(v,p)
  elif isinstance(x,list):
   for v in x:walk(v,p)
 for p in O.rglob('*.json'):walk(read(p),p.relative_to(O).as_posix())
 assert not bad,bad[:8]
 dump(O/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS','explicit_anonymous_scalar_flag_allowlist':True,'private_token_findings':findings,'blocked_fields':bad,'all6032values_16135negatives_1664flags_416rawinvariance':True,'no_private_geometry_or_ID_or_raw_content':True,'first_audit_failure_and_original_lock_retained':True,'no_FINAL_U_T_reads':True})
 dump(O/'protocol/CODE_SNAPSHOT.json',{'own_code_only':[{'path':p.relative_to(O).as_posix(),'sha256':sha(p)} for p in sorted(O.rglob('*.py'))]})
 files=[p for p in sorted(O.rglob('*')) if p.is_file()];dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(O).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True});files=[p for p in sorted(O.rglob('*')) if p.is_file()]
 payload=P/'TREE_PAYLOAD_LOCAL.json';dump(payload,[{'path':PREFIX+'/'+p.relative_to(O).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files])
 for f in read(O/'MANIFEST.json')['files']:assert sha(O/f['path'])==f['sha256'] and (O/f['path']).stat().st_size==f['bytes']
 assert all(v['content']==p.read_text() for v,p in zip(read(payload),files))
 result={'state':'PASS','files':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':payload.stat().st_size,'payload_sha256':sha(payload),'prefix':PREFIX,'all6032values_16135negatives_1664flags_416rawinvariance':True,'privacy_findings':[],'all_manifest_payload_SHAexact':True,'not_published':True};dump(P/'LOCAL_EXPORT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
