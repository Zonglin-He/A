"""Export only own diagnostic code, anonymous scalar traces and visual-audit labels."""
import sys,io,csv,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round28'))
from common28 import *
def main():
 assert read(EXP/'CPU_AUDIT.json')['state']==read(EXP/'POINT_SUPPORT_CPU_AUDIT.json')['state']=='PASSED'
 pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def copy(p,d):
  dest=out/d;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 for folder in [BASE,EXP.parent]:
  for p in sorted(folder.glob('*.py')):copy(p,p.relative_to(ROOT))
 for n in ['PLAN_LOCK.json','FINAL_CODE_SNAPSHOT.json','REVIEW_LIMITATIONS.json']:copy(EXP/n,Path('protocol')/n)
 for n in ['CPU_AUDIT.json','POINT_SUPPORT_CPU_AUDIT.json','DECISION.json','DIAGNOSTIC_SUMMARY.json','FUSION_SUMMARY.json','IDENTITY24_SUMMARY.json','IDENTITY24_ROWS.json','POINT_SUPPORT24_SUMMARY.json','POINT_SUPPORT24_ROWS.json','VISUAL_SAME_INSTANCES_SUMMARY.json','VISUAL_SAME_INSTANCES_ROWS.json','SOURCE_PRIOR_AUDIT.json','CAPTION_LABEL_JOIN_AUDIT.json']:
  copy(EXP/n,Path('results')/n)
 copy(BASE/'blind_review_attempt001/ANNOTATIONS.json','review/SEALED_ASSISTANT_VISUAL_ANNOTATIONS.json');copy(BASE/'blind_review_attempt001/REVIEW_SCOPE.json','review/REVIEW_SCOPE.json');copy(EXP/'REPORT.md','REPORT.md')
 rr=read(EXP/'FUSION_ROWS_PRIVATE.json');scenes={p:sorted({x['scene_PRIVATE'] for x in rr if x['platform']==p}) for p in ['drone','quad']};buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['alias','platform','scene_alias','stage','arm','iou','query_changed_from_original','has_pool','original_mapped'],lineterminator='\n');w.writeheader();n=0
 for r in rr:
  for a,v in r['ious'].items():w.writerow({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_s'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'stage':r['stage'],'arm':a,'iou':v,'query_changed_from_original':r['q'][a]!=r['q']['joint_original'],'has_pool':r['has_pool'],'original_mapped':r['original_mapped']});n+=1
 (out/'results/ALL_COUNTERFACTUAL_SCALAR_ROWS.csv').write_text(buf.getvalue())
 for name in ['FUSION_ALL_NEGATIVES','SOURCE_PRIOR_AUDIT_ROWS']:
  a=read(EXP/(name+'.json'));buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(a[0]),lineterminator='\n');w.writeheader();w.writerows(a);(out/'results'/(name+'.csv')).write_text(buf.getvalue())
 (out/'README.md').write_text('''# R28 — locate R27 failures without new model execution
Offline cached diagnosis over all528R27 observations, plus24input-hash selected
multi-instance blind visual panels (12/platform). Assistant visual review hid
Qwen/source scores, grounding selected boxes,3DGT andIoU; independent detector
proposals were visible. Labels sealed before unblinding. One assistant annotator,
NOT expert-adjudicated identity gold.3uncertain cases retained unchanged.

Fourwrong strict visual rankings,15correct,2ties. Two correct-image/mapped-cue
cases switch to other visual-point support after source fusion; one correct
image escapes to an unmapped zero-cue candidate. Highest cues broadcast to6-23
queries (median14). No independently confirmed complete wrong-instance mapping
in this subset; sharedcamera/masks cannot certify external physical identity.

All387pooled cases have mapped source winners. Originaljoint selects an unmapped
query in34/528cases; Q64has1such query change but bothIoUs0. Anchor/mapped-only
counterfactual adds0.317953pp onOLD400,0.168197pp onALL528,0onQ64. Gains12,harms7
globally; mean/median centering worsensALL528. Fixedcommonoffset stress retained;
no bestoffset selected. These are posthoc diagnostics, NOT new heldout gains.
Sourcefirstnoun BBFtoken readout verified all528; fulltext still encoded. Raw
noun-token probability andQwenbinarylogodds are not same probability space.

Dependencies through19dfcc13c27daab1b823724bdf85dd7cdb360812 and licensed3EED assets.
No newQwen/groundingforward,training,GPUexperiment orTTA. OldproductionG preserved.
Raw RGB panels/captions/pointclouds/calibrationcoords/opaque taskIDs/modelweights
not exported.24case visual-support labels assume same camera/masks;3Dgeometry
checks explicitly proxies. No error partition generalized to all528 or unseen
scenes. All scalar counterfactuals and negatives retained, repeated arms not
independent observations. Read REPORT.md and audit/limitation files.
Workflow: prepare28 -> blind assistant review/seal -> fusion28,identity28,
support28,summary28,prior28,visual_compare28,join28,audit28,audit_support28,
finalize28,publish28. Existing sealed outputs must not be overwritten.
''')
 dump(out/'DEPENDENCIES.json',{'prior_commit':'19dfcc13c27daab1b823724bdf85dd7cdb360812','third_party_code_or_model_weights_redistributed':False,'no_new_method_or_TTA_claim':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];forbidden={str(r[k]) for r in rows() for k in ['id','caption','pointpath','imagepath','metapath','scene']}
 for p in files:assert not any(s and s in p.read_text() for s in forbidden),p
 dump(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True});files=[p for p in sorted(out.rglob('*')) if p.is_file()];prefix='studies/3eed_G_failure_decomposition_20261101'
 dump(pub/'TREE_PAYLOAD_LOCAL.json',[{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files]);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'scalar_rows':n,'unique_observations':528,'blind_visual_cases':24,'privacy_passed':True,'prefix':prefix,'all_negative_records_retained':True});print('EXPORT28',len(files),n)
if __name__=='__main__':main()
