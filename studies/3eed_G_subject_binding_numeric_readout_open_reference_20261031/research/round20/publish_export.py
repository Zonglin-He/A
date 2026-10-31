"""Explicit public code/scalar allowlist, no media/captions/coordinates/weights."""
from pathlib import Path
import sys,shutil,csv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round20'))
from common import *
STUDY='3eed_G_subject_binding_numeric_readout_open_reference_20261031'
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_PRIVATE')}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def main():
 out=EXP.parent/'publication/public_export_attempt001';out.mkdir(parents=True,exist_ok=False)
 for p in list((ROOT/'crossplatform/round20').glob('*.py'))+list((ROOT/'research/round20').glob('*.py')):
  dest=out/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 shutil.copyfile(EXP/'REPORT.md',out/'REPORT.md')
 for p in sorted(EXP.glob('*LOCK*.json'))+[EXP/'AUTHORIZATION.json',EXP/'PROTOCOL_TEXT_ERRATUM.json']:dump(out/'protocol'/p.name,clean(read(p)))
 for name in ('EXPOSURE.json','Z_EXPOSURE.json'):dump(out/'protocol'/name,read(EXP.parent/'input'/name))
 for name in ('DECISION.json','CPU_REPLAY_AUDIT.json','SUBJECT_ATTRIBUTE_AND_REFERENCE_AUDIT.json','MECHANISM_DIAGNOSIS.json','ALL_PAIR_DIAGNOSTICS.json','QUALITATIVE_REVIEW.json'):dump(out/'results'/name,clean(read(EXP/name)))
 dump(out/'results/STRICT144_SUMMARY.json',read(EXP/'scored144_attempt001/SUMMARY.json'));dump(out/'results/STRICT144_TTA_GATE.json',read(EXP/'scored144_attempt001/TTA_PROGRESSION_GATE.json'));dump(out/'results/G2_208_SUMMARY.json',read(EXP/'scored208_attempt002/SUMMARY.json'));dump(out/'results/G2_208_TTA_GATE.json',read(EXP/'scored208_attempt002/TTA_PROGRESSION_GATE.json'))
 scenes={};counts={}
 for folder,dest in [('scored144_attempt001','STRICT144_ALL864_TASK_ARM_ROWS.csv'),('scored208_attempt002','G2_208_ALL1456_TASK_ARM_ROWS.csv')]:
  rows=[]
  for r in read(EXP/folder/'ROWS_PRIVATE.json'):
   key=(r['platform'],r['scene_PRIVATE'])
   if key not in scenes:scenes[key]=f'{r["platform"]}_scene{1+sum(k[0]==r["platform"] for k in scenes):02d}'
   for arm,v in r['ious'].items():rows.append({'alias':r['alias'],'phase':r['phase'],'platform':r['platform'],'scene_alias':scenes[key],'arm':arm,'iou':v,'Acc25':v>.25,'Acc50':v>.5,'gain_pp':100*(v-r['ious']['source']),'query_changed':r['changed'][arm],'G_intervenes':r['G_intervenes']})
  with (out/'results'/dest).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  counts[folder]=len(rows)
 assert counts=={'scored144_attempt001':864,'scored208_attempt002':1456}
 artifacts=[]
 names=['sourceW32_attempt001','sourceZ64_attempt001','text144_attempt001','textZ64_attempt001','pairs144_attempt001','pairsZ64_attempt001','dino144_attempt001','dinoZ64_attempt001','visual4_attempt001','visual144_attempt001','visualZ64_attempt001','direct144_attempt001','direct208_attempt002']
 for n in names:
  p=ROOT/'crossplatform/round20'/n;s=verify(p);artifacts.append({'artifact':n,'seal_sha256':sha(p/'SEALED.json'),**{k:v for k,v in s.items() if k not in ('files','programs')}})
 dump(out/'results/ARTIFACT_SCALARS.json',artifacts)
 for n in ('DINO_ASSET_RECEIPT.json','STANZA_ASSET_RECEIPT.json'):dump(out/'assets'/n,read(ROOT/'crossplatform/round20/assets'/n))
 dump(out/'DEPENDENCIES.json',{'prior_R19':'https://github.com/Zonglin-He/A/tree/ad5f8da77a3affeef0651e223d7de62357d24d4d/studies/3eed_top1_relational_evidence_pair_execution_20261031','official_3EED':'https://github.com/worldbench/3EED','grounding_DINO_model':'https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71','grounding_DINO_safetensors_sha256':read(ROOT/'crossplatform/round20/assets/DINO_ASSET_RECEIPT.json')['model_sha256'],'stanza_official':'https://github.com/stanfordnlp/stanza','runtime_target_versions':{'stanza':'1.15.0','emoji':'2.16.0','udtools':'0.2.9','orjson':'3.13.0','platformdirs':'4.14.0','networkx':'3.6.1'},'runtime_scope':'Dependencies installed only into round20/runtime; originalenvs unchanged. Source/DINO/CPU use existingPy3.11torch2.7cu128, frozenQwen uses existingmllmPy3.13env.','stanza_cached_models_revision':'88e0cd3ddaa0c5a4b682895527ac013ce0241c21','stanza_lemma_identity':'No oldlemma pickleddict loaded; no safetycheck disabled; POS/dependency extraction uses cachedpublicmodels.','source_checkpoint_sha256':'0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd','executed_G1':'source_capture.py,prepare_pairs.py,text.py,detect.py,visual.py,assemble.py; research/round20/score.py','executed_G2_Z':'sourceZ.py,prepareZ.py,textZ.py,detectZ.py,visualZ.py,assemble_v2.py; research/round20/score_v2.py','TTA_executed':False,'rawassets_weights_private_registrations_not_published':True,'reproduction':'OverlaypriorR19anditsdependencies, provide licensed3EED/frozen source/masks/Qwen and private registrations. InstalllistedNLPdependenciesinto localtargetruntime; obtain officialDINOweightandcachedStanfordmodels matching receipts. Exactprivatecohort requireslocalregistration. Publicbundledcode preservesexecutedhashchecks; not a standalone no-data demo. No paperfulltext/thirdpartycode/weights redistributed.'})
 (out/'README.md').write_text('# G repair: actual subject fields, numerical visual readout, open references\n\n[Full report](REPORT.md). Real Waymo-only→Drone/Quad direct-reranking experiments completed. 208unique observations, frozen source and candidate boxes, zero TTA parameter updates. Old18missing subject-field cases are actually scored, not merely codepatched. FixedYes/No logits replace freegenerations; numericaloutputsfinite, no formattinggate. IndependentGroundingDINO provides open-vocabulary reference regions.\n\nFirststrictDINOgate144 preserved; W32 exposeddevelopment revealed automatic detector-veto misrejection. RevisedG2 keeps mask/pointcorrespondence,2crop appearance/attribute evidence and knownrelationcontradiction; detectorabsence staysunknown. NewZ64fixed before revisedGoutputs: Quad23.649301→30.852559mIoU(+7.203258pp),Drone13.983825→14.051852(+.068027pp). 5interventions/5IoUimprovements, no observedharms, but clustered95%lowerbounds remain0 and prespecifiedTTA gatefails. No TTA effect claimed.\n\n208G2cases:17interventions,16IoUgains,0harms,1tie. Limitedscenes, lowcoverage, historicaltargetVALexposure and repeatedphysicalsubjects mean this is a researchcandidate, not validateddeployment or broadreliability. W32 is development forG2, not confirmation.\n\nNegativecontrols remain: D112unconditionalvisualrerank has4IoUharms; newZ explicit-field legacybinding control exceeds fullG by1.458905pp onQuad (missedrescue Z_quad_026). No-reference ablation identical toG2, so reference localization has no establishedtaskcontribution. Stable regions are not semanticidentity; genericperson/isolatedgraffiti proposals can be wrong or ambiguous.\n\nAll864strictprotocol and1456G2task-armrows, perpair evidence, scenealiases, failed/retainedcontrols, independentCPUaudit, sourceimmutable and hash-boundprotocols are included. These2320evaluationrows overlap inD/W and do not represent2320independent observations. No rawmedia/captions/coordinates/instanceIDs/GT/features/weights/credentials or fullpapertext are exported.\n')
 forbidden={'id_PRIVATE','scene_PRIVATE','caption','pointpath','metapath','imagepath','arm_boxes_raw6','query_indices','source_pool','source_scores'}
 def check(x):
  if isinstance(x,dict):
   assert not set(x)&forbidden,set(x)&forbidden
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
 for p in out.rglob('*.json'):check(read(p))
 inputs=read(EXP.parent/'input/ALL208_PRIVATE.json')
 for p in out.rglob('*'):
  if p.is_file():
   text=p.read_text()
   for r in inputs:assert r['id'] not in text and r['caption'] not in text,p
 fs=[{'path':p.relative_to(out).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'study':STUDY,'files':fs,'unique_current_observations':208,'task_arm_rows':2320,'all_negatives_retained':True,'no_raw_assets_or_weights':True});dump(EXP.parent/'publication/LOCAL_EXPORT_AUDIT.json',{'files':len(fs)+1,'structured_private_key_and_caption_ID_checks_passed':True,'report_sha256':sha(out/'REPORT.md'),'manifest_sha256':sha(out/'MANIFEST.json'),'strict_task_arm_rows':864,'G2_task_arm_rows':1456});print('R20_EXPORT_READY',len(fs)+1)
if __name__=='__main__':main()
