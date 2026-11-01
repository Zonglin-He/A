"""Own code and anonymous scalar results only; raw evidence remains private."""
import sys,json,csv,io,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round22'))
from common22 import *
def main():
    pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
    def copy(src,name):
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for f in sorted(BASE.glob('*.py')):copy(f,f.relative_to(ROOT))
    for f in sorted(EXP.parent.glob('*.py')):copy(f,f.relative_to(ROOT))
    for f in sorted(EXP.glob('*LOCK.json')):copy(f,Path('protocol')/f.name)
    copy(EXP/'AUTHORIZATION.json','protocol/AUTHORIZATION.json')
    for name in ['CPU_AUDIT.json','DECISION.json','TTA_GATE.json','REFERENCE_REJECTION_RECOVERY.json','REGRESSION_CASES.json','NEW_VS_R21_NEGATIVE_TAILS.json','PRIMARY_RELATION_INTERVENTIONS.json','ALL_NONZERO_RELATION_CHANGES.json','PAIR_GATE_DECOMPOSITION.json','OFFLINE_RECOVERABILITY_COUNTS.json','SOURCE_RECHECK.json','QUALITATIVE_REVIEW.json','REVIEW_MANIFEST.json']:copy(EXP/name,Path('results')/name)
    pair=read(EXP/'ALL154_PAIR_DIAGNOSTICS.json');p=out/'results/ALL154_PAIR_DIAGNOSTICS.json';p.write_text(json.dumps(pair,separators=(',',':'),ensure_ascii=False)+'\n')
    scopes=[('cache','scored_cache208_attempt001','drop_increment_no_ref'),('formal','scored_direct208_attempt001','direct_pair_no_ref'),('scene_label_enriched','scored_scene21_attempt001','scene_no_ref')];total=0
    for scope,folder,baseline in scopes:
        for name in ['SUMMARY.json','PRELABEL_LOCK.json']:copy(EXP/folder/name,Path('results')/scope/name)
        rows=read(EXP/folder/'ROWS_PRIVATE.json');buf=io.StringIO();writer=csv.DictWriter(buf,lineterminator='\n',fieldnames=['alias','phase','platform','arm','iou','same_as_source','same_as_no_reference','offered_recoverable25','has_reference_relation','scope']);writer.writeheader()
        for r in rows:
            for a,v in r['ious'].items():writer.writerow({'alias':r['alias'],'phase':r['phase'],'platform':r['platform'],'arm':a,'iou':v,'same_as_source':r['q'][a]==r['q']['source'],'same_as_no_reference':r['q'][a]==r['q'][baseline],'offered_recoverable25':r['offered_recoverable25'],'has_reference_relation':r['has_reference_relation'],'scope':scope});total+=1
        (out/'results'/scope/'ALL_SCALAR_ROWS.csv').write_text(buf.getvalue())
    assert total==4223
    costs={'pair_new_forwards':read(BASE/'pair208_attempt001/SEALED.json')['calls'],'scene_new_forwards':read(BASE/'scene21_attempt001/SEALED.json')['calls'],'pair_seconds':read(BASE/'pair208_attempt001/SEALED.json')['seconds'],'scene_seconds':read(BASE/'scene21_attempt001/SEALED.json')['seconds'],'new_DINO_forwards':0,'source_inference_forwards':0,'TTA_optimizer_steps':0,'free_generated_tokens':0,'model_unchanged':True};dump(out/'results/COSTS.json',costs)
    dump(out/'protocol/FAILURE_HISTORY.json',{'depth_schema_failure':'Object depth_camera is a statistics dictionary; use global camera-depth array indexed by exclusive visible point indices. Failed before first prediction; original script/lock/empty run/log preserved privately.','before_final_prediction_scores':True,'no_threshold_changes_from_failure':True})
    copy(EXP/'REPORT.md','REPORT.md');(out/'README.md').write_text('''# R22 — reference veto removal, pair uncertainty and recoverability

Waymo-only source100→Drone/Quad; source weights and256boxes frozen. Full-phrase vs category increment is removed from new G decisions. All208 observations are historically exposed development; all154 original pairs actually processed. Reference proposal acceptance increases223→275, including7additional person proposals. New pair uncertainty avoids two old Quad harms but loses useful corrections; matched relation increment0 both domains.1194new frozenVLMforwards,0format rejects,0TTA steps, no adoption.

Three evaluations retained: cached veto ablation208×7, real-pair/geometry208×13, and21×3 label-enriched full-scene diagnostic.4223scalar rows reuse208inputs. The21-case diagnostic was selected using known labels to include18recoverable-with-relation cases and3regressions; not source-free test performance or independent confirmation. Both selection and limitations are explicit.

Read REPORT.md and protocol before results. Refer to prior R21 study at commit2e6b017ebf6e0cbbd44284f6f02e3cdce18df3f4 for dependencies. Reproduction requires licensed official data, private manifests and frozen source/model assets. Raw captions/RGB/point clouds/calibration/boxes/opaque IDs/model features/weights are excluded; this scalar export does not pretend to be a self-contained data release.

Execution: recoverability22.py (offline only); cache22.py then score22.py; pair22.py, assemble22.py then score22.py; scene22.py, assemble_scene22.py and score_scene22.py only under label-enriched diagnostic specification. All runtime readouts exclude task GT and modifier-increment decisions. No method novelty, relation contribution, or TTA effect claimed.
''')
    dump(out/'DEPENDENCIES.json',{'previous_R21_commit':'2e6b017ebf6e0cbbd44284f6f02e3cdce18df3f4','new_downloads':0,'new_environment_packages':0,'source_sha256':read(EXP/'SOURCE_RECHECK.json')['checkpoint_sha256'],'frozen_models':'Existing Qwen2.5VL3B and cached GroundingDINO/instance observations; no teacher/EMA','TTA_steps':0})
    files=[f for f in sorted(out.rglob('*')) if f.is_file()];forbidden={str(r[k]) for r in read(ROWS) for k in ['id','caption','pointpath','imagepath','metapath']}
    for f in files:
        text=f.read_text();assert not any(x and x in text for x in forbidden),f
    dump(out/'MANIFEST.json',{'files':[{'path':f.relative_to(out).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files],'private_payload_excluded':True});files=[f for f in sorted(out.rglob('*')) if f.is_file()];prefix='studies/3eed_reference_no_increment_pair_uncertainty_recoverability_20261101';entries=[{'path':prefix+'/'+f.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':f.read_text()} for f in files];dump(pub/'TREE_PAYLOAD_LOCAL.json',entries);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'task_arm_rows':total,'unique_observations':208,'scene_panel_is_subset':True,'privacy_passed':True,'prefix':prefix});print('EXPORT22',len(files),total)
if __name__=='__main__':main()
