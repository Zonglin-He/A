"""Public scalar/code export; preserves every development trial and negative."""
import sys,json,csv,io,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round23'))
from common23 import *
def main():
    pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt002';out.mkdir(exist_ok=False)
    def copy(p,d):
        target=out/d;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for p in sorted(BASE.glob('*.py')):copy(p,p.relative_to(ROOT))
    for p in sorted(EXP.parent.glob('*.py')):copy(p,p.relative_to(ROOT))
    for p in sorted(EXP.glob('*LOCK.json')):copy(p,Path('protocol')/p.name)
    copy(EXP/'AUTHORIZATION.json','protocol/AUTHORIZATION.json');copy(EXP.parent/'input/EXPOSURE.json','protocol/EXPOSURE.json');copy(EXP.parent/'literature/PAPER_AUDIT.json','literature/PAPER_AUDIT.json')
    for name in ['CPU_AUDIT.json','DECISION.json','COSTS.json','ALL_CHANGED_CASES.json','REGRESSION_CASES.json','EFFECTS.json','REVIEW_MANIFEST.json','REFERENCE_ABLATION_AUDIT.json','D208_SUMMARY.json','K64_SUMMARY.json','DEV_GRID_RANKING.json','DEV_GRID_v2_RANKING.json','DEV_GRID_v3_RANKING.json','DEV_GRID_SUMMARY.json','DEV_GRID_v2_SUMMARY.json','DEV_GRID_v3_SUMMARY.json']:copy(EXP/name,Path('results')/name)
    allrows=read(EXP/'D208_ROWS_PRIVATE.json')+read(EXP/'K64_ROWS_PRIVATE.json');scenes={p:sorted({r['scene_PRIVATE'] for r in allrows if r['platform']==p}) for p in ['drone','quad']};total=0
    for name,file,ref in [('grid_v1','DEV_GRID_ROWS_PRIVATE.json','R21_no_ref'),('grid_v2','DEV_GRID_v2_ROWS_PRIVATE.json','R21_no_ref'),('grid_v3','DEV_GRID_v3_ROWS_PRIVATE.json','R21_no_ref'),('D208_selected','D208_ROWS_PRIVATE.json','R21_no_ref'),('K64_fixed','K64_ROWS_PRIVATE.json','R21_no_ref')]:
        rows=read(EXP/file);buf=io.StringIO();writer=csv.DictWriter(buf,lineterminator='\n',fieldnames=['alias','platform','scene_alias','arm','iou','same_as_source','same_as_R21']);writer.writeheader()
        for r in rows:
            for arm,v in r['ious'].items():writer.writerow({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_s'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'arm':arm,'iou':v,'same_as_source':r['q'][arm]==r['q']['source'],'same_as_R21':r['q'][arm]==r['q'][ref]});total+=1
        p=out/'results'/name/'ALL_SCALAR_ROWS.csv';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(buf.getvalue())
    assert total==15200
    copy(EXP/'REPORT.md','REPORT.md');(out/'README.md').write_text('''# R23 — soft evidence ranking and expanded instance pools

Waymo-only source100→Drone/Quad. Latest user criterion is aggregate metric improvement, allowing individual harms. Frozen weights/boxes, direct reranking only.208 exposed development observations and64 fresh current-input content-disjoint observations (historical targetVAL exposure disclosed,2Drone/3Quadscene clusters). Three bounded development grids retained; final globally shared configuration frozen before new64scores.

New64 mean mIoU: Source28.063498%,R21 30.922075%,new31.178995%;net+0.256920pp vsR21,Acc25/50unchanged. Drone2gains/1harm,Quad unchanged. Small effect, scene interval crosses0. Retain as research baseline, not evidence of robust or TTA gain. No-reference has identical taskmetrics;twoD208querychanges bothIoU0,allK64finalchoices identical; reference reasoning contribution is unproven. Bare EDA-style readout does not reproduce EDA dense training.

All15200anonymous scalar rows preserve initial unsuccessful log/linear calibration grids and final selection; they reuse272inputs, not15200independent samples. Scene aliases allow cluster replay. Raw captions/media/pointclouds/calibration/box coordinates/query features/opaque IDs/weights remain private. Own code only; literature audit links official sources rather than redistributing third-party implementation.

Read REPORT.md, protocol/G_SPEC* and SELECTED_v3_LOCK. Reproduce final run with licensed official data and required prior cache/model assets: prepare23.py; source23.py; parse23.py; prepare23.py under crossplatform; detect23.py; evidence23.py; rank23.py; evaluate23.py development; evaluate23_v2.py development; evaluate23_v3.py development then confirm. Old scripts are retained as actual trial history. No independent re-selection after K64scores and no TTA updates.

Dependencies and prior experiments: repository study R22 at commit86cf62fd366ea8d3dc3d919d837236ad833c0ef6. Scalar export is not a raw-data release.
''')
    dump(out/'DEPENDENCIES.json',{'previous_R22_commit':'86cf62fd366ea8d3dc3d919d837236ad833c0ef6','source_checkpoint_sha256':read(EXP/'CPU_AUDIT.json')['source_sha256'],'new_model_downloads':0,'new_packages':0,'frozen_components':'Source3EEDWaymo100,existingQwen2.5VL3B,GroundingDINOtiny,scopedStanza','TTA_steps':0})
    files=[p for p in sorted(out.rglob('*')) if p.is_file()];forbidden={str(r[k]) for r in read(EXP.parent/'input/ALL272_PRIVATE.json') for k in ['id','caption','pointpath','imagepath','metapath']}
    for p in files:
        text=p.read_text();assert not any(x and x in text for x in forbidden),p
    dump(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True});files=[p for p in sorted(out.rglob('*')) if p.is_file()];prefix='studies/3eed_soft_evidence_expanded_instance_G_20261101';entries=[{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files];dump(pub/'TREE_PAYLOAD_LOCAL.json',entries);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'scalar_rows':total,'unique_inputs':272,'all_trials_and_negatives_kept':True,'privacy_passed':True,'prefix':prefix});print('EXPORT23',len(files),total)
if __name__=='__main__':main()
