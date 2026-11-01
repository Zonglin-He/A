"""Explicit public export: own code, fixed protocols, anonymous scalar evidence only."""
from pathlib import Path
import sys,shutil,csv,io,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round21'))
from common21 import *
def main():
    pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);dest=pub/'public_export_attempt002';dest.mkdir(exist_ok=False)
    def copy(src,target):
        target=dest/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    for f in sorted(BASE.glob('*.py')):copy(f,f.relative_to(ROOT))
    for name in ['score21.py','score21_v2.py','score21_v3.py','audit21.py','audit21_v3.py','diagnose21.py','render21.py','render21_v2all.py','render21_final.py','finalize21.py','publish21.py']:copy(EXP.parent/name,Path('research/round21')/name)
    for f in sorted(EXP.glob('*LOCK.json')):copy(f,Path('protocol')/f.name)
    for name in ['AUTHORIZATION.json','TEXT_PREREQUISITE_FAILURE_ATTEMPT001.json','TEXT_v2_PREREQUISITE_AUDIT.json','TEXT_v3_CHANGE_AUDIT.json']:copy(EXP/name,Path('protocol')/name)
    for name in ['DECISION.json','COSTS.json','CPU_AUDIT.json','MECHANISM_DIAGNOSIS.json','ATTRIBUTE_GATE_DIAGNOSTICS.json','ATTRIBUTE_REPAIR_NEGATIVE_TAILS.json','ZERO_IOU_RELATION_CANDIDATE_ORACLE.json','RELATION_INTERVENTIONS.json','REVIEW_MANIFEST.json','SOURCE_RECHECK.json']:copy(EXP/name,Path('results')/name)
    for attempt in [2,3]:
        p=EXP/f'scored208_attempt00{attempt}'
        for name in ['SUMMARY.json','TTA_GATE.json','PRELABEL_LOCK.json']:copy(p/name,Path('results')/f'v{attempt}'/name)
        buf=io.StringIO();writer=csv.DictWriter(buf,lineterminator='\n',fieldnames=['alias','phase','platform','arm','iou','selected_same_as_source','selected_same_as_no_reference']);writer.writeheader()
        for row in read(p/'ROWS_PRIVATE.json'):
            for arm,v in row['ious'].items():writer.writerow({'alias':row['alias'],'phase':row['phase'],'platform':row['platform'],'arm':arm,'iou':v,'selected_same_as_source':row['q'][arm]==row['q']['source'],'selected_same_as_no_reference':row['q'][arm]==row['q']['gate_repair_no_reference']})
        target=dest/'results'/f'v{attempt}'/'ALL2080_SCALAR_ROWS.csv';target.write_text(buf.getvalue())
    pairs=[]
    for r in read(ROWS):
        c=read(BASE/'direct208_attempt003/cases_PRIVATE'/(r['id']+'.json'))
        for p in c['pair_diagnostics']:pairs.append(dict(alias=r['alias'],platform=r['platform'],**{k:v for k,v in p.items() if not k.endswith('_PRIVATE')}))
    dump(dest/'results/ALL154_PAIR_DIAGNOSTICS.json',pairs)
    copy(EXP/'REPORT.md','REPORT.md')
    (dest/'README.md').write_text('''# 3EED G compositional reference and role-binding diagnostic — R21

Waymo-only source100 → Drone/Quad. All208 observations are historically exposed development/regression data. Source boxes frozen. Three mechanical parser revisions and two full10-arm evaluations retained. Full references add exactly0 mIoU over matched repaired-appearance no-reference control; no TTA launched. Appearance-gate relaxation introduces two Quad harms. No method adoption.

Read REPORT.md, protocol/G_SPEC_LOCK.json, results/v2 and results/v3. Each version contains2080 anonymous task-arm rows; versions reuse208 inputs and are not4160 independent samples. All154 pair diagnostics and failed prerequisite histories retained. Local raw RGB, captions, point clouds, calibration/coordinates, model features, opaque IDs and weights excluded.

Code builds on previous studies in this repository, especially R20 commit a5c8d72ead300f4039989a5e9f918ab8ae448ffb. Reproduction requires the licensed official dataset, original private manifests and frozen source/model assets; the scalar export does not pretend to contain raw data. CPU and GPU environments unchanged from R20.

Execution order: parse21.py → detect21.py → reference21.py (preserved first prerequisite attempt), then *_v2, then *_v3. v2 and v3 assembly precede corresponding score programs. v3 only repairs missing same-sentence reference inputs after v2 exposed diagnostics, without threshold change. No independent confirmation or adaptation result claimed.
''')
    dump(dest/'DEPENDENCIES.json',{'previous_R20_commit':'a5c8d72ead300f4039989a5e9f918ab8ae448ffb','source_sha256':'0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd','additional_model_downloads':0,'third_party_sources_included':False,'own_code_only':True,'DINO':'Frozen official GroundingDINO tiny assets and PyTorch2.7cu128 as verified R20','VLM':'Frozen Qwen2.5-VL3B numerical label readout as verified R20','NLP':'Existing scoped R20 Stanza runtime/assets reused'})
    files=[f for f in sorted(dest.rglob('*')) if f.is_file()]
    forbidden={str(r[k]) for r in read(ROWS) for k in ['id','caption','pointpath','imagepath','metapath']}
    for f in files:
        text=f.read_text();assert not any(x and x in text for x in forbidden),f
    dump(dest/'MANIFEST.json',{'files':[{'path':f.relative_to(dest).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files],'private_payload_excluded':True})
    files=[f for f in sorted(dest.rglob('*')) if f.is_file()];prefix='studies/3eed_compositional_reference_role_binding_G_diagnostic_20261101'
    tree=[{'path':prefix+'/'+f.relative_to(dest).as_posix(),'mode':'100644','type':'blob','content':f.read_text()} for f in files];dump(pub/'TREE_PAYLOAD_LOCAL.json',tree);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'privacy_checks_passed':True,'task_arm_rows':4160,'unique_inputs':208,'pair_rows':len(pairs),'all_negative_controls_retained':True,'prefix':prefix});print('EXPORT21',len(files),len(pairs))
if __name__=='__main__':main()
