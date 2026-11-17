"""Public-only code/config/anonymous scalar tables; exclude all raw private data."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round128';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def table(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;O.mkdir(parents=True,exist_ok=False);rows=[];cases=[]
 for f in sorted((H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json')):
  rr=rd(f);clusters=sorted({r['scene_PRIVATE'] for r in rr});public=[]
  for r in rr:
   d={'source':r['source'],'target':r['target'],'row':f"ROW{r['slot']+1:04d}",'scene_cluster':f"CLUSTER{clusters.index(r['scene_PRIVATE'])+1:03d}",'protected_alias_fallback':r['protected_alias']};d.update({k+'__IoU':v for k,v in r['IoUs'].items()});public.append(d);rows.append(d)
  table(O/'tasks'/f.name.replace('_ALL_ROWS_PRIVATE.json','.csv'),public)
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','ALL_SCENE_ROWS','AGGREGATE_ROWS']:table(O/'results'/(name+'.csv'),rd(H/'scores_attempt001'/(name+'.json')))
 prim=rd(H/'experiment/COMPARISON_LOCK.json')['primary_readouts'];ctrl={'Tent':'BN_statistics_only','MEMO':'memo_BNprior_zero','EATA':'eata_zero','MATE':'MATE_prepared_zero','MM-TTA':'MMTTA_independent_frozen_ensemble','BFTT3D':'pair_prepared_zero','CloudFixer':'source','Uni-Adapter':'pair_prepared_zero'}
 for method,k in prim.items():
  for src in ['waymo','drone','quad']:
   for tgt in ['waymo','drone','quad']:
    if src==tgt:continue
    q=[r for r in rows if r['source']==src and r['target']==tgt]
    for kind,order in [('largest_gain',sorted(q,key=lambda r:r[k+'__IoU']-r[ctrl[method]+'__IoU'],reverse=True)),('largest_loss',sorted(q,key=lambda r:r[k+'__IoU']-r[ctrl[method]+'__IoU']))]:
     for r in order[:3]:cases.append({'method':method,'source':src,'target':tgt,'row':r['row'],'kind':kind,'control':ctrl[method],'baseline_IoU':r[ctrl[method]+'__IoU'],'adapted_IoU':r[k+'__IoU'],'delta_IoU':r[k+'__IoU']-r[ctrl[method]+'__IoU'],'identity_audit':'not inferred from IoU alone'})
 table(O/'diagnostics/GOOD_BAD_SCALAR_CASES.csv',cases)
 names=['DECISION','MAIN_COMPARISONS','RESOURCE_SUMMARY','FAST_NATIVE_AUDIT','TASK_AUDIT','SCALAR_AUDIT','SOURCE_INTERFACE_AUDIT','SOURCE_ROLE_CALIBRATION','SOURCE_KNOWN_PROTECTED_ALIAS_METADATA_AUDIT','CLOUDFIXER_CPU_FORWARD_AUDIT','CLOUDFIXER_WEIGHTS_AUDIT','MMTTA_INDEPENDENT_TEXT_AUDIT','OFFICIAL_REPOSITORIES','SOURCE_HISTORY','MATE_CHAMFER_SOURCE_HISTORY','MMTTA_INDEPENDENT_RGB_SOURCE_HISTORY']
 for name in names:dump(O/(name+'.json'),rd(H/(name+'.json')))
 for name in ['REPORT.md','METHOD_CARDS.md']:(O/name).write_text((H/name).read_text())
 for f in sorted((H/'experiment').glob('*.json')):dump(O/'config'/f.name,rd(f))
 files=list(H.glob('*.py'))+[R/'tools/experiment_guard_round128.py']
 for f in files:
  d=O/'own_code'/f.relative_to(R);d.parent.mkdir(parents=True,exist_ok=True);d.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 (O/'REPRODUCTION.md').write_text('''# Reproduction boundary

This is a grounding task adaptation suite, not original-task replication. See METHOD_CARDS.md.
Use the existing 3EED workspace, raw licensed data, the three epoch100 source-only checkpoints,
and the prior immutable source/preprocessing interfaces. Private input registries, captions,
media, point/box coordinates, embeddings, weights and third-party snapshots are not distributed.
Download author repositories at OFFICIAL_REPOSITORIES.json commits; MM-TTA uses Latte reference code.

Ordered stages: prepare128_cpu; train128_prepared (initial attempt retained);
train128_mate_chamfer; calibrate128_roles; run128_fast (engineering then formal);
run128_prepared (engineering then formal); run128_cloudfixer (engineering then formal);
run128_zero_controls; prepare128_independent_RGB; run128_mmtta_cpu --independent;
audit_score128 only AFTER every matrix sealed; audit128_scalars; report128; export128;
audit_public128. Every inference/training/scoring job uses experiment_guard_round128.py.
Prepared RGB/source calibration and all algorithm repairs preceded target-GT scoring.
No post-score parameter selection. Corrected one-forward/independent-RGB MM-TTA is primary;
the older diagnostics are retained as disclosed implementation attempts, not paper comparisons.

Prior own-code dependency: studies/3eed_fixed_R91_three_source_six_direction_TTA_20261114
(core97/factory/interface), plus earlier R76/R91 modules. Native upstream 3EED is obtained
from https://github.com/worldbench/3EED, not republished here. CloudFixer author diffusion
weights provenance/hash are provided; checkpoint bytes are intentionally excluded.
''')
 for f in [H/'engineering_fast_attempt001_FAILED_float64.log',H/'prepare_independent_RGB_attempt003.log']:
  if f.exists():d=O/'attempts'/f.name;d.parent.mkdir(exist_ok=True);d.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 dump(O/'MANIFEST.json',{'prefix':'studies/3eed_eight_external_TTA_grounding_baselines_20261117','files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(O.rglob('*')) if f.is_file()]});print('EXPORT128',len(list(O.rglob('*'))),flush=True)
if __name__=='__main__':main()
