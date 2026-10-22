"""Check and snapshot the sole maintained method and its saved evidence."""
import argparse,json,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REQUIRED=['README.md','AGENTS.md','configs/default.json','docs/history.md','docs/method.md','docs/evaluation.md','docs/assets.md','docs/assets.json','docs/third_party.json','docs/migration.json','results/report.md','results/evaluation.json','results/runtime.json','results/cost.json','results/migration_checks.json','results/score_regression.json','results/cleanup_checks.json','docs/data_integrity.md','results/data_integrity.json','docs/cleanup.json','docs/publication.json']
REQUIRED += ['docs/tubedetr_ablation.md']
REQUIRED += ['docs/tubedetr_vid_development.md','docs/tubedetr_hc2_development.md','docs/tubedetr_spatial_analysis.md','docs/tubedetr_layout_trial.md']
REQUIRED += ['docs/tubedetr_opd_direct_research.md']
REQUIRED += ['docs/tubedetr_reward_mechanism.md']
REQUIRED += ['docs/tubedetr_mask_reward_qualification.md']
REQUIRED += ['docs/tubedetr_executed_candidates.md','docs/tubedetr_observability.md','docs/tubedetr_correspondence.md','docs/tubedetr_reward_extent.md','docs/tubedetr_cdf.md','docs/tubedetr_universal_trial.md','docs/tubedetr_universal_diagnosis.md','docs/tubedetr_universal_confidence.md','docs/tubedetr_propagation_policy.md','docs/tubedetr_propagation_v0_spec.md','docs/tubedetr_boxfixed_development.md']
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
REQUIRED += ['configs/tubedetr.json','docs/tubedetr.md','results/tubedetr/engineering_checks.json','results/tubedetr/STATUS.json','results/tubedetr/teacher_service_check.json','results/tubedetr/qualification.json','results/tubedetr/data_intake.json']
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'} and str(p.relative_to(ROOT)) not in REQUIRED]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_vid_development').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_hc2_development').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for folder in ['results/tubedetr_spatial_analysis','results/tubedetr_vid_spatial_followup']
             for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_layout_trial').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_opd_direct_diagnosis').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_reward_mechanism').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_reward_trajectory').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_mask_reward_qualification').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_executed_candidates').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_observability').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_correspondence').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_reward_extent').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_cdf').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_universal_trial').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_universal_diagnosis').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_universal_confidence').rglob('*')
             if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_propagation_policy').rglob('*') if p.is_file() and p.suffix in {'.json', '.md'}]
REQUIRED += [str(p.relative_to(ROOT)) for p in (ROOT/'results/tubedetr_boxfixed_development').rglob('*') if p.is_file() and p.suffix in {'.json', '.md'}]
def check():
 errors=[];warnings=[]
 for name in REQUIRED:
  if not(ROOT/name).is_file():errors.append('Missing '+name)
 for f in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:
  for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
   target=target.split('#',1)[0]
   if not target or '://'in target:continue
   if not(f.parent/target).exists():errors.append(f'{f.relative_to(ROOT)}: missing link {target}')
 if not errors:
  cfg=json.loads((ROOT/'configs/default.json').read_text());r=json.loads((ROOT/'results/evaluation.json').read_text())
  assert [(cfg['datasets'][ds]['lr'],cfg['datasets'][ds]['tau'],cfg['datasets'][ds]['steps'],cfg['datasets'][ds]['initialization_gamma'],cfg['datasets'][ds]['output_alpha'])for ds in ['hc2','vidstg']]==[(.014,.14,40,.5,1.),(.01,.5,30,1.,1.)]
  for ds,g in r['datasets'].items():
   assert g['parents']==g['queries']==len(g['rows'])==128
   for arm,m in g['means'].items():
    for k,v in m.items():
     x=[row['values'][arm][k]for row in g['rows']if row['values'][arm][k]is not None];assert abs(sum(x)/len(x)-v)<1e-12
  snap=ROOT/'docs/sources.json'
  if snap.exists():
   old=json.loads(snap.read_text())['files'];changed=[p for p,h in old.items()if not(ROOT/p).is_file()or sha(ROOT/p)!=h]
   if changed:warnings.append('Snapshot stale; changed '+str(len(changed))+' files')
 result=dict(status='pass'if not errors else'fail',errors=errors,warnings=warnings,scope='current file links/configuration and independent aggregation of saved scores; no inference or new experiment');print(json.dumps(result,indent=2));return result

def snapshot():
 result=check();assert not result['errors'];files={}
 for p in REQUIRED+[str(f.relative_to(ROOT))for folder in ['stvg','tools','tests','scripts']for f in (ROOT/folder).rglob('*.py')]:files[p]=sha(ROOT/p)
 dest=ROOT/'docs/sources.json';dest.write_text(json.dumps(dict(status='snapshot',files=files),indent=2)+'\n');print('Snapshot written:',len(files),'current sources')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['check','snapshot']);a=p.parse_args()
 if a.action=='snapshot':snapshot()
 else:raise SystemExit(0 if check()['status']=='pass'else 1)
