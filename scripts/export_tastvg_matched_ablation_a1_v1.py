"""Explicit public allowlist: completed ablations and pre-result full evaluation registration."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_matched_ablation_a1_v1';DEST=ROOT/'artifacts/github_public_A_ablation_a1_20260929';PUBLIC=Path('/home/wwww/visual-grounding-public-A')

def run():
 files={}
 def add(rel,data):
  f=DEST/'files'/rel;f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists();f.write_bytes(data);files[rel]=dict(bytes=len(data),sha256=sha(f))
 def anon(x):
  if isinstance(x,dict):return {k:f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v) for k,v in x.items()}
  if isinstance(x,list):return [anon(v) for v in x]
  return x
 code=['protocols/tastvg_matched_ablation_a1_v1.md','tests/test_tastvg_matched_ablation_a1_v1.py','vg_tta/tastvg_matched_ablation_a1_v1.py','scripts/audit_tastvg_matched_ablation_a1_public_v1.py']+[f'scripts/{s}_tastvg_matched_ablation_a1_v1.py' for s in ['run','score','report','export']]
 code+=['protocols/tastvg_full_b1_v1.md','scripts/prepare_tastvg_full_b1_v1.py','scripts/prepare_tastvg_full_b1_subjects_v1.py','scripts/run_tastvg_full_b1_experts_v1.py','scripts/run_tastvg_full_b1_online_v1.py','scripts/validate_tastvg_full_b1_capture_v1.py']
 for rel in code:add(rel,(ROOT/rel).read_bytes())
 result='results/tastvg_matched_ablation_a1/2026-09-29'
 for name in ['ROWS.json','DIAGNOSTICS.json','SUMMARY.json','ACROSS_ORDERS.json','SOURCE_EFFECTS.json','ASSIGNMENTS.json','AUDIT.json','PUBLIC_AUDIT.json','RESOURCES.json','STATE_CHAIN.json','REINSERTION_AUDIT.json','PROVENANCE.json','DECISION.json','CLAIM_READOUT.json']:
  add(f'{result}/{name}',(json.dumps(anon(read(OUT/name)),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode())
 for name in ['REPORT.md','RESEARCH_UPDATE.md','CPU_TESTS.txt']:add(f'{result}/{name}',(OUT/name).read_bytes())
 add('docs/TA_MATCHED_ABLATION_A1_UPDATE.md',(OUT/'RESEARCH_UPDATE.md').read_bytes())
 for name in ['COHORT.json','ORDERS.json','PARSER_EQUIVALENCE.json','CAPTURE_INTEGRATION_AUDIT.json']:
  add(f'results/tastvg_full_b1/2026-09-29/{name}',(json.dumps(anon(read(ROOT/'artifacts/tastvg_full_b1_v1'/name)),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode())
 first,rest=(PUBLIC/'REVIEW_START_HERE.md').read_text().split('\n',1)
 entry='## Latest TA-STVG: frozen-method matched ablations; full test registered\n\n[Matched ablations](results/tastvg_matched_ablation_a1/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_MATCHED_ABLATION_A1_UPDATE.md), [full test protocol](protocols/tastvg_full_b1_v1.md). Five-order future vIoU: Final +0.025505 pp, Random-Rank −0.004011, Off-Policy +0.026227, Direct PL +0.003409. Preference beats this random control and fixed PL; **on-policy superiority is not established**. Frozen recipe unchanged. Full official test registration retains all9411queries/670sources after excluding62current-route development sources;3orders×16conditions. Historical project exposure disclosed; full-run results pending, not claimed complete.\n\n'
 add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
 write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(x['bytes'] for x in files.values()),exclusions=['media','captions/source IDs','GT coordinates','weights','H/masks/raw predictions/parameter tensors','private conversation']))
 for rel in files:
  f=PUBLIC/rel;f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,f)
 print(len(files),sum(v['bytes'] for v in files.values()))
if __name__=='__main__':run()
