"""Unify research entry points after temperature selection, preserving deployment."""
from pathlib import Path
import sys,json,shutil
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
D=R/'artifacts/stvg_referdino_finalize_v1';PUB=Path('/home/wwww/visual-grounding-public-A')
def main():
 sel=read(D/'SELECTION_LOCK.json');assert read(D/'SCIENTIFIC_PHASES_COMPLETE.json')['status'].startswith('all_temperature')
 cfg='methods/stvg_referdino_opd_v1/config.json';p='results/stvg_error_state/2026-10-19/referdino_final';manifest=dict(version='stvg_referdino_opd_v1',role='unique_current_frozen_research_entry',status='frozen_implementation_configuration_not_an_efficacy_promotion',entrypoint='methods.stvg_referdino_opd_v1.predictor.ReferDINOOPDPredictor',run='python -m methods.stvg_referdino_opd_v1.run',offline_score='python -m methods.stvg_referdino_opd_v1.score',configuration=cfg,configuration_sha256=sha(R/cfg),tau=sel['tau'],scales=[.05,.25],lr=.01,steps=40,samples=32,seeds=[41,42,43],default_seed=41,temporary_state_parameters=2048,network_weight_updates=0,cross_query_state='none; state and optimizer reset',temporal='frozen Source finalMBR',E='four original GroundingDINO-T observations + supported global correction',teacher='native ReferDINO ryt_mevis_swinb eight positions; frozen',source='TA-STVG pretrained on VidSTG',primary_target='HC',Vid_gain_required=False,results=p+'/REPORT.md',formal_results=p+'/FORMAL_RESULTS_PUBLIC.json',development=p+'/TEMPERATURE_RESULTS_PUBLIC.json',historical_deployment='methods/CURRENT_METHOD.json',no_more_algorithm_search=True)
 for f in ['methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json','README.md','REVIEW_START_HERE.md']:
  src=R/f if(R/f).exists()else PUB/f;dst=D/'entry_before'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 manifest['evaluation']=read(R/p/'EVALUATION_STATUS.json');manifest['dependency_lock']='methods/stvg_referdino_opd_v1/DEPENDENCIES.json';put(R/'methods/RESEARCH_METHOD.json',manifest)
 for f,role in [('MAIN_METHOD.json','current_research'),('PAPER_METHOD.json','paper_research'),('CURRENT_WORKING_METHOD.json','working_research')]:put(R/'methods'/f,dict(version=manifest['version'],role=role,manifest='methods/RESEARCH_METHOD.json',entrypoint=manifest['entrypoint'],configuration=cfg,status=manifest['status'],report=manifest['results'],historical_before='artifacts/stvg_referdino_finalize_v1/entry_before/methods/'+f),True)
 current=read(R/'methods/CURRENT_METHOD.json');current.update(registry_role='historical_deployment_preserved_not_current_research',active_research_manifest='methods/RESEARCH_METHOD.json',production_algorithm_unchanged=True);put(R/'methods/CURRENT_METHOD.json',current,True)
 for name in ['STVG_DINO_P7A_ONLINE_V1.json','STVG_T2_B_EPISODIC_V1.json']:
  f=R/'methods/paper'/name
  if not f.exists():f.parent.mkdir(exist_ok=True);shutil.copyfile(PUB/'methods/paper'/name,f)
  z=read(f);put(D/'entry_before/methods/paper'/name,z);z.update(record_role='historical_version_not_current_research',current_research_manifest='methods/RESEARCH_METHOD.json');put(f,z,True)
 text='''# STVG: frozen ReferDINO-supervised spatial OPD research version

Current research entry: **`stvg_referdino_opd_v1`**, HC primary. [Authoritative research manifest](methods/RESEARCH_METHOD.json), [single frozen configuration](methods/stvg_referdino_opd_v1/config.json), [method and execution guide](methods/stvg_referdino_opd_v1/README.md), [final results and full costs](results/stvg_error_state/2026-10-19/referdino_final/REPORT.md).

Video + text → frozen TA-STVG Source → four GroundingDINO-T observations and E initialization with Source finalMBR → native ReferDINO eight-frame evidence → dual-scale output-box OPD → full student trajectory. Both expert stages are real dependencies and are charged. E retains the original budget of at most four planned-position calls, including the existing skip/rejection behavior; actual counts are reported and all queries remain in evaluation. All pretrained weights stay frozen. Each query independently resets **8×256=2048 temporary content states** and Adam; no online LayerNorm carry.

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_referdino_opd_v1.run \\
  --input request.json --output prediction.pt --seed 41
```

Input metadata specifies video path, caption, original width/height and physical frame IDs. The [guide](methods/stvg_referdino_opd_v1/README.md) documents input validation, optional cache acceleration and separate sealed-prediction scoring. Runtime weights, local model packages and exact dependency hashes are listed in the [provenance record](results/stvg_error_state/2026-10-19/referdino_final/PROVENANCE_PUBLIC.json). Private weights, media, labels, captions and raw caches are excluded from this repository.

The final round compares only temperatures .05/.10/.20 on exposed HC128 with seeds41/42/43; the selected recipe is evaluated on32 prelocked, disjoint parent videos. Earlier Source/old-method exposure remains disclosed. Configuration freeze is not a claim of a stable2–3pp OPD gain. See the report for signed results, confidence intervals, tails, Direct/fresh Fixed/Live comparisons and total spatial versus incremental OPD effects. Three seeds are experimental repeats; default inference is one fixed seed, not an ensemble.

## Version identities

- `methods/RESEARCH_METHOD.json` is the unique current research definition; MAIN/PAPER/WORKING registries reference it.
- [CURRENT_METHOD.json](methods/CURRENT_METHOD.json) preserves the historical deployed P7-A algorithm and is explicitly marked historical deployment. It is not the current paper recipe.
- [Historical P7-A](methods/stvg_opd_p7a_v1/README.md) uses1792 query/LN parameters and online fractional LN carry; the new method does not.
- Historical T2+B, P7-A, protocols, saved predictions and negative results retain their original paths. Old paused queues are not resumed.
'''
 (R/'README.md').write_text(text)
 old=(PUB/'REVIEW_START_HERE.md').read_text();new='''# Current research review entry — 2026-10-19

Start with [the unique research manifest](methods/RESEARCH_METHOD.json), [frozen config](methods/stvg_referdino_opd_v1/config.json), [predictor guide](methods/stvg_referdino_opd_v1/README.md), and [final temperature/full-chain report](results/stvg_error_state/2026-10-19/referdino_final/REPORT.md).

Current research is HC episodic adaptation of2048 temporary content states with frozen TA-STVG, four-frame DINO E initialization and native eight-frame ReferDINO supervision. The current spatial policy is the target-size-normalized two-scale mixture. Source finalMBR is fixed. No cross-query LN state, new network training or new expert is part of this version.

The implementation/configuration is frozen; efficacy is determined by the signed results and uncertainty in the report, not by this entry. MAIN/PAPER/WORKING all point to the same research manifest. CURRENT_METHOD retains the historical deployment separately.

## Historical review log below

The following dated material records prior research and its then-current status. Words such as “latest”, “running” or “current” below refer to those historical entries and do not authorize execution or override the manifest above. Historical paths and negative results remain available.

'''+old
 (R/'REVIEW_START_HERE.md').write_text(new)
 put(D/'ENTRY_UNIFICATION.json',dict(status='complete',research_manifest=manifest,historical_deployment_algorithm_preserved=True,old_entries_saved_privately=True,old_result_protocol_paths_unchanged=True))
if __name__=='__main__':main()
