"""Publish design/implementation and completed algebra audit, not Full scores."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write
from scripts.report_stvg_paper_frozen_v1 import CODES
from scripts.export_stvg_lite_batch001 import public_source_text
from scripts.stvg_paper_public_privacy_v1 import verify_privacy

def main():
 import numpy as np
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 base=ROOT/'artifacts/stvg_paper_t2_b_v1';out=ROOT/'results/stvg_frozen_paper/2026-10-17/method_lock';out.mkdir(parents=True,exist_ok=False);p=ROOT/'artifacts/stvg_b_score_algebra_v1/CPU_CONTRACT_COMPLETION.json';d=read(p);assert d['status']=='pass' and d['selector_sha256_before']==d['selector_sha256_after']==sha(ROOT/'scripts/stvg_referent_abc_tokens002.py');write(out/'B_SCORE_CPU_CONTRACT.json',d)
 write(out/'PUBLICATION_SCOPE.json',dict(status='complete',scope='paper_method_lock_and_CPU_B_algebra_only',source_experiment_completed_count=0,integrated_Full256_status='running',independent_QCQP32_status='waiting_dependency_CPU_contract_pass',new_model_effect_claim=False,LaTeX_source_saved=True,LaTeX_PDF_compilation_status='unavailable: built-in compiler base bundle not cached and fetch failed',historical_CURRENT_unchanged=True,GT_read=False))
 delta=np.linspace(0,1,200);s=.75*.5-.125*delta;fig,axs=plt.subplots(1,2,figsize=(10,3.8));q,c=np.meshgrid(np.linspace(0,1,101),np.linspace(0,1,101));z=.75*np.minimum(q,c)+.25*np.abs(q-c);im=axs[0].imshow(z,origin='lower',extent=[0,1,0,1],aspect='equal');axs[0].set(xlabel='Query rank',ylabel='Caption rank',title='Original B score');fig.colorbar(im,ax=axs[0]);axs[1].plot(delta,s);axs[1].set(xlabel='Absolute disagreement',ylabel='Score',title='Fixed average relevance = 0.5');fig.tight_layout();fig.savefig(out/'B_score_identity.png',dpi=160);plt.close(fig)
 for name in CODES+['export_stvg_paper_freeze_v1.py']:
  p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
 for name in ('METHOD_LOCK.json','CPU_CONTRACT_COMPLETION.json'):
  p=ROOT/'artifacts/stvg_output_budget32_v1'/name;write(out/('QCQP_'+name),read(p))
 paths={'methods/PAPER_METHOD.json','methods/paper/STVG_T2_B_EPISODIC_V1.json','protocols/stvg_paper_t2_b_v1.md','protocols/stvg_output_budget32_v1.md','paper/README_CURRENT.md','paper/stvg_t2_b_episodic_20261017.tex','docs/STVG_B_SCORE_ALGEBRA_CORRECTION_20261017.md'}|{str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()};verify_privacy(paths);write(base/'publication/freeze/EXPORT_MANIFEST.json',dict(status='complete',count=0,scope='method_design_and_completed_CPU_algebra_not_Full_measurement',paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print('METHOD_LOCK_CPU_ALGEBRA_EXPORT_READY_NOT_FULL_RESULTS')
if __name__=='__main__':main()
