"""Readable gap and GT diagnosis; never alter the model based on these records."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.tastvg_best_quick_common_v1 import *

def run():
    lines=['# Matched best-configuration Frozen/Ours comparison','',
           'Fixed old P1/P5 sources and streams; selected v3 parameters. Historically exposed. ',
           'All predictions were sealed before GT diagnosis. No full official-query claim or parameter reselection.','',
           '| Dataset / cohort | Frozen vIoU % | Ours vIoU % | Delta pp | Paired source 95% CI pp |',
           '|---|---:|---:|---:|---|']
    for ds in DATASETS:
        b=BASE/ds;assert read(b/'ROOT_READBACK.json')['status']=='pass'
        summary=read(b/'SUMMARY.json');diag=read(b/'PIPELINE_DIAGNOSIS.json')
        for cohort in ['full','outside_tuning_sources']:
            for group in ['corruption','clean']:
                z=summary[cohort][group]['all']['metrics'];delta=z['delta_m_vIoU']
                lines.append(f'| {ds} / {cohort} / {group} | {z["Frozen_m_vIoU"]["mean"]*100:.4f} | {z["Ours_m_vIoU"]["mean"]*100:.4f} | {delta["mean"]*100:+.4f} | [{delta["ci95"][0]*100:+.4f}, {delta["ci95"][1]*100:+.4f}] |')
        lines+=['',f'## {ds} pipeline diagnosis','',
                'The decomposition measures inherited boxes, inherited interval and current temporal rerank separately. ',
                'Spatial step utility fixes the original time support; current-query post-update gains are diagnostic, not online output.','']
        for group in ['corruption','clean']:
            z=diag['full'][group]
            lines += [f'### {group}','',
                      'Temporal correct-support/missed-selection and correct-destroyed/rescued:',
                      '```json',__import__('json').dumps(z['temporal'],indent=2),'```','',
                      'Spatial per-step support, teacher errors and harmful updates:',
                      '```json',__import__('json').dumps(z['spatial_by_step'],indent=2),'```','']
    lines+=['## Scope and next decision','',
            'The two panels differ in dataset, checkpoint and stream length/order count. Inspect negative and positive cases before a model change. ',
            'A smaller observed delta does not prove the mechanism impossible; a positive panel does not promote the configuration. ',
            'The previous all-query run is preserved and paused. No automatic continuation or new model is launched by this report.','']
    path=ROOT/'docs/TA_BEST_QUICK_UPDATE.md';path.write_text('\n'.join(lines))
    print(path)

if __name__=='__main__':run()
