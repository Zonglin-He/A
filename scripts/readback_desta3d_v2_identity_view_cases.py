"""Full case and optimization readout after seal-gated scoring; no GT reopened."""
from pathlib import Path
import sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_tta8_identity_view import OUT,NEW_ARM,CONDITIONS,read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once

def main():
    torch.set_num_threads(2)
    score=OUT/'independent_readback_v1'
    assert read(score/'COMPLETE.json')['report_sha']==sha(score/'REPORT.json')
    seal=read(OUT/'ALL_PREDICTIONS_SEAL.json')['pins']
    metrics={(r['key'],r['condition']):r for r in read(score/'QUERY_METRICS.json')}
    diagnostics=read(score/'OUTPUT_DIAGNOSTICS.json')
    cases=[];norms=[];term_history=[]
    arms=['Frozen','sourcefit_noTTA','calibration_alignment',NEW_ARM]
    for condition in CONDITIONS:
        for i,row in enumerate(read(OUT/'INPUTS.json')):
            episode=OUT/'episodes'/condition/f'{i:02}';data={}
            for arm in arms:
                path=episode/(arm+'.pt');assert sha(path)==seal[str(path)]
                data[arm]=torch.load(path,map_location='cpu',weights_only=False)
            u=data[NEW_ARM]['update']
            assert u['parameter_count']==66816 and not u['output_anchor']
            assert [h['actual_Adam_steps'] for h in u['history']]==[[1],[2],[3]]
            norms.extend(h['clip_norm_before'] for h in u['history'])
            term_history.append({'key':row['key'],'condition':condition,'terms':[h['terms_before'] for h in u['history']],
                'terms_after':u['terms_after'],'loss_before':u['history'][0]['loss_before'],'loss_after':u['loss_after']})
            m=metrics[(row['key'],condition)]['arms']
            cases.append({'key':row['key'],'source':row['source'],'condition':condition,
                'intervals':{a:data[a]['interval'] for a in arms},
                'v_percent':{a:100*m[a]['v'] for a in arms},
                'v_delta_pp':{a:100*(m[NEW_ARM]['v']-m[a]['v']) for a in arms if a!=NEW_ARM},
                's_delta_pp':{a:100*(m[NEW_ARM]['s']-m[a]['s']) for a in arms if a!=NEW_ARM},
                't_delta_pp':{a:100*(m[NEW_ARM]['t']-m[a]['t']) for a in arms if a!=NEW_ARM},
                'diagnostic':next(d for d in diagnostics if d['key']==row['key'] and d['condition']==condition and d['arm']==NEW_ARM)})
    report={'status':'passed','time':time.time(),'cases':cases,'term_history':term_history,
        'new_steps':len(norms),'clip_count':sum(x>1 for x in norms),'gradient_norm_min_max':[min(norms),max(norms)],
        'loss_decreased':sum(r['loss_after']<r['loss_before'] for r in term_history),
        'case_count':24,'GT_reopened':False,'limits':'saved update norms are logs, no raw parameter gradient arrays in this ablation; task scores independently audited separately'}
    save_once(OUT/'ROOT_CASE_READBACK.json',report)
    print({k:v for k,v in report.items() if k not in ['cases','term_history']})

if __name__=='__main__':main()
