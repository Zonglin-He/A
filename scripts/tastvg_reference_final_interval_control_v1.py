"""Declared post-primary CPU sensitivity readout at the unchanged A Fast interval.

This supplementary readout is added after viewing the primary qualification.
It cannot change the cohort, rewards, selection, primary endpoint, or method.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from tastvg_reference_common_v1 import *
import numpy as np
from vg_tta.tastvg_reference_selection_v1 import pairwise
def run():
    import torch
    torch.set_num_threads(2);verify();assert (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists()
    from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
    from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from scripts.score_tastvg_best_quick_v1 import source_summary
    result={};checks=0;same=0;error=0.
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');old=read(ROOT/p['row_plan']);gt=read(POOL/ds/'GT_LABELS_search.json');rows=[]
        for c,main in zip(p['cells'],read(PUBLIC/ds/'ROWS.json')):
            row=old['rows'][c['parent']];x=load(ROOT/c['payload']);g=gt[str(c['parent'])];truth={int(k):v for k,v in g['truth'].items()};ids=np.asarray(row['frame_ids'])
            same+=sum(z['prediction']['indices']==x['slow']['indices'] for z in x['update_steps'][0]['candidates'])
            utility=[]
            for z in x['update_steps'][0]['candidates']:
                bb=z['prediction']['boxes'];a=evaluator(bb,row,truth,g['span'],ds=='hc2')(x['final_indices'])
                pix=xyxy(bb,row['input']['width'],row['input']['height']);pix=np.maximum(pix,0) if ds=='hc2' else pix
                idx=x['final_indices'];ind=dense_official_metrics(pix,ids,[int(ids[idx[0]]),int(ids[idx[1]])+1],truth,g['span'])
                e=abs(a['v']-ind['m_vIoU']);assert e<1e-10;error=max(error,e);checks+=1;utility.append(a['v'])
            r={k:main[k] for k in ['cell','source_id','order','condition']};r.update(utilities=utility)
            for strategy in ['Uniform','Routed']:
                sel=main[strategy+'_selected'];a=pairwise(main[strategy+'_rewards'],utility)
                r.update({strategy+'_v':utility[sel],strategy+'_gain':utility[sel]-utility[0],strategy+'_regret':max(utility)-utility[sel],strategy+'_pairwise':a['pairwise_accuracy']})
            r['delta_v']=r['Routed_v']-r['Uniform_v'];r['delta_pairwise']=r['Routed_pairwise']-r['Uniform_pairwise'] if r['Routed_pairwise'] is not None else None
            rows.append(r)
        summary={}
        for group in ['corruption','clean']:
            rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption')];summary[group]={}
            for k in ['Uniform_v','Routed_v','Uniform_gain','Routed_gain','Uniform_regret','Routed_regret','delta_v','delta_pairwise']:
                take=[r for r in rr if r[k] is not None];summary[group][k]=source_summary(take,[k])
        write(PUBLIC/ds/'FINAL_INTERVAL_CONTROL.json',dict(post_primary_supplement=True,primary_endpoint_unchanged=True,rows=rows,summary=summary));result[ds]=summary
    write(BASE/'FINAL_INTERVAL_CONTROL_READBACK.json',dict(status='pass',post_primary_supplement=True,dense_scalar_checks=checks,max_error=error,
        own_interval_matches_native=same,total_candidates=540,GT_cannot_affect_prediction=True,GPU_initialized=torch.cuda.is_initialized()))
    write(PUBLIC/'FINAL_INTERVAL_CONTROL_READBACK.json',read(BASE/'FINAL_INTERVAL_CONTROL_READBACK.json'))
    print('SUPPLEMENT', {ds:z['corruption']['delta_v']['metrics']['delta_v'] for ds,z in result.items()},'matching own interval',same,540)
if __name__=='__main__':run()
