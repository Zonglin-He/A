"""Read every sealed source case; no new forward, labels or state selection."""
import sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
RUN=ROOT/'artifacts/desta3d_v2/tta_v2/source_fp32_head_control_v1'
OLD=ROOT/'artifacts/desta3d_v2/tta_v2/source_task_control_v2'
NEW='supervised_fp32'
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def array(x):return x.detach().cpu().double().numpy()
def cosine(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
def main():
    torch.set_num_threads(2)
    report=read(RUN/'independent_readback_v1/REPORT.json')
    assert read(RUN/'ROOT_SCORE_CROSSCHECK.json')['status']=='passed'
    for p,h in read(RUN/'ALL_PREDICTIONS_SEAL.json')['pins'].items():assert sha(Path(p))==h
    cfg=read(RUN/'CONFIG.json');initial=load(cfg['checkpoint']['checkpoint'])['adapter']
    rows=read(RUN/'INPUTS.json');cases=[];transitions=[];norms=[];max_final_error=0.
    for i,row in enumerate(rows):
        ep=RUN/'episodes'/f'{i:02}';oe=OLD/'episodes'/f'{i:02}'
        ns=[load(ep/NEW/f'step{j}.pt') for j in (1,2,3)]
        os=[load(oe/'supervised'/f'step{j}.pt') for j in (1,2,3)]
        summary=read(ep/NEW/'SUMMARY.json');final=load(ep/NEW/'FINAL_CALIBRATION.pt')
        ordered=ns[0]['ordered_names'];base=np.concatenate([array(initial[n]).ravel() for n in ordered])
        endpoint=np.concatenate([array(final[n]).ravel() for n in ordered])
        d=sum(array(s['raw']['actual_delta']) for s in ns)
        err=float(np.max(np.abs(base+d-endpoint)));max_final_error=max(max_final_error,err);assert err<1e-10
        # Initial-point precision change: same state/prefix, not later diverged trajectories.
        oldg=array(os[0]['raw']['task_event'])+array(os[0]['raw']['task_spatial'])
        newg=array(ns[0]['raw']['task_event'])+array(ns[0]['raw']['task_spatial'])
        oldd=sum(array(s['raw']['actual_delta']) for s in os)
        first={'cosine':cosine(oldg,newg),'norm_ratio':float(np.linalg.norm(newg)/np.linalg.norm(oldg)),
               'relative_L2':float(np.linalg.norm(newg-oldg)/np.linalg.norm(oldg))}
        norms.append(first)
        for j in (0,1):
            transitions.append({'key':row['key'],'step':j+1,
                'FP32_CE_change_same_grad_mode':ns[j+1]['FP32']['total']-ns[j]['FP32']['total'],
                'BF16_CE_change_same_no_grad_mode':ns[j+1]['BF16']['total']-ns[j]['BF16']['total']})
        predictions={a:load(ep/(a+'.pt')) for a in ['no_update','supervised',NEW]}
        def eq(x,y):return torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y
        comparison={}
        for ref in ['no_update','supervised']:
            p,q=predictions[NEW],predictions[ref]
            comparison[ref]={k:eq(p[k],q[k]) for k in ['boxes_cxcywh','positions','geometry_valid','interval','format_ok']}
        metrics={a:next(x['metrics'] for x in report['arms'][a]['query_rows'] if x['key']==row['key']) for a in report['arms']}
        branches={}
        oldce=read(ep/'OLD_SUPERVISED_DUAL_CE.json')
        for dtype in ['BF16','FP32']:
            branches[dtype]={b:{'old_change':oldce[dtype][b]['ce']-ns[0][dtype][b]['ce'],
                'new_change':summary[dtype+'_after'][b]['ce']-ns[0][dtype][b]['ce']} for b in ['event','spatial']}
        cases.append({'key':row['key'],'source':row['source'],'frames':len(row['input']['frame_ids']),
            'initial_task_gradient_precision_change':first,'final_delta_cosine_old_new':cosine(oldd,d),
            'final_delta_relative_L2':float(np.linalg.norm(d-oldd)/np.linalg.norm(oldd)),
            'intervals':{a:p['interval'] for a,p in predictions.items()},'geometry_equality':comparison,
            'metrics':metrics,'new_v_delta_B1_pp':100*(metrics[NEW]['vIoU']-metrics['no_update']['vIoU']),
            'new_v_delta_old_supervised_pp':100*(metrics[NEW]['vIoU']-metrics['supervised']['vIoU']),
            'branch_CE_changes_same_definition':branches})
    result={'status':'passed_all16_saved_case_and_raw_readback','time':time.time(),'script_sha':sha(Path(__file__)),
        'report_sha':sha(RUN/'independent_readback_v1/REPORT.json'),'queries':16,'new_steps':48,'new_forwards':0,
        'target_data':False,'final_parameter_reconstruction_error':max_final_error,
        'initial_gradient':{k:{'min':min(x[k] for x in norms),'median':float(np.median([x[k] for x in norms])),
            'max':max(x[k] for x in norms)} for k in norms[0]},
        'first_two_step_CE_decreases':{'denominator':32,
            'FP32':sum(x['FP32_CE_change_same_grad_mode']<0 for x in transitions),
            'BF16':sum(x['BF16_CE_change_same_no_grad_mode']<0 for x in transitions)},
        'native_full_geometry_equal_count':{r:sum(all(c['geometry_equality'][r].values()) for c in cases) for r in ['no_update','supervised']},
        'transitions':transitions,'cases':cases,'limitations':'GT-prefix CE versus original free native; one seed and source-training panel, no causal claim from gradient similarity'}
    save_once(RUN/'ROOT_CASE_AND_OBJECTIVE_READBACK.json',result)
    print({k:v for k,v in result.items() if k not in ['cases','transitions']})
if __name__=='__main__':main()
