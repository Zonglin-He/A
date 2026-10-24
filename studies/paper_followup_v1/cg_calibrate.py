"""Finite CG-specific development search; new-CG-parent confirmation stays separate."""
from common import *
import argparse,time,subprocess

def prepare():
    import numpy as np,torch
    from stvg.consensus import encode
    check();old=read(OLD/'selection.json');selection={};cal={};cfg=read(R/'configs/default.json');seal=read(OLD/'cgstvg_v9_screen_sealed.json')
    for ds,groups in old.items():
        used={str(r['metadata']['source'])for rows in groups.values()for r in rows};parents={}
        for r in records(ds):
            parent=str(r['metadata']['source'])
            if parent not in used:parents.setdefault(parent,[]).append(r)
        h=lambda x:hashlib.sha256(f'{SEED}:{ds}:{x}'.encode()).hexdigest()
        chosen=sorted(parents,key=h)[:(35 if ds=='hc2'else 128)]
        selection[ds]=dict(dev=groups['screen'],confirm=[min(parents[x],key=lambda r:h(r['id']))for x in chosen])
        assert len(chosen)==(35 if ds=='hc2'else 128)
        residual=[]
        for row in groups['screen']:
            e=seal['files'][ds][row['id']];assert sha(OLD/e['capture_path'])==e['capture_sha256'];cap=torch.load(OLD/e['capture_path'],map_location='cpu',weights_only=False)
            tp=OLD/'cgstvg_teacher_cache'/(cap['teacher_evidence_key']+'.pt');assert sha(tp)==read(tp.with_suffix('.json'))['sha256'];teacher=torch.load(tp,map_location='cpu',weights_only=False)
            d=encode(np.asarray(teacher['boxes']))-encode(cap['Source'])[cap['positions8']];residual.append(d-np.median(d,axis=0))
        residual=np.concatenate(residual);scale=np.maximum(np.quantile(abs(residual),.68,axis=0),1e-6);cut=max(1.,float(np.quantile(((residual/scale)**2).sum(-1),.95)))
        cal[ds]=dict(Transferred=cfg['datasets'][ds]['E_calibration'],CGDevRobust=dict(scale=scale.tolist(),truncation=cut,provenance='64 CG target-development query Source-expert residuals; NO GT; not source-training calibration'))
    atomic(D/'cg_selection.json',selection);atomic(D/'cg_calibrations.json',cal)
    trials=[dict(id=f'{name}_g{gamma}',calibration=name,gamma=gamma,link=.25,context=1.5,kappa=64,beta=1.)for name in ['Transferred','CGDevRobust']for gamma in [0.,.125,.25,.5,1.]]
    p=freeze('cg',dict(seed=SEED,selection_sha256=sha(D/'cg_selection.json'),calibration_sha256=sha(D/'cg_calibrations.json'),old_screen_seal_sha256=sha(OLD/'cgstvg_v9_screen_sealed.json'),stageA=trials,stageB='chosen A plus link0/.5,context1/2,kappa16/256,beta.25/.5/2; 10 finite trials, no posthoc combinations',selection_rule='per domain highest development mean Full vIoU, ties lower severe union then declared order; all trials retained',confirmation='35 remaining HC parents and 128 hash-selected remaining Vid parents; excluded ALL192 prior CG parents/domain before scoring; all TA fullsets previously exposed, NOT historically unseen',budget=dict(cached_dev_config_queries=2560,new_confirm_Source_max=167,new_same_ReferDINO8_max=163),original_negative_CG_preserved=True,production_unchanged=True),['common.py','cg_calibrate.py'])
    atomic(D/'cg_state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'],flush=True)

def feedback(cap,teacher,ds,trial):
    import numpy as np
    from stvg.initialization import initialize
    from stvg.predictor import output_writeback
    from stvg.evidence import SourceEvidence
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.prototype import one_step
    from stvg.temporal import final_mbr
    from cgstvg_interfaces_v9 import pool_single_grid
    from cgstvg_runtime_v9 import clip_native_boxes
    source=cap['Source'];pos=cap['positions8'];ids=cap['frame_ids'];ref=np.asarray(teacher['boxes'],np.float64)
    raw,init=initialize(source,pos,ref,read(D/'cg_calibrations.json')[ds][trial['calibration']]);E=output_writeback(source,raw,trial['gamma']);g=cap['grids']
    ev=SourceEvidence.from_native_masks(g['spatial'],g['spatial_valid'],ids,pos,teacher['native_mask_logits'],teacher['input_shape'][-2:],g['shape'],selected_query=teacher['selected_query']);geo=SpatialGeometry(ev,E,ref)
    def solve(iv):
        bank=geo.for_interval(iv);bank.score_pair*=trial['link']/.25
        return evidence_dp(bank)['final'].numpy()
    aux=solve([ids[0],ids[-1]+1]);poolboxes=aux.copy();poolboxes[:,2:]*=trial['context']/1.5
    features=pool_single_grid(g['projected'],g['valid'],poolboxes);q,stats=one_step(features,cap['q0'],kappa=trial['kappa'],beta=trial['beta']);iv=final_mbr(ids,q)['interval']
    return dict(Full=clip_native_boxes(solve(iv)),Full_interval=np.asarray(iv),Space=clip_native_boxes(solve(cap['native_mbr'])),Space_interval=np.asarray(cap['native_mbr']),SharedCorrection=clip_native_boxes(E),SharedCorrection_interval=np.asarray(cap['native_mbr']),Source=source.copy(),Source_interval=np.asarray(cap['native_mbr']),SourceNative=source.copy(),SourceNative_interval=np.asarray(cap['native_map']),frame_ids=np.asarray(ids),positions8=np.asarray(pos))

def trial_list(stage,ds):
    p=validate('cg')
    if stage=='A':return p['stageA']
    base=read(D/'cg_A_selected.json')[ds];out=[dict(base,id='base')]
    for k,values in [('link',[0.,.5]),('context',[1.,2.]),('kappa',[16,256]),('beta',[.25,.5,2.])]:
        for v in values:out.append(dict(base,**{k:v,'id':f'{k}_{v}'}))
    return out

def predict(stage,ds):
    import numpy as np,torch
    torch.set_num_threads(1);p=validate('cg');assert sha(D/'cg_selection.json')==p['selection_sha256']and sha(D/'cg_calibrations.json')==p['calibration_sha256'];seal=read(OLD/'cgstvg_v9_screen_sealed.json');assert sha(OLD/'cgstvg_v9_screen_sealed.json')==p['old_screen_seal_sha256']
    files={};started=time.perf_counter();trials=trial_list(stage,ds);rows=read(D/'cg_selection.json')[ds]['dev'];atomic(D/f'cg_{stage}_{ds}_worker.json',identity());base_gamma=read(R/'configs/default.json')['datasets'][ds]['initialization_gamma']
    for i,row in enumerate(rows,1):
        key=row['id'];e=seal['files'][ds][key];assert sha(OLD/e['capture_path'])==e['capture_sha256']and sha(OLD/e['path'])==e['sha256'];cap=torch.load(OLD/e['capture_path'],map_location='cpu',weights_only=False);tp=OLD/'cgstvg_teacher_cache'/(cap['teacher_evidence_key']+'.pt');assert sha(tp)==read(tp.with_suffix('.json'))['sha256'];teacher=torch.load(tp,map_location='cpu',weights_only=False);files[key]={}
        for t in trials:
            tick=time.perf_counter();out=D/'cg_dev'/stage/ds/t['id']/(key+'.npz');out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists();arrays=feedback(cap,teacher,ds,t)
            if stage=='A'and t['calibration']=='Transferred'and t['gamma']==base_gamma:
                with np.load(OLD/e['path'])as z:assert all(np.array_equal(arrays[k],z[k])for k in arrays),'Transferred baseline mismatch'
            if stage=='B'and t['id']=='base':
                previous=read(D/f'cg_A_{ds}_sealed.json')['files'][key][read(D/'cg_A_selected.json')[ds]['id']]
                with np.load(D/previous['path'])as z:assert all(np.array_equal(arrays[k],z[k])for k in arrays),'Stage A/B base mismatch'
            np.savez_compressed(out,**arrays);files[key][t['id']]=dict(path=str(out.relative_to(D)),sha256=sha(out),CPU_seconds=time.perf_counter()-tick,capture_sha256=e['capture_sha256'],teacher_sha256=sha(tp),new_Source=0,new_expert=0)
        atomic(D/f'cg_{stage}_{ds}_progress.json',dict(completed=i,total=len(rows),trials=len(trials),seconds=time.perf_counter()-started,GT_read=False,**identity()));print(ds,stage,i,flush=True)
    atomic(D/f'cg_{stage}_{ds}_sealed.json',dict(files=files,trials=trials,fingerprint=p['fingerprint'],queries=len(rows),GT_read=False,seconds=time.perf_counter()-started))

def score(stage):
    import numpy as np,torch,csv
    from full_score import metric
    from cgstvg_score import native_official_metric
    torch.set_num_threads(1);p=validate('cg');seals={ds:read(D/f'cg_{stage}_{ds}_sealed.json')for ds in ['hc2','vidstg']};selected={};checks=0;maxerr=0.
    for ds,seal in seals.items():
        assert seal['queries']==64 and not seal['GT_read'];truth=labels(ds);allrows={t['id']:[]for t in seal['trials']}
        for row in read(D/'cg_selection.json')[ds]['dev']:
            g=truth[row['id']];gt={int(k):v for k,v in g['boxes'].items()}
            for t in seal['trials']:
                e=seal['files'][row['id']][t['id']];assert sha(D/e['path'])==e['sha256'];vals={}
                with np.load(D/e['path'])as z:
                    r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
                    for arm in ['SourceNative','Source','SharedCorrection','Space','Full']:
                        iv=z[arm+'_interval'].tolist();vals[arm],err=metric(z[arm],r,gt,g['interval'],iv,z['positions8']);ref=native_official_metric(z[arm],z['frame_ids'].tolist(),r['metadata'],gt,g['interval'],iv,ds);err=max(err,max(abs(vals[arm][k]-ref[k])for k in ['v','t','s']));assert err<2e-10;maxerr=max(maxerr,err);checks+=1
                allrows[t['id']].append(dict(id=row['id'],parent=str(g['parent']),values=vals,duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s'))
        ranking=[]
        for order,t in enumerate(seal['trials']):
            rr=allrows[t['id']];res=report(rr,list(rr[0]['values']),[('Full','Source'),('Full','Space')],P/'cg_dev'/stage/ds/t['id']);ranking.append(dict(trial=t,mean_v=res['means']['Full']['v'],severe=res['severe_vs_Source']['Full']['union'],order=order));atomic(D/'cg_dev'/stage/ds/(t['id']+'_metrics.json'),dict(private_rows=rr,report=res))
        best=min(ranking,key=lambda x:(-x['mean_v'],x['severe'],x['order']));selected[ds]=best['trial'];atomic(P/'cg_dev'/stage/ds/'ranking.json',dict(ranking=ranking,selected=best['trial'],scope='development selection, NOT independent evidence'))
    atomic(D/f'cg_{stage}_selected.json',selected);atomic(P/f'cg_{stage}_selected.json',selected);atomic(D/f'cg_{stage}_score_receipt.json',dict(checks=checks,max_error=maxerr,seal_sha256={ds:sha(D/f'cg_{stage}_{ds}_sealed.json')for ds in seals},GT_after_all128devsealed=True));print(selected,flush=True)

def run():
    p=validate('cg');atomic(D/'cg_launch.json',identity())
    for stage in ['A','B']:
        atomic(D/'cg_state.json',dict(status='predict_'+stage,**identity(),fingerprint=p['fingerprint']));children=[]
        for ds in ['hc2','vidstg']:
            f=(D/f'cg_{stage}_{ds}.log').open('a');c=subprocess.Popen([sys.executable,'-B',__file__,'predict','--stage',stage,'--dataset',ds],stdout=f,stderr=subprocess.STDOUT);children.append(c);atomic(D/f'cg_{stage}_{ds}_child.json',identity(c.pid))
        codes=[c.wait()for c in children];assert codes==[0,0],codes
        c=subprocess.Popen([sys.executable,'-B',__file__,'score','--stage',stage]);atomic(D/f'cg_{stage}_score_child.json',identity(c.pid));assert c.wait()==0
    atomic(D/'cg_state.json',dict(status='development_selected_awaiting_new_parent_confirmation',selected=read(D/'cg_B_selected.json'),**identity(),fingerprint=p['fingerprint']))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','predict','score','run']);a.add_argument('--stage',choices=['A','B']);a.add_argument('--dataset',choices=['hc2','vidstg']);z=a.parse_args();{'prepare':prepare,'predict':lambda:predict(z.stage,z.dataset),'score':lambda:score(z.stage),'run':run}[z.command]()
