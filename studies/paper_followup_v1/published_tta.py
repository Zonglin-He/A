"""Pinned official Tent-LN and SAR on native STVG endpoint categoricals.

The published optimizer algorithms are imported verbatim, not reimplemented.
STVG output mapping and final-decoder update scope are explicitly task adapters.
"""
from common import *
import argparse,math,time,subprocess,importlib.util

def official():
    audit=read(D/'reference_vendor/audit.json');modules={}
    for name in ['tent','sar','sam']:
        path=D/'reference_vendor/mr-eggplant_SAR'/(name+'.py');entry=audit['mr-eggplant/SAR/'+name+'.py'];assert sha(path)==entry['sha256']
        spec=importlib.util.spec_from_file_location('followup_official_'+name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);modules[name]=mod
    return modules

def configs():
    return [dict(id=f'{method}_lr{lr}_n{steps}_m{margin}',method=method,lr=lr,steps=steps,margin_fraction=margin)for method in ['Tent','SAR']for lr in [1e-4,1e-3,1e-2]for steps in [1,5]for margin in ([None]if method=='Tent'else [.4,.8])]

def prepare():
    check();audit=read(D/'reference_vendor/audit.json');official();selection=read(OLD/'selection.json');atomic(D/'tta_selection.json',selection)
    p=freeze('tta',dict(seed=SEED,selection_sha256=sha(D/'tta_selection.json'),support_seal_sha256=sha(OLD/'supports_sealed.json'),official_source=audit,trials=configs(),baseline='Existing Tent (ICLR2021) and SAR (ICLR2023). Official SAR-repository Tent implements LN/GN support; official files imported verbatim.',scope='TA-STVG primary backbone; actual final native temporal-decoder LayerNorm affine only, early context/spatial boxes fixed, no expert or new Source. Task mapping: four endpoint categorical rows, invalid padding finfo.min. Episodic query reset; evaluate AFTER updates, not canonical continual-online prediction.',mode='Grounder dropout remains original eval; LayerNorm train/eval arithmetic identical. Norm affine grads enabled by official configure_model, official collect_params unchanged (SAR exclusion rules preserved).',optimizer='Official SGD momentum.9; SAR official SAM rho.05/adaptiveFalse/reliable entropy/SAM/EMA recovery, reset constant.2; official code unchanged including empty-filter behavior; log reliability counts and any numerical failures, never drop queries.',selection_rule='All 18 trials across64 development parents/domain seal before scoring; per method/domain highest dev vIoU, tie severe union then declared order; freeze before independent128 fixed confirm parents/domain. Both cohorts historically TA-exposed; confirm not used for new tuning.',budgets=dict(dev_queries=128,confirm_queries=256,development_config_queries=2304,max_optimizer_iterations=9472,new_expert=0,new_Source=0),input_restore_sha256=sha(R/'studies/entropy_tta_v2/run.py'),production_unchanged=True,not_PseudoBox=True),['common.py','published_tta.py'])
    atomic(D/'tta_state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'],flush=True)

def make_wrapper(dec,head):
    import torch
    class Endpoints(torch.nn.Module):
        def __init__(self):
            super().__init__();self.decoder=dec;self.head=head;self.calls=0;self.trace=[];self.last=[]
        def forward(self,kwargs):
            zz=[self.head(self.decoder(**kw))[-1]for kw in kwargs];self.calls+=len(zz);self.last=zz;maximum=max(z.shape[1]for z in zz);rows=[]
            for z in zz:
                assert z.shape[0]==1 and z.shape[2]==2
                v=z[0].T;rows.append(torch.nn.functional.pad(v,(0,maximum-v.shape[1]),value=torch.finfo(v.dtype).min))
            out=torch.cat(rows);entropy=-(out.softmax(1)*out.log_softmax(1)).sum(1);self.trace.append(dict(entropy=entropy.detach().cpu().tolist()));return out
    return Endpoints()

def cpu_audit():
    import torch
    mods=official();torch.manual_seed(SEED);results={}
    for method in ['Tent','SAR']:
        model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.LayerNorm(8),torch.nn.Linear(8,5));mod=mods[method.lower()];mod.configure_model(model);params,names=mod.collect_params(model)
        opt=torch.optim.SGD(params,lr=.001,momentum=.9)if method=='Tent'else mods['sam'].SAM(params,torch.optim.SGD,lr=.001,momentum=.9)
        cls=mod.Tent if method=='Tent'else mod.SAR;kw={}if method=='Tent'else dict(margin_e0=.8*math.log(5));adapt=cls(model,opt,steps=1,episodic=False,**kw);initial={k:v.clone()for k,v in model.state_dict().items()};x=torch.randn(4,4);adapt(x);assert all(torch.isfinite(v).all()for v in model.parameters());adapt.reset();assert all(torch.equal(v,initial[k])for k,v in model.state_dict().items());results[method]=dict(official_class_executed=True,reset_exact=True,parameters=names)
    a=torch.randn(2,7);b=torch.nn.functional.pad(a,(0,2),value=torch.finfo(a.dtype).min);assert torch.allclose(mods['sar'].softmax_entropy(a),mods['sar'].softmax_entropy(b),atol=1e-6,rtol=0);atomic(D/'tta_CPU_audit.json',dict(status='passed',results=results,padding_entropy_error_below_1e6=True,real_model=False));print('Official CPU audit passed',flush=True)

def predict(stage):
    import torch,numpy as np,gc
    from stvg.runtime import setup,lease,load_source
    from stvg.temporal import probabilities,mass_table,final_mbr
    from studies.entropy_tta_v2.run import restore_inputs
    p=validate('tta');assert sha(D/'tta_selection.json')==p['selection_sha256']and sha(OLD/'supports_sealed.json')==p['support_seal_sha256']and sha(R/'studies/entropy_tta_v2/run.py')==p['input_restore_sha256'];setup();torch.set_num_threads(1);gpu=lease();mods=official();support=read(OLD/'supports_sealed.json');selection=read(D/'tta_selection.json');files={};counts=dict(queries=0,configs=0,optimizer_iterations=0,decoder_calls=0,Source_forwards=0,new_expert=0);started=time.perf_counter();atomic(D/f'tta_{stage}_worker.json',identity())
    try:
        for ds,groups in selection.items():
            model=load_source(read(R/'configs/default.json')['datasets'][ds]);dec=model.ground_decoder.time_decoder;head=model.temp_embed;initial={k:v.detach().clone()for k,v in dec.state_dict().items()};files[ds]={};trials=p['trials']if stage=='dev'else list(read(D/'tta_selected.json')[ds].values());cases=groups['screen'if stage=='dev'else'confirm'];names_by_method={}
            for i,row in enumerate(cases,1):
                key=row['id'];e=support['files'][ds]['screen'if stage=='dev'else'confirm'][key];assert sha(OLD/e['shared_path'])==e['shared_sha256'];shared=torch.load(OLD/e['shared_path'],map_location='cpu',weights_only=False);assert sha(shared['parent_capture'])==shared['parent_capture_sha256'];cap=torch.load(shared['parent_capture'],map_location='cpu',weights_only=False);kw=[restore_inputs(off,'cuda')for off in shared['temporal']['offsets']];native=[off['native_logits']for off in shared['temporal']['offsets']];files[ds][key]={}
                with torch.no_grad():assert all(torch.equal(head(dec(**v))[-1].cpu(),n)for v,n in zip(kw,native))
                counts['decoder_calls']+=2
                for trial in trials:
                    dest=D/'tta'/stage/ds/trial['id'];dest.mkdir(parents=True,exist_ok=True);out=dest/(key+'.npz');assert not out.exists();tick=time.perf_counter();dec.load_state_dict(initial,strict=True);mod=mods[trial['method'].lower()];mod.configure_model(dec);dec.eval();head.eval();params,names=mod.collect_params(dec);names_by_method[trial['method']]=dict(names=names,scalars=sum(x.numel()for x in params));wrapper=make_wrapper(dec,head);opt=torch.optim.SGD(params,lr=trial['lr'],momentum=.9)if trial['method']=='Tent'else mods['sam'].SAM(params,torch.optim.SGD,lr=trial['lr'],momentum=.9);margin=(trial['margin_fraction']or.4)*math.log(max(z.shape[1]for z in native));adapter=mod.Tent(wrapper,opt,steps=1,episodic=False)if trial['method']=='Tent'else mod.SAR(wrapper,opt,steps=1,episodic=False,margin_e0=margin,reset_constant_em=.2);iterations=[];status='ok'
                    after=None
                    try:
                        for step in range(trial['steps']):
                            before=len(wrapper.trace);adapter(kw);counts['optimizer_iterations']+=1;events=wrapper.trace[before:];iterations.append(dict(step=step,forwards=events,reliable_counts=[sum(v<margin for v in ev['entropy'])for ev in events]if trial['method']=='SAR'else None));assert all(torch.isfinite(x).all()for x in dec.parameters()),'Nonfinite official-method state'
                        with torch.no_grad():wrapper(kw);after=[z.detach().cpu().clone()for z in wrapper.last]
                        _,q=mass_table(probabilities(cap['records'],after));iv=final_mbr(cap['frame_ids'],q)['interval'];delta=sum(float((v-initial[k]).square().sum())for k,v in dec.state_dict().items())**.5
                    except (FloatingPointError,AssertionError)as exc:
                        status='numerical_failure_Source_fallback';iv=list(cap['native_interval']);delta=None;iterations.append(dict(error=str(exc)))
                    finally:
                        counts['decoder_calls']+=wrapper.calls;dec.load_state_dict(initial,strict=True);dec.requires_grad_(False);head.requires_grad_(False);dec.eval();head.eval();opt.zero_grad(set_to_none=True)
                        for par in dec.parameters():par.grad=None
                    with torch.no_grad():reset=[head(dec(**v))[-1]for v in kw];assert all(torch.equal(x.cpu(),y)for x,y in zip(reset,native))
                    counts['decoder_calls']+=2;assert all(torch.equal(v,initial[k])for k,v in dec.state_dict().items());np.savez_compressed(out,Source=cap['Source'],Source_interval=np.asarray(cap['native_interval']),Adapted_interval=np.asarray(iv),positions8=cap['positions8'],frame_ids=cap['frame_ids']);trace=dest/(key+'.trace.json');atomic(trace,dict(status=status,trial=trial,iterations=iterations,parameter_delta=delta,reset_exact=True,official_source_unmodified=True,seconds=time.perf_counter()-tick,GT_read=False));files[ds][key][trial['id']]=dict(path=str(out.relative_to(D)),sha256=sha(out),trace_path=str(trace.relative_to(D)),trace_sha256=sha(trace),status=status);counts['configs']+=1;del wrapper,adapter,opt,params,after;gc.collect()
                counts['queries']+=1;atomic(D/f'tta_{stage}_progress.json',dict(dataset=ds,domain_completed=i,counts=counts,seconds=time.perf_counter()-started,GT_read=False,**identity()));print(stage,ds,i,counts,flush=True);del shared,cap,kw,reset;gc.collect();torch.cuda.empty_cache()
            atomic(D/f'tta_{stage}_{ds}_parameters.json',names_by_method);del model,dec,head,initial;gc.collect();torch.cuda.empty_cache()
        atomic(D/f'tta_{stage}_sealed.json',dict(files=files,counts=counts,GT_read=False,fingerprint=p['fingerprint'],seconds=time.perf_counter()-started))
    finally:gpu.close()

def score(stage):
    import torch,numpy as np
    from full_score import metric
    torch.set_num_threads(1);p=validate('tta');seal=read(D/f'tta_{stage}_sealed.json');assert not seal['GT_read']and seal['counts']['queries']==(128 if stage=='dev'else 256);selection=read(D/'tta_selection.json');chosen={};checks=0;maxerr=0.
    for ds,groups in selection.items():
        gt=labels(ds);trials=p['trials']if stage=='dev'else list(read(D/'tta_selected.json')[ds].values());rows_by_trial={t['id']:[]for t in trials};old={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']}
        for row in groups['screen'if stage=='dev'else'confirm']:
            g=gt[row['id']]
            for t in trials:
                e=seal['files'][ds][row['id']][t['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256'];vals={a:old[row['id']]['values'][a]for a in ['Source','Full']}
                with np.load(D/e['path'])as z:vals[t['method']],err=metric(z['Source'],row,{int(k):v for k,v in g['boxes'].items()},g['interval'],z['Adapted_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err)
                rows_by_trial[t['id']].append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals,status=e['status']))
        chosen[ds]={};ranking=[]
        for order,t in enumerate(trials):
            rr=rows_by_trial[t['id']];res=report(rr,list(rr[0]['values']),[(t['method'],'Source'),('Full',t['method'])],P/'tta'/stage/ds/t['id']);res['numerical_failures']=sum(r['status']!='ok'for r in rr);atomic(D/'tta'/stage/ds/(t['id']+'_metrics.json'),dict(report=res,private_rows=rr));ranking.append(dict(trial=t,mean_v=res['means'][t['method']]['v'],severe=res['severe_vs_Source'][t['method']]['union'],failures=res['numerical_failures'],order=order))
        if stage=='dev':
            for method in ['Tent','SAR']:
                complete=[x for x in ranking if x['trial']['method']==method and x['failures']==0];assert complete,'No complete '+method+' config';chosen[ds][method]=min(complete,key=lambda x:(-x['mean_v'],x['severe'],x['order']))['trial']
        atomic(P/'tta'/stage/ds/'ranking.json',dict(ranking=ranking,selection_scope='development only'if stage=='dev'else'fixed confirmation; no selection'))
    if stage=='dev':atomic(D/'tta_selected.json',chosen);atomic(P/'tta_selected.json',chosen)
    atomic(D/f'tta_{stage}_score_receipt.json',dict(checks=checks,max_error=maxerr,seal_sha256=sha(D/f'tta_{stage}_sealed.json'),all_stage_sealed_before_GT=True));print('Scored',stage,checks,flush=True)

def run():
    p=validate('tta');atomic(D/'tta_launch.json',identity())
    for stage in ['dev','confirm']:
        for command in ['predict','score']:
            atomic(D/'tta_state.json',dict(status=stage+'_'+command,fingerprint=p['fingerprint'],**identity()));child=subprocess.Popen([sys.executable,'-B',__file__,command,'--stage',stage]);atomic(D/f'tta_{stage}_{command}_child.json',identity(child.pid));assert child.wait()==0,'Preserve failure; no blind retry'
    atomic(D/'tta_state.json',dict(status='complete_scored_pending_publication',fingerprint=p['fingerprint'],**identity()))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','cpu_audit','predict','score','run']);a.add_argument('--stage',choices=['dev','confirm']);z=a.parse_args();{'prepare':prepare,'cpu_audit':cpu_audit,'predict':lambda:predict(z.stage),'score':lambda:score(z.stage),'run':run}[z.command]()
