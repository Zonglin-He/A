"""Pinned official Tent/SAR with native readout and real episodic STVG updates."""
from common import *
import argparse,math,time,importlib.util,gc,traceback
FOLLOW=R/'outputs/paper_followup_v1';G=D/'published_TTA'
def official():
    audit=read(FOLLOW/'reference_vendor/audit.json');modules={}
    for name in ['tent','sar','sam']:
        path=FOLLOW/'reference_vendor/mr-eggplant_SAR'/(name+'.py');entry=audit['mr-eggplant/SAR/'+name+'.py'];assert sha(path)==entry['sha256']
        spec=importlib.util.spec_from_file_location('followup_official_'+name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);modules[name]=mod
    return modules

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

def prepare():
 official();p=freeze('published_TTA',dict(cohort='Existing confirm128query/parent/domain; olddevselected fixed Tentlr1e-4steps5 and SARlr1e-4steps5margin.8, no new tuning.',algorithms='Pinned official SAR-repository Tent-LN/SGDmomentum.9 and SAR/SAMrho.05/reliableentropy/EMAresetconstant.2 source imported verbatim; same selected configurations, genuine updates.',scope='TA original native temporaldecoder LayerNormaffine only, cached groundercontext+Sourceboxes fixed. This is a disclosed STVG task adapter, not canonical BN-Tent/fullnetwork/continual-online reproduction.',readout='Primary TentNative/SARNative use originalperoffsetlegalMAP+hull afteractualupdates. JointMAP counterparts use sameupdatedlogits with q0jointMAP; separate readoutcontrols, no MBR. FullMAP compares matchedfixedconfirm cohort, no universalTTA ranking.',reset='Episodic query reset of parameters/grads/optimizer/SARstate; originalnative logits bitwisebefore/afterreset. Originaleval/dropout arithmetic preserved, normaffinegradsenabled.',failure='Officialemptyfilter/nonfinite errors preserved and hardstop, no Sourcefallback/skip/unscoredquery suppression. Trace reliability/entropy/parameterdelta/forwardcost/reset.',cost='Zero newSource/expert/RGB; real GPUdecoderforward/backward optimizersteps logged, inheritedcontextacquisition cost notzero deployment. Oursuses8expert, optimizers0; TTAuses0expert, actual5updates, reportbudgetdifference.'),[Path(__file__),S/'common.py',R/'studies/entropy_tta_v2/run.py',FOLLOW/'reference_vendor/audit.json',*[(FOLLOW/'reference_vendor/mr-eggplant_SAR')/(n+'.py')for n in ['tent','sar','sam']]])
 atomic(G/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.temporal import probabilities,mass_table
 from stvg.readout import prediction
 from studies.entropy_tta_v2.run import restore_inputs
 from map_core import map_interval
 p=validate('published_TTA');assert not(G/'worker.json').exists(),'No blind repeat optimizerwork';setup();torch.set_num_threads(1);gpu=lease();who=identity();atomic(G/'worker.json',who);mods=official();selection=read(D/'selection.json');support=read(OLD/'supports_sealed.json');trials=read(FOLLOW/'tta_selected.json');files={};counts=dict(queries=0,configs=0,optimizer_iterations=0,decoder_calls=0,new_Source=0,new_expert=0);start=time.perf_counter()
 try:
  for ds,groups in selection.items():
   model=load_source(read(R/'configs/default.json')['datasets'][ds]);dec=model.ground_decoder.time_decoder;head=model.temp_embed;initial={k:v.detach().clone()for k,v in dec.state_dict().items()};files[ds]={}
   for row in groups['confirm']:
    key=row['id'];e=support['files'][ds]['confirm'][key];assert sha(OLD/e['shared_path'])==e['shared_sha256'];shared=torch.load(OLD/e['shared_path'],map_location='cpu',weights_only=False,mmap=True);c,parent=load_parent(ds,row);kw=[restore_inputs(off,'cuda')for off in shared['temporal']['offsets']];native=[off['native_logits']for off in shared['temporal']['offsets']];dest=G/ds;out=dest/'predictions'/(key+'.npz');lp=dest/'adapted_logits'/(key+'.pt');rp=dest/'receipts'/(key+'.json');assert not out.exists()and not lp.exists()and not rp.exists();attempt=dest/'attempts'/(key+'.json');assert not attempt.exists();atomic(attempt,dict(worker=who,before_updates=True,GT_read=False));arrays=dict(Source=np.asarray(c['Source']),Source_interval=np.asarray(prediction(c['native_logits'],torch.as_tensor(c['Source']),c['records'],c['frame_ids'])['physical_interval']),frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']));alllogs={};traces={}
    with torch.no_grad():assert all(torch.equal(head(dec(**v))[-1].cpu(),n)for v,n in zip(kw,native))
    for method in ['Tent','SAR']:
     trial=trials[ds][method];tick=time.perf_counter();dec.load_state_dict(initial,strict=True);mod=mods[method.lower()];mod.configure_model(dec);dec.eval();head.eval();params,names=mod.collect_params(dec);wrapper=make_wrapper(dec,head);opt=torch.optim.SGD(params,lr=trial['lr'],momentum=.9)if method=='Tent'else mods['sam'].SAM(params,torch.optim.SGD,lr=trial['lr'],momentum=.9);margin=(trial['margin_fraction']or.4)*math.log(max(z.shape[1]for z in native));adapter=mod.Tent(wrapper,opt,steps=1,episodic=False)if method=='Tent'else mod.SAR(wrapper,opt,steps=1,episodic=False,margin_e0=margin,reset_constant_em=.2);steps=[]
     try:
      for step in range(trial['steps']):
       before=len(wrapper.trace);adapter(kw);counts['optimizer_iterations']+=1;events=wrapper.trace[before:];steps.append(dict(step=step,forwards=events,reliable_counts=[sum(v<margin for v in ev['entropy'])for ev in events]if method=='SAR'else None));assert all(torch.isfinite(x).all()for x in dec.parameters()),'Official adaptation produced nonfinite model'
      with torch.no_grad():wrapper(kw);after=[x.detach().cpu().clone()for x in wrapper.last]
      assert all(torch.isfinite(x).all()for x in after);delta=sum(float((v-initial[k]).square().sum())for k,v in dec.state_dict().items())**.5;iv=prediction(after,torch.as_tensor(c['Source']),c['records'],c['frame_ids'])['physical_interval'];_,q=mass_table(probabilities(c['records'],after));joint=map_interval(c['frame_ids'],q);arrays[method+'Native']=np.asarray(c['Source']);arrays[method+'Native_interval']=np.asarray(iv);arrays[method+'JointMAP']=np.asarray(c['Source']);arrays[method+'JointMAP_interval']=np.asarray(joint);alllogs[method]=after;traces[method]=dict(trial=trial,steps=steps,parameter_delta=delta,parameters=names,parameter_scalars=sum(x.numel()for x in params),seconds=time.perf_counter()-tick,decoder_calls=wrapper.calls,official_source_unmodified=True)
     finally:
      counts['decoder_calls']+=wrapper.calls;dec.load_state_dict(initial,strict=True);dec.requires_grad_(False);head.requires_grad_(False);dec.eval();head.eval();opt.zero_grad(set_to_none=True)
      for par in dec.parameters():par.grad=None
     with torch.no_grad():reset=[head(dec(**v))[-1]for v in kw];assert all(torch.equal(x.cpu(),y)for x,y in zip(reset,native));assert all(torch.equal(v,initial[k])for k,v in dec.state_dict().items());traces[method]['reset_exact']=True;counts['configs']+=1;del wrapper,adapter,opt,params,after;gc.collect()
    lp.parent.mkdir(parents=True,exist_ok=True);tmp=lp.with_suffix('.tmp');torch.save(dict(Source=np.asarray(c['Source']),records=c['records'],frame_ids=c['frame_ids'],positions8=c['positions8'],Source_native_interval=arrays['Source_interval'],adapted_logits=alllogs,GT_read=False),tmp);os.replace(tmp,lp);out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
    os.replace(out.with_suffix('.tmp'),out);rec=dict(path=str(out.relative_to(D)),sha256=sha(out),logits_path=str(lp.relative_to(D)),logits_sha256=sha(lp),parent=parent,shared_sha256=e['shared_sha256'],traces=traces,worker=who,GT_read=False);atomic(rp,rec);files[ds][key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));counts['queries']+=1;atomic(G/'progress.json',dict(counts=counts,total=256,worker=who,GT_read=False));print(ds,key,counts,flush=True);del c,shared,kw,reset,arrays,alllogs;gc.collect();torch.cuda.empty_cache()
   del model,dec,head,initial;gc.collect();torch.cuda.empty_cache()
  assert counts['queries']==256 and counts['configs']==512;atomic(G/'sealed.json',dict(files=files,counts=counts,GT_read=False,fingerprint=p['fingerprint'],seconds=time.perf_counter()-start))
 except BaseException:atomic(G/'state.json',dict(status='failed_preserved_no_fallback',error=traceback.format_exc(),counts=counts,worker=who));raise
 finally:gpu.close()
def score():
 import torch,numpy as np
 from full_score import metric
 torch.set_num_threads(1);validate('published_TTA');seal=read(G/'sealed.json');assert seal['counts']['queries']==256 and not seal['GT_read'];selection=read(D/'selection.json');checks=0;maxerr=0.
 for ds,groups in selection.items():
  truth=labels(ds);old={r['id']:r for r in read(MAP/'TA'/ds/'metrics.json')['private_rows']};rr=[]
  for row in groups['confirm']:
   e=seal['files'][ds][row['id']];assert sha(D/e['path'])==e['sha256'];g=truth[row['id']];vals={'FullMAP':old[row['id']]['values']['FullMAP'],'SourceJointMAP':old[row['id']]['values']['SourceJointMAP']}
   with np.load(D/e['path'])as z:
    r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
    for arm in ['Source','TentNative','TentJointMAP','SARNative','SARJointMAP']:vals[arm],err=metric(z[arm],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);assert err<2e-10;checks+=1;maxerr=max(maxerr,err)
   assert vals['Source']==old[row['id']]['values']['Source'];rr.append(dict(id=row['id'],parent=str(g['parent']),duration_group=old[row['id']]['duration_group'],values=vals))
  pairs=[(a,'Source')for a in ['TentNative','TentJointMAP','SARNative','SARJointMAP']]+[('FullMAP',a)for a in ['Source','SourceJointMAP','TentNative','TentJointMAP','SARNative','SARJointMAP']];folder=P/'published_TTA'/ds;res=parent_report(rr,list(rr[0]['values']),pairs,folder);signed_tails(rr,pairs,folder);atomic(G/ds/'metrics.json',dict(private_rows=rr,report=res))
 atomic(G/'score_receipt.json',dict(status='scored',checks=checks,max_error=maxerr,whole256sealed_before_GT=True,seal_sha256=sha(G/'sealed.json')))
def audit():
 import torch,numpy as np
 from stvg.temporal import probabilities,mass_table
 from stvg.readout import prediction
 from map_core import map_interval
 torch.set_num_threads(1);validate('published_TTA');seal=read(G/'sealed.json');score=read(G/'score_receipt.json');assert score['seal_sha256']==sha(G/'sealed.json');checks=0;costs={}
 for ds,files in seal['files'].items():
  costs[ds]={}
  for key,e in files.items():
   for f,h in [('path','sha256'),('logits_path','logits_sha256'),('receipt_path','receipt_sha256')]:assert sha(D/e[f])==e[h];checks+=1
   c=torch.load(D/e['logits_path'],map_location='cpu',weights_only=False,mmap=True)
   with np.load(D/e['path'])as z:
    for method,logs in c['adapted_logits'].items():
     iv=prediction(logs,torch.as_tensor(c['Source']),c['records'],c['frame_ids'])['physical_interval'];_,q=mass_table(probabilities(c['records'],logs));joint=map_interval(c['frame_ids'],q);assert np.array_equal(iv,z[method+'Native_interval'])and np.array_equal(joint,z[method+'JointMAP_interval']);assert np.array_equal(z[method+'Native'],c['Source'])and e['traces'][method]['reset_exact'];cc=costs[ds].setdefault(method,dict(queries=0,adaptation_seconds=0.,decoder_calls=0));cc['queries']+=1;cc['adaptation_seconds']+=e['traces'][method]['seconds'];cc['decoder_calls']+=e['traces'][method]['decoder_calls']
  rr=read(G/ds/'metrics.json')['private_rows'];res=read(P/'published_TTA'/ds/'report.json')
  for arm,fields in res['means'].items():
   for f,v in fields.items():
    vv=[r['values'][arm][f]for r in rr if r['values'][arm][f]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 rec=dict(status='verified_official_Tent_SAR_STVG_adapters',queries=256,configs=512,SHA_checks=checks,adapted_logit_native_and_joint_readouts_replayed_exact=True,all_means_recomputed=True,all_episodic_resets_exact=True,actual_counts=seal['counts'],costs=costs,score_sha256=sha(G/'score_receipt.json'),new_Source=0,new_expert=0,new_GT_scoring=0,scope='Pinned official algorithms, native temporaldecoder LN task adapter; cached earlycontext/spatialboxes; no canonicalfullnetwork/BN/continualclaim; different expert andoptimizerbudgets disclosed.');atomic(G/'reporting_receipt.json',rec);atomic(P/'published_TTA/reporting_receipt.json',rec);atomic(G/'state.json',dict(status='complete_scored_audited'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','predict','score','audit']);z=a.parse_args();dict(prepare=prepare,predict=predict,score=score,audit=audit)[z.command]()
