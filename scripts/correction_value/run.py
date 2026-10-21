"""Frozen-student single-action interventions, complete sealing before GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import read,sha,atomic,savept
from scripts.evidence_policy.run import validate as validate_previous,TRIAL as PREVIOUS,BANKS
from scripts.box_propagation.bank import CandidateBank
from scripts.correction_value.core import intervene,BRANCHES,continue_student,_prepare
HERE=Path(__file__).resolve().parent;TRIAL=ROOT/'outputs/correction_value_hc64'


def lock_inputs():
    old=read(PREVIOUS/'lock.json');validate_previous(PREVIOUS,old)
    cfg=read(HERE/'config.json');assert cfg['queries']==64 and cfg['maximum_interventions_per_query']==4
    paths=[p for p in HERE.glob('*.py')if not p.name.startswith('test_')]+[HERE/'config.json']
    sources={**old['sources'],**{str(p.relative_to(ROOT)):sha(p)for p in paths}}
    for p,h in sources.items():assert sha(ROOT/p)==h,p
    previous_seal=read(PREVIOUS/'sealed.json');bank_seal=read(BANKS/'sealed.json')
    inputs=[]
    for item in old['inputs']:
        k=item['id'];pr=read(PREVIOUS/'receipts'/f'{k}.json');br=read(BANKS/'receipts'/f'{k}.json')
        inputs.append(dict(item,previous_trace_path=pr['trace_path'],previous_trace_sha256=pr['trace_sha256'],
            previous_prediction_path=previous_seal['files'][k]['path'],previous_prediction_sha256=previous_seal['files'][k]['sha256'],
            bank_path=br['bank_path'],bank_sha256=br['bank_sha256']))
    ident=dict(config=cfg,sources=sources,previous_lock_sha256=sha(PREVIOUS/'lock.json'),
        previous_seal_sha256=sha(PREVIOUS/'sealed.json'),previous_fingerprint=old['fingerprint'],
        bank_seal_sha256=sha(BANKS/'sealed.json'),formal_state_sha256=old['formal_state_sha256'],
        registered_config_sha256=old['registered_config_sha256'],manifest_sha256=old['manifest_sha256'],
        inputs=inputs,ids=old['ids'],GT_read=False)
    L={**ident,'fingerprint':hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()}
    if(TRIAL/'lock.json').exists():assert read(TRIAL/'lock.json')==L,'Immutable intervention trial changed'
    else:
        atomic(TRIAL/'lock.json',L)
        for p in paths:
            d=TRIAL/'source_snapshot'/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(p.read_bytes())
    return cfg,L


def check_sources(L):
    for p,h in L['sources'].items():assert sha(ROOT/p)==h,p
    for p,k in [(PREVIOUS/'lock.json','previous_lock_sha256'),(PREVIOUS/'sealed.json','previous_seal_sha256'),
        (BANKS/'sealed.json','bank_seal_sha256'),(ROOT/'outputs/formal/state.json','formal_state_sha256'),
        (ROOT/'configs/default.json','registered_config_sha256'),(ROOT/'data/evaluation/development.jsonl','manifest_sha256')]:assert sha(p)==L[k],str(p)
    assert not torch.cuda.is_initialized()


def validate_one(item,L):
    k=item['id'];r=read(TRIAL/'receipts'/f'{k}.json');assert r['fingerprint']==L['fingerprint']and r['GT_read']is False
    for key,base,hashkey in [('bank_path',BANKS,'bank_sha256'),('previous_trace_path',PREVIOUS,'previous_trace_sha256'),
        ('previous_prediction_path',PREVIOUS,'previous_prediction_sha256')]:assert sha(base/item[key])==item[hashkey]
    assert sha(TRIAL/r['prediction_path'])==r['prediction_sha256']and sha(TRIAL/r['trace_path'])==r['trace_sha256']
    assert r['parameter_updates']==r['RGB_reads']==r['expert_calls']==r['decoder_calls']==r['model_calls']==0
    trace=torch.load(TRIAL/r['trace_path'],map_location='cpu',weights_only=False)
    assert trace['GT_read']is False and len(trace['branches'])==r['interventions']<=4
    with np.load(TRIAL/r['prediction_path'],allow_pickle=False)as p,np.load(PREVIOUS/item['previous_prediction_path'],allow_pickle=False)as previous:
        assert all(np.array_equal(p[n],previous[n])for n in ('frame_ids','positions8','Strong_interval'))
        assert np.array_equal(p['CurrentGreedy'],previous['B_OPD'])
        assert np.array_equal(p['BaseStudent'],trace['base']['boxes'].numpy())
        M=len(trace['branches']);T=len(p['frame_ids']);assert p['intervention_nodes'].shape==(M,)
        for name in BRANCHES:assert p[name].shape==(M,T,4)and p[name+'_actions'].shape==(M,T)
        assert len({b['point']['node']for b in trace['branches']})==M
        for i,entry in enumerate(trace['branches']):
            point=entry['point'];t=point['node'];assert t==p['intervention_nodes'][i]
            assert p['active'][t]and (p['active'][t+1:]&~p['observed'][t+1:]).any()
            assert point['student_action']!=point['teacher_action']
            for name in BRANCHES:
                path=entry['paths'][name];proof=entry['proof'][name]
                assert all(proof[n]for n in ('prefix_and_incoming_state_identical','common_random_numbers','same_frozen_student_weights','single_forced_action','inactive_E_no_memory_write'))
                assert np.array_equal(p[name][i],path['boxes'].numpy())and np.array_equal(p[name+'_actions'][i],path['actions'].numpy())
            assert np.array_equal(p['Student'][i],p['BaseStudent'])
            if point['random_action']==point['teacher_action']:
                assert np.array_equal(p['Teacher'][i],p['Random'][i])and np.array_equal(p['Teacher_actions'][i],p['Random_actions'][i])
    return r


def validate(trial=TRIAL,lock=None):
    assert Path(trial).resolve()==TRIAL.resolve();L=read(TRIAL/'lock.json')if lock is None else lock;check_sources(L)
    seal=read(TRIAL/'sealed.json');assert seal['queries']==64 and seal['GT_read']is False and seal['fingerprint']==L['fingerprint']
    assert seal['lock_sha256']==sha(TRIAL/'lock.json')and set(seal['files'])==set(L['ids'])
    receipts={}
    for item in L['inputs']:
        k=item['id'];receipts[k]=validate_one(item,L);e=seal['files'][k]
        assert sha(TRIAL/e['path'])==e['sha256']and sha(TRIAL/e['receipt_path'])==e['receipt_sha256']
    return receipts


def main(run=False):
    cfg,L=lock_inputs()
    if not run:print(json.dumps(dict(status='prepared_not_run',fingerprint=L['fingerprint'])));return
    assert not(TRIAL/'run.json').exists()and not(TRIAL/'sealed.json').exists(),'No overwrite/restart'
    from scripts.propagation_opd.run import check_idle
    check_idle();torch.set_num_threads(1);begin=time.perf_counter();files={};policy_cfg=read(PREVIOUS/'lock.json')['config']
    atomic(TRIAL/'run.json',dict(status='running',PID=os.getpid(),fingerprint=L['fingerprint'],GT_read=False))
    try:
        for i,item in enumerate(L['inputs']):
            t0=time.perf_counter();k=item['id'];bank=CandidateBank.from_state(torch.load(BANKS/item['bank_path'],map_location='cpu',weights_only=False))
            old=torch.load(PREVIOUS/item['previous_trace_path'],map_location='cpu',weights_only=False)
            phi=old['B_OPD']['finaltheta'].detach().clone();del old
            with np.load(PREVIOUS/item['previous_prediction_path'],allow_pickle=False)as prev:
                frames=prev['frame_ids'].copy();out=intervene(bank,phi,policy_cfg,item['key'],frames)
                b=_prepare(bank)
                with torch.no_grad():current=__import__('scripts.evidence_policy.policy',fromlist=['rollout']).rollout(bank,phi,policy_cfg,variant='B',collect_only=True,prepared=b)
                assert np.array_equal(current['boxes'].numpy(),prev['B_OPD'])
                M=len(out['branches']);T=len(frames)
                values=dict(frame_ids=frames,positions8=prev['positions8'].copy(),Strong_interval=prev['Strong_interval'].copy(),
                    CurrentGreedy=current['boxes'].numpy(),BaseStudent=out['base']['boxes'].numpy(),active=b['active'].numpy(),observed=b['observed'].numpy(),
                    intervention_nodes=np.array([entry['point']['node']for entry in out['branches']],dtype=np.int64))
                for name in BRANCHES:
                    values[name]=np.stack([entry['paths'][name]['boxes'].numpy()for entry in out['branches']])if M else np.empty((0,T,4),dtype=np.float64)
                    values[name+'_actions']=np.stack([entry['paths'][name]['actions'].numpy()for entry in out['branches']])if M else np.empty((0,T),dtype=np.int64)
            out['current_greedy']=current;out['phi_sha256']=hashlib.sha256(phi.numpy().tobytes()).hexdigest()
            tp=TRIAL/'traces'/f'{k}.pt';savept(tp,out);pp=TRIAL/'predictions'/f'{k}.npz';pp.parent.mkdir(parents=True,exist_ok=True)
            with pp.with_suffix('.tmp').open('wb')as f:np.savez(f,**values)
            os.replace(pp.with_suffix('.tmp'),pp)
            r=dict(fingerprint=L['fingerprint'],prediction_path=str(pp.relative_to(TRIAL)),prediction_sha256=sha(pp),trace_path=str(tp.relative_to(TRIAL)),trace_sha256=sha(tp),
                bank_sha256=item['bank_sha256'],previous_trace_sha256=item['previous_trace_sha256'],phi_sha256=out['phi_sha256'],interventions=M,
                eligible_disagreements=len(out['eligible_disagreement_nodes']),points=[entry['point']for entry in out['branches']],
                proofs=[entry['proof']for entry in out['branches']],seconds=time.perf_counter()-t0,parameter_updates=0,model_calls=0,expert_calls=0,decoder_calls=0,RGB_reads=0,GT_read=False)
            rp=TRIAL/'receipts'/f'{k}.json';atomic(rp,r);validate_one(item,L)
            files[k]=dict(path=str(pp.relative_to(TRIAL)),sha256=sha(pp),receipt_path=str(rp.relative_to(TRIAL)),receipt_sha256=sha(rp))
            atomic(TRIAL/'progress.json',dict(completed=i+1,total=64,seconds=time.perf_counter()-begin,GT_read=False))
            print(json.dumps(dict(completed=i+1,interventions=M,total=64,seconds=time.perf_counter()-begin)),flush=True)
            del bank,out,current,values;gc.collect()
        atomic(TRIAL/'sealed.json',dict(queries=64,fingerprint=L['fingerprint'],lock_sha256=sha(TRIAL/'lock.json'),files=files,GT_read=False));validate()
        atomic(TRIAL/'run.json',dict(status='completed_sealed',queries=64,fingerprint=L['fingerprint'],seconds=time.perf_counter()-begin,GT_read=False))
    except BaseException:
        atomic(TRIAL/'failure.json',dict(completed=len(files),fingerprint=L['fingerprint'],traceback=traceback.format_exc(),GT_read=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();main(a.run)
