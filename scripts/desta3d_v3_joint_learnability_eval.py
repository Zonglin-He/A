"""Free native inference for the locked source GT-privileged confirmation set."""
import argparse,gc,json,os,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_joint_learnability import D,ROSTER,OUT,load,setup,example,infer,ADAPTER_SHA
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,local_dependencies,allocation,tensor_sha
E=D/'evaluation'
ARMS=['B1','seed20260928','seed20260929']

def run(name):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from vg_tta.desta3d_v3_data import SourcePool
    from scripts.desta3d_v2_source_fit import prediction_record
    from scripts.desta3d_v2_reference_audit_cached_v3 import details
    from vg_tta.desta3d_v2_prediction_contract import validate_prediction
    from scripts.desta3d_v2_p0 import adapter_sha256
    cfg=read(D/'CONFIG.json');check_pins(read(D/'LOCK.json')['pins'])
    finals={a:D/a/'FINAL.pt' for a in ARMS[1:]}
    for a,p in finals.items():
        done=read(p.parent/'COMPLETE.json');assert done['cursor']==618 and done['windows']==155 and done['final_sha']==sha(p)
    if not (E/'LOCK.json').exists():
        paths=local_dependencies([Path(__file__),ROOT/'scripts/score_desta3d_v3_joint_learnability.py',*finals.values(),D/'LOCK.json'])
        write(E/'LOCK.json',dict(pins={str(p):sha(p) for p in paths}))
    check_pins(read(E/'LOCK.json')['pins']);d=OUT/name;assert not d.exists()
    write(d/'CONFIG.json',{**cfg,'mode':'confirmation_inference'})
    write(d/'LOCK.json',dict(pins={**read(E/'LOCK.json')['pins'],str(E/'LOCK.json'):sha(E/'LOCK.json')}))
    write(d/'REGISTRATION.json',dict(time=time.time(),GT_privileged_evidence=True,metrics_computed=False))
    with allocation(d) as (cfg,guard):
        pr,model,adapter,mixer=setup(cfg['seeds'][0]);pool=SourcePool(load_annotations=False)
        db=sqlite3.connect(f'file:{ROSTER/"SOURCE.sqlite"}?mode=ro',uri=True)
        rows=read(D/'VALIDATION_INPUTS.json');provenance=read(D/'VALIDATION_PROVENANCE.json')
        start=time.monotonic();completed=0
        for i,row0 in enumerate(rows):
            guard();ep=E/'episodes'/f'{i:04}'
            if (ep/'COMPLETE.json').exists():
                check_pins({str(ep/p):h for p,h in read(ep/'COMPLETE.json')['files'].items()});completed+=1;continue
            if time.monotonic()-start>cfg['phase_seconds']-120:break
            assert not ep.exists(),'Partial episode requires explicit recovery, never silent replay'
            ep.mkdir(parents=True)
            row=json.loads(json.dumps(row0));path,media=pool.media(provenance[row['key']],materialize=True)
            assert media['sha256']==row['input']['video_sha256'];row['input']['video_path']=str(path)
            label=json.loads(db.execute('select labels_json from examples where key=?',(row['key'],)).fetchone()[0])
            prompt,pre,fields,args=example(pr,model,adapter,row,label)
            support={k:tensor_sha(v) for k,v in fields.items() if isinstance(v,torch.Tensor)}
            write(ep/'INPUT.json',dict(key=row['key'],source=row['source'],support=support,preprocess=pre,frame_ids=row['input']['frame_ids'],
                GT_used='source evidence at inference, not decoder prefix or update',evidence_sha=tensor_sha(args[-2])))
            for arm in ARMS:
                if arm!='B1':mixer.load_state_dict(load(finals[arm])['mixer'])
                result,inj=infer(model,pr,adapter,mixer,prompt,fields,args,base=arm=='B1')
                pred=prediction_record(result,row,pre,ADAPTER_SHA)
                pred.update(arm=arm,readout=details(result),GT_read=True,GT_purpose='privileged source evidence',decoder_GT_prefix=False,
                    target_read=False,optimizer_steps=0,support=support,mixer_sha=None if arm=='B1' else sha(finals[arm]),injection=inj)
                torch.save(pred,ep/(arm+'.pt'));validate_prediction(pred,len(row['input']['frame_ids']))
                assert adapter_sha256(adapter)==ADAPTER_SHA and all(p.grad is None for p in model.parameters())
                del pred,result;gc.collect();torch.cuda.empty_cache()
            files={p.name:sha(p) for p in ep.iterdir() if p.is_file()};write(ep/'COMPLETE.json',dict(index=i,files=files))
            completed+=1;print('JOINT_CONFIRMATION',completed,len(rows),flush=True)
            del args,fields,prompt;gc.collect();torch.cuda.empty_cache()
        if completed==len(rows):
            files={str(p.relative_to(E)):sha(p) for p in (E/'episodes').rglob('*') if p.is_file()}
            write(E/'PREDICTIONS_SEAL.json',dict(files=files,predictions=len(rows)*3,source_GT_privileged=True,metrics_computed=False))
            write(E/'COMPLETE.json',dict(queries=len(rows),parents=31,predictions=len(rows)*3,seal_sha=sha(E/'PREDICTIONS_SEAL.json')))
        write(d/'COMPLETE.json',dict(status='complete' if completed==len(rows) else 'safe_pause',completed_queries=completed))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)
