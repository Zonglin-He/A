"""One bounded GPU owner, sequential fixed evidence chunks; no auto retries."""
from pathlib import Path
import argparse,subprocess,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch-authorization',required=True);a=ap.parse_args()
    grant=read(a.batch_authorization)
    assert grant['state']=='ROOT_AUTHORIZED_R50_SEQUENTIAL_FIXED_EVIDENCE_BATCH'
    assert grant['program_sha256']==sha(__file__)
    planpath=ROOT/grant['evidence_plan'];authpath=ROOT/grant['evidence_authorization'];plan=read(planpath)
    assert sha(planpath)==grant['evidence_plan_sha256'] and sha(authpath)==grant['evidence_authorization_sha256']
    driver=ROOT/'crossplatform/round50/full_evidence50.py';assert sha(driver)==grant['driver_sha256']
    keys=grant['chunks'];assert keys and len(set(keys))==len(keys)
    stage=grant['stage'];assert stage in ('dino','numeric') and all(plan['chunks'][k]['stage']==stage for k in keys)
    assert sum(plan['chunks'][k]['max_new_calls'] for k in keys)==grant['registered_batch_new_call_cap']
    runtime=grant['runtime'];assert Path(runtime)==Path('LOCAL_HOME/mllm/.venv/bin/python')
    start=time.perf_counter();exp=ROOT/'research/round50/experiment';run=ROOT/plan['output_root']
    for key in keys:
        log=exp/f'FULL_{key.upper()}_ATTEMPT001.log';assert not log.exists()
        assert not (run/plan['chunks'][key]['output_subdir']).exists()
        with log.open('w') as stream:
            subprocess.run([runtime,str(driver),'--plan',str(planpath),'--authorization',str(authpath),
                            '--stage',stage,'--chunk',key],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('EVIDENCE50_BATCH_COMPLETE',stage,key,flush=True)
    if grant['finalize_stage']:
        assert all(j['stage']==stage for j in plan['chunks'].values())
        index={};seals={};n=physical=logical=historical=0
        for key,job in plan['chunks'].items():
            folder=run/job['output_subdir'];seal=read(folder/'SEALED.json');counts=read(folder/'COUNTS.json')
            assert seal['plan_sha256']==sha(planpath) and counts['model_unchanged']
            for rel,digest in seal['files'].items():assert sha(folder/rel)==digest
            rows=read(ROOT/job['rows_file']);assert len(rows)==counts['n']
            for row in rows:
                sid=row['id'];assert sid not in index;p=folder/'cases_PRIVATE'/(sid+'.json')
                index[sid]={'file':p.relative_to(ROOT).as_posix(),'sha256':sha(p)}
            seals[folder.relative_to(ROOT).as_posix()]=sha(folder/'SEALED.json')
            n+=counts['n'];physical+=counts['new_input_forwards_all_processes_of_this_chunk']
            logical+=counts['logical_requests'];historical+=counts['historical_exact_cache_logical_uses']+counts.get('historical_whole_case_logical_uses',0)
        assert n==len(index)==5912 and physical<=sum(j['max_new_calls'] for j in plan['chunks'].values())
        root=run/('dino_chunks' if stage=='dino' else 'evidence_chunks')
        dump(root/'CASE_INDEX_PRIVATE.json',index)
        dump(root/'ALL_CHUNKS_SEALED.json',{'state':'R50_FULL_'+stage.upper()+'_ALL5912_CHUNKS_COMPLETE',
            'n':n,'chunks':len(seals),'sealed_chunks':seals,'physical_new_forwards':physical,'logical_requests':logical,
            'historical_exact_logical_uses':historical,'case_index_sha256':sha(root/'CASE_INDEX_PRIVATE.json'),
            'evidence_plan_sha256':sha(planpath),'current_batch_wall_seconds':time.perf_counter()-start,
            'GT_access':False,'optimizer_steps':0})
        print('EVIDENCE50_ALL_CHUNKS_SEALED',stage,n,physical,logical,flush=True)
if __name__=='__main__':main()
