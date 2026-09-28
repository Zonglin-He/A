"""Finite three-baseline HC1 run; no GT in prediction, no new full VidSTG."""
import argparse
import collections
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha

BASE=ROOT/'artifacts/decota_natural_paper_v1'
OUT=BASE/'native_baselines_v1'
PARENT=ROOT/'artifacts/decota_final_simplification_v1'
METHODS=['Tent','MEMO','SAR']
CODE=['vg_tta/native_baselines_paper_v1.py','scripts/run_native_baselines_paper_v1.py',
      'vg_tta/native_probability_interface_v1.py','tests/test_native_baselines_paper_v1.py',
      'protocols/decota_paper_native_baselines_v1.md']


def prepare():
    from vg_tta.native_baselines_paper_v1 import DEFAULTS
    from methods.decota_final_simplified_v1.release import verify_release
    verify_release()
    if (OUT/'LOCK.json').exists(): return verify()
    p=read(PARENT/'LOCK.json'); dev=[r for r in p['rows']['hcstvg1_test'] if r['f44_role']=='development'][:8]
    assert len(dev)==len({r['input']['source'] for r in dev})==8
    write(OUT/'LOCK.json',dict(created=time.time(),methods=METHODS,config=DEFAULTS,dev=dev,
          full=p['full_rows']['hcstvg1_test'],direction='vid_to_hc1',GT_online=False,
          code_pins={n:sha(ROOT/n) for n in CODE},
          pins={str(f):sha(f) for f in [PARENT/'LOCK.json',PARENT/'FINAL_CONFIG.json',
               ROOT/'methods/CURRENT_METHOD.json',ROOT/'methods/CURRENT_WORKING_METHOD.json']},
          labels=p['labels'],labels_sha256=p['labels_sha256'],deferred_new_full_vid=True,
          no_method_tuning=True,no_recurring_task=True,finite_hours=36,
          dependency=str(BASE/'continuation_20260916/STATUS.json')))
    print('PREPARED HC1 native baseline ports',METHODS,flush=True)


def verify():
    from methods.decota_final_simplified_v1.release import verify_release
    verify_release();p=read(OUT/'LOCK.json')
    for n,h in {**p['pins'],**p['code_pins']}.items():
        if sha(ROOT/n)!=h: raise RuntimeError('Pinned baseline input/code changed: '+n)
    return p


def model_load():
    import numpy as np
    import torch
    from methods.decota_final_simplified_v1.config import MethodConfig
    from methods.decota_final_simplified_v1._tastvg_load import load_model_on_device
    from methods.decota_final_simplified_v1.observations import QuerySubjectParser
    cfg=MethodConfig.for_direction('vid_to_hc1'); path=ROOT/cfg.checkpoint
    assert sha(path)==cfg.checkpoint_sha256
    torch.set_num_threads(4);torch.manual_seed(20260910);np.random.seed(20260910)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model,_,_=load_model_on_device(cfg.source_dataset,ROOT/'artifacts/tastvg_runtime/vidstg',
        checkpoint=path,device='cuda',source_dataset=cfg.source_dataset)
    return model.eval().requires_grad_(False),QuerySubjectParser(ROOT/'.cache/stanza')


def receipt(path):
    if path.with_suffix('.json').exists():
        r=read(path.with_suffix('.json'))
        if sha(path)!=r['sha256'] or r['lock_sha256']!=sha(OUT/'LOCK.json'):
            raise RuntimeError('Baseline receipt mismatch')
        return r
    if path.exists(): raise RuntimeError('Unreceipted output preserved')
    return None


def parity(actual,expected):
    import torch
    if actual['indices']!=expected['indices'] or not torch.equal(actual['boxes'],expected['boxes']):
        raise RuntimeError('Frozen boxes/interval do not match paper baseline')
    if any(not torch.equal(a,b) for a,b in zip(actual['logits'],expected['logits'])):
        raise RuntimeError('Frozen logits differ')


def run(phase,limit=0):
    import torch
    from scripts.run_final_simplification_v1 import lease
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.native_baselines_paper_v1 import episode
    p=verify()
    if phase=='full':
        assert read(OUT/'dev/BARRIER.json')['functional_validation_passed']
    gpu=lease()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active: raise RuntimeError('GPU not exclusive: '+active)
    model,parser=model_load();rows=p[phase];receipts=[];pixels=collections.OrderedDict();media={};done=0
    started=time.time()
    for at,row in enumerate(rows):
        path=OUT/phase/f'{row["ordinal"]:06d}.pt';r=receipt(path)
        if r: receipts.append(r);continue
        if time.time()-started>p['finite_hours']*3600: raise TimeoutError('Finite baseline stage cap')
        if shutil.disk_usage(ROOT).free<40*2**30: raise RuntimeError('Disk guard')
        q=row['input'];x=dict(key=row['key'],input=q,GT_online=False,phase=phase,methods={},controls={},
                             lock_sha256=sha(OUT/'LOCK.json'),source=q['source'])
        if row.get('input_unavailable'):
            x.update(status='known_input_unavailable',reason=row['input_unavailable'])
        else:
            if q['video_path'] not in media: media[q['video_path']]=sha(q['video_path'])
            assert media[q['video_path']]==q['video_sha256']
            pk=(q['video_sha256'],tuple(q['frame_ids']));tick=time.perf_counter()
            if pk in pixels: frames,ids=pixels.pop(pk)
            else: frames,ids=decode(q)
            pixels[pk]=(frames,ids)
            while len(pixels)>2: pixels.popitem(last=False)
            x['decode_seconds']=time.perf_counter()-tick
            parent_path=PARENT/'full/hcstvg1_test'/path.name
            parent_r=read(parent_path.with_suffix('.json'));assert sha(parent_path)==parent_r['sha256']
            parent=load(parent_path);assert parent['input']==q
            frozen=parent['predictions']['Frozen'];x['parent_receipt']=parent_r
            for method in METHODS:
                torch.cuda.synchronize();tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                result=episode(model,parser,frames,ids,q,method,p['config'][method])
                torch.cuda.synchronize()
                result.update(seconds_with_parity_forward=time.perf_counter()-tick,
                              peak_memory_allocated=torch.cuda.max_memory_allocated(),
                              peak_memory_reserved=torch.cuda.max_memory_reserved())
                parity(result['Frozen'],frozen)
                assert result['audit']['source_restored'] and result['audit']['expert_calls']==0
                assert torch.isfinite(result['prediction']['boxes']).all()
                if method in ('Tent','MEMO'):
                    assert result['audit']['backwards']>0 and max(result['audit']['gradient_norms'])>0
                x['methods'][method]=result
                if phase=='dev' and at==0:
                    for ctrl,override in [('steps0',dict(steps=0)),('lr0',dict(lr=0.))]:
                        z=episode(model,parser,frames,ids,q,method,{**p['config'][method],**override})
                        parity(z['prediction'],frozen)
                        assert not z['audit']['parameter_changed']
                        x['controls'][method+'_'+ctrl]=z['audit']
            x['status']='available'
        save(path,x);r=dict(key=x['key'],path=str(path),sha256=sha(path),completed=time.time(),
            lock_sha256=x['lock_sha256'],status=x['status'])
        write(path.with_suffix('.json'),r);receipts.append(r);done+=1
        status(OUT/phase/'STATUS.json',dict(done=len(receipts),total=len(rows),key=row['key'],pid=os.getpid(),updated=time.time()))
        print(phase,len(receipts),'/',len(rows),row['key'],x['status'],flush=True)
        del x;gc.collect();torch.cuda.empty_cache()
        if limit and done>=limit: break
    if len(receipts)==len(rows) and not (OUT/phase/'BARRIER.json').exists():
        write(OUT/phase/'BARRIER.json',dict(complete=True,receipts=receipts,queries=len(rows),
            functional_validation_passed=True,selection_by_GT=False,config_sha256=sha(OUT/'LOCK.json')))
    gpu.close()


def score(phase):
    from scripts.analyze_spatial10_components_v1 import checked_score
    from scripts.score_decota_final_freeze_v1 import aggregate
    p=verify();barrier=read(OUT/phase/'BARRIER.json');dest=OUT/phase/'analysis'
    if (dest/'COMPLETION.json').exists(): return
    assert sha(p['labels'])==p['labels_sha256'];labels=read(p['labels']);rows=[];missing=[];audits=[]
    for r in barrier['receipts']:
        assert receipt(Path(r['path']))==r;x=load(r['path'])
        if x['status']!='available': missing.append(dict(key=x['key'],reason=x['reason']));continue
        parent=load(x['parent_receipt']['path']); gt=labels[x['key']];ids=x['input']['frame_ids']
        predictions={m:z['prediction'] for m,z in x['methods'].items()}
        predictions.update(Frozen=parent['predictions']['Frozen'],DeCoTA=parent['predictions']['Full_DeCoTA'])
        vals={m:checked_score(z['boxes'],gt,ids,z['indices'])[0] for m,z in predictions.items()}
        rows.append(dict(key=x['key'],source=x['source'],arms=vals))
        audits.append(dict(key=x['key'],methods={m:z['audit'] for m,z in x['methods'].items()}))
    sources=[r['source'] for r in rows];metrics=['vIoU_corrected','sIoU','tIoU'];names=['Frozen',*METHODS,'DeCoTA']
    arms={a:{m:aggregate([r['arms'][a][m] for r in rows],sources) for m in metrics} for a in names}
    paired={a+' - Frozen':{m:aggregate([r['arms'][a][m]-r['arms']['Frozen'][m] for r in rows],sources)
            for m in metrics} for a in names if a!='Frozen'}
    write(dest/'ALL_QUERY_RESULTS.json',dict(rows=rows,missing=missing))
    write(dest/'ALL_SOURCE_RESULTS.json',dict(queries=len(rows),sources=len(set(sources)),arms=arms,paired=paired))
    write(dest/'AUDIT.json',dict(rows=audits,missing=missing,ViTTA_complete=False,
                               historical_exposure=True,GT_online=False))
    lines=['# Native STVG baseline ports: HC1 '+phase,'',f'{len(rows)} usable queries / {len(set(sources))} parent sources; {len(missing)} unavailable.',
           'Exact matched Frozen, fixed defaults; exposed pool, not untouched confirmation. ViTTA is not completed by this runner.',
           '', '| Method | Source vIoU % | Source sIoU % | Source tIoU % |','|---|---:|---:|---:|']
    for a in names: lines.append('|'+a+'|'+'|'.join(f"{100*arms[a][m]['mean']:.3f}" for m in metrics)+'|')
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(dest/'COMPLETION.json',dict(status='scored_complete',files={str(f):sha(f) for f in dest.iterdir() if f.is_file()}))
    print('SCORED',phase,len(rows),len(set(sources)),flush=True)


def queue():
    p=verify();start=time.time();write(OUT/'PROCESS.json',dict(pid=os.getpid(),started=start,recurring=False))
    while True:
        dep=read(p['dependency'])
        if dep['status']=='failed': raise RuntimeError('Prerequisite queue failed; no GPU contention/retry')
        if dep['status']=='finite_sequence_complete': break
        if time.time()-start>48*3600: raise TimeoutError('Finite dependency wait expired')
        status(OUT/'STATUS.json',dict(status='waiting_for_K_and_Sports_queue',updated=time.time(),pid=os.getpid()))
        time.sleep(30)
    for phase in ['dev','full']:
        for action in ['run','score']:
            status(OUT/'STATUS.json',dict(status='running',phase=phase,action=action,updated=time.time()))
            result=subprocess.run([sys.executable,'-B',__file__,action,'--phase',phase],cwd=ROOT)
            if result.returncode:
                status(OUT/'STATUS.json',dict(status='failed',phase=phase,action=action,returncode=result.returncode,updated=time.time()))
                raise SystemExit(result.returncode)
    status(OUT/'STATUS.json',dict(status='three_HC1_ports_complete',updated=time.time(),
        remaining=['ViTTA_port','HC2_safety','exclusive_efficiency'],full_vid_new_inference='deferred_by_user'))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=['prepare','run','score','queue'])
    a.add_argument('--phase',choices=['dev','full'],default='dev');a.add_argument('--limit',type=int,default=0);v=a.parse_args()
    if v.action=='prepare':prepare()
    elif v.action=='queue':queue()
    elif v.action=='run':run(v.phase,v.limit)
    else:score(v.phase)
