"""Fixed eight-parent source gradient panel; no optimizer or target data."""
import argparse,fcntl,json,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.desta3d_v2_p0 import read,write,sha,adapter_sha256
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts
PARENT=ROOT/'artifacts/desta3d_v2';BASE=PARENT/'gradient_scale_panel_v1'

def register():
    assert not BASE.exists();rows=[];seen=set()
    for r in sorted(read(PARENT/'source_fit/INPUTS.json'),key=lambda r:r['key']):
        if r['split']=='train' and r['source'] not in seen:rows.append(r);seen.add(r['source'])
        if len(rows)==8:break
    assert len(rows)==len(seen)==8
    ck=read(PARENT/'shared_reference_v1/CHECKPOINT.json')
    first=PARENT/'shared_reference_v1/pilot003/SOURCE_GRADIENT_DECOMPOSITION.json'
    assert read(first)['key']==rows[0]['key']
    c={'query_selection':'lexical metadata first8 unique training parents; no metric/GT filtering',
       'queries':8,'parents':8,'checkpoint':ck,'seed':20260927,'source_GT_use':'only three fixed loss gradients, no optimizer',
       'reuse_first_completed':str(first),'first_exposure':'first-query scale already observed before panel registration; additional7 selected by metadata; diagnostic development, not confirmatory holdout',
       'parameter_scope':'shared_stem only','losses':['event PTD task','spatial PTD task','referent+event auxiliary'],
       'weighted_ratio':'.1 aux vs event+spatial on identical parameters, per-query; no claim about accumulated windows',
       'target_GT_read':False,'validation_GT_read':False,'optimizer_steps':0,'phase_seconds':1800,'cumulative_cap_seconds':None}
    write(BASE/'CONFIG.json',c);write(BASE/'INPUTS.json',rows)
    paths=[Path(__file__),ROOT/'scripts/desta3d_v2_source_gradient_audit.py',ROOT/'vg_tta/desta3d_v2.py',ROOT/'vg_tta/desta3d_v2_ptd.py',ROOT/'vg_tta/desta3d_v2_training.py',ROOT/'vg_tta/desta3d_v2_source.py',ROOT/'scripts/ptd_8b_teacher_feasibility_v1.py',ROOT/'scripts/desta3d_source_fit_v1.py',ROOT/'scripts/ptd_spatial_adapter_ab_v1.py',PARENT/'source_fit/LOCK.json',PARENT/'shared_reference_v1/CHECKPOINT.json',Path(ck['checkpoint']),first,BASE/'CONFIG.json',BASE/'INPUTS.json',ROOT/'methods/CURRENT_METHOD.json']
    write(BASE/'LOCK.json',{'pins':{str(p):sha(p) for p in paths}});print('REGISTERED',BASE,flush=True)

def run():
    c=read(BASE/'CONFIG.json');rows=read(BASE/'INPUTS.json')
    for p,h in read(BASE/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    assert not (BASE/'STARTED.json').exists()
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='running';failure=None;prior=sum(scan_nested_gpu_receipts(p)[0] for p in [ROOT/'artifacts/desta3d_v1',PARENT])
    try:
        assert shutil.disk_usage(ROOT).free>8*2**30
        write(BASE/'STARTED.json',{'pid':os.getpid(),'time':time.time(),'prior_seconds':prior})
        torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.cuda.reset_peak_memory_stats()
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from scripts.desta3d_v2_source_gradient_audit import audit_source_gradient
        pr=processor_load();model=model_load();adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        adapter.load_state_dict(torch.load(c['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter'])
        assert adapter_sha256(adapter)==c['checkpoint']['adapter_sha256']
        reports=[read(Path(c['reuse_first_completed']))]
        write(BASE/'queries/000/SOURCE_GRADIENT_DECOMPOSITION.json',reports[0])
        for i,row in enumerate(rows[1:],start=1):
            if time.monotonic()-start>c['phase_seconds']:raise RuntimeError('bounded phase review; partial outputs saved')
            assert shutil.disk_usage(ROOT).free>8*2**30
            model.eval();adapter.eval();frames,ids=frames_for(row,'clean');prompt,prep=inputs_for(row,pr,frames)
            fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
            out=BASE/'queries'/f'{i:03}';audit_source_gradient(model,pr,adapter,row,fields,out)
            write(out/'INPUT_IDENTITY.json',{'key':row['key'],'source':row['source'],'frame_ids':ids,'preprocess':prep,'adapter_sha':adapter_sha256(adapter),'video_sha256':row['input']['video_sha256']})
            report=read(out/'SOURCE_GRADIENT_DECOMPOSITION.json');reports.append(report)
            print('GRADIENT',i,row['key'],report.get('B_weighted_aux_to_sum_task_norm_ratio'),flush=True)
            del frames,prompt,fields
        import statistics
        ratios=[x['B_weighted_aux_to_sum_task_norm_ratio'] for x in reports if x.get('B_weighted_aux_to_sum_task_norm_ratio') is not None]
        norms=[x['norms'] for x in reports if 'norms' in x]
        summary={'queries':len(rows),'parents':len({r['source'] for r in rows}),'valid_task_gradients':len(ratios),
          'weighted_aux_task_ratio':{'values':ratios,'median':statistics.median(ratios) if ratios else None,'min':min(ratios) if ratios else None,'max':max(ratios) if ratios else None},
          'task_aux_cosines':[x['angles']['task_sum__aux_unweighted']['cosine'] for x in reports if 'angles' in x],
          'all_adapter_unchanged':all(x.get('adapter_unchanged',True) for x in reports),'source_GT_used':True,'target_GT_read':False,
          'optimizer_steps':0,'scope':'fixed small source development panel, not trajectory effect or evidence for a particular optimizer remedy'}
        write(BASE/'SUMMARY.json',summary);write(BASE/'COMPLETE.json',{'status':'completed','queries':len(rows),'pins':{str(p):sha(p) for p in sorted((BASE/'queries').rglob('*.json'))}});status='completed'
    except BaseException:
        status='failed';failure=traceback.format_exc();write(BASE/'FAILURE.json',{'failure':failure});raise
    finally:
        seconds=time.monotonic()-start;write(PARENT/'receipts/gradient_scale_panel_v1.json',{'status':status,'seconds':seconds,'failure':failure,'prior_seconds':prior,'cumulative_seconds':prior+seconds,'cumulative_cap_seconds':None,'peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None});lease.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);a=p.parse_args();register() if a.action=='register' else run()
