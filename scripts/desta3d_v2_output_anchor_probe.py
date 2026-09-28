"""Source-only actual cached-output KL connectivity and scale probe."""
from __future__ import annotations
import argparse, fcntl, gc, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts,tensor_sha256,make_mild_photometric_view
from vg_tta.optimizer_checkpoint import cpu_clone
PARENT=ROOT/'artifacts/desta3d_v2';BASE=PARENT/'tta_v2'
SOURCE=BASE/'source_moments_B1_v1'


def save_pt(p,x):
    assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    torch.save(cpu_clone(x),p)


def register(out):
    assert not out.exists()
    row=read(SOURCE/'INPUTS.json')[0];assert row['split']=='train' and '10016' in row['key']
    c={'checkpoint':read(SOURCE/'CONFIG.json')['checkpoint'],'query':row['key'],
       'seed':20260927,'phase_seconds':900,'free_disk_bytes':8*2**30,
       'cumulative_cap_seconds':None,'target_data_read':False,'GT_read':False,
       'perturbation':'existing calibration-alignment fixed3 AdamW1e-5; ephemeral and reset',
       'same_state_logit_atol':1e-5,'same_state_KL_atol':1e-7,
       'readout':'official cached schedule replay with gradients, fixed B1 teacher tokens',
       'parameter_count':66816,'gates_frozen':True}
    save_once(out/'CONFIG.json',c);save_once(out/'INPUT.json',row)
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2_output_anchor.py',ROOT/'tests/test_desta3d_v2_output_anchor.py',
       ROOT/'protocols/desta3d_v2_output_anchor_probe_v1.md',ROOT/'vg_tta/desta3d_v2.py',
       ROOT/'vg_tta/desta3d_v2_ptd.py',ROOT/'vg_tta/desta3d_v2_shared_reference.py',
       ROOT/'vg_tta/desta3d_v2_shared_reference_cached.py',ROOT/'vg_tta/desta3d_v2_tta_pilot.py',
       ROOT/'vg_tta/desta3d_v2_tta_objective.py',ROOT/'vg_tta/optimizer_checkpoint.py',
       ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py',
       ROOT/'scripts/ptd_spatial_adapter_ab_v1.py',SOURCE/'MOMENTS.json',Path(c['checkpoint']['checkpoint']),
       ROOT/'methods/CURRENT_METHOD.json',out/'CONFIG.json',out/'INPUT.json']
    save_once(out/'LOCK.json',{'pins':{str(p):sha(p) for p in paths}})
    save_once(out/'REGISTRATION.json',{'status':'registered_before_GPU','time':time.time(),
       'GT_read':False,'target_data_read':False,'parents':1,'queries':1})


def run(out,name):
    assert not (out/'STARTED.json').exists()
    c=read(out/'CONFIG.json');row=read(out/'INPUT.json')
    for p,h in read(out/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='running';failure=None
    prior=sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ('desta3d_v1','desta3d_v2'))
    try:
        save_once(out/'STARTED.json',{'pid':os.getpid(),'time':time.time(),'prior_seconds':prior})
        def guard():
            assert shutil.disk_usage(ROOT).free>=c['free_disk_bytes']
            assert time.monotonic()-start<c['phase_seconds']
        guard();torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        torch.cuda.reset_peak_memory_stats()
        from scripts.ptd_spatial_adapter_ab_v1 import model_load,processor_load,frames_for,inputs_for
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_output_anchor import capture_teacher,replay_branch,output_kl
        from vg_tta.desta3d_v2_tta_pilot import configure,adapt,forward
        from vg_tta.desta3d_v2_tta_objective import calibration_objective,CalibrationWeights,gradient_groups
        pr=processor_load();model=model_load().eval().requires_grad_(False)
        a=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        initial=torch.load(c['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        a.load_state_dict(initial);a.set_train_stage('frozen');digest=adapter_sha256(a)
        assert digest==c['checkpoint']['adapter_sha256']
        frames,ids=frames_for(row,'clean');prompt,pre=inputs_for(row,pr,frames)
        view_prompt,view_pre=inputs_for(row,pr,make_mild_photometric_view(frames))
        f=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
        vf=capture_stock_fields(model,pr,view_prompt,row['input']['caption'],ids,row['input']['fps'])
        save_once(out/'INPUT_IDENTITY.json',{'key':row['key'],'source':row['source'],'frame_ids':ids,
            'preprocess':pre,'view_preprocess':view_pre,'visual_grid_sha':tensor_sha256(f['visual_grid']),
            'query_tokens_sha':tensor_sha256(f['query_tokens']),'frame_times_sha':tensor_sha256(f['frame_times']),
            'GT_read':False,'target_data_read':False})
        native,trace=capture_teacher(model,pr,prompt,a,f)
        save_pt(out/'TEACHER.pt',{'native':native,'trace':trace})
        assert native['format_ok'] and len(trace['branches'])==2
        anchor,count=configure(a,'calibration');assert count==66816
        with torch.no_grad():teacher=forward(a,f)
        names=[n for n,p in a.named_parameters() if p.requires_grad]
        params=[p for p in a.parameters() if p.requires_grad]
        assert set(names)==set(anchor)
        equal={}
        for branch,index,kind in [('event',0,'time'),('spatial',1,'coordinate')]:
            guard();value,cache=replay_branch(model,prompt,a,f,trace,branch)
            ref=trace['branches'][index]['logits'][kind]
            error=float((value.detach().cpu()-ref).abs().max());kl=float(output_kl(value,ref).detach())
            equal[branch]={'max_logit_error':error,'KL':kl,'cache':cache,'requires_grad':value.requires_grad}
            save_pt(out/f'IDENTITY_{branch}.pt',{'student':value,'teacher':ref,'audit':equal[branch]})
            assert value.requires_grad and error<=c['same_state_logit_atol'] and abs(kl)<=c['same_state_KL_atol'],equal[branch]
            del value;gc.collect();torch.cuda.empty_cache()
        save_once(out/'SAME_STATE.json',equal)
        moments=read(SOURCE/'MOMENTS.json')
        perturb=adapt(a,f,vf,interface='calibration',alignment=.01,moments=moments,steps=3,lr=1e-5)
        save_once(out/'PERTURBATION.json',perturb)
        # Re-enable exactly the declared subset, but retain the original B1 anchor.
        configure(a,'calibration');assert adapter_sha256(a)!=digest
        summaries={}
        def collect(label,loss):
            a.zero_grad(set_to_none=True);loss.backward()
            raw=torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
            assert torch.isfinite(raw).all() and raw.numel()==66816
            groups=gradient_groups(a)
            summaries[label]={'loss':float(loss.detach()),'gradient_l2':float(raw.norm()),'groups':groups}
            save_pt(out/f'GRADIENT_{label}.pt',{'ordered_names':names,'gradient':raw,'summary':summaries[label]})
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            assert all(p.grad is None and not p.requires_grad for p in a.parameter_groups()['gates'])
            a.zero_grad(set_to_none=True)
        loss,terms=calibration_objective(a,forward(a,vf),teacher,anchor,
            weights=CalibrationWeights(alignment=.01),source_moments=moments)
        collect('pre_gate_total',loss);del loss,terms;gc.collect()
        loss=sum((p-anchor[n]).square().sum() for n,p in a.named_parameters() if p.requires_grad)/66816
        collect('parameter_anchor',loss);del loss;gc.collect()
        for branch,index,kind in [('event',0,'time'),('spatial',1,'coordinate')]:
            guard();value,cache=replay_branch(model,prompt,a,f,trace,branch)
            ref=trace['branches'][index]['logits'][kind]
            save_pt(out/f'PERTURBED_{branch}.pt',{'student':value,'teacher':ref,'cache':cache})
            loss=output_kl(value,ref);collect(branch,loss)
            assert summaries[branch]['gradient_l2']>0
            del loss,value;gc.collect();torch.cuda.empty_cache()
        a.load_state_dict(initial);a.zero_grad(set_to_none=True);a.set_train_stage('frozen')
        assert adapter_sha256(a)==digest
        save_once(out/'REPORT.json',{'status':'passed_source_only_interface','same_state':equal,
            'gradient_summaries':summaries,'adapter_reset_exact':digest,'GT_read':False,'target_data_read':False,
            'optimizer_steps':3,'source_queries':1,'source_parents':1,
            'scope':'source connectivity/scale only; no target efficacy or coefficient selection'})
        raw=sorted(out.glob('*.pt'))
        save_once(out/'COMPLETE.json',{'status':'completed_source_probe','pins':{str(p):sha(p) for p in raw},'report_sha':sha(out/'REPORT.json')})
        status='completed';print('OUTPUT_ANCHOR_PROBE_COMPLETE',flush=True)
    except BaseException:
        status='failed';failure=traceback.format_exc();save_once(out/'FAILURE.json',{'failure':failure,'time':time.time()});raise
    finally:
        seconds=time.monotonic()-start
        save_once(PARENT/f'receipts/output_anchor_{name}.json',{'status':status,'seconds':seconds,'prior_seconds':prior,
            'cumulative_seconds':prior+seconds,'cumulative_cap_seconds':None,'failure':failure,
            'peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None})
        lease.close()


def launch(out,name):
    assert not (out/'STARTED.json').exists()
    start=time.monotonic();child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run','--name',name],cwd=ROOT)
    wall=time.monotonic()-start;receipt=PARENT/f'receipts/output_anchor_{name}.json'
    worker=read(receipt)['seconds'] if receipt.exists() else 0.
    save_once(PARENT/f'receipts/output_anchor_{name}_wrapper.json',{'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-worker),'worker_seconds':worker,'child_wall_seconds':wall,'returncode':child.returncode})
    raise SystemExit(child.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch']);p.add_argument('--name',required=True)
    args=p.parse_args();assert args.name.replace('_','').isalnum()
    out=BASE/'output_anchor_probe_v1'/args.name
    if args.action=='register':register(out)
    elif args.action=='run':run(out,args.name)
    else:launch(out,args.name)
