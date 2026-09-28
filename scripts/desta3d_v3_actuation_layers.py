"""No-update forward readback of the saved oracle's actual residual chain."""
import sys,argparse,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,PANEL,read,write,sha,allocation,local_dependencies,tensor_sha,verify_seal
from scripts.desta3d_v3_privileged_ptd_qualification import B1,ADAPTER_SHA
CASES=[(7,'event'),(9,'spatial'),(3,'spatial')]

def register(name):
    d=OUT/name;assert not d.exists();rows=read(PANEL/'INPUTS.json');verify_seal(OUT/'oracle001')
    write(d/'CONFIG.json',{'phase_seconds':600,'minimum_free_bytes':8*2**30,'cumulative_cap':None,'maximum_new_bytes':2*2**30,
        'source_GT_in_worker':True,'target_input':False,'target_GT':False,'optimizer_steps':0,'new_native':0,
        'purpose':'Replay3 source stock prefills and original adapter chain; use sealed native logits, not new generation'})
    write(d/'INPUTS.json',[dict(rows[i],original_index=i,diagnostic_branch=b) for i,b in CASES])
    paths=[Path(__file__),ROOT/'protocols/desta3d_v3_actuation_layers_v1.md',B1,d/'CONFIG.json',d/'INPUTS.json',OUT/'FREE_ACTUATION_CPU_PREFLIGHT.json',OUT/'oracle001/PREDICTIONS_SEAL.json']
    write(d/'LOCK.json',{'pins':{str(p):sha(p) for p in local_dependencies(paths)}});write(d/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU','no_optimizer':True})

def run(name):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';d=OUT/name
    with allocation(d) as (cfg,guard):
        import torch,gc
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,inputs_for
        from scripts.desta3d_v2_p0 import adapter_sha256
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from vg_tta.desta3d_v3_latent_oracle import privileged_branch_latents
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        pr=processor_load();model=model_load().eval().requires_grad_(False)
        a=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        a.load_state_dict(torch.load(B1,map_location='cpu',weights_only=False)['adapter']);a.set_train_stage('frozen')
        records=[];pins={}
        for j,row in enumerate(read(d/'INPUTS.json')):
            guard();ep=d/'episodes'/f'{j:02}';ep.mkdir(parents=True)
            old=OUT/'oracle001/episodes'/f"{row['original_index']:02}";branch=row['diagnostic_branch']
            pred=torch.load(old/'original.pt',map_location='cpu',weights_only=False)
            frames,ids=decode(row['input']);prompt,pre=inputs_for(row,pr,frames)
            f=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
            assert all(tensor_sha(f[k])==v for k,v in pred['support'].items() if k in f)
            assert tensor_sha(prompt['pixel_values_videos'])==pred['support']['pixel_values_videos']
            masks=torch.load(old/'ORACLE_MASKS.pt',weights_only=False,map_location='cpu');wrong=torch.load(old/'WRONG_MASKS.pt',weights_only=False,map_location='cpu')
            bname='temporal' if branch=='event' else 'spatial';gate=float(getattr(a,'gate_'+branch).sigmoid())
            data={};stats={}
            for arm,mm in [('original',None),(bname,masks),('wrong_'+bname,wrong[bname])]:
                cm=privileged_branch_latents(a,mm,event=branch=='event',spatial=branch=='spatial') if mm else __import__('contextlib').nullcontext()
                captured={}
                def prehook(module,args):captured['z']=args[0].detach().cpu().clone()
                h=getattr(a,'out_proj_'+branch).register_forward_pre_hook(prehook)
                # Register observer inside privilege so it sees the modulated tensor.
                h.remove()
                with cm:
                    h=getattr(a,'out_proj_'+branch).register_forward_pre_hook(prehook)
                    try:
                        with torch.no_grad():r=a(f['visual_grid'],f['query_tokens'],query_mask=f['query_mask'],frame_times=f['frame_times'])
                    finally:h.remove()
                proposal=r['proposal_delta_'+branch].detach().cpu();updated=r['updated_tokens_'+branch].detach().cpu();cast=updated.to(torch.bfloat16)
                oldeffect=torch.load(old/(arm+'_INJECTION_EFFECT.pt'),weights_only=False,map_location='cpu')[branch]
                assert tensor_sha(cast)==oldeffect['current_sha'],'Layer replay differs from actual sealed merger endpoint'
                data[arm]={'z':captured['z'],'proposal':proposal,'updated':updated,'cast':cast}
                z=data[arm]['z'];stats[arm]={'z_norm':float(z.double().norm()),'proposal_norm':float(proposal.double().norm()),
                    'gated_norm':float((proposal*gate).double().norm()),'updated_norm':float(updated.double().norm()),
                    'gate':gate,'exact_old_postcast_hash':True}
                if arm!='original':
                    base=data['original'];deltaz=z-base['z'];deltap=proposal-base['proposal'];dc=cast.float()-base['cast'].float()
                    stats[arm].update(z_delta_norm=float(deltaz.double().norm()),projection_delta_norm=float(deltap.double().norm()),
                       gated_delta_norm=float((deltap*gate).double().norm()),continuous_delta_norm=float((updated-base['updated']).double().norm()),
                       postcast_delta_norm=float(dc.double().norm()),postcast_changed=int((dc!=0).sum()),postcast_numel=dc.numel())
            torch.save(data,ep/'LAYERS.pt');pins[str(ep/'LAYERS.pt')]=sha(ep/'LAYERS.pt')
            records.append({'key':row['key'],'branch':branch,'stats':stats});del data,f,prompt,r;gc.collect();torch.cuda.empty_cache()
            assert sum(p.stat().st_size for p in d.rglob('*') if p.is_file())<cfg['maximum_new_bytes']
            print('LAYERS',row['key'],branch,flush=True)
        assert adapter_sha256(a)==ADAPTER_SHA and all(p.grad is None for p in a.parameters())
        write(d/'REPORT.json',{'status':'GPU_chain_replay_complete','cases':records,'pins':pins,'new_native':0,'optimizer_steps':0})
        write(d/'COMPLETE.json',{'status':'completed_needs_CPU_readback','cases':3,'report_sha':sha(d/'REPORT.json'),'pins':pins})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);p.add_argument('--name',required=True);a=p.parse_args();globals()[a.action](a.name)
