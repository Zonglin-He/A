"""Bounded fresh actual four-candidate decoder forwards at sealed A boundaries."""
import collections,gc,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_forward_candidate_common_v1 import *

def run():
    import torch
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old();verify();prior.standby()
    from scripts.run_stvg_opd_expanded_after_p8_priority001 import gpu
    handle=gpu()
    from scripts.run_decota_paper_main_v1 import model_for,read_row,frames_for,unpack_expert
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    from vg_tta.c1_enabling_tricks_v1 import TrickReplay
    from vg_tta.decota_optimizer_posterior_r1_v1 import Energy
    from methods.decota_final_simplified_v1.tensors import state_hash,detached
    from vg_tta.stvg_opd_forward_candidate_update_v1 import ARMS,candidate_states,metrics,choose,useful
    from scripts.decota_paper_common_v1 import budget
    rr=read(BASE/'DESIGN_LOCK.json')['records'];cache=collections.OrderedDict()
    model=None;ds=None;count=0;captures=0;bindings={};start=time.time()
    try:
        for r in rr:
            budget();prior.standby()
            if ds!=r['dataset']:
                if model is not None:
                    assert state_hash(model.state_dict())==mh;del model;cache.clear();gc.collect();torch.cuda.empty_cache()
                torch.manual_seed(20260920);model=model_for(r['source']);mh=model._fixed_full_state_hash;ds=r['dataset']
            inp=normalized(load(ROOT/r['originals'][0]['input_path']));expert=unpack_expert(inp['expert'])
            A=load(ROOT/r['A_path'])['fit'];assert mh==r['source_model_sha256']==inp['source_model_state_sha256']
            for repeat in [0,1]:
                row=read_row(ds,r['query_ordinal']);frames,ids,_=frames_for(ds,row,cache);shifted,pixel,spec=observation(ds,row,'clean',frames)
                batch,recs,base=frozen_forward(model,shifted,row);captures+=1
                assert pixel==r['pixel_sha256']==inp['pixel_sha256'] and ids==inp['frame_ids'] and spec is None
                assert torch.equal(base.zero['boxes'].cpu(),inp['native_boxes'])
                del frames,shifted,batch
                base.restore(A['initial']);rp=TrickReplay(base,ids,row['key'],{})
                with torch.no_grad():entering=rp.values()['boxes'].detach().clone()
                assert torch.equal(entering.cpu(),A['before'])
                support=Energy(expert,entering,'top1');positions=[int(t[0]) for t in support.frames]
                targets=torch.cat([t[1] for t in support.frames],0).detach() if positions else entering.new_empty((0,4))
                assert positions==A['positions']
                U=[i for i in range(len(ids)) if i not in set(r['visited'])]
                for k in [0,r['config']['steps']//2,r['config']['steps']-1]:
                    budget();prior.standby();before=A['path'][k]['state'];after=A['path'][k+1]['state']
                    rp.restore(before)
                    with torch.no_grad():current=rp.values()['boxes'].detach().clone()
                    assert torch.equal(current.cpu(),A['path'][k]['boxes'])
                    torch.cuda.synchronize();tick=time.perf_counter();cs=candidate_states(before,after)
                    outputs={};scores={};costs={}
                    for name in ARMS:
                        torch.cuda.synchronize();t=time.perf_counter();rp.restore(cs[name])
                        with torch.no_grad():boxes=rp.values()['boxes'].detach().clone()
                        scores[name]=metrics(boxes,current,entering,positions,targets,U)
                        outputs[name]=boxes.cpu();torch.cuda.synchronize();costs[name]=time.perf_counter()-t
                    selected,feasible,reason=choose(scores);better=useful(scores,feasible)
                    rp.restore(cs[selected]);torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                    assert torch.equal(outputs['A'],A['path'][k+1]['boxes'])
                    assert prior.old.digest(rp.state())==prior.old.digest(cs[selected])
                    witness=load(prior.BASE/'qualification'/f"{r['index']:05}"/f'P_k{k}_repeat{repeat}.pt')['witness']
                    if witness['updated']:
                        assert torch.equal(outputs['A'],witness['original_proposal_boxes'])
                        raw=torch.cat([(after[n]-before[n]).flatten() for n in before])
                        assert torch.equal(raw,A['path'][k]['update']['raw'])
                    else:
                        assert torch.equal(outputs['A'],current.cpu()) and prior.old.digest(before)==prior.old.digest(after)
                    z=dict(index=r['index'],dataset=ds,repeat=repeat,boundary=k,frame_ids=ids,
                      entering=entering.cpu(),current=current.cpu(),positions=positions,visited=r['visited'],unvisited=U,targets=targets.cpu(),
                      before_state=before,original_after_state=after,candidate_states=cs,outputs=outputs,scores=scores,
                      selected=selected,feasible=feasible,reason=reason,useful_alternatives=better,
                      candidate_seconds=costs,screen_seconds=elapsed,original_informative=witness['updated'],
                      A_complete_fit_sha256=r['A_sha256'],proposal_witness_sha256=sha(prior.BASE/'qualification'/f"{r['index']:05}"/f'P_k{k}_repeat{repeat}.pt'),
                      original_noise_sha256=prior.old.digest(A['rounds'][k]['rollout']) if witness['updated'] else None,
                      pixel_sha256=pixel,source_model_sha256=mh,Native_interval=inp['interval'],runtime_sha256=sha(BASE/'RUNTIME.json'),
                      actual_full_fit=False,new_backward_calls=0,new_optimizer_steps=0,new_DINO_calls=0,GT_read=False)
                    if repeat:
                        first=load(BASE/'comparisons'/f"{r['index']:05}"/f'boundary{k}_repeat0.pt')
                        z['repeat_exact']=all(prior.old.digest(first[n])==prior.old.digest(z[n]) for n in ['entering','current','before_state','original_after_state','candidate_states','outputs','scores','selected','feasible','reason','useful_alternatives'])
                    p=BASE/'comparisons'/f"{r['index']:05}"/f'boundary{k}_repeat{repeat}.pt';save(p,z)
                    write(p.with_suffix('.json'),dict(status='accepted_complete_real_forward_comparison',count=count+1,total=48,sha256=sha(p),bytes=p.stat().st_size,GT_read=False,time=time.time()))
                    bindings[str(p.relative_to(ROOT))]=sha(p);count+=1
                    status(BASE/'GPU_STATUS.json',dict(status='running_real_forward_candidates',dataset=ds,done=count,total=48,count=captures,pid=os.getpid(),GT_read=False,time=time.time()))
                    print('FORWARD_CANDIDATES',count,48,ds,k,selected,reason,'alternatives',len(better),flush=True)
                    del z,cs,outputs,witness
                base.restore(base.initial);assert state_hash(model.state_dict())==mh
                del rp,base,entering,current;gc.collect();torch.cuda.empty_cache()
            del A,inp,expert
        assert count==48 and captures==16
        verify();write(BASE/'GLOBAL_READOUT_BARRIER.json',dict(status='sealed',scope='all8 fixed queries/two captures/48 boundary comparisons/192 true forward candidate readouts; zero new fits or GT',files=bindings,count=count,total=48,
          candidate_readouts=192,actual_frozen_captures=captures,new_backward_calls=0,new_optimizer_steps=0,new_DINO_calls=0,
          source_state_restored=True,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False,elapsed_seconds=time.time()-start,time=time.time()))
        status(BASE/'STATUS.json',dict(status='pending_actual_CPU_root_view_public_archive',done=48,total=48,GT_read=False,time=time.time()))
        write(BASE/'worker_exit.json',dict(status='pass',scope='finite actual forward-only comparison finished; source restored; cancelled baseline standby unchanged',count=48,total=48,pid=os.getpid(),GT_read=False,time=time.time()))
    finally:
        if model is not None:del model
        cache.clear();gc.collect();torch.cuda.empty_cache();handle.close()

if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare':prepare()
        else:run()
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc());raise
