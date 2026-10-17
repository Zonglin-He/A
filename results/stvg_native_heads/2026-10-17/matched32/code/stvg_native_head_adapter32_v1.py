"""Fixed32 native-head qualification and matched controlled outputs."""
from pathlib import Path
import gc,inspect,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
BASE=ROOT/'artifacts/stvg_native_head_adapter32_v1'
R1=ROOT/'artifacts/stvg_r1_direct_reliable32_v1'
B3=ROOT/'artifacts/stvg_temporal_marginal_decode256_v1'
B2=ROOT/'artifacts/stvg_temporal_marginal_train256_v1'

def verify():
    d=read(BASE/'EXECUTION_RUNTIME.json')
    for p,h in d['pins'].items():assert sha(ROOT/p)==h,p
    if (BASE/'ACTIVE_EXECUTION.json').exists():assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']

def init():
    from vg_tta.stvg_native_head_residual32_v1 import cpu_contracts
    assert not BASE.exists();BASE.mkdir()
    for n in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','TEMPORAL_LOCK.json','R1_FIXED_REFERENCE_LOCK.json','INPUT_JOB_LOCK.json']:(BASE/n).write_bytes((R1/n).read_bytes())
    rr=read(BASE/'ROSTER_LOCK.json')['records'];ids={r['probe_alias'] for r in rr};br={r['probe_alias']:r for r in read(B3/'ROSTER_LOCK.json')['records']}
    assert len(ids)==32 and all(br[r['probe_alias']]['source_case_path']==r['source_case_path'] and br[r['probe_alias']]['parent_sha256']==r['parent_sha256'] for r in rr)
    names=['stvg_native_head_adapter32_v1.py','continue_stvg_native_heads_qwen32_v1.py','report_stvg_native_head_adapter32_v1.py']
    paths=[ROOT/'scripts'/n for n in names]+[ROOT/'vg_tta/stvg_native_head_residual32_v1.py',ROOT/'protocols/stvg_native_head_adapter32_v1.md',ROOT/'methods/CURRENT_METHOD.json',ROOT/'methods/PAPER_METHOD.json',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',ROOT/'scripts/stvg_temporal_marginal_train_fit256_v1.py',ROOT/'scripts/stvg_temporal_marginal_decode_util256_v1.py']
    paths += [BASE/n for n in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','TEMPORAL_LOCK.json','R1_FIXED_REFERENCE_LOCK.json','INPUT_JOB_LOCK.json']]
    paths += [R1/'arms/R1_FIXED_OPD/trials/G32'/n for n in ['GLOBAL_PREDICTION_BARRIER.json','ROOT_READBACK.json','DESIGN_LOCK.json']]
    paths += [B3/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json',B3/'PRIMARY_ROOT_READBACK.json',B2/'FIT_BARRIER.json',B2/'TEACHER_BARRIER.json',B2/'INPUT_BARRIER.json']
    write(BASE/'METHOD_LOCK.json',dict(status='locked_before_new_fits',count=32,per_domain=16,Temporal='existing native temp_embed input residual256x2 exact B3 reparameterization',Spatial='final bbox Linear input residual shared bias-free, native module dimensions',new_Spatial_fits_max=32,new_Teacher_calls=0,reference_clock='sealed T2 spatial inputs fixed; B3 temporal evidence fixed',four_cells_are_controlled_not_fresh_linked_Full=True,GT_read=False))
    paths.append(BASE/'METHOD_LOCK.json');write(BASE/'EXECUTION_RUNTIME.json',dict(status='locked',count=32,pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False))
    write(BASE/'CPU_CONTRACT_COMPLETION.json',cpu_contracts());write(BASE/'STATUS.json',dict(status='prepared',count=32,GT_read=False),mutable=True)

def worker():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts.stvg_temporal_semantic_grounder32_v1 import cached_source
    from scripts.stvg_temporal_semantic_grounder32_v2 import clean
    from scripts.run_decota_paper_main_v1 import model_for
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_native_head_residual32_v1 import BoxHeadResidualReplay,TemporalHeadResidual,fit
    from scripts.stvg_temporal_marginal_decode_util256_v1 import decode
    verify();clean();sys.addaudithook(guard);torch.set_num_threads(2)
    oldmap={f['case_id']:f for f in read(R1/'arms/R1_FIXED_OPD/trials/G32/GLOBAL_PREDICTION_BARRIER.json')['files']}
    bmap={f['case_id']:f for f in read(B3/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json')['files']};fmap={f['case_id']:f for f in read(B2/'FIT_BARRIER.json')['files']}
    refs={r['case_id']:r['anchors']for r in read(BASE/'R1_FIXED_REFERENCE_LOCK.json')['records']};design=read(R1/'arms/R1_FIXED_OPD/trials/G32/DESIGN_LOCK.json')
    files=[];qual=[];model=None;source=None
    def load(f):assert sha(ROOT/f['path'])==f['sha256'];return torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        verify();cid=rec['probe_alias'];c,_=load_case(rec);src='vidstg' if rec['dataset']=='hc2'else'hcstvg2'
        if src!=source:
            if model is not None:assert state_hash(model.state_dict())==model._fixed_full_state_hash;del model;gc.collect();torch.cuda.empty_cache()
            model=model_for(src);source=src
        base=cached_source(model,c);old=load(oldmap[cid]);bd=load(bmap[cid]);tf=load(fmap[cid])['fits']['HYP'];oldfit=old['fits']['M2'];head=base.head
        for _,p in base.named:p.requires_grad_(False)
        zero=BoxHeadResidualReplay(base)
        with torch.no_grad():
            z=zero.values();full=zero.full_values()
        assert torch.equal(z['boxes'],base.zero['boxes']) and torch.equal(full['boxes'],base.zero['boxes'])
        assert all(torch.equal(a,b)for a,b in zip(full['logits'],base.zero['logits']))
        base.head=TemporalHeadResidual(head,tf['theta']*0)
        with torch.no_grad():tzero=base.reference_values(capture=False)
        assert all(torch.equal(a,b)for a,b in zip(tzero['logits'],base.zero['logits']))
        assert torch.equal(tzero['boxes'],base.zero['boxes'])
        source_native=prediction(full['logits'],full['boxes'],c['records'],list(map(int,c['frame_ids'])))
        assert source_native['physical_interval']==list(c['Native_interval'])
        base.head=TemporalHeadResidual(head,tf['theta'])
        tg=torch.autograd.grad(base.head(base.temporal_inputs[0]).sum(),base.head.weight)[0]
        assert torch.isfinite(tg).all() and torch.count_nonzero(tg)>0
        with torch.no_grad():tv=base.reference_values(capture=False)
        assert torch.equal(tv['boxes'],base.zero['boxes'])
        assert all(torch.equal(a.cpu(),b)for a,b in zip(tv['logits'],tf['final_logits']))
        tiv=decode(c['records'],tv['logits'])[0];assert tiv==bd['arms']['B_MARGINAL_DECODE']['interval'];base.head=head
        anchors=[dict(position=a['position'],box=torch.tensor(a['box_cxcywh'],device=base.zero['boxes'].device,dtype=torch.float32),weight=a['weight'])for a in refs[cid]]
        first=not any(q['dataset']==rec['dataset']for q in qual)
        if first:
            # Fixed, reversible engineering perturbation, not accepted method state.
            with torch.no_grad():zero.weight.fill_(1e-4);test=zero.full_values();zero.weight.zero_()
            support={a['position']for a in anchors};non=[i for i in range(len(c['frame_ids']))if i not in support]
            assert any(not torch.equal(test['boxes'][i],base.zero['boxes'][i])for i in non)
        nf=None
        if anchors:
            nf=fit(base,base.initial,anchors,list(map(int,c['frame_ids'])),oldfit['seed_key'],design['configs'][rec['dataset']]['M2'],design['response_settings'][rec['dataset']]['M2'],qualification=first)
            rp=BoxHeadResidualReplay(base);rp.restore(nf['final_state'])
            with torch.no_grad():v=rp.full_values()
            assert torch.equal(v['boxes'].cpu(),nf['final_cached_decoder_boxes'])
            assert all(torch.equal(a,b)for a,b in zip(v['logits'],base.zero['logits']))
            bp=prediction(v['logits'],v['boxes'],c['records'],list(map(int,c['frame_ids'])))['boxes'].cpu().numpy()
        else:bp=np.asarray(c['native_boxes_xyxy_normalized']).copy()
        shape=zero.shape;oldboxes=np.asarray(old['arms']['TS_M2']['boxes']);native=np.asarray(c['native_boxes_xyxy_normalized']);miv=bd['arms']['MARGINAL_NO_UPDATE']['interval'];t2=old['arms']['T']['interval']
        arms=dict(Frozen=dict(boxes=native,interval=list(c['Native_interval'])),MARGINAL_NO_UPDATE=dict(boxes=native,interval=miv),B3_TEMPORAL=dict(boxes=native,interval=tiv),NEW_TEMPORAL=dict(boxes=native,interval=tiv))
        for label,boxes in [('ORIGINAL',oldboxes),('TEMPORAL_HEAD',oldboxes),('SPATIAL_HEAD',bp),('BOTH_HEADS',bp)]:arms[label]=dict(boxes=boxes,interval=tiv)
        for label,boxes in [('ORIGINAL_S_FIXED_CLOCK',oldboxes),('NEW_S_FIXED_CLOCK',bp)]:arms[label]=dict(boxes=boxes,interval=t2)
        non=[i for i in range(len(c['frame_ids']))if i not in {a['position']for a in anchors}]
        diagnostics=dict(Temporal_parameters=512,Spatial_old_parameters=1792,Spatial_new_parameters=shape['parameters'],shape=shape,anchors=len(anchors),nonanchor_frames=len(non),nonanchor_changed_frames=sum(not np.array_equal(bp[i],native[i])for i in non),zero_native_exact=True,Temporal_reparameterization_exact=True,full_final_decoder_boxes_exact=True,new_fit=nf is not None,accepted=nf['optimizer_steps']if nf else 0,proposed=nf['proposed_optimizer_steps']if nf else 0,Temporal_old_optimizer_steps=tf['optimizer_steps'],Temporal_old_final_loss=tf['path'][-1]['loss'],Temporal_boundary_changed=tiv!=miv)
        if first:qual.append(dict(dataset=rec['dataset'],**diagnostics,actual_nonanchor_influence_qualification=True,GT_read=False))
        assert state_hash(model.state_dict())==model._fixed_full_state_hash
        f=save_pt(BASE/'predictions'/(cid+'.pt'),dict(case_id=cid,dataset=rec['dataset'],arms=arms,fit=nf,diagnostics=diagnostics,support=[a['position']for a in anchors],source_case_sha256=c['source_case_sha256'],reference_sha256=sha(BASE/'R1_FIXED_REFERENCE_LOCK.json'),GT_read=False));f['case_id']=cid;files.append(f)
        write(BASE/'STATUS.json',dict(status='running',stage='head_fit_and_full_readout',done=len(files),total=32,count=32,GT_read=False),mutable=True);print('NATIVE_HEAD',len(files),flush=True)
        del c,base,old,bd,tf,nf,zero,z,full,tv,arms;gc.collect();torch.cuda.empty_cache()
    write(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,records=qual,GT_read=False));write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False));write(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,actual_native_head_reinsertion=True,zero_recovery=True,Temporal_exact_reparameterization=True,full_vs_cached_Spatial_exact=True,GT_read=False))

def score():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    verify();assert read(BASE/'ROOT_READBACK.json')['status']=='pass';bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert len(bar['files'])==bar['count']==32
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    write(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,all_arms_sealed_before_GT=True,GT_read=False))
    rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,bindings=truth._truths_after_seal(cases);off,ind,_=truth._metric_functions();rows=[];diags=[]
    for r,f in zip(rr,bar['files']):
        cid=r['probe_alias'];assert f['case_id']==cid;d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
        for arm,p in d['arms'].items():
            v,_=metric._score_student_arm(p,cases[cid],targets[cid],set(d['support']),off,ind);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5));rows.append(dict(case_id=cid,dataset=r['dataset'],arm=arm,metrics=v))
        diags.append(dict(case_id=cid,dataset=r['dataset'],**d['diagnostics']))
    summary,paired=subset_aggregate(rows);write(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=paired,usage=diags,GT_bindings=bindings,GT_read=True,new_Teacher_calls=0,development_not_confirmation=True,controlled_fixed_evidence_not_fresh_linked_Full=True));write(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True))

if __name__=='__main__':globals()[sys.argv[1]]()
