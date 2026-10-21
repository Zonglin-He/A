"""CPU-only proposal seal -> held teacher feedback seal -> offline GT evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import read,sha,atomic,savept,OLD
V0=ROOT/'outputs/propagation_v0_hc64'
from scripts.score_audit.proposal import retained_teacher,build
from scripts.score_audit.cross_feedback import source_only_affinity,predict_held_support,held_feedback
HERE=Path(__file__).resolve().parent;TR=ROOT/'outputs/score_audit_probe';PUB=ROOT/'results/score_audit'

def lock():
    cfg=read(HERE/'config.json');old=read(V0/'lock.json')
    oldseal=read(V0/'stage1/sealed.json')
    assert oldseal['lock_sha256']==sha(V0/'lock.json') and oldseal['fingerprint']==old['fingerprint']
    assert len(oldseal['files'])==64 and oldseal['GT_read'] is False
    # Hash-only lineage checks: do not numerically load the old all-eight banks or paths.
    for item in old['inputs']:
        e=oldseal['files'][item['id']];p=V0/'stage1'/e['path'];rr=read(p.with_suffix('.json'))
        assert sha(p)==e['sha256']==rr['sha256'] and sha(p.with_suffix('.json'))==e['receipt_sha256']
        for field in ('bank','trace','evidence'):assert sha(V0/'stage1'/rr[field+'_path'])==rr[field+'_sha256']
        for field in ('cache_sha256','native_mask_sha256','reference_sha256'):assert rr[field]==item[field]
        assert sha(OLD/'student_cache'/f"{item['id']}.pt")==item['cache_sha256']
        assert sha(OLD/'research/native_masks'/item['mask_path'])==item['native_mask_sha256']
        assert sha(OLD/item['reference_path'])==item['reference_sha256']
    sources={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'config.json',HERE/'run.py',HERE/'proposal.py',HERE/'cross_feedback.py',HERE/'evaluate.py']}
    sources={**old['sources'],**old['frozen_sources'],**sources}
    for p,h in sources.items():assert sha(ROOT/p)==h,p
    ident=dict(config=cfg,sources=sources,old_lock_sha256=sha(V0/'lock.json'),old_seal_sha256=sha(V0/'stage1/sealed.json'),formal_sha256=sha(ROOT/'outputs/formal/state.json'),default_sha256=sha(ROOT/'configs/default.json'),inputs=old['inputs'],manifest_sha256=old['manifest_sha256'],GT_read=False)
    ident['fingerprint']=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()
    if (TR/'lock.json').exists():assert read(TR/'lock.json')==ident,'Immutable experiment identity changed'
    else:atomic(TR/'lock.json',ident)
    return ident

def validate(stage,L):
    s=read(TR/stage/'sealed.json');assert s['fingerprint']==L['fingerprint'] and s['queries']==64 and s['folds']==512
    assert s['lock_sha256']==sha(TR/'lock.json') and s['GT_read'] is False
    assert set(s['files'])=={x['id'] for x in L['inputs']}
    for k,e in s['files'].items():assert sha(TR/stage/e['path'])==e['sha256']
    for p,h in L['sources'].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/'outputs/formal/state.json')==L['formal_sha256'] and sha(ROOT/'configs/default.json')==L['default_sha256']
    return s

def seal(stage,files,L,begin):
    assert len(files)==64
    atomic(TR/stage/'sealed.json',dict(queries=64,folds=512,fingerprint=L['fingerprint'],lock_sha256=sha(TR/'lock.json'),GT_read=False,files=files,seconds=time.perf_counter()-begin,model_calls=0,expert_calls=0,decoder_calls=0,RGB_reads=0))
    validate(stage,L)

def proposals():
    L=lock();folder=TR/'proposals';assert not(folder/'sealed.json').exists(),'Do not overwrite sealed proposals'
    begin=time.perf_counter();files={};torch.set_num_threads(1)
    for n,item in enumerate(L['inputs']):
        k=item['id'];rr=read(V0/'stage1/predictions'/f'{k}.json')
        ev=torch.load(V0/'stage1'/rr['evidence_path'],map_location='cpu',weights_only=False)
        cp=OLD/'student_cache'/f'{k}.pt';mp=OLD/'research/native_masks'/item['mask_path']
        assert sha(cp)==item['cache_sha256'] and sha(mp)==item['native_mask_sha256']
        cache=torch.load(cp,map_location='cpu',weights_only=False);pack=torch.load(mp,map_location='cpu',weights_only=False)
        assert cache['GT_read'] is pack['GT_read'] is ev['GT_read'] is False
        raw=pack.get('raw_out',pack.get('out'));positions=list(cache['planned_positions'])
        source=torch.as_tensor(cache['Source'],dtype=torch.float64);ids=torch.as_tensor(cache['frame_ids'],dtype=torch.long)
        with np.load(OLD/item['reference_path']) as gold:assert np.array_equal(source.numpy(),gold['Source'])
        F=ev['features'].float();valid=ev['valid'].bool();H,W=ev['seeds'].shape[-2:]
        assert torch.equal(ids,ev['frame_ids']) and len(positions)==8
        span=np.asarray(cache['interval']);event=(ids>=span[0])&(ids<span[1]);pairs={};folds=[]
        for h in range(8):
            keep=[j for j in range(8) if j!=h];pos7=[positions[j] for j in keep]
            # Slice BEFORE query selection, interpolation, support, cycle or proposals.
            l7=raw['pred_logits'][0,keep].clone();b7=raw['pred_boxes'][0,keep].clone();m7=raw['pred_masks'][0,keep].clone()
            q,tb,seeds=retained_teacher(l7,b7,m7,tuple(pack['nested_tensor_shape'][-2:]),(H,W),valid[pos7])
            b=build(source,ids,F,valid,pos7,seeds,tb,event,(H,W),pairs)
            assert not b['observed'][positions[h]] and not (b['parents']==positions[h]).any()
            predicted=torch.zeros(len(ids),3,H,W,dtype=torch.float64)
            for t in range(len(ids)):
                if t==positions[h]:continue
                a=source_only_affinity(F[t],F[positions[h]],valid[t],valid[positions[h]],source_node=t,held_node=int(positions[h]))
                predicted[t]=predict_held_support(b['boxes'][t],a,source_shape=(H,W),held_shape=(H,W))
            b.update(held_index=h,held_node=positions[h],selected_query=q,retained_indices=keep,retained_positions=pos7,predicted_held=predicted)
            folds.append(b)
        dest=folder/f'{k}.pt';savept(dest,dict(folds=folds,source=source,frame_ids=ids,positions8=positions,interval=span,shape=(H,W),fingerprint=L['fingerprint'],GT_read=False))
        files[k]=dict(path=dest.name,sha256=sha(dest));atomic(folder/'progress.json',dict(completed=n+1,total=64,folds=8*(n+1),GT_read=False))
        print(json.dumps(dict(stage='proposals',completed=n+1,seconds=time.perf_counter()-begin)),flush=True)
    seal('proposals',files,L,begin);assert not torch.cuda.is_initialized()

def feedback():
    L=lock();pseal=validate('proposals',L);folder=TR/'feedback';assert not(folder/'sealed.json').exists()
    begin=time.perf_counter();files={};torch.set_num_threads(1)
    for n,item in enumerate(L['inputs']):
        k=item['id'];P=torch.load(TR/'proposals'/pseal['files'][k]['path'],map_location='cpu',weights_only=False)
        mp=OLD/'research/native_masks'/item['mask_path'];assert sha(mp)==item['native_mask_sha256']
        pack=torch.load(mp,map_location='cpu',weights_only=False);raw=pack.get('raw_out',pack.get('out'))
        rr=read(V0/'stage1/predictions'/f'{k}.json');ev=torch.load(V0/'stage1'/rr['evidence_path'],map_location='cpu',weights_only=False)
        folds=[]
        for b in P['folds']:
            h=b['held_index'];t=b['held_node'];q=b['selected_query'];H,W=P['shape']
            # First permitted held-mask access: AFTER ALL 512 proposals are sealed.
            import torch.nn.functional as TF
            full=TF.interpolate(raw['pred_masks'][0,h,q][None,None].float(),size=tuple(pack['nested_tensor_shape'][-2:]),mode='bilinear',align_corners=False).sigmoid()
            target=TF.interpolate(full,size=(H,W),mode='area')[0,0].double()*ev['valid'][t].reshape(H,W)
            cross=torch.zeros(len(P['source']),3,dtype=torch.float64)
            for j in range(len(cross)):
                if j==t:continue
                a=source_only_affinity(ev['features'][j],ev['features'][t],ev['valid'][j],ev['valid'][t],source_node=j,held_node=int(t))
                cross[j]=held_feedback(b['predicted_held'][j],target,a)
            eligible=b['active'].clone();eligible[t]=False
            outs={};acts={}
            for name,s in [('Projection',b['projection']-.1*b['guard']),('CrossAnchor',cross-.1*b['guard'])]:
                s=s.masked_fill(~b['valid'],-torch.inf);a=s.argmax(-1)
                gain=s.gather(1,a[:,None]).squeeze(1)-s[:,0]
                a=torch.where(eligible & (gain>1e-12),a,torch.zeros_like(a))
                acts[name]=a;outs[name]=b['boxes'][torch.arange(len(a)),a]
                assert torch.equal(outs[name][~eligible],P['source'][~eligible])
            folds.append(dict(held_index=h,held_node=t,selected_query=q,actions=acts,outputs=outs,cross=cross,eligible=eligible))
        dest=folder/f'{k}.pt';savept(dest,dict(folds=folds,proposal_sha256=pseal['files'][k]['sha256'],GT_read=False,fingerprint=L['fingerprint']))
        files[k]=dict(path=dest.name,sha256=sha(dest));print(json.dumps(dict(stage='feedback',completed=n+1)),flush=True)
    seal('feedback',files,L,begin);assert validate('proposals',L)==pseal

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['proposals','feedback']);a=p.parse_args()
    proposals() if a.stage=='proposals' else feedback()
