"""Finite teacher audit. Pool creation is GT-free; scoring reads isolated labels."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts.ptd_joint_box_opd_v1 import rows,file,checked,put
from scripts.prepare_ptd_joint_box_opd_v1 import OUT as OLD,digest
from vg_tta.ptd_joint_box_opd_v1 import SEEDS,sampled
from scripts.score_ptd_spatial_adapter_ab_v1 import truth,iou,evaluate
OUT=ROOT/'artifacts/ptd_teacher_qualification_v1'

def teacher_folder(teacher):return 'univg_pairs_v2' if teacher=='univg' else teacher
def path(r,folder):return OUT/teacher_folder(folder)/(digest(r['key'])+'.pt')
def point_path(r,teacher,j):return OUT/(teacher_folder(teacher)+'_points')/digest(r['key'])/f'{j}.pt'
def teacher_barrier(teacher,stage):return OUT/f'{teacher_folder(teacher)}_{stage}_BARRIER.json'
def stats(v):
    v=np.asarray(v,float)
    if not len(v):return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(20260925);boot=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return dict(n=len(v),mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),positive=int((v>1e-14).sum()),negative=int((v < -1e-14).sum()),zero=int((abs(v)<=1e-14).sum()),min=float(v.min()),max=float(v.max()))

def verify():
    r=read(OUT/'REGISTRATION.json');pins=dict(r['pins'])
    for p in sorted((OUT/'amendments').glob('*.json')):pins.update(read(p).get('pins',{}))
    for p,h in {**pins,**r['protected'],**r['parents']}.items():assert sha(ROOT/p)==h,p
    return r

def register():
    # No GT is read by registration or pool generation.
    pins=['protocols/ptd_teacher_qualification_v1.md','scripts/ptd_teacher_qualification_v1.py','scripts/download_univg_qualification_v1.py',
          'scripts/ptd_joint_box_opd_v1.py','scripts/score_ptd_spatial_adapter_ab_v1.py','vg_tta/ptd_joint_box_opd_v1.py']
    protected=['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json',
               'artifacts/ptd_joint_box_opd_v1/REGISTRATION.json','artifacts/ptd_joint_box_opd_v1/FINAL_COMPLETION.json']
    parents=['artifacts/ptd_joint_box_opd_v1/INPUTS.json','artifacts/ptd_joint_box_opd_v1/SELECTION.json']
    write(OUT/'REGISTRATION.json',dict(time=time.time(),pins={p:sha(ROOT/p) for p in pins},protected={p:sha(ROOT/p) for p in protected},parents={p:sha(ROOT/p) for p in parents},
        request='./private_authorization_notes/authorization.txt',sources=256,development=64,qualification=192,HC_train=78,
        GPU_seconds=21600,engineering_seconds=1800,output_bytes=8*2**30,disk_floor=25*2**30,download_seconds=14400,
        gamma='exact conditional coordinate for D1; split independent MC groups for joint',copy_shuffle_steps=3,absorption_sources=8,
        GT_training='only explicitly isolated optional plasticity diagnostic; no deployed state',historically_exposed=True,new_unseen_confirmation=False))

def pools():
    verify();files={};count=0
    for r in rows():
        dest=path(r,'pools')
        if dest.exists():checked(dest);files[str(dest)]=sha(dest);continue
        z=checked(file(r,'native'));parents={str(file(r,'native')):sha(file(r,'native'))}
        if not z['format_ok']:
            put(dest,dict(key=r['key'],format_ok=False,positions=[],fallback='native',parents=parents,GT_read=False));files[str(dest)]=sha(dest);continue
        base=z['base_tokens'].long();assert torch.equal(base,z['logits'].argmax(-1))
        lp=z['logits'].log_softmax(-1);samples=torch.cat([sampled(lp,r['key'],s,0) for s in SEEDS],1)
        trajectories=[]
        for s in SEEDS:
            fitted=checked(file(r,'fits','D1',s));parents[str(file(r,'fits','D1',s))]=sha(file(r,'fits','D1',s))
            trajectories.extend([h['tokens'].long() for h in fitted['history'][1:]])
        real=torch.cat([base[:,None],samples,torch.stack(trajectories,1)],1)
        names=['native']+[f'p0_seed{s}_sample{k}' for s in SEEDS for k in range(4)]+[f'D1_seed{s}_step{k}' for s in SEEDS for k in [1,2,3]]
        control=[base];cnames=['native']
        for c in range(4):
            for sign in [-1,1]:
                v=base.clone();v[:,c]+=10*sign;control.append(v.clamp(0,1000));cnames.append(f'edge{c}_{10*sign}')
        for dim in [0,1]:
            for sign in [-1,1]:
                v=base.clone();v[:,[dim,dim+2]]+=10*sign;control.append(v.clamp(0,1000));cnames.append(f'translate{dim}_{10*sign}')
        for scale in [.9,1.1]:
            b=base.double();center=(b[:,:2]+b[:,2:])/2;half=(b[:,2:]-b[:,:2])/2*scale
            control.append(torch.cat([center-half,center+half],-1).round().clamp(0,1000).long());cnames.append(f'scale{scale}')
        if r['split']=='development':
            for si,s in enumerate(SEEDS):
                for j in range(len(base)):
                    p=OLD/'joint_points'/digest(r['key'])/f'{s}_0_{j}.pt';v=checked(p)
                    assert torch.equal(v['actions'],samples[j,si*4:si*4+4]);parents[str(p)]=sha(p)
        put(dest,dict(key=r['key'],format_ok=True,positions=z['positions'],real=real,control=torch.stack(control,1),samples=samples,real_names=names,control_names=cnames,
            GT_read=False,parents=parents,qualification_samples_new=r['split']=='confirmation',controlled_geometry_not_repaired=True))
        files[str(dest)]=sha(dest);count+=1
        print('POOL',r['key'],len(base),flush=True)
    write(OUT/'POOL_BARRIER.json',dict(time=time.time(),files=files,sources=256,GT_read=False))

def geometry(tokens,gt):
    xy=np.asarray(tokens,dtype=np.float64)/1000;valid=(xy[...,2:]>xy[...,:2]).all(-1)
    b=np.concatenate([(xy[...,:2]+xy[...,2:])/2,xy[...,2:]-xy[...,:2]],-1);b=np.where(valid[...,None],b,0.)
    return iou(b,gt)

def support(r,z,g):
    ids=np.asarray(r['input']['frame_ids']);pos=np.asarray(z.get('positions',[]),int)
    if z.get('interval') is None:return pos,np.zeros(len(pos),bool),1
    s,e=z['interval'];a,b=ids[s],ids[e]+1;c,d=g['interval'];inside=(ids>=a)&(ids<b)
    return pos,g['valid'][pos]&inside[pos],max(1,int(((ids>=min(a,c))&(ids<max(b,d))).sum()))

def utility(r,z,g):
    pos,valid,denom=support(r,z,g);base=z['base_tokens'].numpy();native=geometry(base,g['boxes'][pos])
    allboxes=np.repeat(base[:,None,None,:],4,axis=1);allboxes=np.repeat(allboxes,1001,axis=2)
    for c in range(4):allboxes[:,c,:,c]=np.arange(1001)
    values=geometry(allboxes,g['boxes'][pos,None,None]);u=(values-native[:,None,None])/denom*valid[:,None,None]
    # u is R(intervention)-R(native); the dropped constant cancels both advantages and gradients.
    return torch.from_numpy(u),values,native,valid,denom

def coordinate_math(logits,logq,u):
    lp=logits.double().log_softmax(-1);p=lp.exp();q=logq.double();r=q-lp
    ag=u-(p*u).sum(-1,keepdim=True);ar=r-(p*r).sum(-1,keepdim=True)
    g=p*ag;d=p*ar;gamma=(g*d).sum(-1);cov=(p*ag*ar).sum(-1)
    mass=p*ag.abs();agree=(mass*((ag*ar)>0)).sum(-1);total=mass.sum(-1)
    return dict(lp=lp,p=p,AGT=ag,AOPD=ar,g=g,d=d,gamma=gamma,cov=cov,agreement_numerator=agree,agreement_denominator=total,KL=(p*(lp-q)).sum(-1))

def joint_math(lp,actions,logq,reward):
    # Full joint objective, not independent teacher marginals. Unknown logZ is a baseline.
    p=lp.double().exp();act=actions.long();sel=lp.double()[:,None].expand(-1,12,-1,-1).gather(-1,act[...,None]).squeeze(-1).sum(-1)
    score=torch.nn.functional.one_hot(act,1001).double()-p[:,None]
    cost=sel-logq.double();est=[]
    for a,b in [(0,6),(6,12)]:
        r=reward[:,a:b].double();c=cost[:,a:b];s=score[:,a:b]
        g=((r-r.mean(1,keepdim=True))[:,:,None,None]*s).sum(1)/5
        d=-((c-c.mean(1,keepdim=True))[:,:,None,None]*s).sum(1)/5;est.append((g,d))
    gamma=((est[0][0]*est[1][1]).sum((1,2))+(est[1][0]*est[0][1]).sum((1,2)))/2
    return gamma

def pool_result(pool,scores,r,z,g):
    pos,valid,denom=support(r,z,g);values=geometry(pool.numpy(),g['boxes'][pos,None]);ix=scores.argmax(-1).numpy()
    native=values[:,0];picked=values[np.arange(len(ix)),ix];oracle=values.max(1);dv=(picked-native)*valid/denom;op=(oracle-native)*valid/denom
    base=evaluate(z,z['base_tokens'],r,g);selected=evaluate(z,pool[torch.arange(len(ix)),torch.as_tensor(ix)],r,g)
    assert abs((selected['v']-base['v'])-dv.sum())<1e-6
    good=valid&(native>=.5);bad=valid&(native<.5)
    return dict(dv=float(dv.sum()),oracle=float(op.sum()),regret=float((op-dv).sum()),native_v=base['v'],selected_v=selected['v'],
        ds=float((picked[valid]-native[valid]).mean()) if valid.any() else None,support=int(valid.sum()),positions=pos.tolist(),indices=ix.tolist(),
        native_iou=native.tolist(),selected_iou=picked.tolist(),oracle_iou=oracle.tolist(),valid=valid.tolist(),
        native_good=int(good.sum()),native_good_retained=int((good&(picked>=.5)).sum()),native_good_harmed=int((good&(picked<native-1e-10)).sum()),
        native_good_harm_gt5pp=int((good&(picked<native-.05)).sum()),native_bad=int(bad.sum()),native_bad_rescued=int((bad&(picked>=.5)).sum()),
        native_bad_delta_sum=float((picked[bad]-native[bad]).sum()),all_frame_loss_gt5pp=int((valid&(picked<native-.05)).sum()))

def summary(records):
    out={}
    for name,rs in [('all',records),('HC',[r for r in records if r['domain']=='HC']),('Vid',[r for r in records if r['domain']=='Vid']),
                    ('HC_test',[r for r in records if r['cohort']=='hcstvg1_test']),('HC_train',[r for r in records if r['cohort']=='hcstvg1_train'])]:
        if not rs:continue
        block=dict(sources=len(rs),invalid_native=sum(not r['format_ok'] for r in rs),pools={})
        for pool in ['real','control']:
            ps=[r[pool] for r in rs];block['pools'][pool]=dict(dv=stats([p['dv'] for p in ps]),oracle=stats([p['oracle'] for p in ps]),regret=stats([p['regret'] for p in ps]),
                ds=stats([p['ds'] for p in ps if p['ds'] is not None]),support=sum(p['support'] for p in ps),native_good=sum(p['native_good'] for p in ps),
                native_good_retained=sum(p['native_good_retained'] for p in ps),native_good_harmed=sum(p['native_good_harmed'] for p in ps),
                native_good_harm_gt5pp=sum(p['native_good_harm_gt5pp'] for p in ps),native_bad=sum(p['native_bad'] for p in ps),native_bad_rescued=sum(p['native_bad_rescued'] for p in ps),
                native_bad_delta_sum=sum(p['native_bad_delta_sum'] for p in ps),source_loss_gt5pp=sum(p['dv']<-.05 for p in ps))
        block['gamma']=stats([r.get('gamma',0.) for r in rs]);block['joint_gamma']=stats([r['joint_gamma'] for r in rs if r.get('joint_gamma') is not None])
        for k in ['legal_coordinates','harmful_coordinates','large_KL_coordinates','large_KL_harmful','teacher_good_coordinates','teacher_good_harmful','teacher_good_tube_coordinates','teacher_good_tube_harmful','advantage_agree_weight','advantage_total_weight','important_agree_weight','important_total_weight']:
            block[k]=sum(r.get(k,0) for r in rs)
        out[name]=block
    return out

def empty_pool():return dict(dv=0.,oracle=0.,regret=0.,ds=None,support=0,native_good=0,native_good_retained=0,native_good_harmed=0,native_good_harm_gt5pp=0,native_bad=0,native_bad_rescued=0,native_bad_delta_sum=0.)

def audit(stage):
    verify();bar=read(OUT/'POOL_BARRIER.json');assert len(bar['files'])==256
    labels=read(OLD/'LABELS_SCORER_ONLY.json');records=[];start=time.monotonic();numerical=[]
    for r in rows(stage):
        z=checked(file(r,'native'));pool=checked(path(r,'pools'));rec={k:r[k] for k in ['key','source','cohort','domain','split']};rec['format_ok']=z['format_ok']
        if not z['format_ok']:
            rec.update(real=empty_pool(),control=empty_pool(),gamma=0.,joint_gamma=None,reason='native_format_invalid');records.append(rec);continue
        q=checked(file(r,'teacher'))['logp'];g=truth(r,labels);u,values,native,valid,denom=utility(r,z,g);m=coordinate_math(z['logits'],q,u)
        for kind in ['real','control']:
            actions=pool[kind];scores=q[:,None].expand(-1,actions.shape[1],-1,-1).gather(-1,actions[...,None]).squeeze(-1).sum(-1)
            rec[kind]=pool_result(actions,scores,r,z,g)
        gamma=m['gamma'].numpy();kl=m['KL'].numpy();legal=np.repeat(valid[:,None],4,axis=1);harm=gamma < -1e-14;large=kl>=np.quantile(kl,.75)
        teacher_good=valid&(np.asarray(rec['real']['selected_iou'])>=.5);goodmask=np.repeat(teacher_good[:,None],4,axis=1)
        tube=rec['real']['selected_v']>=.5;important=torch.from_numpy(abs(values-native[:,None,None])>=.05)
        mass=m['p']*m['AGT'].abs();agree=(m['AGT']*m['AOPD'])>0
        rec.update(gamma=float(gamma.sum()/gamma.size),legal_coordinates=int(legal.sum()),harmful_coordinates=int((legal&harm).sum()),
            large_KL_coordinates=int((legal&large).sum()),large_KL_harmful=int((legal&large&harm).sum()),
            teacher_good_coordinates=int(goodmask.sum()),teacher_good_harmful=int((goodmask&harm).sum()),
            teacher_good_tube_coordinates=int(legal.sum()) if tube else 0,teacher_good_tube_harmful=int((legal&harm).sum()) if tube else 0,
            advantage_agree_weight=float((mass*agree).sum()),advantage_total_weight=float(mass.sum()),
            important_agree_weight=float((mass*agree*important).sum()),important_total_weight=float((mass*important).sum()),
            coordinate_gamma=gamma.tolist(),coordinate_KL=kl.tolist(),coordinate_covariance=m['cov'].tolist(),joint_gamma=None)
        if r['split']=='development':
            jq=[]
            for j in range(len(native)):
                jq.append(torch.cat([checked(OLD/'joint_points'/digest(r['key'])/f'{s}_0_{j}.pt')['raw'] for s in SEEDS]))
            rew=torch.from_numpy((geometry(pool['samples'].numpy(),g['boxes'][np.asarray(z['positions']),None])-native[:,None])*valid[:,None]/denom)
            jg=joint_math(m['lp'],pool['samples'],torch.stack(jq),rew);rec['joint_gamma']=float(jg.mean()/4);rec['joint_frame_gamma']=jg.tolist()
        if len(numerical)<8:
            # Independent autograd and finite differences at a deterministic legal coordinate.
            inds=np.argwhere(legal)
            if len(inds):
                j,c=inds[0];l=z['logits'][j,c].double().clone().requires_grad_();lp=l.log_softmax(-1)
                J=(lp.exp()*u[j,c]).sum();K=(lp.exp()*(lp-q[j,c].double())).sum();gg,=torch.autograd.grad(J,l,retain_graph=True);kk,=torch.autograd.grad(K,l)
                err=max(float((gg-m['g'][j,c]).abs().max()),float((-kk-m['d'][j,c]).abs().max()));assert err<1e-12
                eps=1e-4;d=m['d'][j,c];fd=float((((l.detach()+eps*d).softmax(-1)*u[j,c]).sum()-((l.detach()-eps*d).softmax(-1)*u[j,c]).sum())/(2*eps))
                ge=float(m['gamma'][j,c]);assert abs(fd-ge)<1e-9
                value=731;t=z['base_tokens'].clone();t[j,c]=value
                exact=evaluate(z,t,r,g)['v']-evaluate(z,z['base_tokens'],r,g)['v'];taskerr=abs(float(u[j,c,value])-exact);assert taskerr<1e-6
                numerical.append(dict(key=r['key'],coordinate=[int(j),int(c)],gradient_error=err,finite_difference_error=abs(fd-ge),task_intervention_error=taskerr))
        dest=path(r,'ordinary_marginal_math')
        if not dest.exists():put(dest,dict(key=r['key'],u_delta=u,gamma=m['gamma'],g=m['g'],d=m['d'],AGT=m['AGT'],AOPD=m['AOPD'],GT_offline_only=True))
        records.append(rec);print('AUDIT',r['key'],round(rec['real']['dv']*100,4),f'gamma={rec["gamma"]:.3g}',flush=True)
    result=dict(time=time.time(),stage=stage,teacher='ordinary Qwen3-VL-8B exact D1 marginals',summary=summary(records),records=records,
        numerical_checks=numerical,CPU_wall_seconds=time.monotonic()-start,new_teacher_forwards=0,GT_predictor=False,pool_barrier_sha=sha(OUT/'POOL_BARRIER.json'))
    write(OUT/f'ORDINARY_{stage}.json',result);status(OUT/'STATUS.json',dict(state='ordinary_cache_audit_completed',stage=stage,time=time.time()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['register','pools','audit']);ap.add_argument('--stage',default='development');a=ap.parse_args();torch.set_num_threads(4)
    if a.action=='register':register()
    elif a.action=='pools':pools()
    else:audit(a.stage)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(OUT/'failures'/f'{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
