"""One frozen cross-frame signal qualification; no student fit or new teacher."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup,lease
from stvg.tube_runner import write
from tools.layout_trial.worker import checked_plan,records,load_feature
from tools.mask_reward import MaskAreaReward
PUBLIC=ROOT/'results/tubedetr_correspondence';OUT=ROOT/'outputs/tubedetr_correspondence';PLAN=PUBLIC/'plan.json';CACHE=ROOT/'outputs/tubedetr_layout_trial'

def inside(points,box):
    return ((points>=box[:2]-box[2:]/2)&(points<=box[:2]+box[2:]/2)).all(-1)

def transport(features,boxes,ratio,max_side):
    """Mutual nearest nodes; seed in the OTHER frame's fixed teacher box.

    Neither correspondence nor teacher seeds depend on candidate boxes. Thus
    continuous area reward gradients are equally available to OPD and Direct.
    No assumption that mutual matches are true identity correspondence.
    """
    assert not features.requires_grad and torch.allclose(ratio,ratio[:1].expand_as(ratio))
    valid_hw=features.new_tensor(features.shape[-2:])*ratio[0].flip(0)
    hw=(valid_hw/valid_hw.max()*max_side).round().clamp_min(1).long().tolist();h,w=hw
    y,x=torch.meshgrid((torch.arange(h,device=features.device)+.5)/h,(torch.arange(w,device=features.device)+.5)/w,indexing='ij')
    points=torch.stack([x,y],-1).reshape(-1,2)
    grid=(points[None]*ratio[:,None]*2-1).reshape(8,h,w,2)
    feat=F.grid_sample(features,grid,align_corners=False).flatten(2).transpose(1,2)
    feat=F.normalize(feat,dim=-1);maps=torch.zeros(8,h*w,device=features.device);degree=torch.zeros(8,1,device=features.device);edges=[]
    for t in range(7):
        sim=feat[t]@feat[t+1].T;forward=sim.argmax(1);reverse=sim.argmax(0);src=torch.arange(len(points),device=features.device)
        valid=reverse[forward]==src;a=src[valid];b=forward[valid];edges.append((a.cpu().numpy(),b.cpu().numpy()))
        maps[t+1,b[inside(points[a],boxes[t])]]+=1;degree[t+1]+=1
        maps[t,a[inside(points[b],boxes[t+1])]]+=1;degree[t]+=1
    maps=(maps/degree).reshape(8,1,h,w)
    maps=F.avg_pool2d(maps,3,stride=1,padding=1)
    return maps,points,edges

def inputs():
    p=json.loads(PLAN.read_text())
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    for folder,h in p['parent_final_seals'].items():
        f=ROOT/folder/'final_seal.json';assert sha(f)==h;s=json.loads(f.read_text());assert s['status']=='complete'
        for name,v in s['files'].items():assert sha(ROOT/name)==v,name
    for name,h in p['reference_seals'].items():assert sha(CACHE/name)==h,name
    old=checked_plan(ROOT/'results/tubedetr_layout_trial/plan.json');rows=records(old);assert len(rows)==128
    return p,old,rows

def worker():
    p,old,rows=inputs();setup();lock=lease();fs=json.loads((CACHE/'features_seal.json').read_text());ds=json.loads((CACHE/'diagnostic_seal.json').read_text());entries={}
    for index,row in enumerate(rows):
        f=load_feature(row,CACHE,fs);e=ds['files'][row['id']];assert sha(CACHE/e['path'])==e['sha256']
        with np.load(CACHE/e['path'])as z:
            candidates=torch.from_numpy(z['boxes'].copy()).cuda();fids=z['observed_frame_ids'].copy()
        assert np.array_equal(fids,f['observed_frame_ids'])
        maps,points,edges=transport(torch.from_numpy(f['features']).cuda(),torch.from_numpy(f['boxes']).cuda(),torch.from_numpy(f['valid_ratio']).cuda(),p['max_grid_side'])
        reward=MaskAreaReward(maps,torch.ones(8,2,device=maps.device),torch.zeros(8,0,device=maps.device,dtype=torch.bool))
        values=dict(TransportDice=reward.scores(candidates)['MaskDice'].cpu().numpy(),observed_frame_ids=fids,points=points.cpu().numpy(),transport_mass=maps.sum((1,2,3)).cpu().numpy())
        for t,(a,b)in enumerate(edges):values[f'edge_{t}_source']=a;values[f'edge_{t}_target']=b
        path=OUT/'rewards'/f"{row['id']}.npz";path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez(path,**values)
        entries[row['id']]=dict(path=str(path.relative_to(OUT)),sha256=sha(path))
        write(OUT/'worker_progress.json',dict(complete=index+1,total=128,labels_read=False));print('Correspondence qualification',index+1,flush=True)
    write(OUT/'sealed.json',dict(status='complete',queries=128,files=entries,plan_sha256=sha(PLAN),labels_read=False,new_teacher_forwards=0,student_fits=0));lock.close()

def score():
    p,old,rows=inputs();seal=json.loads((OUT/'sealed.json').read_text());assert seal['status']=='complete' and seal['queries']==128 and seal['plan_sha256']==sha(PLAN)
    for e in seal['files'].values():assert sha(OUT/e['path'])==e['sha256']
    ds=json.loads((CACHE/'diagnostic_seal.json').read_text())
    for e in ds['files'].values():assert sha(CACHE/e['path'])==e['sha256']
    lp=ROOT/old['labels'];assert sha(lp)==old['labels_sha256'];labels={v['id']:v for v in map(json.loads,lp.read_text().splitlines())}
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    from tools.layout_trial.diagnostics import paired_ranking,mean
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    names=['Geometry','Layout3','TransportDice'];details=[];empty=0;edge_counts=[];frame_gaps=[]
    for row in rows:
        t=labels[row['id']];vals={n:[]for n in names};edge_quality=[]
        with np.load(OUT/seal['files'][row['id']]['path'])as new,np.load(CACHE/ds['files'][row['id']]['path'])as z:
            rewards=dict(Geometry=z['geometry_reward'],Layout3=z['layout3_reward'],TransportDice=new['TransportDice']);empty+=int((new['transport_mass']==0).sum());frame_gaps.extend(np.diff(new['observed_frame_ids']).tolist())
            for i,fid in enumerate(z['observed_frame_ids']):
                if str(int(fid))not in t['boxes']:continue
                iou=box_iou(xyxy(z['boxes'][i],row['metadata']['width'],row['metadata']['height']),t['boxes'][str(int(fid))])
                for name in names:
                    s=rewards[name][i,:32];tie=np.isclose(s,s.max(),rtol=1e-10,atol=1e-12);vals[name].append({**paired_ranking(s,iou[:32]),'top1_IoU_regret':float(iou[:32].max()-iou[:32][tie].mean())})
            points=new['points'];w=row['metadata']['width'];h=row['metadata']['height']
            for i in range(7):
                a=new[f'edge_{i}_source'];b=new[f'edge_{i}_target'];edge_counts.append(len(a));f0,f1=map(lambda x:str(int(x)),new['observed_frame_ids'][i:i+2])
                if f0 not in t['boxes']or f1 not in t['boxes']:continue
                def in_gt(pts,box):
                    # GT is stored xyxy in image coordinates, unlike predicted cxcywh.
                    lo=np.array(box[:2])/[w,h];hi=np.array(box[2:])/[w,h];return ((pts>=lo)&(pts<=hi)).all(-1)
                s=in_gt(points[a],t['boxes'][f0]);d=in_gt(points[b],t['boxes'][f1])
                if s.any():edge_quality.append(float(d[s].mean()))
            details.append(dict(id=row['id'],parent=t['parent'],GT_eligible_observations=len(vals['Geometry']),GT_conditional_correspondence_precision=mean(edge_quality),rewards={n:{k:mean([v[k]for v in vs])for k in ['pair_accuracy','spearman','top1_IoU_regret']}for n,vs in vals.items()}))
    result=dict(status='complete_signal_qualification',queries=128,rewards={n:{k:mean([v['rewards'][n][k]for v in details])for k in ['pair_accuracy','spearman','top1_IoU_regret']}for n in names},empty_transport_frames=empty,mean_mutual_edges_per_pair=float(np.mean(edge_counts)),mean_observation_frame_gap=float(np.mean(frame_gaps)),GT_conditional_correspondence_precision=mean([r['GT_conditional_correspondence_precision']for r in details]),GT_only_after_complete_seal=True,no_adaptation=True,no_new_expert=True,not_independent_confirmation=True)
    ref=result['rewards']['Geometry'];n=result['rewards']['TransportDice'];result['passed_development_signal_gate']=n['pair_accuracy']>=ref['pair_accuracy']and n['top1_IoU_regret']<=ref['top1_IoU_regret']and(n['pair_accuracy']>ref['pair_accuracy']or n['top1_IoU_regret']<ref['top1_IoU_regret'])
    result['differences']={k:cluster_difference([None if r['rewards']['TransportDice'][k]is None or r['rewards']['Geometry'][k]is None else r['rewards']['TransportDice'][k]-r['rewards']['Geometry'][k]for r in details],[r['parent']for r in details])for k in ['pair_accuracy','top1_IoU_regret']}
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result)
    (PUBLIC/'report.md').write_text('# 跨帧对应传递信号资格\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n仅原Vid128/八帧/已有ReferDINO特征；双向互近邻不保证语义身份。连续面积积分可同样给Direct使用。未做学生适应或新全量。\n')
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},response_seal_sha256=sha(OUT/'sealed.json'),publication_pending=True))

def controller():
    inputs()
    for stage in ['worker','score']:
        command=[sys.executable,'-B','-u','tools/tubedetr_correspondence_qualification.py','--stage',stage];log=ROOT/'logs'/f'tubedetr_correspondence_{stage}.log';env=dict(os.environ)
        if stage=='score':env['CUDA_VISIBLE_DEVICES']=''
        write(OUT/'status.json',dict(status=stage,no_adaptation=True))
        with log.open('a')as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);path=ROOT/'runtime/tubedetr_active_jobs.json';r=json.loads(path.read_text());r['jobs']['correspondence_worker']=dict(pid=process.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage);write(path,r)
            if process.wait():raise RuntimeError('Correspondence stage failed: '+stage)
    write(OUT/'status.json',dict(status='complete',queries=128,no_adaptation=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['worker','score']);a=p.parse_args()
    try:globals()[a.stage]()if a.stage else controller()
    except Exception as e:
        v=dict(status='failed',error=type(e).__name__,message=str(e));write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
