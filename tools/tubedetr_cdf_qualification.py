"""Two fixed boundary-CDF signals, sealed before independent offline scoring."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup
from stvg.tube_runner import write
from tools.layout_trial.worker import checked_plan,records
from tools.cdf_extent_reward import CDFExtentReward
from tools.mask_reward import MaskAreaReward
PUBLIC=ROOT/'results/tubedetr_cdf';OUT=ROOT/'outputs/tubedetr_cdf';PLAN=PUBLIC/'plan.json'
FOLDERS=dict(base='tubedetr_layout_trial',mask='tubedetr_mask_reward_qualification',transport='tubedetr_correspondence')

def inputs():
    p=json.loads(PLAN.read_text())
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    for folder,h in p['parent_final_seals'].items():
        path=ROOT/folder/'final_seal.json';assert sha(path)==h;s=json.loads(path.read_text());assert s['status']=='complete'
        for name,v in s['files'].items():assert sha(ROOT/name)==v,name
    seals={}
    for alias,name in [('base','diagnostic_seal.json'),('mask','mask_seal.json'),('transport','sealed.json')]:
        path=ROOT/'outputs'/FOLDERS[alias]/name;assert sha(path)==p['reference_seals'][alias];s=json.loads(path.read_text());assert s['status']=='complete' and s['queries']==128;seals[alias]=s
    old=checked_plan(ROOT/'results/tubedetr_layout_trial/plan.json');rows=records(old);assert len(rows)==128
    return p,old,rows,seals

def read(seals,alias,q):
    e=seals[alias]['files'][q];path=ROOT/'outputs'/FOLDERS[alias]/e['path'];assert sha(path)==e['sha256']
    with np.load(path)as z:return {k:z[k].copy()for k in z.files}

def restored_transport(data,teacher):
    points=torch.from_numpy(data['points']);h=len(torch.unique(points[:,1]));w=len(torch.unique(points[:,0]));assert h*w==len(points)
    maps=torch.zeros(8,h*w);degree=torch.zeros(8,1)
    for t in range(7):
        a=torch.from_numpy(data[f'edge_{t}_source']);b=torch.from_numpy(data[f'edge_{t}_target'])
        def inside(p,box):return ((p>=box[:2]-box[2:]/2)&(p<=box[:2]+box[2:]/2)).all(-1)
        maps[t+1,b[inside(points[a],teacher[t])]]+=1;degree[t+1]+=1
        maps[t,a[inside(points[b],teacher[t+1])]]+=1;degree[t]+=1
    return F.avg_pool2d((maps/degree).reshape(8,1,h,w),3,stride=1,padding=1)

def worker():
    p,old,rows,seals=inputs();setup();entries={};error=0.
    for index,row in enumerate(rows):
        q=row['id'];base=read(seals,'base',q);mask=read(seals,'mask',q);trans=read(seals,'transport',q)
        assert np.array_equal(base['observed_frame_ids'],mask['observed_frame_ids'])and np.array_equal(base['observed_frame_ids'],trans['observed_frame_ids'])
        b=torch.from_numpy(base['boxes']);maps=restored_transport(trans,torch.from_numpy(mask['boxes'][:,0]))
        check=MaskAreaReward(maps,torch.ones(8,2),torch.zeros(8,0,dtype=torch.bool)).scores(b)['MaskDice'].numpy()
        error=max(error,float(abs(check-trans['TransportDice']).max()));assert error<=p['transport_reconstruction_atol']
        rewards={};empty={}
        for name,mass,ratio in [('MaskCDF',torch.from_numpy(mask['masks'][:,:1]),torch.from_numpy(mask['valid_ratio'])),('TransportCDF',maps,torch.ones(8,2))]:
            r=CDFExtentReward(mass.double(),ratio.double(),p['tail_probability']);scores=r.scores(b)
            assert scores.shape==b.shape[:2] and b.shape[0]==8 and b.shape[1]>=34 and torch.isfinite(scores).all();rewards[name]=scores.numpy();empty[name]=(~r.nonempty).numpy()
        path=OUT/'rewards'/f'{q}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez(path,**rewards,**{k+'_empty':v for k,v in empty.items()},observed_frame_ids=base['observed_frame_ids'])
        entries[q]=dict(path=str(path.relative_to(OUT)),sha256=sha(path));write(OUT/'worker_progress.json',dict(complete=index+1,total=128,labels_read=False,no_GPU=True));print('CDF qualification',index+1,flush=True)
    write(OUT/'sealed.json',dict(status='complete',queries=128,files=entries,plan_sha256=sha(PLAN),labels_read=False,no_GPU=True,new_expert_forwards=0,student_fits=0,max_transport_dice_reconstruction_error=error))

def score():
    p,old,rows,seals=inputs();seal=json.loads((OUT/'sealed.json').read_text());assert seal['status']=='complete'and seal['queries']==128 and seal['plan_sha256']==sha(PLAN)
    for e in seal['files'].values():assert sha(OUT/e['path'])==e['sha256']
    # Verify all bank hashes before accessing the independent labels.
    for alias in ['base','transport']:
        for e in seals[alias]['files'].values():assert sha(ROOT/'outputs'/FOLDERS[alias]/e['path'])==e['sha256']
    lp=ROOT/old['labels'];assert sha(lp)==old['labels_sha256'];labels={v['id']:v for v in map(json.loads,lp.read_text().splitlines())}
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    from tools.layout_trial.diagnostics import paired_ranking,mean
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    names=['Geometry','Layout3','TransportDice','MaskCDF','TransportCDF'];details=[];empty={n:0 for n in names[-2:]}
    keys=['pair_accuracy','spearman','top1_IoU_regret','GT_rectangle_coverage','box_precision','log_area_ratio','area_smaller_than_GT']
    for row in rows:
        q=row['id'];truth=labels[q];base=read(seals,'base',q);trans=read(seals,'transport',q)
        with np.load(OUT/seal['files'][q]['path'])as z:new={k:z[k].copy()for k in z.files}
        rewards=dict(Geometry=base['geometry_reward'],Layout3=base['layout3_reward'],TransportDice=trans['TransportDice'],MaskCDF=new['MaskCDF'],TransportCDF=new['TransportCDF']);values={n:[]for n in names};teacher=[]
        for n in names[-2:]:empty[n]+=int(new[n+'_empty'].sum())
        for i,fid in enumerate(base['observed_frame_ids']):
            if str(int(fid))not in truth['boxes']:continue
            gt=np.array(truth['boxes'][str(int(fid))]);pred=xyxy(base['boxes'][i],row['metadata']['width'],row['metadata']['height']);iou=box_iou(pred,gt);teacher.append(float(iou[32]-iou[33]));b=pred[:32]
            inter=np.maximum(np.minimum(b[:,2:],gt[2:])-np.maximum(b[:,:2],gt[:2]),0).prod(-1);area=np.maximum(b[:,2:]-b[:,:2],0).prod(-1);gtarea=np.prod(gt[2:]-gt[:2])
            for n in names:
                r=rewards[n][i,:32];tie=np.isclose(r,r.max(),rtol=1e-10,atol=1e-12)
                values[n].append(dict(**paired_ranking(r,iou[:32]),top1_IoU_regret=float(iou[:32].max()-iou[:32][tie].mean()),GT_rectangle_coverage=float((inter/gtarea)[tie].mean()),box_precision=float((inter/area)[tie].mean()),log_area_ratio=float(np.log(area/gtarea)[tie].mean()),area_smaller_than_GT=float((area<gtarea)[tie].mean())))
        details.append(dict(id=q,parent=truth['parent'],eligible_frames=len(teacher),teacher_minus_source=mean(teacher),rewards={n:{k:mean([v[k]for v in values[n]])for k in keys}for n in names}))
    def aggregate(subset):return {n:{k:mean([v['rewards'][n][k]for v in subset])for k in keys}for n in names}
    result=dict(status='complete_signal_qualification',queries=128,eligible_queries=sum(r['eligible_frames']>0 for r in details),rewards=aggregate(details),qualification={},empty_frames=empty,groups={},tail_probability=p['tail_probability'],no_GPU=True,no_adaptation=True,no_parameter_search=True,GT_only_after_complete_seal=True,not_independent_confirmation=True)
    baseline=result['rewards']['Geometry'];parents=[v['parent']for v in details]
    for n in names[-2:]:
        v=result['rewards'][n];passed=v['pair_accuracy']>=baseline['pair_accuracy']and v['top1_IoU_regret']<=baseline['top1_IoU_regret']and(v['pair_accuracy']>baseline['pair_accuracy']or v['top1_IoU_regret']<baseline['top1_IoU_regret'])
        result['qualification'][n]=dict(passed_development_signal_gate=passed,differences={k:cluster_difference([None if r['rewards'][n][k]is None or r['rewards']['Geometry'][k]is None else r['rewards'][n][k]-r['rewards']['Geometry'][k]for r in details],parents)for k in ['pair_accuracy','top1_IoU_regret']})
    for group,fn in dict(teacher_better=lambda r:r['teacher_minus_source']is not None and r['teacher_minus_source']>0,teacher_not_better=lambda r:r['teacher_minus_source']is not None and r['teacher_minus_source']<=0,no_GT_observation=lambda r:r['eligible_frames']==0).items():
        subset=[r for r in details if fn(r)];result['groups'][group]=dict(queries=len(subset),rewards=aggregate(subset))
    prior=json.loads((ROOT/'results/tubedetr_correspondence/summary.json').read_text())
    for n,v in prior['rewards'].items():
        for k in ['pair_accuracy','spearman','top1_IoU_regret']:assert abs(result['rewards'][n][k]-v[k])<1e-12
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result)
    (PUBLIC/'report.md').write_text('# 内容边界CDF奖励资格\n\n同一已曝光Vid128/八帧/32候选，固定tail=.025，零新教师/适应/参数搜索。Direct获得同样连续积分梯度；越界CDF饱和、不额外混几何惩罚。全量奖励封存后独立CPU标签评分。\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n资格通过不等于适应收益、OPD优势或独立确认；教师分组仅诊断，不能GT门控。\n')
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},reward_seal_sha256=sha(OUT/'sealed.json'),publication_pending=True))

def controller():
    inputs()
    for stage in ['worker','score']:
        command=[sys.executable,'-B','-u','tools/tubedetr_cdf_qualification.py','--stage',stage];log=ROOT/'logs'/f'tubedetr_cdf_{stage}.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='');write(OUT/'status.json',dict(status=stage,no_GPU=True,no_adaptation=True))
        with log.open('a')as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);path=ROOT/'runtime/tubedetr_active_jobs.json';r=json.loads(path.read_text());r['jobs']['cdf_worker']=dict(pid=process.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage);write(path,r)
            if process.wait():raise RuntimeError('CDF stage failed: '+stage)
    write(OUT/'status.json',dict(status='complete',queries=128,no_GPU=True,no_adaptation=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['worker','score']);a=p.parse_args()
    try:globals()[a.stage]()if a.stage else controller()
    except Exception as e:
        v=dict(status='failed',error=type(e).__name__,message=str(e));write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
