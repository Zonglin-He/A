"""Freeze source-only trained choices, predict all400targets, then CPU-score."""
import sys,time,numpy as np,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round25'),str(ROOT/'research/round5/audit')]
from common25 import *
from fusion25 import bank,part
from model25 import make_model,state_path
from independent_core_v2 import candidate_ious,transform_gt

def predict():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;spec=read(EXP/'FINAL_SOURCE_SELECTED_LOCK.json');assert spec['selection_source']=='WaymoVAL512 only';out=BASE/'target400_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();rr=rows('T400');data=bank(rr);scores={'source':data['scores'].numpy()};timings={};start=time.perf_counter()
 for c in spec['selected']:
  m=make_model(c['family']).cuda().eval();sp=state_path(c);assert sha(sp)==c['state_sha256'];m.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in load_npz(sp).items()});m.requires_grad_(False);base=c['variant'];seed=c['seed'];modes=[base] if base=='no_rgb' else ['rgb','zero_rgb','permute_rgb','global_rgb']
  for mode in modes:
   name=(f'noRGB_source_train_seed{seed}' if base=='no_rgb' else f'fusion_{mode}_seed{seed}');ss=[];gg=[];t=time.perf_counter()
   with torch.no_grad():
    for i in range(len(rr)):
     b=part(data,slice(i,i+1));L=int((~b['padding'][0]).sum());b['tokens']=b['tokens'][:,:L];b['mask']=b['mask'][:,:L];b['padding']=b['padding'][:,:L];s,d=m(b,mode=mode);ss.append(s.cpu().numpy());gg.append(d['gate'].cpu().numpy())
   scores[name]=np.concatenate(ss);timings[name]={'head_seconds':time.perf_counter()-t,'mean_gate':float(np.concatenate(gg).mean()),'seed':seed,'source_checkpoint_name':c['state_name']};print('PREDICT25',name,flush=True)
  del m
 # A stronger independently source-selected noRGB family is retained if different.
 if spec['stronger_noRGB_family']!=spec['family']:
  for c in spec['stronger_noRGB_configs']:
   m=make_model(c['family']).cuda().eval();m.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in load_npz(state_path(c)).items()});m.requires_grad_(False);ss=[]
   with torch.no_grad():
    for i in range(len(rr)):
     b=part(data,slice(i,i+1));L=int((~b['padding'][0]).sum());b['tokens']=b['tokens'][:,:L];b['mask']=b['mask'][:,:L];b['padding']=b['padding'][:,:L];v,_=m(b,mode='no_rgb');ss.append(v.cpu().numpy())
   scores['strong_noRGB_seed'+str(c['seed'])]=np.concatenate(ss)
 names=list(scores)
 for i,r in enumerate(rr):
  z=feature(r);ix=np.array([scores[n][i].argmax() for n in names]);np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(names),query_indices=ix,boxes=z['boxes'][ix],scores=np.stack([scores[n][i] for n in names]))
 dump(out/'HEAD_COSTS.json',timings);seal(out,'R25_all400_frozen_trimodal_predictions_before_target_GT',n=400,source_selected_sha256=sha(EXP/'FINAL_SOURCE_SELECTED_LOCK.json'),target_optimizer_steps=0,no_Qwen=True,boxes_from_unchanged_source_pool=True,seconds=time.perf_counter()-start);dump(EXP/'TARGET_PRELABEL_LOCK.json',{'prediction_seal_sha256':sha(out/'SEALED.json'),'source_selection_sha256':sha(EXP/'FINAL_SOURCE_SELECTED_LOCK.json'),'target_labels_not_used_to_select':True,'N64_new_content_historical_fullVAL_exposure':True});print('TARGET25_PREDICTIONS_SEALED',flush=True)

def score():
 out=BASE/'target400_attempt001';verify(out);index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};records=[]
 for r in rows('T400'):
  ix=index[r['id']];path=ix['meta_path'];assert sha(path)==ix['metadata_container_sha256']
  if path not in containers:containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));z=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));names=z['arm_names'].tolist();q=z['query_indices'];b=z['boxes'].copy();assert np.array_equal(b,feature(r)['boxes'][q]);b[:,3:]=np.maximum(b[:,3:],1e-6);v=candidate_ious(b,gt)[0];records.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'stage':r['stage'],'q':dict(zip(names,map(int,q))),'ious':dict(zip(names,map(float,v)))})
 dump(EXP/'T400_ROWS_PRIVATE.json',records)
 for phase,rr in [('ALL400',records),('OLD336',[r for r in records if r['stage']!='N']),('NEW64',[r for r in records if r['stage']=='N'])]:
  summary={}
  for p in ['drone','quad']:
   r=[r for r in rr if r['platform']==p];summary[p]={}
   for n in r[0]['q']:
    v=np.array([x['ious'][n] for x in r]);d=v-np.array([x['ious']['source'] for x in r]);summary[p][n]={'n':len(r),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gains_vs_source':int((d>1e-10).sum()),'harms_vs_source':int((d< -1e-10).sum()),'query_changes':sum(x['q'][n]!=x['q']['source'] for x in r),'scene_clusters':len({x['scene_PRIVATE'] for x in r})}
  dump(EXP/(phase+'_SUMMARY.json'),summary);print(phase,{p:{n:round(v['mIoU_percent'],4) for n,v in s.items()} for p,s in summary.items()},flush=True)
if __name__=='__main__':predict() if sys.argv[1]=='predict' else score()
