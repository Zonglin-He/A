"""Frozen B=1 official weights/heads. GPU sees XYZ+text, never true class/bbox.
All six candidate class maps are predetermined from caption/dictionary, then sealed.
True class selects a score column only in an independent post-seal CPU evaluator.
"""
from pathlib import Path
import argparse,json,hashlib,os,time
import numpy as np
import torch
from common import R,ROUND,P,U,sha,CLASS_NAMES,HEAD_NAMES,SENTINEL,class_masks,input_masks,sample_points
from models.bdetr import BeaUTyDETR

def file_hashes(out):return {p.name:sha(p) for p in sorted(out.glob('*.npz'))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);a=ap.parse_args();assert sha(a.plan)==a.plan_sha256;plan=json.loads(a.plan.read_text());assert plan['GPU_execution_authorized'] is True and plan['batch_size']==1
 assert sha(a.manifest)==plan['manifest_sha256'];assert plan['conditions']==['full','noun']
 for f,h in plan['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in plan['upstream_code_sha256'].items():assert sha(P/f)==h,f
 samples=json.loads(a.manifest.read_text());assert len(samples)==plan['expected_samples'];assert len({s['id'] for s in samples})==len(samples)
 out=a.output;out.mkdir(parents=True,exist_ok=False);(out/'registered_plan.json').write_bytes(a.plan.read_bytes());(out/'manifest_input.json').write_bytes(a.manifest.read_bytes());start_all=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(plan['seed']);os.chdir(U)
 model=BeaUTyDETR(num_class=256,num_obj_class=485,input_feature_dim=3,num_queries=256,num_decoder_layers=6,contrastive_align_loss=True,butd=False,self_attend=True,roberta_config_only=True)
 ck=P/'assets/final_6384/ckpt_6384.pth';assert sha(ck)==plan['checkpoint_sha256']
 with torch.serialization.safe_globals([argparse.Namespace]):state=torch.load(ck,map_location='cpu',weights_only=True)['model']
 state={k.removeprefix('module.'):v for k,v in state.items()};assert len(state)==1034 and all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());model.load_state_dict(state,strict=True);del state;model.requires_grad_(False).cuda().eval();param_versions={n:p._version for n,p in model.named_parameters()};buffers={n:b.clone() for n,b in model.named_buffers()};first=None;trace=[];raw_hash_cache={};point_rows=[];model_times={};bridges=[]
 def infer(pc,caption,platform,bridge=False):
  text=caption+SENTINEL;cm,found,nonzero,spans,mapids=class_masks(model.tokenizer,caption,platform);nm,ids,attn,offset=input_masks(model.tokenizer,text);T=len(ids);assert T<=256
  inp={'point_clouds':torch.from_numpy(pc)[None].cuda(),'text':[text]};cmg=torch.from_numpy(cm).cuda();nmg=torch.from_numpy(nm).cuda()
  with torch.inference_mode():
   torch.cuda.synchronize();model_start=time.perf_counter();o=model(inp);torch.cuda.synchronize();model_times['last']=time.perf_counter()-model_start;allboxes=[];allraw=[];bbf=[];bbs=[];n_bbf=[];n_bbs=[];ord_bbf=[];ord_bbs=[];ord_nb=[];ord_ns=[]
   for h in HEAD_NAMES:
    raw=torch.cat([o[h+'center'],o[h+'pred_size']],-1)[0];boxes=raw.clone();boxes[:,3:6].clamp_(min=1e-6)
    p=((o[h+'proj_queries']@o['proj_tokens'].transpose(-1,-2))/.07).softmax(-1)[0];pad=torch.zeros((256,256),device='cuda',dtype=p.dtype);pad[:,:T]=p
    soft=o[h+'sem_cls_scores'][0].softmax(-1);fb=(pad[None]*cmg[:,None]).sum(-1);sb=(soft[None]*cmg[:,None]).sum(-1);fn=(pad*nmg[None]).sum(-1);sn=(soft*nmg[None]).sum(-1)
    
    if bridge:
     ref_fb=torch.stack([(pad*mask[None]).sum(-1) for mask in cmg]);ref_sb=torch.stack([(soft*mask[None]).sum(-1) for mask in cmg]);assert torch.equal(ref_fb,fb) and torch.equal(ref_sb,sb)
     bridges.append({'head':h,'all6class_BBF_max_abs':float((ref_fb-fb).abs().max()),'all6class_BBS_max_abs':float((ref_sb-sb).abs().max()),'mask_source':'released loader get_positive_map with leading-space format; no actualclass selection'})
    allraw.append(raw.cpu().numpy());allboxes.append(boxes.cpu().numpy());bbf.append(fb.cpu().numpy());bbs.append(sb.cpu().numpy());n_bbf.append(fn.cpu().numpy());n_bbs.append(sn.cpu().numpy());ord_bbf.append(torch.argsort(fb,dim=-1,descending=True)[:,:10].cpu().numpy());ord_bbs.append(torch.argsort(sb,dim=-1,descending=True)[:,:10].cpu().numpy());ord_nb.append(torch.argsort(fn,descending=True)[:10].cpu().numpy());ord_ns.append(torch.argsort(sn,descending=True)[:10].cpu().numpy())
  z={'head_names':np.array(HEAD_NAMES),'class_names':np.array(CLASS_NAMES),'boxes':np.stack(allboxes),'boxes_raw':np.stack(allraw),'bbf_scores_by_class':np.stack(bbf),'bbs_scores_by_class':np.stack(bbs),'firstnoun_bbf':np.stack(n_bbf),'firstnoun_bbs':np.stack(n_bbs),'bbf_native_top10_by_class':np.stack(ord_bbf),'bbs_native_top10_by_class':np.stack(ord_bbs),'firstnoun_bbf_native_top10':np.stack(ord_nb),'firstnoun_bbs_native_top10':np.stack(ord_ns),'class_masks':cm,'class_synonym_found':found,'class_positive_map_nonzero':nonzero,'class_span_chars':spans,'firstnoun_mask256':nm,'input_ids':ids,'attention_mask':attn,'offset_mapping':offset,'class_map_input_ids':mapids}
  assert all(np.isfinite(v).all() for v in z.values() if v.dtype.kind not in 'US');return z
 for j,s in enumerate(samples):
  start=time.perf_counter();fp=Path(s['lidar_path']);raw_hash_cache.setdefault(str(fp),sha(fp));pc,ix,nraw=sample_points(s);ph=hashlib.sha256(np.ascontiguousarray(pc).tobytes()).digest();ih=hashlib.sha256(np.ascontiguousarray(ix).tobytes()).digest();point_rows.append({'id':s['id'],'raw_point_sha256':raw_hash_cache[str(fp)],'raw_N':nraw,'sampling_seed':s['sampling_seed'],'point_sample_sha256':ph.hex(),'raw_indices_sha256':ih.hex(),'n_unique_sample_indices':len(np.unique(ix))})
  for condition,key in [('full','caption'),('noun','noun_caption')]:
   torch.cuda.synchronize();t=time.perf_counter();torch.cuda.reset_peak_memory_stats();z=infer(pc,s[key],s['platform'],bridge=(j==0));torch.cuda.synchronize();seconds=time.perf_counter()-t;z['point_sample_sha256']=np.frombuffer(ph,dtype=np.uint8);z['raw_indices_sha256']=np.frombuffer(ih,dtype=np.uint8);name=s['id']+'_'+condition+'.npz';np.savez_compressed(out/name,**z)
   entry={'id':s['id'],'platform':s['platform'],'scene':s['scene'],'object_index':s['object_index'],'condition':condition,'forward_and_fixed_scores_seconds':seconds,'model_only_seconds':model_times['last'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'tokens':len(z['input_ids']),'nonpositive_raw_size_count_by_head':np.any(z['boxes_raw'][:,:,3:6]<=0,axis=-1).sum(-1).tolist(),'nonfinite_raw_boxes':False};trace.append(entry)
   with open(out/'trace.jsonl','a') as f:f.write(json.dumps(entry)+'\n')
   if first is None:first={k:v.copy() for k,v in z.items()};firstpc=pc.copy()
  trace[-1]['sample_end_to_end_seconds_including_sampling_saving']=time.perf_counter()-start
  if j%25==0 or j+1==len(samples):print(json.dumps({'completed_samples':j+1,'expected_samples':len(samples),'completed_forwards':len(trace),'elapsed_seconds':time.perf_counter()-start_all,'latest_platform':s['platform']}),flush=True)
 repeat=infer(firstpc,samples[0]['caption'],samples[0]['platform']);dif={k:float(np.max(np.abs(v.astype(float)-repeat[k].astype(float)))) for k,v in first.items() if v.dtype.kind not in 'US' and k not in ['point_sample_sha256','raw_indices_sha256']};assert max(dif.values())<=1e-6,dif;assert all(np.array_equal(first[k],repeat[k]) for k in ['head_names','class_names'])
 assert param_versions=={n:p._version for n,p in model.named_parameters()};assert all(torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items());(out/'repeat_control.json').write_text(json.dumps({'max_differences':dif,'bitexact':all(x==0 for x in dif.values()),'firstsample_extra_frozen_forward':1},indent=2)+'\n');(out/'native_readout_bridge.json').write_text(json.dumps({'first_sample_both_conditions_all7heads_all6class':bridges,'all_bitexact':True},indent=2)+'\n');(out/'point_provenance.json').write_text(json.dumps(point_rows,indent=2)+'\n');(out/'trace.json').write_text(json.dumps(trace,indent=2)+'\n');files=file_hashes(out);assert len(files)==2*len(samples)
 seal={'state':'sealed_frozen_baseline_no_labels_selected','n':len(samples),'n_samples':len(samples),'conditions':['full','noun'],'n_prediction_caches':len(files),'head_names':HEAD_NAMES,'class_names':CLASS_NAMES,'plan_sha256':sha(a.plan),'manifest_sha256':sha(a.manifest),'checkpoint_sha256':plan['checkpoint_sha256'],'strict_state_entries':1034,'no_actual_target_class_or_numeric_GT_read':True,'all_six_candidate_class_scores_precomputed_without_GT_class':True,'all_parameters_frozen':True,'parameter_versions_unchanged':True,'buffers_bitexact':True,'point_provenance_sha256':sha(out/'point_provenance.json'),'repeat_control_sha256':sha(out/'repeat_control.json'),'trace_sha256':sha(out/'trace.json'),'native_readout_bridge_sha256':sha(out/'native_readout_bridge.json'),'elapsed_through_cache_and_repeat_seconds':time.perf_counter()-start_all,'files':files};(out/'sealed.json').write_text(json.dumps(seal,indent=2)+'\n');print('BASELINE_SEALED',len(samples),sha(out/'sealed.json'),flush=True)
if __name__=='__main__':main()
