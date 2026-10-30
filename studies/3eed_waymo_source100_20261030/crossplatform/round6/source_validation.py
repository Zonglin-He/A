"""Source-Waymo-only eval predictions. Dynamic padding and original span kept."""
from pathlib import Path
import json,hashlib,re,time
import numpy as np
import torch
from factory import P,sha,to_cuda,state_fingerprint
LEXICON=('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman','people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv','jeep','trailer','walker','worker')
HEADS=['last_','proposal_']

def predict_source_val(m,dl,out,epoch,checkpoint_path,contract):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);model=m.module if hasattr(m,'module') else m;before=state_fingerprint(model);model.eval();versions={n:p._version for n,p in model.named_parameters()};buffers={n:b.clone() for n,b in model.named_buffers()};index_path=P.parent/'research/round6/source_data_audit/OFFICIAL_VAL_ANNO_ORDER_PRIVATE.json';index=json.loads(index_path.read_text());count=0;tr=[];start=time.perf_counter()
 with torch.no_grad():
  for bidx,batch in enumerate(dl):
   batch=to_cuda(batch);o=m({'point_clouds':batch['point_clouds'].float(),'text':batch['utterances']});T=o['proj_tokens'].shape[1];assert T<=256;N=len(batch['utterances']);maps=(batch['positive_map'][:,0]>0);enc=model.tokenizer(batch['utterances'],padding='longest',return_offsets_mapping=True,return_tensors='pt');offsets=enc['offset_mapping'].numpy();mask=np.zeros((N,256),bool)
   for i,text in enumerate(batch['utterances']):
    match=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',text,re.I);off=offsets[i];mask[i,:T]=((off[:,1]>match.start())&(off[:,0]<match.end())) if match else (off[:,1]>off[:,0])
   textmask=torch.as_tensor(mask,device='cuda');arrays={k:[] for k in ['boxes_raw','boxes','official_bbf','official_bbs','text_bbf','text_bbs','official_bbf_top10','official_bbs_top10','text_bbf_top10','text_bbs_top10']}
   for h in HEADS:
    raw=torch.cat([o[h+'center'],o[h+'pred_size']],-1);boxes=raw.clone();boxes[:,:,3:6].clamp_(min=1e-6);contrast=((o[h+'proj_queries']@o['proj_tokens'].transpose(-1,-2))/.07).softmax(-1);pad=torch.zeros((N,256,256),device='cuda');pad[:,:,:T]=contrast;soft=o[h+'sem_cls_scores'].softmax(-1);values={'boxes_raw':raw,'boxes':boxes,'official_bbf':(pad*maps[:,None]).sum(-1),'official_bbs':(soft*maps[:,None]).sum(-1),'text_bbf':(pad*textmask[:,None]).sum(-1),'text_bbs':(soft*textmask[:,None]).sum(-1)}
    for key in ['official_bbf','official_bbs','text_bbf','text_bbs']:values[key+'_top10']=torch.argsort(values[key],dim=-1,descending=True)[:,:10]
    for key,v in values.items():arrays[key].append(v.cpu().numpy())
   arrays={k:np.stack(v,axis=1) for k,v in arrays.items()}
   for i in range(N):
    assert Path(batch['meta_path'][i]).as_posix()==index[count]['meta_path'];pc=batch['point_clouds'][i].cpu().contiguous().numpy();pm=batch['positive_map'][i,0].cpu().contiguous().numpy();pointsha=hashlib.sha256(pc.tobytes()).digest();pmsha=hashlib.sha256(pm.tobytes()).digest();assert pmsha.hex()==index[count]['positive_map_sha256'];z={k:v[i] for k,v in arrays.items()};z.update(head_names=np.asarray(HEADS),anno_index=np.array(count),point_sample_sha256=np.frombuffer(pointsha,np.uint8),positive_map_sha256=np.frombuffer(pmsha,np.uint8),T_batch=np.array(T),actual_text_token_length=np.array(int(enc['attention_mask'][i].sum())));assert all(np.isfinite(v).all() for v in z.values() if v.dtype.kind not in 'US');np.savez_compressed(out/(f'{count:06d}.npz'),**z);count+=1
   tr.append({'batch':bidx,'B':N,'T':T,'completed':count});print(json.dumps({'event':'source_val_progress','epoch':epoch,'batch':bidx,'completed':count,'n':len(dl.dataset)}),flush=True) if bidx%25==0 else None
 assert count==len(dl.dataset)==3782;assert versions=={n:p._version for n,p in model.named_parameters()} and all(torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items());assert state_fingerprint(model)==before;(out/'trace.json').write_text(json.dumps(tr,indent=2)+'\n');seal={'n':count,'epoch':epoch,'head_names':HEADS,'checkpoint_sha256':sha(checkpoint_path),'model_state_fingerprint':before,'val_order_sha256':contract['order_sha256'],'private_val_index_sha256':sha(index_path),'source_only':True,'all_inference_no_grad':True,'parameters_buffers_unchanged':True,'dynamic_padding_preserved':True,'actual_batch_size':24,'last_batch':tr[-1]['B'],'no_validation_metric_read_for_primary_selection':True,'seconds':time.perf_counter()-start,'files':{p.name:sha(p) for p in sorted(out.glob('*.npz'))}};(out/'sealed.json').write_text(json.dumps(seal,indent=2)+'\n');model.train();return seal
