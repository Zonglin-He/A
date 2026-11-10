"""Fusion-before-seed ablation with frozen spatial anchors, no stale query states.
Not raw-object deletion: the frozen PointNet's exterior receptive fields remain.
"""
import torch,numpy as np

def capture(model,points,indices,phrase):
 import sys
 from pathlib import Path
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'crossplatform/round51'))
 from model51 import source_capture
 payload={};hooks=[]
 def enc(_m,_a,k):
  for old,new in [('vis_feats','vis0'),('pos_feats','pos0'),('text_feats','lang0'),('text_padding_mask','textpad')]:payload[new]=k[old].detach().clone().cpu().numpy()
 def backbone(_m,_a,o):payload['xyz']=o['fp2_xyz'].detach().clone().cpu().numpy()
 hooks.extend([model.cross_encoder.register_forward_pre_hook(enc,with_kwargs=True),model.backbone_net.register_forward_hook(backbone)])
 for j,dec in enumerate(model.decoder):
  def pos(_m,a,k,j=j):payload['position'+str(j)]=a[3].detach().clone().cpu().numpy()
  hooks.append(dec.register_forward_pre_hook(pos,with_kwargs=True))
 try:z,receipt=source_capture(model,points,indices,phrase)
 finally:
  for h in hooks:h.remove()
 for k in ['h','query','tokens','token_mask','boxes','scores','query_points_sample_inds','query_anchor_xyz','point_indices']:payload[k]=z[k]
 assert np.array_equal(payload['xyz'][0,payload['query_points_sample_inds']],payload['query_anchor_xyz']);return payload,receipt

def class_language(model,phrase):
 from model51 import firstnoun_mask,SENTINEL
 text=phrase+SENTINEL;device=next(model.parameters()).device;tok=model.tokenizer.batch_encode_plus([text],padding='longest',return_tensors='pt').to(device)
 with torch.no_grad():lang=model.text_projector(model.text_encoder(**tok).last_hidden_state)
 mask,rec=firstnoun_mask(model,text);return {'class_lang0':lang.cpu().numpy(),'class_textpad':tok.attention_mask.ne(1).cpu().numpy(),'class_token_mask':mask.numpy()},rec

def replay(model,z,region_mask=None,category=False,alternate_vis0=None):
 device=next(model.parameters()).device;t=lambda x:torch.as_tensor(x,device=device);V=t(z['vis0'] if alternate_vis0 is None else alternate_vis0);pos=t(z['pos0']);L=t(z['class_lang0'] if category else z['lang0']);tp=t(z['class_textpad'] if category else z['textpad']);ids=t(z['query_points_sample_inds']).long()[None];xyz=t(z['xyz']);mask=torch.zeros(V.shape[:2],device=device,dtype=torch.bool) if region_mask is None else t(np.asarray(region_mask,bool))[None];active=bool(mask.any());qm=mask.gather(1,ids) if active else None;proof={'masked_seed_n':int(mask.sum()),'masked_query_keys_n':int(qm.sum()) if active else 0,'visual_attention_calls':0,'query_attention_calls':0,'max_blocked_attention':0.};hooks=[]
 def inspect(_m,a,k,o):
  expected=qm if _m in [d.self_attn for d in model.decoder] else mask;assert torch.equal(k['key_padding_mask'],expected);w=o[1];bad=w.masked_select(expected[:,None].expand_as(w));proof['max_blocked_attention']=max(proof['max_blocked_attention'],float(bad.abs().max()) if bad.numel() else 0.);proof['query_attention_calls' if expected is qm else 'visual_attention_calls']+=1
 if active:
  for enc in model.cross_encoder.layers:
   for attn in [enc.self_attention_visual.self_attn,enc.cross_layer.cross_lv]:hooks.append(attn.register_forward_hook(inspect,with_kwargs=True))
  for dec in model.decoder:
   def supply(_m,a,k):return a,dict(k,key_padding_mask=mask)
   hooks.append(dec.cross_v.register_forward_pre_hook(supply,with_kwargs=True));hooks.append(dec.cross_v.register_forward_hook(inspect,with_kwargs=True));hooks.append(dec.self_attn.register_forward_hook(inspect,with_kwargs=True))
 try:
  if active:
   for enc in model.cross_encoder.layers:
    V=V.masked_fill(mask[:,:,None],0);V,L=enc(V,pos,mask,L,tp,{},detected_feats=None,detected_mask=None);V=V.masked_fill(mask[:,:,None],0)
  else:V,L=model.cross_encoder(vis_feats=V,pos_feats=pos,padding_mask=mask,text_feats=L,text_padding_mask=tp,end_points={},detected_feats=None,detected_mask=None)
  _,f,_=model.gsample_module(xyz,V.transpose(1,2).contiguous(),ids.int());Q=model.decoder_query_proj(f).transpose(1,2).contiguous()
  # The original post-fusion query is never an input. Masked query initialization
  # contains only the projection's source bias; blocked query keys cannot serve others.
  for j,dec in enumerate(model.decoder):Q=dec(Q,V,L,t(z['position'+str(j)]),qm,tp,detected_feats=None,detected_mask=None)
  tokens=torch.nn.functional.normalize(model.contrastive_align_projection_text(L),dim=-1);queries=torch.nn.functional.normalize(model.contrastive_align_projection_image(Q),dim=-1);m=t(z['class_token_mask'] if category else z['token_mask']).bool()[None,None];logits=queries@tokens.transpose(-1,-2)/.07;s=(torch.logsumexp(logits.masked_fill(~m,-torch.inf),-1)-torch.logsumexp(logits.masked_fill(m,-torch.inf),-1))[0];p=s.double().softmax(-1)
  if active:assert proof['visual_attention_calls']==12 and proof['query_attention_calls']==6 and proof['max_blocked_attention']==0.
  return {'h':Q[0].detach().cpu().numpy(),'query':queries[0].detach().cpu().numpy(),'tokens':tokens[0].detach().cpu().numpy(),'logodds':s.detach().cpu().numpy(),'p':p.detach().cpu().numpy(),'proof':proof}
 finally:
  for hk in hooks:hk.remove()
