"""Controlled finite computational interventions, not fitted corrections."""
import importlib.util,torch,numpy as np
from common61 import *
def modules(parameters,device):
 spec=importlib.util.spec_from_file_location('native_R61_decoder',ROOT/'crossplatform/round3/p0/upstream/models/encoder_decoder_layers.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 decoder=mod.BiDecoderLayer(288,8,dim_feedforward=256,dropout=.1,self_position_embedding='loc_learned',butd=False)
 head=torch.nn.Sequential(torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,64))
 decoder.load_state_dict({k[len('decoder.5.'):]:torch.tensor(v) for k,v in parameters.items() if k.startswith('decoder.5.')});head.load_state_dict({k[len('contrastive_align_projection_image.'):]:torch.tensor(v) for k,v in parameters.items() if k.startswith('contrastive_align_projection_image.')})
 return decoder.eval().requires_grad_(False).to(device),head.eval().requires_grad_(False).to(device)
def source_fingerprint(*mods):
 h=hashlib.sha256()
 for m in mods:
  for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def decoder_forward(dec,z,v):
 args=[torch.tensor(z[v+'_'+k],device=next(dec.parameters()).device) if k!='prefix_query_mask' else None for k in ['prefix_query','prefix_vis','prefix_lang','prefix_pos','prefix_query_mask','prefix_text_padding']]
 return dec(*args,detected_feats=None,detected_mask=None)
def compose(raw,z,g):
 parts=[];device=next(iter(raw.values())).device
 for v in 'ABC':
  q=torch.nn.functional.normalize(raw[v],dim=-1);t=torch.tensor(z[v+'_tokens'],device=device)[None];mask=torch.tensor(z[v+'_token_mask'],device=device).bool()[None,None];logits=q@t.transpose(-1,-2)/.07;zz=(torch.logsumexp(logits.masked_fill(~mask,-torch.inf),-1)-torch.logsumexp(logits.masked_fill(mask,-torch.inf),-1))[0];prob=zz.double().softmax(-1).cpu().numpy()
  if v!='A':
   prob=g[v+'_W']@prob;mass=float(g[v+'_fixed_mass']);prob=mass*prob/max(float(prob.sum()),1e-300)+(1-mass)*g['base']
  parts.append(prob)
 return sum(parts)/3
def selector_indices(z,head,det,patch):
 result={};n=det['noise_energy']-det['between_candidate_signal'];idx=np.argsort(-n,kind='stable')[:32];result['mlp_noise32_off']=(2,idx)
 result['mlp_noise8_off']=(2,np.flatnonzero(patch['noise_dynamic']));result['mlp_common8_off']=(2,np.flatnonzero(patch['common_dynamic']));result['mlp_random8_off']=(2,np.flatnonzero(patch['noise_random']))
 with torch.no_grad():
  for stage,label in [(0,'h1'),(2,'h2')]:
   acts=[]
   for v in 'ABC':
    x=torch.tensor(z[v+'_h'],device=next(head.parameters()).device)[None];h1=head[1](head[0](x));h2=head[3](head[2](h1));acts.append((h1 if stage==0 else h2)[0])
   weight=head[2 if stage==0 else 4].weight;energy=torch.cat(acts).square().mean(0)*weight.square().sum(0);result['mlp_'+label+'_energy32_off']=(stage,torch.argsort(energy,descending=True,stable=True)[:32].cpu().numpy())
 return result
def run_intervention(dec,head,z,g,arm,selectors):
 handles=[];saved=None;edited=None;module=None
 try:
  if arm in HEAD_ARMS:
   stage,idx=selectors[arm];module=head[2 if stage==0 else 4];saved=module.weight.detach().clone();edited=saved.clone();edited[:,idx]=0;module.weight.copy_(edited)
  elif arm.startswith('dec_vision_head'):
   index=int(arm[len('dec_vision_head')]);module=dec.cross_v.out_proj;saved=module.weight.detach().clone();edited=saved.clone();edited[:,index*36:(index+1)*36]=0;module.weight.copy_(edited)
  elif arm in DECODER_ARMS:
   _,name,gain=arm.split('_');factor={'off':0.,'half':.5,'boost':1.5}[gain];branch={'self':dec.self_attn,'language':dec.cross_l,'vision':dec.cross_v,'ffn':dec.ffn}[name]
   def hook(_m,args,output):return (output[0]*factor,output[1]) if isinstance(output,tuple) else output*factor
   handles=[branch.register_forward_hook(hook)]
  hs={v:torch.tensor(z[v+'_h'],device=next(head.parameters()).device)[None] if arm in HEAD_ARMS else decoder_forward(dec,z,v) for v in 'ABC'};raw={v:head(hs[v]) for v in 'ABC'};scores=compose(raw,z,g);delta=0. if saved is None else float((edited-saved).norm().item()/saved.norm().item())
  return scores,{'max_h_change':max(float((hs[v][0]-torch.tensor(z[v+'_h'],device=hs[v].device)).abs().max()) for v in 'ABC'),'relative_edited_matrix_norm':delta,'actual_parameter_assignment':saved is not None,'residual_gain_hook':arm in DECODER_ARMS}
 finally:
  for h in handles:h.remove()
  if saved is not None:module.weight.copy_(saved)
