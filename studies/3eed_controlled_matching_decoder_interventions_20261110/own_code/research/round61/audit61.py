"""Independent functional last-decoder and head reconstruction, no task labels.

No intervention producer imported. Residual changes are evaluated explicitly;
individual attention heads use functional multi-head attention with source
out-projection columns removed. Independent score composition preserves R57.
"""
import argparse,importlib.util,time,torch
import torch.nn.functional as F
from common61 import *

def create(parameters):
 spec=importlib.util.spec_from_file_location('audit_native_definition61',ROOT/'crossplatform/round3/p0/upstream/models/encoder_decoder_layers.py');lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib);d=lib.BiDecoderLayer(288,8,256,.1,self_position_embedding='loc_learned',butd=False).cuda().eval().requires_grad_(False)
 sd={k[10:]:torch.tensor(v,device='cuda') for k,v in parameters.items() if k.startswith('decoder.5.')};d.load_state_dict(sd);head={k[len('contrastive_align_projection_image.'):]:torch.tensor(v,device='cuda') for k,v in parameters.items() if k.startswith('contrastive_align_projection_image.')};return d,head
def attention(m,q,k,v,padding=None,head_off=None):
 outW=m.out_proj.weight
 if head_off is not None:
  outW=outW.clone();outW[:,head_off*36:(head_off+1)*36]=0
 return F.multi_head_attention_forward(q,k,v,288,8,m.in_proj_weight,m.in_proj_bias,None,None,False,.1,outW,m.out_proj.bias,training=False,key_padding_mask=padding,need_weights=True,attn_mask=None)[0]
def decoder(d,z,v,arm):
 to=lambda k:torch.tensor(z[v+'_'+k],device='cuda');q=to('prefix_query').transpose(0,1);pos=d.self_posembed(to('prefix_pos')).transpose(1,2).contiguous().transpose(0,1);gains=dict(self=1.,language=1.,vision=1.,ffn=1.)
 if arm in DECODER_ARMS:
  _,k,value=arm.split('_');gains[k]={'off':0.,'half':.5,'boost':1.5}[value]
 h=attention(d.self_attn,q+pos,q+pos,q);q=d.norm1(q+gains['self']*h)
 l=to('prefix_lang').transpose(0,1);h=attention(d.cross_l,q+pos,l,l,to('prefix_text_padding'));q=d.norm_l(q+gains['language']*h)
 x=to('prefix_vis').transpose(0,1);off=int(arm[len('dec_vision_head')]) if arm in HEAD_OFF_ARMS else None;h=attention(d.cross_v,q+pos,x,x,head_off=off);q=d.norm_v(q+gains['vision']*h);q=d.norm2(q+gains['ffn']*d.ffn(q));return q.transpose(0,1).contiguous()
def head_forward(h,H,stage=None,indices=None):
 weights={k:H[k] for k in H}
 if stage is not None:
  layer=2 if stage==0 else 4;key=str(layer)+'.weight';weights[key]=weights[key].clone();weights[key][:,indices]=0
 for k in [0,2,4]:
  h=F.linear(h,weights[str(k)+'.weight'],weights[str(k)+'.bias'])
  if k<4:h=F.relu(h)
 return h
def score(raw,z,g):
 prob=[]
 for v in 'ABC':
  t=torch.tensor(z[v+'_tokens'],device='cuda')[None];mask=torch.tensor(z[v+'_token_mask'],device='cuda').bool()[None,None];logits=F.normalize(raw[v],dim=-1).matmul(t.transpose(-1,-2))/.07;s=(torch.logsumexp(logits.masked_fill(~mask,-torch.inf),-1)-torch.logsumexp(logits.masked_fill(mask,-torch.inf),-1))[0].double().softmax(-1).cpu().numpy()
  if v!='A':
   s=g[v+'_W'].dot(s);m=float(g[v+'_fixed_mass']);s=m*s/max(s.sum(),1e-300)+(1-m)*g['base']
  prob.append(s)
 return (prob[0]+prob[1]+prob[2])/3
def selector_check(z,H,det,patch):
 out={'mlp_noise8_off':(2,np.flatnonzero(patch['noise_dynamic'])),'mlp_noise32_off':(2,np.argsort(-(det['noise_energy']-det['between_candidate_signal']),kind='stable')[:32]),'mlp_common8_off':(2,np.flatnonzero(patch['common_dynamic'])),'mlp_random8_off':(2,np.flatnonzero(patch['noise_random']))}
 for layer,name in [(0,'h1'),(2,'h2')]:
  hh=[]
  for v in 'ABC':
   h=torch.tensor(z[v+'_h'],device='cuda')[None];first=F.relu(F.linear(h,H['0.weight'],H['0.bias']));second=F.relu(F.linear(first,H['2.weight'],H['2.bias']));hh.append((first if layer==0 else second)[0])
  w=H['2.weight' if layer==0 else '4.weight'];energy=torch.cat(hh).square().mean(0)*w.square().sum(0);out['mlp_'+name+'_energy32_off']=(layer,torch.argsort(energy,descending=True,stable=True)[:32].cpu().numpy())
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 cap=ROOT/p['capture_root'];pr=ROOT/p['prediction_root'];cs=read(cap/'SEALED.json');ps=read(pr/'SEALED.json');assert cs['n']==ps['n']==336 and ps['arm_names']==ARMS;parameters=load(cap/'DECODER_AND_MLP_PARAMETERS_PRIVATE.npz')
 sourceplan=read(ROOT/p['source_plan']);assert sha(sourceplan['source_checkpoint_path'])==sourceplan['source_checkpoint_sha256'];checkpoint=torch.load(sourceplan['source_checkpoint_path'],map_location='cpu',weights_only=True);expected={k:v.numpy() for k,v in checkpoint['model'].items() if k.startswith(('decoder.5.','contrastive_align_projection_image.'))};assert set(expected)==set(parameters) and all(np.array_equal(parameters[k],expected[k]) for k in expected);del expected,checkpoint
 d,H=create(parameters);rows=read(ROOT/p['manifest']);start=time.perf_counter();maxerr=0.;maxh=0.;actual=0
 for i,row in enumerate(rows):
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel] and sha(pr/rel)==ps['files'][rel];z=load(cap/rel);pred=load(pr/rel);groot=ROOT/p['association_root'];assert sha(groot/rel)==read(groot/'SEALED.json')['files'][rel];g=load(groot/rel);oldroot=ROOT/p['r59_prediction_root'];assert sha(oldroot/rel)==read(oldroot/'SEALED.json')['files'][rel];old=load(oldroot/rel);old60root=ROOT/p['r60_prediction_root'];old60=load(old60root/rel);patch=load(old60root/'patches_PRIVATE'/(sid+'.npz'));det=load(ROOT/p['r60_detection_root']/rel);selected=load(pr/'selectors_PRIVATE'/(sid+'.npz'));selectors=selector_check(z,H,det,patch)
  ab=load(ROOT/p['AB_root']/rel);c=load(ROOT/p['C_root']/rel)
  for v,original,pre in [('A',ab,'A_'),('B',ab,'B_'),('C',c,'')]:
   for k in ['h','tokens','token_mask']:assert np.array_equal(z[v+'_'+k],original[pre+k]),('original_view_state',v,k)
  for arm,(layer,indices) in selectors.items():assert np.array_equal(indices,selected[arm+'_indices'])
  assert np.array_equal(pred['source_boxes'],old['source_boxes']) and np.array_equal(pred['source_scores'],old['source_scores']);assert np.array_equal(pred['selected_original_boxes'],old['source_boxes'][pred['queries']])
  for j,arm in enumerate(REFERENCES):
   ref=old60 if arm.startswith('r60_') else old;name={'r60_noise':'noise_dynamic','r60_common':'common_dynamic','r59_direct_all':'direct_all'}.get(arm,arm);assert np.array_equal(pred['scores'][j],ref['scores'][list(ref['arm_names']).index(name)])
  basehs={v:decoder(d,z,v,'zero_replay') for v in 'ABC'}
  for v in 'ABC':maxh=max(maxh,float((basehs[v][0]-torch.tensor(z[v+'_h'],device='cuda')).abs().max()))
  for j,arm in enumerate(ARMS[len(REFERENCES):],len(REFERENCES)):
   if arm in HEAD_ARMS:
    stage,indices=selectors[arm];raw={v:head_forward(torch.tensor(z[v+'_h'],device='cuda')[None],H,stage,indices) for v in 'ABC'}
   else:
    hs=basehs if arm=='zero_replay' else {v:decoder(d,z,v,arm) for v in 'ABC'};raw={v:head_forward(hs[v],H) for v in 'ABC'}
   s=score(raw,z,g);error=float(abs(s-pred['scores'][j]).max());maxerr=max(maxerr,error);assert error<3e-5 and s.argmax()==pred['queries'][j],('READOUT',i,arm,error);actual+=1
  assert pred['scores'].argmax(1).tolist()==pred['queries'].tolist()
  if (i+1)%32==0:print('AUDIT61',i+1,336,flush=True)
 assert maxh<1e-4
 out=ROOT/'research/round61/prediction_audit_attempt001';assert not out.exists();out.mkdir(parents=True);result={'state':'PASS_R61_INDEPENDENT_FUNCTIONAL_DECODER_HEAD_AND_ALL_READOUTS_AUDIT','n':336,'arm_names':ARMS,'independent_noninherited_readouts':actual,'source_prefix_hidden_max_error':maxh,'independent_native_score_max_error':maxerr,'all_selected_original_queries_exact':True,'fixed_ABC_correspondence_UNKNOWN_and_boxes':True,'all_selectors_reconstructed_without_task_labels':True,'no_new_full_source_forward_or_optimizer':True,'prediction_seal_sha256':sha(pr/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result,flush=True)
if __name__=='__main__':main()
