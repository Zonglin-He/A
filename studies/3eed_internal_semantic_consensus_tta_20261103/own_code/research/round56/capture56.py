"""Frozen internal semantics: proposal reconstruction, decoder trace, subject phrase."""
import argparse,sys,time,torch
from common56 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint,native_matching,SENTINEL
sys.path.insert(0,str(ROOT/'research/round53'));from support53 import entity_masks

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--limit',type=int);ap.add_argument('--resume',action='store_true');a=ap.parse_args();p=read(a.plan);pin(p);out=ROOT/p['output_root'];rows=read(ROOT/p['manifest']);binding=sha(a.plan)
 if out.exists():assert a.resume and read(out/'RUN_BINDING.json')['plan_sha256']==binding and not (out/'SEALED.json').exists()
 else:out.mkdir(parents=True);dump(out/'RUN_BINDING.json',{'plan_sha256':binding})
 for d in ['samples_PRIVATE','calls_PRIVATE','cases_PRIVATE']:(out/d).mkdir(exist_ok=True)
 start=time.perf_counter();model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);versions=[(n,x._version) for n,x in model.named_parameters()];ls=read(ROOT/p['latent_root']/'SEALED.json');gs=read(ROOT/p['graph_root']/'SEALED.json');done=[];new=0
 try:
  for row in rows:
   sid=row['id'];sf=out/'samples_PRIVATE'/(sid+'.npz');cf=out/'cases_PRIVATE'/(sid+'.json')
   if cf.exists():
    c=read(cf);assert a.resume and c['state']=='COMPLETED' and sha(sf)==c['sample_sha256'];done.append(sid);continue
   rel='samples_PRIVATE/'+sid+'.npz';lp=ROOT/p['latent_root']/rel;assert sha(lp)==ls['files'][rel]
   keys=['seed_features','fp2_xyz','text_memory','tokens','token_mask','query_points_sample_inds','boxes','scores','h','input_sample_xyz','point_indices']+['decoder_query_'+str(i) for i in range(6)]
   z=load(lp,keys);gp=ROOT/p['graph_root']/('attributes_PRIVATE/'+sid+'.json');assert sha(gp)==gs['files']['attributes_PRIVATE/'+sid+'.json'];graph=read(gp)['text_graph_unchanged'];lc=read(ROOT/p['latent_root']/('cases_PRIVATE/'+sid+'.json'));masks,status=entity_masks(graph,row['caption'],np.asarray(lc['token_offsets_PRIVATE']));names=['T']+[r['entity'] for r in graph['references'] if masks[r['entity']].any()];role_masks=np.stack([z['token_mask']]+[masks[r] for r in names[1:]])
   T=lambda v:torch.as_tensor(v,device='cuda');point_feats=T(z['seed_features'])[None];xyz=T(z['fp2_xyz'])[None];indices=T(z['query_points_sample_inds']).int()[None];tokens=T(z['tokens'])[None];stage_h=[];stage_boxes=[]
   with torch.no_grad():
    pts,features,got=model.gsample_module(xyz,point_feats,indices);assert torch.equal(got,indices)
    q=model.decoder_query_proj(features).transpose(1,2).contiguous();stage_h.append(q[0]);ep={};center,size=model.proposal_head(features,base_xyz=pts,end_points=ep,prefix='proposal_');stage_boxes.append(torch.cat([center,size],-1)[0]);pos=torch.cat([center,size],-1)
    for i,layer in enumerate(model.decoder):
     q=layer(q,point_feats.transpose(1,2).contiguous(),T(z['text_memory'])[None],pos,None,torch.zeros((1,len(z['tokens'])),device='cuda',dtype=torch.bool),detected_feats=None,detected_mask=None)
     assert torch.equal(q[0],T(z['decoder_query_'+str(i)])),('stage_trace',i)
     stage_h.append(q[0]);ep={};center,size=model.prediction_heads[i](q.transpose(1,2).contiguous(),base_xyz=pts,end_points=ep,prefix='s_');pos=torch.cat([center,size],-1);stage_boxes.append(pos[0])
    projected=[];stage_z=[];stage_p=[]
    for hh in stage_h:
     qq=torch.nn.functional.normalize(model.contrastive_align_projection_image(hh[None]),dim=-1);pairs=[native_matching(qq,tokens,T(mm).bool()) for mm in role_masks];projected.append(qq[0].cpu().numpy());stage_p.append(pairs[0][0].cpu().numpy());stage_z.append(torch.stack([u[1] for u in pairs]).cpu().numpy())
    assert np.array_equal(stage_p[-1],z['scores']) and np.array_equal(stage_boxes[-1].cpu().numpy(),z['boxes'])
   phrase=graph['base_text']['subject'].get('text','').strip(' ,.;:');assert phrase and len(phrase)<4096
   text=phrase+SENTINEL;tokenized=model.tokenizer([text],return_tensors='pt',return_offsets_mapping=True);offset=tokenized['offset_mapping'][0].numpy();positive=(offset[:,1]>offset[:,0])&(offset[:,0]<len(phrase));assert positive.any() and (~positive).any();call=out/'calls_PRIVATE'/(sid+'.json');assert not call.exists(),'No silent retry of entered short-phrase call'
   dump(call,{'state':'STARTED','entered':True,'phrase_sha256':sha_bytes(phrase.encode()),'point_sample_sha256':sha_bytes(z['input_sample_xyz'].tobytes())});new+=1
   with torch.no_grad():
    output=model({'point_clouds':T(z['input_sample_xyz'])[None],'text':[text]});pp,zz=native_matching(output['last_proj_queries'],output['proj_tokens'],T(positive).bool());sb=torch.cat([output['last_center'],output['last_pred_size']],-1)[0]
   np.savez_compressed(sf,stage_role_z=np.stack(stage_z),stage_target_probability=np.stack(stage_p),stage_proj_queries=np.stack(projected),stage_boxes=np.stack([v.cpu().numpy() for v in stage_boxes]),role_masks=role_masks,entity_names=np.asarray(names),subject_phrase_boxes=sb.cpu().numpy(),subject_phrase_scores=pp.cpu().numpy(),subject_phrase_z=zz.cpu().numpy(),subject_phrase_proj_queries=output['last_proj_queries'][0].cpu().numpy(),subject_phrase_proj_tokens=output['proj_tokens'][0].cpu().numpy(),subject_phrase_token_mask=positive,source_boxes=z['boxes'],source_scores=z['scores'])
   dump(call,{'state':'COMPLETED','entered':True,'phrase_sha256':sha_bytes(phrase.encode()),'sample_sha256':sha(sf)});dump(cf,{'state':'COMPLETED','sample_sha256':sha(sf),'platform':row['platform'],'n_roles':len(names),'all_six_decoder_replays_bitexact':True,'proposal_reconstructed_from_original_native_gather_and_projection':True,'subject_phrase_positive_all_content_before_sentinel':True,'entity_mask_status_PRIVATE':status,'point_indices_sha256':sha_bytes(z['point_indices'].tobytes())});done.append(sid)
   if len(done)%16==0:print('CAPTURE56',len(done),336,flush=True)
   if a.limit is not None and new>=a.limit:break
  assert versions==[(n,x._version) for n,x in model.named_parameters()] and fingerprint(model)==fp
  counts={'n':len(done),'new_subject_phrase_forwards_this_process':new,'full_description_forwards':0,'six_decoder_replays':len(done)*6,'source_fingerprint':fp,'model_unchanged':True,'seconds_this_process':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())}
  if len(done)==336:dump(out/'COSTS.json',counts);seal(out,'R56_FROZEN_STAGE_AND_SUBJECT_PHRASE_SEMANTICS_NO_GT',**counts,plan_sha256=binding)
  else:dump(out/'ENGINEERING_STOP.json',counts)
  print(json.dumps(counts),flush=True)
 except BaseException as e:dump(out/('FAILED_'+str(time.time_ns())+'.json'),{'type':type(e).__name__,'error':str(e),'n':len(done),'new_subject_forwards':new,'GT_read':False});raise

def sha_bytes(x):import hashlib;return hashlib.sha256(x).hexdigest()
if __name__=='__main__':main()
