"""Frozen source full/masked features. No target inputs or labels imported."""
from core89 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));torch.manual_seed(0);m=Fusion89(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=load_R76(m);fp=fingerprint(base);rows=source_rows('TRAIN512')+source_rows('VAL128')
if args.engineering:rows=rows[:2]
O=B89/('engineering_cache_source_attempt001' if args.engineering else 'source_features640_attempt001');(O/'samples_PRIVATE').mkdir(parents=True,exist_ok=False);(O/'routes_PRIVATE').mkdir();counts=[];calls=0;start=time.perf_counter();error=0.
with torch.no_grad():
 for nr,r in enumerate(rows):
  assert r['platform']=='waymo';z0=prefix(r);z=tensor_prefix(z0);g=load(casepath(r));d=tensor_data(g,r);graph=read(B89/'source_roles640_attempt001/cases_PRIVATE'/(r['id']+'.json'))['graph'];doc=read(B89/'source_roles640_attempt001/dependencies_PRIVATE'/(r['id']+'.json'));m.route=None;m.use_interface=False;att=[];hook=base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(lambda _m,_a,o:att.append(o[1].cpu().numpy()[0]));p0,_,_=m.main(base,z,d,'bidirectional');hook.remove();calls+=1;roles,offset=token_roles88(base,r,z0,graph,doc);a,info=route89(r,{**z0,'frozen_probability':p0.cpu().numpy()},g,graph,roles,att[0],offset,doc);m.route=tensor_route89(a);m.use_interface=True;region={'point':a['region_point'],'image':a['region_image']};ab=[mask89(r,d,region,k) for k in [89001,89002]];payload={}
  for v,(pm,im) in enumerate([(None,None)]+ab):
   p,q,e=m.main(base,z,d,'bidirectional',point_mask=pm,image_mask=im);calls+=1
   if v==0:error=max(error,float(abs(p-p0).max()));assert error<1e-15 and int(p.argmax())==int(p0.argmax())
   for key,x in [('h',e['query_hidden']),('u',e['projected_text']),('late',e['late_RGB_delta']),('local',e['local']),('own',e['own']),('text',e['text']),('relation_text',e['relation_text']),('baseline',e['base_probability']),('mask',z['token_mask'])]:payload[str(v)+'_'+key]=x.cpu().numpy()
  np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),**payload);np.savez_compressed(O/'routes_PRIVATE'/(r['id']+'.npz'),**a);counts.append({'row':'SOURCE'+str(nr+1).zfill(4),'split':'TRAIN512' if nr<512 else 'VAL128',**info})
  if (nr+1)%128==0:print('CACHE_SOURCE89',nr+1,flush=True)
assert fp==fingerprint(base) and all(np.array_equal(v.detach().cpu().numpy(),S[k]) for k,v in m.state_dict().items() if not k.startswith('interface.'));dump(O/'COUNTS.json',{'n':len(rows),'native_suffix_calls':calls,'seconds':time.perf_counter()-start,'zero_initialization_max_probability_error':error,'source_R76_parameters_changed':False,'target_inputs_read':False,'labels_read':False,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});seal(O,'R89_SOURCE_FROZEN_FULL_AND_TWO_MASKED_FEATURES_SEALED',n=len(rows));dump(H89/('ENGINEERING_SOURCE_SUPPORT_ROWS.json' if args.engineering else 'SOURCE_SUPPORT_ROWS.json'),counts);print(read(O/'COUNTS.json'))
