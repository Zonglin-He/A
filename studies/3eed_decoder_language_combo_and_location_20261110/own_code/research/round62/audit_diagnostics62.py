"""Independent saved-only role/readout and main summary checks; no model/GPU."""
import argparse,re,time
from common62 import *
H=ROOT/'research/round62'
def own_mask(caption,offsets,text,bounds=None):
 result=np.zeros(len(offsets),bool)
 if not isinstance(text,str) or not text:return result
 if bounds is not None:
  if not isinstance(bounds,list) or len(bounds)!=2 or not 0<=bounds[0]<bounds[1]<=len(caption):return result
  left,right=bounds;loc=[i for i in range(left,right) if caption.startswith(text,i) and i+len(text)<=right]
  if len(loc)!=1:return result
  a=loc[0];b=a+len(text)
  if re.search('[^\\W_]|_',caption[left:a]+caption[b:right],flags=re.UNICODE):
   if any(c.isalnum() for c in caption[left:a]+caption[b:right]):return result
 else:
  loc=[i for i in range(len(caption)) if caption.startswith(text,i)]
  if len(loc)!=1:return result
  a=loc[0];b=a+len(text)
 for j,(s,e) in enumerate(offsets):result[j]=e>s and e>a and s<b
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args();p=read(args.plan);pin(p);start=time.perf_counter();rows=read(ROOT/p['manifest']);tasks=read(H/'score_semantics336_attempt001/ALL_TASK_ROWS_PRIVATE.json');public=read(H/'role_diagnostics_attempt001/ALL_ROLE_SCALARS_PUBLIC_SAFE.json');savedm=read(H/'role_diagnostics_attempt001/MASKS_PRIVATE.json');weights=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');mx=0.;checks=0
 for i,(r,t) in enumerate(zip(rows,tasks)):
  sid=r['id'];assert sid==t['id_PRIVATE'];z=load(ROOT/p['capture_root']/'samples_PRIVATE'/(sid+'.npz'),['token_offsets','A_tokens']);st=load(ROOT/p['semantics_root']/'states_PRIVATE'/(sid+'.npz'),['source_A_h','native_A_h']);att=load(ROOT/p['semantics_root']/'attention_PRIVATE'/(sid+'.npz'));g=read(ROOT/'crossplatform/round51/grouped336_attempt001/attributes_PRIVATE'/(sid+'.json'))['text_graph_unchanged'];b=g['base_text'];sm=own_mask(r['caption'],z['token_offsets'],b['subject'].get('text',''),b['subject'].get('span'));am=own_mask(r['caption'],z['token_offsets'],b.get('own_attributes_literal',''))
  if np.array_equal(sm,am):am[:]=False
  rm=np.zeros(len(sm),bool);qm=rm.copy()
  for x in g['references']:rm|=own_mask(r['caption'],z['token_offsets'],x.get('phrase',''),x.get('span'))
  for x in g['reference_relations']+g['global_relations']:qm|=own_mask(r['caption'],z['token_offsets'],x.get('quote',''))
  mm=[sm,am,rm,qm&~(sm|am|rm)];q0=t['queries']['frozen_ABC'];q1=t['queries']['selected_native15'];scores={}
  for k in ['source','native']:
   h=st[k+'_A_h'].astype(np.float32)
   for l in [0,2,4]:
    h=h@weights[str(l)+'.weight'].T+weights[str(l)+'.bias']
    if l<4:h=np.maximum(h,0)
   h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12);scores[k]=h@z['A_tokens'].T/.07
  for v in 'ABC':assert np.array_equal(att['source_'+v+'_selected'],att['native_'+v+'_selected'])
  for j,mask in enumerate(mm):
   rr=public[4*i+j];ss=savedm[4*i+j];assert ss['id_PRIVATE']==sid and np.array_equal(mask,ss['mask']);assert rr['available']==bool(mask.any()) and rr['role_token_count']==int(mask.sum());assert rr['query_changed']==(q0!=q1);assert abs(rr['IoU_delta_pp']-100*(t['IoU']['selected_native15']-t['IoU']['frozen_ABC']))<1e-10
   if mask.any():
    for k in ['source','native']:
     x=float(np.mean(scores[k][q1,mask]-scores[k][q0,mask]));e=abs(x-rr[k+'_role_margin_selected_pair']);mx=max(mx,e);assert e<1e-4
    for q,n in [(q0,'q0'),(q1,'q1')]:assert abs(rr['source_attention_mass_'+n]-float(att['source_A_selected'][q,mask].sum()))<1e-10
   checks+=1
 # Reconstruct all overall main/semantic numbers, rescue overlaps and scenes
 # from audited saved task values. No producers or scoring modules imported.
 for mode,name in [('layers','ANALYSIS_MAIN.json'),('semantics','ANALYSIS_SEMANTICS.json')]:
  tt=read(H/('score_'+mode+'336_attempt001/ALL_TASK_ROWS_PRIVATE.json'));a=read(H/name)
  for m in a['metrics']:
   arm=m['arm'];base=np.array([x['IoU']['frozen_ABC'] for x in tt]);v=np.array([x['IoU'][arm] for x in tt]);balanced=np.mean([np.mean([x['IoU'][arm] for x in tt if x['platform']==pl])*100 for pl in ['drone','quad']]);assert abs(balanced-m['balanced_mIoU_percent'])<1e-10;assert m['IoU_improved_vs_ABC']==int((v>base+1e-10).sum()) and m['IoU_worsened_vs_ABC']==int((v<base-1e-10).sum());assert m['rescued25']==int(((base<=.25)&(v>.25)).sum());assert m['broken25']==int(((base>.25)&(v<=.25)).sum())
  if mode=='layers':
   old=np.array([x['IoU']['native_l5'] for x in tt]);rescueold=(base<=.25)&(old>.25)
   for rr in a['rescue_overlap']:
    v=np.array([x['IoU']['native_l'+str(rr['layer'])] for x in tt]);rescue=(base<=.25)&(v>.25);assert rr['rescue25_new_vs_last']==int((rescue&~rescueold).sum());assert rr['rescue25_shared_with_last']==int((rescue&rescueold).sum())
 out=H/'independent_diagnostics_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R62_INDEPENDENT_SAVED_ROLE_MASK_NUMERIC_AND_SUMMARY_AUDIT','n':336,'role_rows':checks,'max_role_margin_numeric_error':mx,'all_selected_language_attentions_exactly_unchanged':True,'all_main_and_semantic_metrics_and_rescue_overlap_verified':True,'no_GT_geometry_models_GPU_or_network':True,'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result)
if __name__=='__main__':main()
