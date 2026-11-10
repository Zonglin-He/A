"""Post-seal A-view role-score diagnostics, never supervision or identity labels."""
import argparse,re,torch
from common62 import *
from compute62 import cpu_head
H=ROOT/'research/round62'
def literal_mask(caption,offsets,literal,span=None):
 mask=np.zeros(len(offsets),bool)
 if not isinstance(literal,str) or not literal:return mask
 if span is not None:
  if not isinstance(span,list) or len(span)!=2 or not 0<=span[0]<span[1]<=len(caption):return mask
  a,b=span;fragment=caption[a:b];starts=[m.start() for m in re.finditer(re.escape(literal),fragment)]
  if len(starts)!=1:return mask
  left=a+starts[0];right=left+len(literal)
  if any(c.isalnum() for c in caption[a:left]+caption[right:b]):return mask
 else:
  starts=[m.start() for m in re.finditer(re.escape(literal),caption)]
  if len(starts)!=1:return mask
  left=starts[0];right=left+len(literal)
 return (offsets[:,1]>left)&(offsets[:,0]<right)&(offsets[:,1]>offsets[:,0])
def masks(graph,caption,offsets):
 base=graph['base_text'];subject=base['subject'];sm=literal_mask(caption,offsets,subject.get('text',''),subject.get('span'))
 attr=literal_mask(caption,offsets,base.get('own_attributes_literal',''))
 if np.array_equal(attr,sm):attr[:]=False
 refs=np.zeros(len(offsets),bool)
 for r in graph['references']:refs|=literal_mask(caption,offsets,r.get('phrase',''),r.get('span'))
 relation=np.zeros(len(offsets),bool)
 for r in graph['reference_relations']+graph['global_relations']:relation|=literal_mask(caption,offsets,r.get('quote',''))
 relation&=~(sm|attr|refs)
 return {'subject':sm,'literal_attribute':attr,'reference':refs,'relation_quote_remainder':relation}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args();p=read(args.plan);pin(p);torch.set_num_threads(4);torch.set_grad_enabled(False)
 assert read(H/'scalar_semantics_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_')
 out=H/'role_diagnostics_attempt001';assert not out.exists();out.mkdir();rows=read(ROOT/p['manifest']);tasks=read(H/'score_semantics336_attempt001/ALL_TASK_ROWS_PRIVATE.json');head=cpu_head(load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz'));public=[];private=[];maxatt=0.
 for idx,(r,t) in enumerate(zip(rows,tasks)):
  sid=r['id'];assert sid==t['id_PRIVATE'];z=load(ROOT/p['capture_root']/'samples_PRIVATE'/(sid+'.npz'),['token_offsets','A_tokens']);st=load(ROOT/p['semantics_root']/'states_PRIVATE'/(sid+'.npz'),['source_A_h','native_A_h']);att=load(ROOT/p['semantics_root']/'attention_PRIVATE'/(sid+'.npz'));graph=read(ROOT/'crossplatform/round51/grouped336_attempt001/attributes_PRIVATE'/(sid+'.json'))['text_graph_unchanged'];mm=masks(graph,r['caption'],z['token_offsets']);q0=t['queries']['frozen_ABC'];q1=t['queries']['selected_native15'];delta=t['IoU']['selected_native15']-t['IoU']['frozen_ABC'];changed=q0!=q1
  scores={}
  for kind in ['source','native']:
   q=torch.nn.functional.normalize(head(torch.tensor(st[kind+'_A_h'])),dim=-1).numpy();scores[kind]=q@z['A_tokens'].T/.07
  for v in 'ABC':maxatt=max(maxatt,float(abs(att['source_'+v+'_selected']-att['native_'+v+'_selected']).max()))
  for role,mask in mm.items():
   rec={'row':f'DEV336_{idx+1:05d}','platform':r['platform'],'role':role,'available':bool(mask.any()),'role_token_count':int(mask.sum()),'query_changed':changed,'IoU_delta_pp':float(delta*100),'source_role_margin_selected_pair':float((scores['source'][q1,mask]-scores['source'][q0,mask]).mean()) if mask.any() else None,'native_role_margin_selected_pair':float((scores['native'][q1,mask]-scores['native'][q0,mask]).mean()) if mask.any() else None,'source_attention_mass_q0':float(att['source_A_selected'][q0,mask].sum()) if mask.any() else None,'source_attention_mass_q1':float(att['source_A_selected'][q1,mask].sum()) if mask.any() else None}
   public.append(rec);private.append({'id_PRIVATE':sid,'role':role,'mask':mask.tolist()})
 summary=[]
 for role in mm:
  for group in ['all','changed','changed_improved','changed_worsened','changed_IoU_unchanged']:
   rr=[x for x in public if x['role']==role and (group=='all' or x['query_changed']) and (group!='changed_improved' or x['IoU_delta_pp']>1e-8) and (group!='changed_worsened' or x['IoU_delta_pp']< -1e-8) and (group!='changed_IoU_unchanged' or abs(x['IoU_delta_pp'])<=1e-8)];known=[x for x in rr if x['available']];d=[x['native_role_margin_selected_pair']-x['source_role_margin_selected_pair'] for x in known];summary.append({'role':role,'group':group,'n':len(rr),'available_n':len(known),'UNKNOWN_n':len(rr)-len(known),'mean_role_margin_delta':float(np.mean(d)) if d else None,'positive_margin_delta_n':sum(v>1e-8 for v in d),'negative_margin_delta_n':sum(v< -1e-8 for v in d)})
 dump(out/'ALL_ROLE_SCALARS_PUBLIC_SAFE.json',public);dump(out/'MASKS_PRIVATE.json',private);dump(out/'SUMMARY_PUBLIC_SAFE.json',{'post_seal_descriptive_only':True,'A_view_only':True,'ABC_selected_pair_not_per_view_identity':True,'n':336,'changed_n':sum(t['queries']['frozen_ABC']!=t['queries']['selected_native15'] for t in tasks),'selected_layer_attention_max_abs_change':maxatt,'relation_role_is_quote_remainder_not_predicate_parse':True,'groups':summary});seal(out,'R62_POST_SEAL_ROLE_DIAGNOSTICS_NOT_METHOD_SELECTION',n=336);print(read(out/'SUMMARY_PUBLIC_SAFE.json'))
if __name__=='__main__':main()
