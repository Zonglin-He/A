"""Independent cached bbox domain audit; no model/media/task GT access."""
import sys,json,hashlib,math,time,copy,collections
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[3];H=Path(__file__).resolve().parent;E=R/'research/round47/experiment';B=R/'crossplatform/round47/predictions208_attempt001';S=R/'crossplatform/round45/bboxA208_attempt001';P3=R/'crossplatform/round39/inference208_attempt002';PRE=R/'crossplatform/round37/prepared208_attempt001'
sys.path[:0]=[str(R/'research/round39/independent_audit'),str(R/'research/round40/independent_audit')]
from audit39_center_independent import parse,bbox_readout,security
from audit40_independent import readout,fixtures
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(p):
 s=read(p/'SEALED.json')
 for rel,h in s['files'].items():assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(p/rel)==h,(p,rel)
 return s
NEW=[f'{c}_bbox_domain_{m}_{g}' for c in ['3B','7B'] for m in ['center','rectIoU'] for g in ['fullmap','oldsubset']]
def domain(raw,wh):
 p=parse(raw,wh);f={'strict_status':p['status'],'strict_reason':p.get('reason'),'outside_only_recovered':p['status']=='invalid_box' and p.get('reason')=='outside_declared_input_range','declared_range_compliant':p['status']=='parsed','original_values_unchanged':True,'not_original_prompt_compliance_certification':True}
 p=copy.deepcopy(p)
 if f['outside_only_recovered']:p.update(status='parsed',reason=None)
 return p,f
def assign(c,case,z):
 p=c['parsed'];grid=c['image_grid_thw'][0];W,H=case['image_size_wh'];wp,hp=grid[2]*14,grid[1]*14;assert c['model_resized_wh']==[wp,hp];d={'status':p['status'],'mask_index':None,'bbox_resized_unchanged':p['value'],'bbox_original_xyxy':None,'mask_IoU':None,'top_IoU':None,'top2_margin':None,'exact_max_tie_n':None,'all_masks_including_unmappable':True}
 if p['status']!='parsed':return d
 x1,y1,x2,y2=p['value'];assert all(math.isfinite(v) for v in p['value']) and x1<x2 and y1<y2;b=np.array([x1*W/wp,y1*H/hp,x2*W/wp,y2*H/hp],float);d['bbox_original_xyxy']=b.tolist()
 if not np.isfinite(b).all():d.update(status='nonfinite_geometry_unknown',bbox_original_xyxy=None);return d
 rect=z.get('mask_boxes');n=len(z['masks_bool'])
 if rect is None or np.shape(rect)!=(n,4) or not np.isfinite(rect).all() or np.any(rect[:,2:]<=rect[:,:2]):d['status']='missing_or_invalid_detector_box_inventory';return d
 if not n:d.update(status='no_independent_masks',mask_IoU=[],exact_max_tie_n=0);return d
 cand={x['mask_index']:x for x in case['candidates']};assert set(cand)==set(range(n)) and all(np.array_equal(cand[m]['bbox_xyxy'],rect[m]) for m in range(n));rr=np.array(rect,float)
 with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
  area=(b[2:]-b[:2]).prod();intersection=np.maximum(np.minimum(b[2:],rr[:,2:])-np.maximum(b[:2],rr[:,:2]),0).prod(1);union=area+(rr[:,2:]-rr[:,:2]).prod(1)-intersection;v=np.divide(intersection,union,out=np.zeros(n,float),where=union>0)
 if not math.isfinite(float(area)) or not np.isfinite(v).all() or not np.all((v>=0)&(v<=1)):d['status']='nonfinite_geometry_unknown';return d
 mx=float(v.max());hit=np.flatnonzero(v==mx);o=np.sort(v);d.update(mask_IoU=v.tolist(),top_IoU=mx,top2_margin=float(mx-o[-2]) if n>1 else None,exact_max_tie_n=len(hit))
 if mx<=0:d['status']='all_rectangle_IoU_zero'
 elif len(hit)!=1:d['status']='exact_max_rectangle_IoU_tie'
 else:d.update(status='unique_positive_rectangle_IoU',mask_index=int(hit[0]))
 return d
def main():
 start=time.perf_counter();seal=verify(B);auth=read(E/'POSTSEAL_EVALUATION_AUTHORIZATION.json');assert auth['independent_CPU_audit_authorized'] and auth['prediction_seal_sha256']==sha(B/'SEALED.json');plan=read(E/'PLAN_LOCK.json');impl=read(E/'IMPLEMENTATION_LOCK.json');assert seal['plan_sha256']==sha(E/'PLAN_LOCK.json') and seal['implementation_sha256']==sha(E/'IMPLEMENTATION_LOCK.json')
 p0,f0=domain('{"bbox_2d":[-10,10,40,40]}',[100,100]);assert p0['status']=='parsed' and p0['value']==[-10,10,40,40] and f0['outside_only_recovered']
 for raw in ['{"bbox_2d":[4,1,3,2]}','{"bbox_2d":[0,0,Infinity,2]}','{"bbox_2d":[0,0,true,2]}','[{"bbox_2d":[0,0,2,2]},{"bbox_2d":[0,0,2,2]}]','{"bbox_2d":[0,0,2,2],"cmd":"run"}']:
  pp,ff=domain(raw,[100,100]);assert pp['status']!='parsed' and not ff['outside_only_recovered']
 for rel,h in {**plan['input_sha256'],**impl['program_sha256']}.items():assert sha(R/rel)==h,rel
 for p in [S,P3,PRE]:verify(p)
 rows=read(R/'research/round27/input/ALL528_PRIVATE.json')[:208];output=[];counts={c:collections.Counter() for c in ['3B','7B']};rawbindings=[];validexact=0;emptymax=0;maxerr=0.;arms=None
 for r in rows:
  old=npz(S/'predictions_PRIVATE'/(r['id']+'.npz'));new=npz(B/'predictions_PRIVATE'/(r['id']+'.npz'));arms=old['arm_names'].tolist()+NEW;assert new['arm_names'].tolist()==arms and np.array_equal(old['queries'],new['queries'][:21]) and np.array_equal(old['boxes'],new['boxes'][:21]);z=npz(PRE/'arrays_PRIVATE'/(r['id']+'.npz'));case=read(PRE/'cases_PRIVATE'/(r['id']+'.json'));saved=read(B/'cases_PRIVATE'/(r['id']+'.json'));rd={};flags={}
  for cap,path in [('3B',P3/'calls_PRIVATE'/(r['id']+'_bbox.json')),('7B',S/'calls_PRIVATE'/(r['id']+'_bbox.json'))]:
   call=read(path);call=call['result'] if cap=='3B' else call;raw=call['raw_answer_PRIVATE'] if 'raw_answer_PRIVATE' in call else call['raw'];p,f=domain(raw,call['model_resized_wh']);author=saved['details'][cap];assert p['status']==author['domain_parsed']['status'] and p['value']==author['domain_parsed']['value'] and f==author['domain_flags'];assert sha(path)==saved['raw_refs'][cap]['sha256'];counts[cap][f['strict_status']]+=1;counts[cap]['recovered_outside_only']+=int(f['outside_only_recovered']);flags[cap]=f;cc=dict(call,parsed=p);a=assign(cc,case,z);aa=author['assignment']
   for k in a:
    if k=='mask_IoU' and a[k] is not None:err=float(abs(np.asarray(a[k])-aa[k]).max(initial=0));maxerr=max(maxerr,err);assert err<1e-12
    elif k in ['top_IoU','top2_margin'] and a[k] is not None:assert abs(a[k]-aa[k])<1e-12
    else:assert a[k]==aa[k],(r['alias'],cap,k)
   if a['mask_index'] is not None and not case['full_groups'].get(str(a['mask_index']),[]):emptymax+=1
   center=cc
   if p['status']=='parsed' and not all(math.isfinite((p['value'][j]+p['value'][j+2])/2) for j in [0,1]):center=copy.deepcopy(cc);center['parsed']['status']='nonfinite_center_geometry_unknown'
   for sub in [False,True]:
    g='oldsubset' if sub else 'fullmap';rd[f'{cap}_bbox_domain_center_{g}']=bbox_readout(center,case,z,sub);rd[f'{cap}_bbox_domain_rectIoU_{g}']=readout(a,case,z,sub)
    if f['strict_status']=='parsed':
     for mode in ['center','rectIoU']:
      strict=('bbox_' if mode=='center' else 'bbox_IoU_')+g if cap=='3B' else f'7B_bbox_{mode}_{g}';assert rd[f'{cap}_bbox_domain_{mode}_{g}']['query']==old['queries'][old['arm_names'].tolist().index(strict)];validexact+=1
   rawbindings.append({'alias':r['alias'],'platform':r['platform'],'capacity':cap,'source_raw_container_sha256':sha(path),'raw_unchanged':True,'strict_status':f['strict_status'],'outside_only_recovered':f['outside_only_recovered'],'values_unchanged':True})
  assert rd==saved['readouts'];q=list(map(int,old['queries']))+[rd[a]['query'] for a in NEW];assert np.array_equal(new['queries'],q) and np.array_equal(new['boxes'],z['source_boxes'][q]);output.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':dict(zip(arms,q)),'readouts':rd,'domain_flags':flags})
 result={'state':'PASS_R47_INDEPENDENT_CACHED_DOMAIN_CONTRACT','n':208,'arms':29,'fixed_boxes':6032,'strict21_control_queries_boxes_exact':True,'raw_source_calls_unchanged':416,'new_readouts_recomputed':1664,'strict_valid_readouts_exact':validexact,'status_counts':{c:dict(v) for c,v in counts.items()},'rectangle_IoU_vector_maxerror':maxerr,'empty_max_groups_not_skipped':emptymax,'no_clip_reorder_unit_change':True,'strict_badJSON_nonfinite_nonpositive_not_recovered':True,'software_security':security(),'old_association_software_fixtures':fixtures(),'prediction_seal_sha256':sha(B/'SEALED.json'),'no_GPU_model_optimizer_media_taskGT_FINAL_U_T':True,'seconds':time.perf_counter()-start};dump(H/'AUDIT_RESULT.json',result);dump(H/'AUDITED_SELECTED_QUERIES_PRIVATE.json',output);dump(H/'ALL416_RAW_BINDING_SCALARS.json',rawbindings);print(json.dumps(result))
if __name__=='__main__':main()
