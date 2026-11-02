"""UNEXECUTED root-granted CPU fixed episodic structure producer.

Reads only DEV336 registration, sealed raw-safe/pre-existing source A/B/safeG
arrays, a saved direct-G gate, and original raw XYZ. No RGB, camera metadata,
external model caches, text parsing, GT, source forward or optimizer is used.
"""
from pathlib import Path
import argparse,collections,hashlib,json,time
import numpy as np
from structure51 import make_structure

ROOT=Path(__file__).resolve().parents[2]
MANIFEST='research/round24/input/ALL336_PRIVATE.json'
FACTORS={.5:'replace_all_c05',1.:'replace_all_c1',2.:'replace_all_c2'}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def save(path,arrays):
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 tmp.replace(path)
def path(value):
 p=(ROOT/value).resolve()
 assert p.is_relative_to(ROOT) and 'FINAL' not in str(p) and 'round46/final' not in str(p)
 return p
def exact(a,b,label):
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label
def npz(p,keys):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}


class Ledger:
 def __init__(self,out):self.out=out;self.rows=[]
 def bind(self,p,expected,role,sid=None,fields=None):
  digest=sha(p);self.rows.append({'path_PRIVATE':str(p),'sha256':digest,'role':role,'id_PRIVATE':sid,'selected_fields':fields})
  dump(self.out/'INPUT_LEDGER_PRIVATE.json',{'items':self.rows,'GT_read':False})
  assert digest==expected,role
  return p
 def bound(self,folder,seal,rel,role,sid=None,fields=None):
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
  return self.bind(folder/rel,seal['files'][rel],role,sid,fields)


def context(args):
 pp=path(args.plan);grant=read(path(args.authorization));plan=read(pp)
 assert pp.parent==ROOT/'research/round51/experiment'
 assert grant['state']=='ROOT_AUTHORIZED_R51_DEV336_FIXED_POINT_CARRIER_STRUCTURES_NO_GT'
 assert grant['plan_sha256']==sha(pp) and grant['program_sha256']==sha(__file__)
 assert plan['state']=='ROOT_LOCKED_R51_DEV336_FIXED_POINT_CARRIER_STRUCTURES'
 assert plan['n']==336 and plan['manifest']==MANIFEST
 assert plan['selected_appearance_factor'] in FACTORS and plan['selected_G_arm']==FACTORS[plan['selected_appearance_factor']]
 assert plan['safe_relation_operator']=='literal_R23_weighted/(1+total)'
 assert plan['readout_addition_order']=='native_probability_plus_2V_then_2P_then_2R'
 assert plan['direct_G_gate_PASS'] is True and plan['no_GT_RGB_external_models_newG_or_optimizer'] is True
 rootkeys=('prepared_root','capture_A_root','capture_B_root','safe_root')
 required={MANIFEST,plan['direct_G_gate_receipt']}|{plan[k]+'/SEALED.json' for k in rootkeys}
 assert set(plan['inputs_sha256'])==required
 for group in ('inputs_sha256','programs_sha256'):
  for rel,digest in plan[group].items():assert sha(path(rel))==digest,rel
 assert {'crossplatform/round51/'+n for n in ('structure_prepare51.py','structure51.py','instances51.py')}<=set(plan['programs_sha256'])
 assert read(path(plan['direct_G_gate_receipt']))['gate_PASS'] is True
 rows=read(ROOT/MANIFEST);assert len(rows)==336 and len({r['id'] for r in rows})==336
 assert set(r['platform'] for r in rows)=={'drone','quad'}
 out=path(plan['output_root']);assert out.parent==ROOT/'crossplatform/round51' and not out.exists()
 roots={k:path(plan[k]) for k in rootkeys};assert all(p.parent==ROOT/'crossplatform/round51' for p in roots.values())
 seals={k:read(p/'SEALED.json') for k,p in roots.items()}
 expected={'prepared_root':'R51_RAW_SAFE_INSTANCE_ATTRIBUTE_PREPARATION_COMPLETE_NO_GT',
  'capture_A_root':'R51_DEV336_SINGLE_WAYMO_MODEL_SUBJECT_REFERENCE_HYPOTHESES_NO_GT',
  'capture_B_root':'R51_CONDITIONAL_DEV336_B_ONLY_SOURCE_PREFIX_CAPTURE_NO_GT',
  'safe_root':'R51_FINAL_SAFE_THREE_FACTOR_PREDICTIONS_NO_GT'}
 for k,s in seals.items():assert s['state']==expected[k] and s['n']==336
 assert seals['prepared_root']['external_vision_outputs_or_mask_derived_cache_read'] is False
 assert seals['safe_root']['external_vision_outputs_or_mask_derived_caches_read'] is False
 assert seals['safe_root']['relation_operator']==plan['safe_relation_operator']
 return plan,rows,out,roots,seals,pp


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--plan',required=True);parser.add_argument('--authorization',required=True)
 args=parser.parse_args();plan,rows,out,roots,seals,pp=context(args)
 out.mkdir();(out/'cases_PRIVATE').mkdir();(out/'arrays_PRIVATE').mkdir()
 ledger=Ledger(out);start=time.perf_counter();counts=collections.Counter();done=[]
 try:
  for rel,digest in plan['inputs_sha256'].items():ledger.bind(path(rel),digest,'REGISTERED_INPUT')
  for row in rows:
   sid=row['id'];tick=time.perf_counter();prep=roots['prepared_root'];ps=seals['prepared_root']
   pcpath=ledger.bound(prep,ps,'cases_PRIVATE/'+sid+'.json','RAW_SAFE_CASE',sid);pc=read(pcpath)
   papath=ledger.bound(prep,ps,'arrays_PRIVATE/'+sid+'.npz','RAW_SAFE_STRICT_CARRIERS',sid)
   assert pc['arrays_sha256']==sha(papath) and pc['platform']==row['platform']
   assert pc['caption_sha256']==hashlib.sha256(row['caption'].encode()).hexdigest()
   a=npz(papath,['source_boxes','source_scores','query_instance_ids','instance_point_offsets','instance_point_indices'])
   ar=roots['capture_A_root'];ass=seals['capture_A_root'];br=roots['capture_B_root'];bs=seals['capture_B_root']
   acp=ledger.bound(ar,ass,'cases_PRIVATE/'+sid+'.json','SOURCE_A_CASE',sid);ac=read(acp)
   bcp=ledger.bound(br,bs,'cases_PRIVATE/'+sid+'.json','SOURCE_B_CASE',sid);bc=read(bcp)
   ap=ledger.bound(ar,ass,ac['subject']['file'],'SOURCE_A_ORIGINAL_BOXES_P',sid,['boxes','scores'])
   bp=ledger.bound(br,bs,bc['B_file'],'SOURCE_B_ORIGINAL_BOXES_P',sid,['boxes','scores'])
   assert sha(ap)==ac['subject']['sha256']==bc['A_source_file_sha256'] and sha(bp)==bc['B_file_sha256']
   assert ac['A_seed']==row['seed'] and bc['same_coordinates_no_rotation_translation_or_flip'] is True
   ca=npz(ap,['boxes','scores']);cb=npz(bp,['boxes','scores'])
   exact(ca['boxes'],a['source_boxes'],'A_canonical_boxes');exact(ca['scores'],a['source_scores'],'A_canonical_nativep')
   sr=roots['safe_root'];ss=seals['safe_root']
   scp=ledger.bound(sr,ss,'cases_PRIVATE/'+sid+'.json','FROZEN_SAFE_G_CASE',sid);sc=read(scp)
   sp=ledger.bound(sr,ss,'samples_PRIVATE/'+sid+'.npz','FROZEN_SELECTED_SAFE_G',sid,
    ['source_boxes','source_scores','scores','components','arm_names','appearance_factor_values','queries'])
   assert sc['samples_sha256']==sha(sp) and sc['input_prepared_case_sha256']==sha(pcpath)
   gdata=npz(sp,['source_boxes','source_scores','scores','components','arm_names','appearance_factor_values','queries'])
   exact(ca['boxes'],gdata['source_boxes'],'G_original_boxes');exact(ca['scores'],gdata['source_scores'],'G_original_nativep')
   factors=gdata['appearance_factor_values'].tolist();fi=factors.index(plan['selected_appearance_factor'])
   assert gdata['arm_names'].tolist()[fi+1]==plan['selected_G_arm']
   cues=np.asarray(gdata['components'][fi],np.float64)
   literal=ca['scores'].astype(np.float64)+2*cues[0]+2*cues[1]+2*cues[2]
   exact(literal,gdata['scores'][fi+1],'ORIGINAL_G_COMPONENT_ADDITION_ORDER_BYTE_BRIDGE')
   assert int(literal.argmax())==int(gdata['queries'][fi+1])
   rawpath=Path(row['pointpath']);ledger.bind(rawpath,pc['raw_bindings']['point_sha256'],'ORIGINAL_RAW_XYZ_ONLY',sid)
   if rawpath.suffix=='.npy':raw=np.load(rawpath,allow_pickle=False)
   elif rawpath.suffix=='.bin':raw=np.fromfile(rawpath,np.float32).reshape(-1,4)
   else:raise ValueError('Only registered .npy or .bin pointclouds')
   assert raw.ndim==2 and raw.shape[1]>=3 and len(raw)==ac['raw_point_count']==bc['raw_point_count']
   raw=np.ascontiguousarray(raw[:,:3],np.float32)
   info,arrays=make_structure(raw_points=raw,platform=row['platform'],prepared_A=a,prepared_receipt=pc,
    cache_A=ca,cache_B=cb,fixed_g=literal-ca['scores'].astype(np.float64),fixed_components=cues)
   info.update(id_PRIVATE=sid,platform=row['platform'],selected_G_arm=plan['selected_G_arm'],
    selected_appearance_factor=plan['selected_appearance_factor'],input_bindings={
     'A_file_sha256':sha(ap),'B_file_sha256':sha(bp),'B_capture_seal_sha256':sha(br/'SEALED.json'),
     'A_capture_seal_sha256':sha(ar/'SEALED.json'),'prepared_case_sha256':sha(pcpath),
     'prepared_arrays_sha256':sha(papath),'safe_sample_sha256':sha(sp),'safe_case_sha256':sha(scp),
     'raw_point_file_sha256':pc['raw_bindings']['point_sha256'],'raw_points_array_sha256':pc['raw_points_sha256'],
     'safe_seal_sha256':sha(sr/'SEALED.json')},GT_read=False,
    new_G_reference_RGB_or_external_model_calls=0,seconds=time.perf_counter()-tick)
   dest=out/'arrays_PRIVATE'/(sid+'.npz');save(dest,arrays);info['arrays_sha256']=sha(dest)
   dump(out/'cases_PRIVATE'/(sid+'.json'),info);done.append(sid);counts['processed_n']+=1
   counts['accepted_feedback']+=info['accepted'];counts['rejected_feedback']+=not info['accepted']
   counts['A_strict_queries']+=int((arrays['A_strict_instance_ids']>=0).sum())
   counts['B_mapped_queries']+=int((arrays['B_canonical_instance_ids']>=0).sum())
   for reason in info['rejection_reasons']:counts['rejection_'+reason]+=1
   if len(done)%16==0:print('R51_STRUCTURE_PREPARE',len(done),336,flush=True)
  dump(out/'COUNTS.json',dict(counts));dump(out/'COSTS.json',{'seconds':time.perf_counter()-start,
   'new_grounder_forwards':0,'new_G_or_reference_calls':0,'optimizer_steps':0,'RGB_reads':0,'metadata_reads':0,'GT_read':False})
  dump(out/'SEALED.json',{'state':'R51_DEV336_FIXED_POINT_CARRIER_STRUCTURES_BEFORE_TTA_NO_GT','n':336,
   'plan_sha256':sha(pp),'selected_G_arm':plan['selected_G_arm'],'selected_appearance_factor':plan['selected_appearance_factor'],
   'direct_G_gate_PASS':True,'original_G_components_and_source_bytes_bridged':True,
   'all_unknown_and_rejected_cases_retained':True,'task_GT_read':False,
   'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as error:
  dump(out/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'completed_n':len(done),
   'automatic_retry':False,'GT_read':False});raise


if __name__=='__main__':main()
