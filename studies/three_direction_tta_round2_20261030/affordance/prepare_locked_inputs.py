"""Rebind the published hashed cohort to legally acquired native data in a FRESH replica.
Only XYZ is copied from annotation containers. CoT/image/GT stay local; no model is run.
This produces NEW executable locks because absolute paths necessarily differ; original
hash receipts remain under protocols and are never represented as identical lock bytes.
"""
import argparse,pathlib,json,hashlib,numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=pathlib.Path,default=ROOT.parent/'datasets/PIADv2_Unseen_aff/Unseen_aff');a=ap.parse_args();dataset=a.dataset.resolve()
 assert not (ROOT/'EXECUTION_LOCK.json').exists(),'Use a fresh replica; never replace an execution lock'
 out=ROOT/'cohorts';assert not out.exists(),'Use a fresh replica; existing cohort cannot be replaced';out.mkdir();(out/'xyz').mkdir()
 names=['Img_test.txt','Point_test.txt','hk_test.txt','ok_test.txt'];images,points,human,obj=[(dataset/n).read_text().splitlines() for n in names];assert len(images)==len(points)==len(human)==len(obj)==2013
 def native(s):return dataset/s.replace('\\','/').split('Unseen_aff/',1)[1]
 co=json.loads((ROOT/'protocols/cohort_lock_public.json').read_text());co['manifests']={};inputs={}
 for tag in ['confirm','dev']:
  members=json.loads((ROOT/f'protocols/{tag}_membership.json').read_text());inference=[];scoring=[];metadata=[]
  for m in members:
   i=m['native_index'];ident=m['id'];image=native(images[i]);point=native(points[i]);assert sha(image)==m['image_sha256'],('Native image mismatch',i)
   raw=np.load(point,mmap_mode='r',allow_pickle=False);assert raw.shape==(2048,4) and raw.dtype==np.float64
   xyz=np.array(raw[:,:3],dtype=np.float64,copy=True);assert np.isfinite(xyz).all();geo=out/'xyz'/(ident+'.npy');np.save(geo,xyz);assert sha(geo)==m['xyz_npy_sha256'],('Native geometry mismatch',i)
   assert len(human[i].split(','))==m['human_phrase_count'];assert point.parts[-4]==m['object_class'] and image.parts[-2]==m['affordance']
   inference.append({'id':ident,'parent_id':m['parent_id'],'component_id':m['component_id'],'image':str(image),'geometry':str(geo),'human':human[i],'object':obj[i],'seed':m['seed'],'native_index':i})
   scoring.append({'id':ident,'parent_id':m['parent_id'],'native_annotation_container':str(point),'target_columns':'3:','object_class':m['object_class'],'affordance':m['affordance'],'native_index':i})
   metadata.append({k:v for k,v in m.items() if k not in ['native_index','seed','xyz_npy_sha256']});inputs[str(image)]=sha(image);inputs[str(geo)]=sha(geo)
  for typ,value in [('inference',inference),('scoring',scoring),('metadata',metadata)]:
   p=out/f'{tag}_{typ}.json';write(p,value);co['manifests'][p.name]=sha(p)
 co['replica_note']='Same original selected parents; paths rebound locally. Original source evidence hash receipt preserved in protocols.';write(out/'COHORT_LOCK.json',co)
 mechanism=json.loads((ROOT/'protocols/mechanism_lock_public.json').read_text());mechanism['cohort_lock_sha256']=sha(out/'COHORT_LOCK.json');mechanism['gpu_flock']='<set local shared GPU lock before launch>';write(ROOT/'MECHANISM_LOCK.json',mechanism)
 original=json.loads((ROOT/'protocols/execution_lock_public.json').read_text());sources=original['source_sha256']
 for f,h in sources.items():assert sha(ROOT/f)==h,('Executed source changed or missing',f)
 write(ROOT/'EXECUTION_LOCK.json',{'schema_version':1,'status':'replica_locked_before_any_real_forward','source_sha256':sources,'input_sha256':inputs,'cohort_lock_sha256':sha(out/'COHORT_LOCK.json'),'mechanism_lock_sha256':sha(ROOT/'MECHANISM_LOCK.json'),'original_execution_lock_sha256':original['original_lock_sha256']})
 write(ROOT/'REPLICA_PREPARATION_RECEIPT.json',{'new_execution_lock_sha256':sha(ROOT/'EXECUTION_LOCK.json'),'same_published_image_and_xyz_hashes':True,'GT_columns_copied_to_inference':False,'native_texts_loaded_only_locally':True,'new_model_calls':0})
 print('Prepared same 48 hashed parents with local paths and NEW replica locks; no model or GT scoring executed.')
if __name__=='__main__':main()
