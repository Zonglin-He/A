"""CPU-only actual official Waymo train/VAL qualification and preprocessing audit.
Training-source labels are authorized; no Drone/Quad annotations are opened.
No learned model is constructed or called, and CUDA is hidden for this process.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,hashlib,collections,re,time
import numpy as np
import torch
R=Path(__file__).resolve().parent;B=R.parents[2];P=B/'crossplatform';U=P/'upstream';D=P/'dataset_full/3eed_dataset';sys.path[:0]=[str(U),str(P)];os.chdir(U)
from src.joint_det_dataset import Joint3DDataset,WAYMO_SYNONYMS,get_positive_map
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature(meta,caption,cls,bbox):return (str(meta),caption,cls,json.dumps(bbox,separators=(',',':')))
def front(raw,suffix):
 b=np.asarray(raw[:7],np.float64).copy();a=np.deg2rad([0,-45,45,-90,90][suffix]);c,s=np.cos(a),np.sin(a);b[:2]=np.array([[c,-s],[s,c]])@b[:2];b[6]+=a;return b.astype(np.float32)
def nheader(p):
 with p.open('rb') as f:
  v=np.lib.format.read_magic(f);shape,_,dtype=np.lib.format._read_array_header(f,v);offset=f.tell()
 assert len(shape)==2 and shape[1]>=4 and shape[0]>0 and not dtype.hasobject
 assert p.stat().st_size==offset+int(np.prod(shape))*dtype.itemsize
 return int(shape[0]),str(dtype),list(shape)
def main():
 start=time.perf_counter();assert not torch.cuda.is_initialized();report={};indices={};bridges=[];datasets={};allscenes={};allframes={};allmetahashes={};rawpaths={}
 for part in ['train','val']:
  split=D/'splits'/('waymo_'+part+'.txt');scenes=[s.strip() for s in split.read_text().splitlines() if s.strip()];present=[s for s in scenes if (D/'waymo'/s).is_dir()];missing=[s for s in scenes if s not in present];allscenes[part]=set(present);unmatched=[];rawrows=[];mapping=collections.defaultdict(list);frames=[];headers={}
  # Independent eligibility reconstruction follows exact loader dictionary priority.
  for scene in scenes:
   sd=D/'waymo'/scene
   if not sd.is_dir():continue
   for frame in sd.iterdir():
    if not frame.is_dir():continue
    mp=frame/'meta_info.json';lp=frame/'lidar.npy';assert mp.is_file() and lp.is_file();frames.append((scene,frame.name));headers[str(lp)]=nheader(lp);rawpaths[str(lp)]={'bytes':lp.stat().st_size,'N':headers[str(lp)][0]};allmetahashes[str(mp)]=sha(mp);container=json.loads(mp.read_text())
    for oi,obj in enumerate(container['ground_info']):
     caption=obj['caption'];cls=obj['class'].lower();bbox=obj['bbox_3d'];b=np.asarray(bbox,float);assert b.shape==(7,) and np.isfinite(b).all() and np.all(b[3:6]>0);formatted=' '+' '.join(caption.replace(',',' ,').split())+' ';positions=[]
     for word in WAYMO_SYNONYMS.get(cls,[cls]):positions.extend((m.start(),m.end(),word) for m in re.finditer(r'\b'+re.escape(word)+r'\b',formatted,re.I))
     row={'platform':'waymo','split':part,'scene':scene,'frame':frame.name,'object_index':oi,'meta_path':str(mp),'meta_sha256':allmetahashes[str(mp)],'lidar_path':str(lp),'caption_sha256':hashlib.sha256(caption.encode()).hexdigest(),'true_source_class':cls,'raw_point_count':headers[str(lp)][0],'waymo_view_suffix':int(frame.name.split('_')[-1]),'source_GT7_float32':front(bbox,int(frame.name.split('_')[-1])).tolist(),'official_eligible':bool(positions)};rawrows.append(row)
     if positions:
      row['official_chosen_synonym']=positions[0][2];row['official_span_chars']=[positions[0][0],positions[0][1]];mapping[signature(mp,caption,cls,bbox)].append(row)
     else:unmatched.append({k:row[k] for k in ['scene','frame','object_index','true_source_class','caption_sha256']})
  ds=Joint3DDataset(dataset_dict={'waymo':1},test_dataset={'waymo':1},split=part,data_path=str(D),split_dir=str(D/'splits'),use_color=False,use_height=False,use_multiview=False,debug=False);datasets[part]=ds;annos=ds.annos;order=[];empty=0;countmap=collections.Counter()
  for ai,a in enumerate(annos):
   assert a['dataset']=='waymo' and Path(a['meta_path']).is_relative_to(D/'waymo');key=signature(a['meta_path'],a['utterance'],a['target'],a['gt_bbox']);assert mapping[key],('actual loader annotation not reconstructed',part,ai);row=mapping[key].pop(0);row={**row,'anno_index':ai};formatted=ds._format_caption(a['utterance']);token=ds.tokenizer(formatted,return_tensors='pt');expected=get_positive_map(token,[torch.tensor(row['official_span_chars'])])[0];assert np.array_equal(expected,np.asarray(a['pred_pos_map'])[0]);row['positive_map_nonzero']=bool((expected>0).any());row['positive_map_sha256']=hashlib.sha256(expected.tobytes()).hexdigest();row['actual_model_token_length']=len(ds.tokenizer(' '.join(a['utterance'].replace(',',' ,').split())+' . not mentioned')['input_ids']);assert row['actual_model_token_length']<=256;order.append(row);empty+=not row['positive_map_nonzero'];countmap[a['target']]+=1
  assert not any(mapping.values()),'Independent eligible query absent from actual loader';assert len(annos)==len(rawrows)-len(unmatched);indices[part]=order;allframes[part]=set(frames);(R/f'OFFICIAL_{part.upper()}_ANNO_ORDER_PRIVATE.json').write_text(json.dumps(order,indent=2));npoints=[r['raw_point_count'] for r in order];report[part]={'official_listed_scenes':len(scenes),'present_scenes':len(present),'missing_release_scenes':missing,'all_actual_frames':len(frames),'raw_objects':len(rawrows),'eligible_actual_loader_annos':len(annos),'ineligible_no_class_synonym_span':len(unmatched),'eligible_empty_positive_map':empty,'class_counts':dict(countmap),'raw_N_per_eligible_expression_min_median_max':[min(npoints),float(np.median(npoints)),max(npoints)],'eligible_expressions_sampling_with_replacement':sum(n<16384 for n in npoints),'original_loader_anno_order_sha256':hashlib.sha256(json.dumps([(r['meta_path'],r['object_index'],r['caption_sha256'],r['true_source_class']) for r in order],separators=(',',':')).encode()).hexdigest(),'private_order_index_sha256':sha(R/f'OFFICIAL_{part.upper()}_ANNO_ORDER_PRIVATE.json'),'split_sha256':sha(split),'ineligible_rows_private':unmatched,'batch24_train_steps_drop_last' if part=='train' else 'batch24_val_steps_no_drop':len(annos)//24 if part=='train' else (len(annos)+23)//24,'train_drop_last_per_epoch' if part=='train' else 'val_last_batch_size':len(annos)%24}
  # One actual getitem per available Waymo view suffix; no model, source labels only.
  chosen=[]
  for suffix in range(5):
   ai=next((i for i,r in enumerate(order) if r['waymo_view_suffix']==suffix),None)
   if ai is not None:chosen.append(ai)
  for ai in chosen:
   seed=20261030+ai;np.random.seed(seed);item=ds[ai];row=order[ai];raw=np.load(row['lidar_path'],allow_pickle=False);np.random.seed(seed);ix=np.random.choice(len(raw),16384,replace=len(raw)<16384);pc=raw[ix,:3].copy();angle=np.deg2rad([0,-45,45,-90,90][row['waymo_view_suffix']]);c,s=np.cos(angle),np.sin(angle);pc=(pc@np.array([[c,-s,0],[s,c,0],[0,0,1.]]).T).astype(np.float32);assert np.array_equal(item['point_clouds'],pc);assert item['point_clouds'].shape==(16384,3) and np.isfinite(pc).all();assert np.array_equal(item['gt_bboxes'][0],np.asarray(row['source_GT7_float32'],np.float32));assert np.array_equal(item['center_label'][0],item['gt_bboxes'][0,:3]) and np.array_equal(item['size_gts'][0],item['gt_bboxes'][0,3:6]);assert item['box_label_mask'].sum()==1 and set(np.unique(item['point_instance_label'])).issubset({-1,0});assert item['positive_map'].shape==(132,256) and np.count_nonzero(item['positive_map'][1:])==0
   bridges.append({'split':part,'anno_index':ai,'suffix':row['waymo_view_suffix'],'sourceXYZ_randomsample_frontrotation_bitexact':True,'sourceGTcenter_size_yaw_float32_bitexact':True,'target_points_in_sample':int((item['point_instance_label']==0).sum()),'positive_map_nonzero':bool(item['positive_map'][0].any()),'no_model_or_GPU':True})
 assert not allscenes['train']&allscenes['val'];assert not allframes['train']&allframes['val'];assert report['train']['eligible_actual_loader_annos']==3682 and report['val']['eligible_actual_loader_annos']==3782;assert not torch.cuda.is_initialized();(R/'GETITEM_BRIDGES.json').write_text(json.dumps(bridges,indent=2));receipt={'state':'actual_official_Waymo_source_train_val_CPU_audit_complete','scope':'Only sourceWaymo train andVAL labels. NoDrone/Quadmetadata opened, no learned model constructed, noGPU initialized.','parts':report,'train_val_present_scene_intersection':0,'train_val_frame_intersection':0,'source_labels_used_for_train_qualification_and_training':True,'source_val_labels_allowed_for_source_monitoring_only':True,'source_val_not_training_data':True,'no_target_domain_labels_or_selection':True,'augmentation_actual':'__getitem__ calls random16384sampling and fixed Waymo suffixXYfrontrotation only. self.augment flag exists but _augment/aug_points never called.','input_channels_actual':'XYZ only; reflectance computed/discarded; noRGB/height/multiview branch in getter; backbone hardcodes3channels','GT_readout':'Source trueclass chooses caption synonym span by dictionarypriority, leading/trailing spaces andget_positive_map. Positive_map0 is used for officialclass-assistedreadout. Puretextfirstnoun must remain separatelynamed.','global_RNG':'Officialnp.random.choice uses worker/globalNP RNG; do not silently substitute perexpressionGenerator in training.','annotation_order':'os.listdir/Path.iterdir insertion order, not sorted in originalloader. Privateindex above binds exact current order and rawobjectindex forsourceVAL join.','metadata_source_file_hashes_sha256':hashlib.sha256(json.dumps(sorted(allmetahashes.items()),separators=(',',':')).encode()).hexdigest(),'point_header_inventory_sha256':hashlib.sha256(json.dumps(sorted(rawpaths.items()),separators=(',',':')).encode()).hexdigest(),'getitem_bridges_sha256':sha(R/'GETITEM_BRIDGES.json'),'code_sha256':sha(Path(__file__)),'upstream_sha256':{name:sha(U/name) for name in ['src/joint_det_dataset.py','utils/transform_waymo.py','utils/box_util.py','train_dist_mod.py','main_utils.py','scripts/train_waymo.sh']},'tokenizer_sha256':{n:sha(U/'data/roberta_base'/n) for n in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']},'seconds':time.perf_counter()-start};(R/'SOURCE_DATA_AUDIT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'parts':{p:{k:v for k,v in q.items() if k!='ineligible_rows_private'} for p,q in report.items()},'bridges':len(bridges),'seconds':receipt['seconds'],'audit_sha256':sha(R/'SOURCE_DATA_AUDIT.json')},indent=2))
if __name__=='__main__':main()
