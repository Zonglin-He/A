"""CPU data qualification and original-loader bridges. Protected frames excluded before loader access."""
from pathlib import Path
import os,sys,json,hashlib,re,collections,time,zipfile
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';D=R/'crossplatform/dataset_full/3eed_dataset';V=B/'data_view';E=H/'experiment';sys.path.insert(0,str(B));from factory import configure,dataset,sha,order_contract,torch,np
U=R/'crossplatform/round3/p0/upstream';sys.path.insert(0,str(U));from src.joint_det_dataset import WAYMO_SYNONYMS,M3ED_SYNONYMS,get_positive_map
rd=lambda p:json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
start=time.perf_counter();assert not V.exists();V.mkdir();(V/'splits').symlink_to(D/'splits',target_is_directory=True);configure({'seed':0,'torch_num_threads':4})
# Read only the already-registered protected frame identities for exclusion, never media or target labels.
protected={str(Path(r['metapath']).resolve()) for r in rd(R/'research/NIGHT_20261101/input/FINAL384_PRIVATE.json')};counts={};inventories={}
for pl in ['waymo','drone','quad']:
 (V/pl).mkdir();counts[pl]={}
 for part in ['train','val']:
  ss=[x.strip() for x in (D/'splits'/f'{pl}_{part}.txt').read_text().splitlines() if x.strip()];n=excluded=0;missing=[]
  for sc in ss:
   sd=D/pl/sc
   if not sd.exists():missing.append(sc);continue
   vd=V/pl/sc;vd.mkdir(exist_ok=True)
   for f in sd.iterdir():
    if not f.is_dir():continue
    if str((f/'meta_info.json').resolve()) in protected:
     assert part=='val';excluded+=1;continue
    (vd/f.name).symlink_to(f,target_is_directory=True);n+=1
  counts[pl][part]={'split_listed_scenes':len(ss),'missing_release_scenes':len(missing),'retained_frames':n,'protected_frames_excluded_before_opening':excluded,'split_sha256':sha(D/'splits'/f'{pl}_{part}.txt')}
 assert not set((D/'splits'/f'{pl}_train.txt').read_text().split()) & set((D/'splits'/f'{pl}_val.txt').read_text().split())
for pl in ['drone','quad','waymo']:
 for part in (['train','val'] if pl!='waymo' else ['val']):
  ds=dataset(pl,part);syn=WAYMO_SYNONYMS if pl=='waymo' else M3ED_SYNONYMS;mapping=collections.defaultdict(list);raw=missing=0
  for sc in (D/'splits'/f'{pl}_{part}.txt').read_text().split():
   sd=V/pl/sc
   if not sd.exists():continue
   for f in sd.iterdir():
    mp=f/'meta_info.json';actual=str(mp.resolve());assert actual not in protected
    data=rd(mp)
    for oi,o in enumerate(data['ground_info']):
     raw+=1;text=o['caption'];cls=o['class'].lower();formatted=' '+' '.join(text.replace(',',' ,').split())+' ';pos=[]
     for word in syn.get(cls,[cls]):pos.extend((m.start(),m.end(),word) for m in re.finditer(r'\b'+re.escape(word)+r'\b',formatted,re.I))
     if not pos:missing+=1;continue
     b=np.asarray(o['bbox_3d'],np.float64);assert b.shape==((7,) if pl=='waymo' else (9,)) and np.isfinite(b).all() and (b[3:6]>0).all();gt=b[:7].copy()
     if pl=='drone':gt[2]+=1.8
     if pl=='waymo':
      angle=np.deg2rad([0,-45,45,-90,90][int(f.name.split('_')[-1])]);c,s=np.cos(angle),np.sin(angle);gt[:2]=np.array([[c,-s],[s,c]])@gt[:2];gt[6]+=angle
     key=(str(mp),text,cls,json.dumps(o['bbox_3d'],separators=(',',':')));mapping[key].append({'platform':pl,'split':part,'scene':sc,'frame':f.name,'object_index':oi,'meta_path':str(mp),'original_meta_path':actual,'caption_sha256':hashlib.sha256(text.encode()).hexdigest(),'true_source_class':cls,'source_GT7_float32':gt.astype(np.float32).tolist(),'span_chars':[pos[0][0],pos[0][1]],'meta_sha256':sha(mp)})
  rows=[];lens=[]
  for i,a in enumerate(ds.annos):
   key=(a['meta_path'],a['utterance'],a['target'],json.dumps(a['gt_bbox'],separators=(',',':')));assert mapping[key];row=mapping[key].pop(0);tok=ds.tokenizer(ds._format_caption(a['utterance']),return_tensors='pt');pm=get_positive_map(tok,[torch.tensor(row['span_chars'])])[0];assert np.array_equal(pm,np.asarray(a['pred_pos_map'])[0]);row.update(anno_index=i,positive_map_sha256=hashlib.sha256(pm.tobytes()).hexdigest());assert (pm>0).any();n=len(ds.tokenizer(' '.join(a['utterance'].replace(',',' ,').split())+' . not mentioned')['input_ids']);assert n<=256;npoints=Path(a['pcd_path']).stat().st_size//16 if pl!='waymo' else len(np.load(a['pcd_path'],mmap_mode='r'));assert npoints>0;row.update(text_token_length=n,raw_points=npoints);rows.append(row);lens.append(n)
  assert not any(mapping.values()) and raw-missing==len(rows);out=E/f'OFFICIAL_{pl.upper()}_{part.upper()}_ORDER_PRIVATE.json';dump(out,rows);contract=order_contract(ds,pl,part);bridges=[]
  # Bit-exact actual getter point/box preprocessing checks, including worst text length.
  chosen=sorted(set([0,len(ds)//2,len(ds)-1,int(np.argmax(lens))]))
  for i in chosen:
   seed=960000+i;np.random.seed(seed);it=ds[i];a=ds.annos[i];p=Path(a['pcd_path']);rawpc=np.fromfile(p,np.float32).reshape(-1,4) if pl!='waymo' else np.load(p);np.random.seed(seed);ix=np.random.choice(len(rawpc),16384,replace=len(rawpc)<16384);pc=rawpc[ix,:3].copy()
   if pl=='drone':pc[:,2]+=1.8
   if pl=='waymo':
    angle=np.deg2rad([0,-45,45,-90,90][int(rows[i]['frame'].split('_')[-1])]);c,s=np.cos(angle),np.sin(angle);pc=pc@np.array([[c,-s,0],[s,c,0],[0,0,1.]]).T
   assert np.array_equal(it['point_clouds'],pc.astype(np.float32));assert np.array_equal(it['gt_bboxes'][0],np.asarray(rows[i]['source_GT7_float32'],np.float32));assert it['point_clouds'].shape==(16384,3) and np.isfinite(it['point_clouds']).all();bridges.append({'anno_position':i,'point_and_GT_preprocessing_bitexact':True,'text_token_length':lens[i]})
  counts[pl][part].update(raw_annotations=raw,official_eligible=len(rows),excluded_by_official_no_class_synonym=missing,contract=contract,min_max_model_tokens=[min(lens),max(lens)],train_steps_B24_drop_last=len(rows)//24 if part=='train' else None,bridges=bridges);print({'platform':pl,'split':part,'n':len(rows),'tokens_max':max(lens)},flush=True)
archive=R/'crossplatform/downloads/baseline_ckpt_pred.zip';archive_sha=sha(archive);assert archive_sha=='c9ab8ab90517a44379976a3a9a136a1d84bade4abf6dc2c7123b8b33d2dd79ef'
with zipfile.ZipFile(archive) as z:
 ck=[x.filename for x in z.infolist() if x.filename.endswith(('.pth','.pt'))];cfg=json.loads(z.read('final_6384/config.json'));assert len(ck)==1 and set(cfg['dataset'])=={'waymo','drone','quad'}
out={'state':'PASS_R96_SOURCE_QUALIFICATION_AND_PROTECTED_FRAME_EXCLUSION','platforms':counts,'official_archive_sha256':archive_sha,'official_checkpoint_count':len(ck),'official_checkpoint_trained_platforms':cfg['dataset'],'single_source_public_checkpoint_found':False,'protected_unique_frames_exclusion_only':len(protected),'protected_media_decoded_or_scored':False,'source_model_constructed':False,'CUDA_initialized':torch.cuda.is_initialized(),'point_input_channels':3,'drone_z_shift':1.8,'all_primary_checkpoints_fixed_epoch':100,'seconds':time.perf_counter()-start};assert not out['CUDA_initialized'];dump(E/'DATA_AUDIT.json',out);print(out)
