"""Post-seal label-assisted official readout diagnostic; never a new method."""
from pathlib import Path
import sys,json,hashlib,re,numpy as np,torch,importlib.util
R=Path(__file__).resolve().parent;P=R.parent;sys.path[:0]=[str(P),str(P/'upstream')]
from transformers import RobertaTokenizerFast
from src.joint_det_dataset import WAYMO_SYNONYMS,M3ED_SYNONYMS,get_positive_map
from utils.transform_waymo import transform_to_front_view
spec=importlib.util.spec_from_file_location('exact_3eed_score',P/'score.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);exact_iou=module.exact_iou
seal=json.loads((R/'source/sealed.json').read_text());tokenizer=RobertaTokenizerFast.from_pretrained(P/'upstream/data/roberta_base',local_files_only=True);rows=[]
for s in json.loads((R/'manifest.json').read_text()):
 name=s['id']+'.npz';assert hashlib.sha256((R/'source'/name).read_bytes()).hexdigest()==seal['files'][name];d=np.load(R/'source'/name);meta=json.loads(Path(s['meta_path']).read_text());obj=meta['ground_info'][s['obj_index']];caption=' '+' '.join(obj['caption'].replace(',',' ,').split())+' ';cat=obj['class'].lower();dictionary=WAYMO_SYNONYMS if s['platform']=='waymo' else M3ED_SYNONYMS;words=dictionary.get(cat,[cat]);positions=[]
 for word in words:positions.extend((m.start(),m.end(),word) for m in re.finditer(r'\b'+re.escape(word)+r'\b',caption,re.I))
 source_idx=int(d['top_query_index']);row={'id':s['id'],'platform':s['platform'],'scene':s['scene'],'source_query':source_idx,'official_span_available':bool(positions)}
 if not positions:row['official_would_skip']=True;rows.append(row);continue
 beg,end,word=positions[0];tokenized=tokenizer.batch_encode_plus([caption],padding='longest',return_tensors='pt');pmap=get_positive_map(tokenized,torch.tensor([[beg,end]],dtype=torch.long))[0];pmap=(pmap>0).astype(np.float32);ts=d['token_scores'];officialscore=(ts*pmap[:ts.shape[1]][None]).sum(1);official_idx=int(np.argsort(-officialscore,kind='stable')[0]);gt=np.asarray(obj['bbox_3d'][:7],float)
 if s['platform']=='drone':gt[2]+=1.8
 if s['platform']=='waymo':
  view=['F','FL','FR','SL','SR'][int(Path(s['lidar_path']).parent.name.split('_')[-1])];_,g=transform_to_front_view(np.zeros((1,3)),gt[None],view);gt=g[0]
 a=exact_iou(gt,d['boxes'][source_idx]);b=exact_iou(gt,d['boxes'][official_idx]);row.update({'official_query':official_idx,'same_query':official_idx==source_idx,'same_box_bitexact':np.array_equal(d['boxes'][source_idx],d['boxes'][official_idx]),'text_only_source_iou':a,'GT_class_assisted_source_iou':b,'source_acc25':a>.25,'official_acc25':b>.25,'source_acc50':a>.5,'official_acc50':b>.5,'official_mask_token_indices':np.flatnonzero(pmap).tolist()});rows.append(row)
summary=[]
for platform in ['all','waymo','drone','quad']:
 rr=[r for r in rows if r['official_span_available'] and (platform=='all' or r['platform']==platform)];summary.append({'platform':platform,'n_eligible':len(rr),'same_query_count':sum(r['same_query'] for r in rr),'same_box_bitexact_count':sum(r['same_box_bitexact'] for r in rr),'text_only_source_mean_iou':float(np.mean([r['text_only_source_iou'] for r in rr])),'GT_class_assisted_source_mean_iou':float(np.mean([r['GT_class_assisted_source_iou'] for r in rr])),'text_only_acc25':float(np.mean([r['source_acc25'] for r in rr])),'GT_class_assisted_acc25':float(np.mean([r['official_acc25'] for r in rr])),'text_only_acc50':float(np.mean([r['source_acc50'] for r in rr])),'GT_class_assisted_acc50':float(np.mean([r['official_acc50'] for r in rr]))})
result={'scope':'POSTHOC label-assisted metric/readout diagnostic, not an unlabeled method; original36PCA/source predictions and mainresults unchanged','implementation':'Exactly emulates official synonym-list iterationorder, caption-leading/trailingspace formatter and get_positive_map token indices; selects query from sealed originalcontrastive token_scores. Actual sourceforward input matches officialutterancebody+notmentioned.','n':len(rows),'n_official_skip':sum(not r['official_span_available'] for r in rows),'rows':rows,'summary':summary,'source_seal_sha256':hashlib.sha256((R/'source/sealed.json').read_bytes()).hexdigest()};(R/'readout_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
