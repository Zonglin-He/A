"""Post-seal GT-class-assisted official span diagnostic; never adaptation."""
from pathlib import Path
import argparse,hashlib,json,re,sys
import numpy as np
import torch
R=Path(__file__).resolve().parent;P=R.parent;sys.path[:0]=[str(R),str(P/'upstream')]
from transformers import RobertaTokenizerFast
from src.joint_det_dataset import WAYMO_SYNONYMS,M3ED_SYNONYMS,get_positive_map
from utils.transform_waymo import transform_to_front_view
from score_sealed_double import metric

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['dev','confirm'],required=True);a=ap.parse_args();lock=json.loads((R/'READOUT_DIAGNOSTIC_LOCK.json').read_text());assert sha(Path(__file__))==lock['code_sha256'];assert sha(P/'upstream/src/joint_det_dataset.py')==lock['official_loader_sha256']
    if a.cohort=='confirm':assert (R/'CONFIRMATION_SUPERSEAL.json').is_file()
    pred=R/(a.cohort+'_predictions');assert (pred/'sealed.json').is_file();source=R/(a.cohort+'_source');seal=json.loads((source/'sealed.json').read_text());index=json.loads((R/(a.cohort+'_scoring_index_PRIVATE.json')).read_text());assert len(index)==seal['n'];tokenizer=RobertaTokenizerFast.from_pretrained(P/'upstream/data/roberta_base',local_files_only=True);rows=[]
    for s in index:
        name=s['id']+'.npz';assert sha(source/name)==seal['files'][name];z=np.load(source/name,allow_pickle=False);meta=json.loads(Path(s['meta_path']).read_text());obj=meta['ground_info'][s['object_index']];caption=' '+' '.join(obj['caption'].replace(',',' ,').split())+' ';cat=obj['class'].lower();dictionary=WAYMO_SYNONYMS if s['platform']=='waymo' else M3ED_SYNONYMS;positions=[]
        for word in dictionary.get(cat,[cat]):positions.extend((m.start(),m.end()) for m in re.finditer(r'\b'+re.escape(word)+r'\b',caption,re.I))
        src=int(z['semantic_scores'].argmax());row={'id':s['id'],'platform':s['platform'],'official_span_available':bool(positions),'source_query':src}
        if not positions:row['official_would_skip']=True;rows.append(row);continue
        span=torch.tensor([positions[0]],dtype=torch.long);tokenized=tokenizer.batch_encode_plus([caption],padding='longest',return_tensors='pt');pmap=(get_positive_map(tokenized,span)[0]>0).astype(np.float32);scores=(z['token_scores']*pmap[:z['token_scores'].shape[1]][None]).sum(1);off=int(np.argsort(-scores,kind='stable')[0]);gt=np.asarray(obj['bbox_3d'][:7],float)
        if s['platform']=='drone':gt[2]+=1.8
        if s['platform']=='waymo':_,g=transform_to_front_view(np.zeros((1,3)),gt[None],['F','FL','FR','SL','SR'][s['Waymo_view_suffix']]);gt=g[0]
        si,_=metric(np.r_[z['boxes'][src].astype(float),0.],gt);oi,_=metric(np.r_[z['boxes'][off].astype(float),0.],gt);row.update(official_query=off,same_query=src==off,same_box_bitexact=np.array_equal(z['boxes'][src],z['boxes'][off]),text_only_source_iou=si,GT_class_assisted_source_iou=oi,official_mask_token_indices=np.flatnonzero(pmap).tolist());rows.append(row)
    summary=[]
    for plat in ['all','waymo','drone','quad']:
        rr=[r for r in rows if r['official_span_available'] and (plat=='all' or r['platform']==plat)];summary.append({'platform':plat,'eligible':len(rr),'same_query_count':sum(r['same_query'] for r in rr),'text_only_mean_iou':float(np.mean([r['text_only_source_iou'] for r in rr])),'GT_class_assisted_mean_iou':float(np.mean([r['GT_class_assisted_source_iou'] for r in rr])),'text_only_acc25_count':sum(r['text_only_source_iou']>.25 for r in rr),'GT_class_assisted_acc25_count':sum(r['GT_class_assisted_source_iou']>.25 for r in rr),'text_only_acc50_count':sum(r['text_only_source_iou']>.5 for r in rr),'GT_class_assisted_acc50_count':sum(r['GT_class_assisted_source_iou']>.5 for r in rr)})
    result={'scope':'postseal labelassistedofficialspan diagnostic, notunlabeledmethod nororacle; primarysource/adapterpredictionsunchanged','cohort':a.cohort,'n':len(index),'official_skip_count':sum(not r['official_span_available'] for r in rows),'source_seal_sha256':sha(source/'sealed.json'),'diagnostic_lock_sha256':sha(R/'READOUT_DIAGNOSTIC_LOCK.json'),'rows':rows,'summary':summary};out=R/(a.cohort+'_readout_diagnostic.json');assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
