"""Only aggregate saved task scalars. No GT/metadata/FINAL/geometry access."""
from pathlib import Path
import json,statistics,hashlib,collections,math,time,datetime
R=Path(__file__).resolve().parents[3];I=R/'research/round48/independent_audit';O=Path(__file__).resolve().parent
FILES=['ALL336_TASK_ROWS_PRIVATE.json','TASK_SUMMARY.json','FACTOR_CONTRASTS.json','SCENE_SUMMARIES_PRIVATE.json','ALL_NEGATIVES.json','SCORING_RECEIPT.json','AUDIT_RESULT.json']
def read(p):assert p.parent==I and p.name in FILES;return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmp(a,b):
 if isinstance(a,dict):assert set(a)==set(b);[cmp(a[k],b[k]) for k in a]
 elif isinstance(a,list):assert len(a)==len(b);[cmp(x,y) for x,y in zip(a,b)]
 elif isinstance(a,(int,float)) and not isinstance(a,bool):assert abs(float(a)-float(b))<=1e-10,(a,b)
 else:assert a==b

def main():
 start=time.perf_counter();hashes={n:sha(I/n) for n in FILES};rows=read(I/FILES[0]);a=read(I/'AUDIT_RESULT.json');rec=read(I/'SCORING_RECEIPT.json');assert rec['state']=='PASS_R48_SINGLE_OLD336_TASK_SCORE' and rec['independent_audit_sha256']==sha(I/'AUDIT_RESULT.json');assert a['prediction_seal_sha256']==rec['prediction_seal_sha256']=='f40374b6f47b013d865748e503f763a4a412fbc24a77a2dd587ca6562802d67d';assert len(rows)==336 and len({r['id_PRIVATE'] for r in rows})==336
 arms=list(rows[0]['IoU']);assert len(arms)==5;current='current_proxy_CPU64';parser='text_only_subject_parser';lit='official_LITERAL_indices_DIAGNOSTIC';remap='official_charspan_REMAP_DIAGNOSTIC';bridge='cached_native_current_BRIDGE';assert set(arms)=={current,parser,lit,remap,bridge}
 for r in rows:assert list(r['IoU'])==arms and set(r['queries'])==set(arms) and all(math.isfinite(v) and 0<=v<=1 for v in r['IoU'].values());assert r['IoU'][current]==r['IoU'][parser]==r['IoU'][bridge] and r['queries'][current]==r['queries'][parser]==r['queries'][bridge] and r['IoU'][lit]==r['IoU'][remap] and r['queries'][lit]==r['queries'][remap]
 summary={};factor={}
 for split in ['ALL336','D208','V64','M64']:
  rr=[r for r in rows if split=='ALL336' or r['split']==split];assert len(rr)=={'ALL336':336,'D208':208,'V64':64,'M64':64}[split];summary[split]={};factor[split]={}
  for pl in ['drone','quad']:
   pp=[r for r in rr if r['platform']==pl];assert len(pp)==len(rr)//2;summary[split][pl]={}
   for arm in arms:
    v=[r['IoU'][arm] for r in pp];d=[r['IoU'][arm]-r['IoU'][current] for r in pp];summary[split][pl][arm]={'n':len(pp),'mIoU_percent':statistics.mean(v)*100,'Acc25_percent':sum(x>.25 for x in v)/len(v)*100,'Acc50_percent':sum(x>.5 for x in v)/len(v)*100,'gain_vs_CPU64_current_pp':statistics.mean(d)*100,'improved':sum(x>1e-10 for x in d),'worsened':sum(x< -1e-10 for x in d),'query_changes':sum(r['queries'][arm]!=r['queries'][current] for r in pp),'scene_clusters':len({r['scene_PRIVATE'] for r in pp})}
   mm=lambda arm:summary[split][pl][arm]['mIoU_percent'];factor[split][pl]={'parser_minus_current_CPU64_pp':mm(parser)-mm(current),'official_LITERAL_minus_current_CPU64_pp':mm(lit)-mm(current),'official_REMAP_minus_current_CPU64_pp':mm(remap)-mm(current),'official_REMAP_minus_LITERAL_pp':mm(remap)-mm(lit),'current_CPU64_minus_stored_native_pp':mm(current)-mm(bridge)}
  summary[split]['balanced_mIoU_percent']={arm:(summary[split]['drone'][arm]['mIoU_percent']+summary[split]['quad'][arm]['mIoU_percent'])/2 for arm in arms};factor[split]['balanced']={k:(factor[split]['drone'][k]+factor[split]['quad'][k])/2 for k in factor[split]['drone']}
 cmp(summary,read(I/'TASK_SUMMARY.json'));cmp(factor,read(I/'FACTOR_CONTRASTS.json'))
 scenes=read(I/'SCENE_SUMMARIES_PRIVATE.json');assert len(scenes)==20
 for scene in scenes:
  rr=[r for r in rows if (scene['split']=='ALL336' or r['split']==scene['split']) and r['platform']==scene['platform'] and r['scene_PRIVATE']==scene['scene_PRIVATE']];assert len(rr)==scene['n'];cmp({arm:statistics.mean(r['IoU'][arm] for r in rr)*100 for arm in arms},scene['mIoU_percent'])
 neg=read(I/'ALL_NEGATIVES.json');j=0
 for r in rows:
  for arm in arms:
   for ref in arms:
    if arm!=ref and r['IoU'][arm]<r['IoU'][ref]-1e-10:
     x=neg[j];assert all(x[k]==r[k] for k in ['alias','split','platform']) and x['arm']==arm and x['reference']==ref;assert abs(x['delta_pp']-100*(r['IoU'][arm]-r['IoU'][ref]))<1e-10;j+=1
 assert j==len(neg)==rec['negative_ordered_comparisons']==60;assert rec['task_values']==1680;assert all(r['current_parser_mask_equal'] and r['literal_remap_mask_equal'] for r in rows);unknown=sum(r['parser_status']!='FOUND' for r in rows);assert unknown==a['parser_UNKNOWN']==26
 for n in FILES:assert sha(I/n)==hashes[n]
 out={'state':'PASS_INDEPENDENT_R48_SAVED_SCALAR_AGGREGATION','n':336,'arms':5,'IoU_values_aggregated_not_recomputed':1680,'ordered_negatives_exact':60,'scene_summary_rows_exact':20,'split_platform_summary_and_factor_contrasts_exact':True,'parser_unknown':unknown,'effective_current_parser_masks_and_queries_equal_all336':True,'official_literal_remap_equal_all336':True,'native_bridge_query_equal_all336':True,'no_GTbox_metadata_class_manifest_media_XYZ_FINAL_model_NLP_or_IoUengine_access':True,'frozen24blind_text_annotations_and_rolecomparison_not_changed':True,'input_scalar_sha256':hashes,'seconds':time.perf_counter()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (O/'AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');(O/'RECOMPUTED_TASK_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');(O/'RECOMPUTED_FACTOR_CONTRASTS.json').write_text(json.dumps(factor,indent=2)+'\n');print(json.dumps({'audit':out,'balanced':{k:v['balanced_mIoU_percent'] for k,v in summary.items()}},indent=2))
if __name__=='__main__':main()
