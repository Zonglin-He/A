from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round126';D=R/'crossplatform/round126/posthoc_fit_transfer_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 for name,seal in [('PREDICTIONS_PRIVATE.json','PREDICTIONS_SEALED.json'),('RGB_ADDITIONAL_PREDICTIONS_PRIVATE.json','RGB_ADDITIONAL_SEALED.json')]:assert hashlib.sha256((D/name).read_bytes()).hexdigest()==rd(D/seal)['sha256']
 rr=rd(H/'FIT_TRANSFER_ROWS_PRIVATE.json');old={(r['cohort'],r['source'],r['id_PRIVATE']):r for p in (R/'research/round125/scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json') for r in rd(p)};assert len(rr)==2304
 for r in rr:
  a=old[(r['cohort'],r['source'],r['id_PRIVATE'])]
  for v in ['F','C']:
   for st in ['zero','corr3']:arm=('mean_context' if v=='F' else 'zero_C') if st=='zero' else st+'_'+v;assert r['values'][v+'_full_'+st]==a['arms'][arm]['IoU']
 for fn in ['FIT_TRANSFER_DIAGNOSTIC','RGB_READOUT_SPLIT_DIAGNOSTIC']:
  for x in rd(H/(fn+'.json'))['groups']:
   rows=[r for r in rr if r['cohort']==x['cohort'] and (x['scope']=='all' or r['B_active'] and r['C_available'])];assert len(rows)==x['n'];parts=[]
   for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
    p=[r for r in rows if r['source']==src and r['target']==tgt];parts.append(sum(r['values'][x['view']+'_'+x['readout']+'_corr3']-r['values'][x['view']+'_'+x['readout']+'_zero'] for r in p)/len(p)*100)
   assert len(parts)==6 and abs(sum(parts)/6-x.get('gain_pp',x.get('gain_pp_all_six_equal')))<1e-9
 out={'state':'PASS_POSTHOC_SEALS_PRIMARY_FC_EXACT_BRIDGES_AND_ALL_MATCHED_SUBSET_REDUCTIONS','n':2304,'primary_query_precision_changes':rd(H/'FIT_TRANSFER_DIAGNOSTIC.json')['precision_vs_primary_FC_query_changes'],'used_for_method_selection':False,'new_models_or_parameter_changes':False,'FINAL_access':False};(H/'DIAGNOSTIC_SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
