"""Follow retained harmful parents through the actual final dataset-wide selections."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import P,read,put

def main():
 old=read(P/'FAILURE_PARAMETER_LINKS_PUBLIC.json');final=read(P/'SELECTED_FIVE_ROW_RESULTS_PUBLIC.json');sel=read(P/'SELECTION_PUBLIC.json');out={};lines=['# Final configurations: harmful-parent follow-through','','Parameters are chosen once per dataset on all128 exposed parents. The following retained baseline-harm group is for offline explanation, not a per-video choice. Original and selected initialization are reported separately. All values are complete vIoU percentage points.','']
 for ds,g in final['directions'].items():
  former={r['alias']:r for r in old['datasets'][ds]['all128_signed_parent_explanations']};rows=[]
  for r in g['rows']:
   a=r['alias'];q=former[a];o=r['historical'];v=r['values'];rows.append(dict(alias=a,original_baseline_harmful=q['baseline_harmful'],original_Full_percent=100*o['original_Full']['v'],original_E_percent=100*o['original_ERef_Strong']['v'],selected_E_percent=100*v['Without_spatial_adaptation']['v'],selected_Full_percent=100*v['Full']['v'],original_adaptation_delta_pp=100*(o['original_Full']['v']-o['original_ERef_Strong']['v']),selected_adaptation_delta_pp=100*(v['Full']['v']-v['Without_spatial_adaptation']['v']),selected_total_gain_over_original_Full_pp=100*(v['Full']['v']-o['original_Full']['v']),teacher_same_as_original=True))
  def group(rr):
   return dict(parents=len(rr),mean_gain_vs_original_Full_pp=float(np.mean([r['selected_total_gain_over_original_Full_pp']for r in rr])),improved_vs_original=sum(r['selected_total_gain_over_original_Full_pp']>1e-10 for r in rr),worse_vs_original=sum(r['selected_total_gain_over_original_Full_pp']< -1e-10 for r in rr),original_mean_adaptation_delta_pp=float(np.mean([r['original_adaptation_delta_pp']for r in rr])),selected_mean_adaptation_delta_pp=float(np.mean([r['selected_adaptation_delta_pp']for r in rr])),selected_severe_relative_own_E=sum(r['selected_adaptation_delta_pp']< -5 for r in rr))
  out[ds]=dict(configuration=sel['datasets'][ds]['config'],all_parents=group(rows),original_harmed_group=group([r for r in rows if r['original_baseline_harmful']]),remaining_group=group([r for r in rows if not r['original_baseline_harmful']]),all128_signed_followup=rows)
  lines += [f'## {ds}', '', '| Alias | old E | old Full | selected E | selected Full | selected−old Full | selected OPD−own E |','|---|---:|---:|---:|---:|---:|---:|']
  aliases=list(dict.fromkeys(old['datasets'][ds]['worst_baseline_aliases']+(['BV041']if ds=='vidstg'else[])))
  for a in aliases:
   r=next(x for x in rows if x['alias']==a);lines.append('| '+a+' | '+' | '.join(f'{r[k]:+.5f}'for k in ['original_E_percent','original_Full_percent','selected_E_percent','selected_Full_percent','selected_total_gain_over_original_Full_pp','selected_adaptation_delta_pp'])+' |')
  lines += ['', 'All-parent, original-harm and remaining-parent aggregates: `'+str({k:v for k,v in out[ds].items()if k.endswith('group')or k=='all_parents'})+'`.','']
 put(P/'SELECTED_FAILURE_FOLLOWUP_PUBLIC.json',dict(status='completed_same_retained_harm_cases_under_actual_final_configs',datasets=out,new_GT_read=False,new_model_or_expert_forward=False,per_video_configuration_choice=False));(P/'SELECTED_FAILURE_FOLLOWUP.md').write_text('\n'.join(lines)+'\n');print('SELECTED_HARM_FOLLOWUP_COMPLETE')
if __name__=='__main__':main()
