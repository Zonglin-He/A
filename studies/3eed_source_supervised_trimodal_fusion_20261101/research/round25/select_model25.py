"""Final family/checkpoint selection: Waymo source VAL only, never targets."""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round25'))
from common25 import *

def main():
 candidates={};rank=[]
 for family,folder,lockname in [('token_mass','training_attempt001','SOURCE_SELECTED_LOCK.json'),('residual_rank','training_rank_attempt001','SOURCE_RANK_SELECTED_LOCK.json')]:
  verify(BASE/folder);s=read(EXP/lockname);candidates[family]=[dict(x,family=family,training_folder=folder,selected_optimizer_steps=x['epoch']*116) for x in s['selected']];rank.append({'family':family,'mean_RGB_sourceVAL_mIoU':float(np.mean([x['mIoU_percent'] for x in s['selected'] if x['variant']=='rgb'])),'mean_noRGB_sourceVAL_mIoU':float(np.mean([x['mIoU_percent'] for x in s['selected'] if x['variant']=='no_rgb']))})
 rank.sort(key=lambda x:(-x['mean_RGB_sourceVAL_mIoU'],x['family']));selectedfamily=rank[0]['family'];bestcontrol=max(rank,key=lambda x:x['mean_noRGB_sourceVAL_mIoU'])['family'];dump(EXP/'FINAL_SOURCE_SELECTED_LOCK.json',{'family':selectedfamily,'selected':candidates[selectedfamily],'stronger_noRGB_family':bestcontrol,'stronger_noRGB_configs':[x for x in candidates[bestcontrol] if x['variant']=='no_rgb'],'all_source_choices':rank,'selection_source':'WaymoVAL512 only','target_scores_or_labels_used':False,'target_parameters_updated':False,'primary_seed':0,'second_seed_replication':1,'family_choice_rule_sha256':sha(EXP/'SOURCE_READOUT_EXTENSION_LOCK.json')});print('FINAL_FAMILY25',selectedfamily,rank)
if __name__=='__main__':main()
