from run110 import *
def main():
 p=H110/'experiment/FORMAL_READOUT_LOCK.json';assert not p.exists();jdump(p,{'arms':ARMS110,'n':768,'readouts':768*len(ARMS110),'primary':PRIMARY110,'same_info_direct_three_way_controls_added_before_predictions_or_GT':True,'C_hidden_arrays_and_frozen_references_first_loaded_after_both_states_in_run':True,'buffer_only_TTA':True,'zero_gradient_no_pseudo_labels':True,'registered_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()});print('READOUT_LOCK110',len(ARMS110))
if __name__=='__main__':main()
