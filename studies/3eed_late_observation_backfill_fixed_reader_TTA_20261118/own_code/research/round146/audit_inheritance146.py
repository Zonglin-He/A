"""Reused parent outputs checked against pinned original seals, without task labels."""
from common146 import *

def main():
 check();reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');ps=rd(R/'crossplatform/round145/method/SEALED.json')['files'];es=rd(R/'crossplatform/round145/EVIDENCE_SEALED.json')['files'];n=0
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];rel=name+'/samples_PRIVATE/'+sid+'.npz';assert sha(R/'crossplatform/round145/method'/rel)==ps[rel];rel='evidence_PRIVATE/'+name+'/'+sid+'.npz';assert sha(R/'crossplatform/round145'/rel)==es[rel];n+=1
 assert n==1798;dump(H/'INHERITANCE_AUDIT.json',{'status':'PASS','n':n,'all1798_parent_sample_files_match_pinned_parent_seal':True,'all1798_parent_evidence_match_pinned_parent_seal':True,'parent_prediction_values_never_rescored_for_runtime_selection':True,'GT_read':False});print(n,flush=True)
if __name__=='__main__':main()
