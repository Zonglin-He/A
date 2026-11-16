from core124 import *
def main():
 assert H==R/'research/round124' and B==R/'crossplatform/round124'
 reg=jread(R/'research/round121/SOURCE_CACHE_REGISTRY_PRIVATE.json');assert len(reg)==384
 for e in reg:
  for v in e['files'].values():assert sha(R/v['path'])==v['sha256']
  D=R/'crossplatform/round121/formal'/e['source'];sid=e['row']['id']
  for k in ['samples_PRIVATE','features_PRIVATE']:assert (D/k/(sid+'.npz')).is_file()
 jdump(H/'PREFLIGHT.json',{'n':384,'each_source':128,'source_registry_sha256':sha(R/'research/round121/SOURCE_CACHE_REGISTRY_PRIVATE.json'),'all_original_source_cache_hashes_verified':True,'private_source_grounding_label_arrays_not_decoded':True,'target_data_read':False,'FINAL_access':False})
 print('PREFLIGHT124_PASS384_SOURCE_ONLY',flush=True)
if __name__=='__main__':main()
