"""Require all frozen confirmation arms and common baselines before GT release."""
from pathlib import Path
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    frozen=json.loads((R/'v2/CONFIRMATION_RELEASE.json').read_text());assert frozen['primary']=='roof_constrained3' and frozen['n']==36;assert (R/'CONFIRMATION_RELEASE.json').read_bytes()==(R/'v2/CONFIRMATION_RELEASE.json').read_bytes();assert sha(R/'inputs/confirm_manifest.json')==frozen['input_manifest_sha256']
    dirs={'source':R/'confirm_source','v1':R/'confirm_predictions','v2':R/'v2/confirm_predictions'};seals={k:json.loads((p/'sealed.json').read_text()) for k,p in dirs.items()};assert all(s['n']==36 and len(s['files'])==36 for s in seals.values());assert set(seals['source']['files'])==set(seals['v1']['files'])==set(seals['v2']['files']);count=0
    for name in seals['source']['files']:
        for key,d in dirs.items():assert sha(d/name)==seals[key]['files'][name]
        a=np.load(dirs['v1']/name,allow_pickle=False);b=np.load(dirs['v2']/name,allow_pickle=False)
        for ka,kb in [('source','source'),('pca','pca'),('surface','surface'),('direct_latent','direct_old'),('adapter0','adapter0'),('source_target_control3','source_target_control3')]:
            assert a[ka].dtype==b[kb].dtype and a[ka].tobytes()==b[kb].tobytes(),(name,ka,kb);count+=1
    receipt={'status':'allconfirmationpredictionssealed_commonbaselinesbyteexact_GTscoringpermitted','n':36,'frozen_primary':'roof_constrained3','final_confirmation_lock_sha256':sha(R/'v2/CONFIRMATION_RELEASE.json'),'prediction_seals':{k:sha(d/'sealed.json') for k,d in dirs.items()},'common_baseline_arrays_byteexact':count,'v1_arms':seals['v1']['arms'],'v2_arms':seals['v2']['arms'],'numeric_GT_access_by_this_process':False,'no_method_or_threshold_selection_from_confirmation':True};p=R/'CONFIRMATION_SUPERSEAL.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2));print(json.dumps({'status':receipt['status'],'n':36,'commonbaselinearrays':count,'superseal_sha256':sha(p)},indent=2))
if __name__=='__main__':main()
