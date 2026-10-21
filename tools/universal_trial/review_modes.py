"""Read-only no-GT interpretation of completed parameter-policy components."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write

def cluster(boxes,window,threshold=.9):
    def iou(a,b):
        al=a[:,:2]-a[:,2:]/2;ah=a[:,:2]+a[:,2:]/2;bl=b[:,:2]-b[:,2:]/2;bh=b[:,:2]+b[:,2:]/2
        inter=np.maximum(np.minimum(ah,bh)-np.maximum(al,bl),0).prod(-1);union=a[:,2:].prod(-1)+b[:,2:].prod(-1)-inter
        return float((inter/np.maximum(union,1e-20))[window].mean())
    medoids=[];ids=[]
    for b in boxes:
        found=[i for i,c in enumerate(medoids)if iou(b,c)>=threshold]
        if found:ids.append(found[0])
        else:ids.append(len(medoids));medoids.append(b)
    return ids,len(medoids)

def run():
    roots=[ROOT/'outputs'/n for n in ['tubedetr_universal_trial','tubedetr_universal_diagnosis']];seals=[json.loads((r/'sealed.json').read_text())for r in roots];values=[];mode=[]
    for q,e in seals[0]['files'].items():
        e2=seals[1]['files'][q];assert sha(roots[0]/e['path'])==e['sha256']and sha(roots[1]/e2['path'])==e2['sha256']
        with np.load(roots[0]/e['path'])as z,np.load(roots[1]/e2['path'])as d:
            best=int(z['policy_rewards'].argmax());selected=np.unravel_index(d['Direct_path_rewards'].argmax(),d['Direct_path_rewards'].shape);base=z['Source'];pb=z['policy_boxes'][best];db=d['Direct_path_boxes'][selected]
            values.append([np.abs(z['policy_centers']).sum(-1).mean(),np.abs(z['policy_centers'][best]).sum(),np.linalg.norm(pb-base,axis=-1).mean(),np.linalg.norm(db-base,axis=-1).mean()])
            window=(z['frame_ids']>=z['Strong_interval'][0])&(z['frame_ids']<z['Strong_interval'][1]);ids,n=cluster(z['initial_boxes'],window);_,postn=cluster(z['policy_boxes'],window)
            mixes=sum(len({ids[0],ids[1+a],ids[9+a]})>1 for a in range(8));mode.append([n,postn,mixes/8])
    a=np.mean(values,axis=0);m=np.mean(mode,axis=0);folder=ROOT/'results/tubedetr_universal_confidence'
    scale=dict(status='posthoc_scale_interpretation_no_GT',queries=16,mean_all_policy_center_L1=float(a[0]),mean_selected_policy_center_L1=float(a[1]),mean_selected_policy_box_displacement=float(a[2]),mean_selected_allstart_Direct_box_displacement=float(a[3]),source_prediction_seal_sha256=sha(roots[0]/'sealed.json'),diagnosis_prediction_seal_sha256=sha(roots[1]/'sealed.json'),not_causal_or_method_advantage=True,interpretation='Policy centers move less than selected Direct tubes; this is consistent with conservative update size, not causal proof of exploration advantage.')
    saved=folder/'scale_interpretation.json';assert json.loads(saved.read_text())==scale
    f=folder/'mode_interpretation.json';write(f,dict(status='read_only_no_GT_mode_check',queries=16,mean_initial_trajectory_clusters=float(m[0]),mean_policy_center_trajectory_clusters=float(m[1]),mean_fraction_parameter_components_containing_multiple_initial_trajectory_clusters=float(m[2]),IoU_threshold=.9,clustering='original greedy uniform sampled-frame IoU within fixed Strong window; not semantic identity truth',parameter_component_not_proven_trajectory_mode=True,no_new_experiment=True))
    write(folder/'interpretation_seal.json',dict(status='complete_read_only_interpretation',files={str(p.relative_to(ROOT)):sha(p)for p in [saved,f]},code_sha256=sha(Path(__file__)),parent_prediction_seals=[sha(r/'sealed.json')for r in roots],labels_read=False))

if __name__=='__main__':run()
