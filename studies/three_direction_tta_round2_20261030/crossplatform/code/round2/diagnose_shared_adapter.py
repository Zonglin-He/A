"""Deterministic CPU replay diagnostics; never changes sealed predictions or reads GT."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import torch

R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import residual_adapter as adapter
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['dev','confirm'],required=True);a=ap.parse_args();torch.set_num_threads(4)
    lock=json.loads((R/'ADAPTER_DIAGNOSTIC_LOCK.json').read_text());assert lock['code_sha256']==sha(Path(__file__));assert lock['plan_sha256']==sha(R/'PLAN_LOCK.json')
    out=R/(a.cohort+'_predictions');seal=json.loads((out/'sealed.json').read_text());assert sha(out/'trace.json')==seal['trace_sha256'];trace=json.loads((out/'trace.json').read_text());source=R/(a.cohort+'_source');srcseal=json.loads((source/'sealed.json').read_text());records=[]
    for row in trace:
        name=row['id']+'.npz';assert sha(out/name)==seal['files'][name];assert sha(source/name)==srcseal['files'][name];z=np.load(source/name,allow_pickle=False);teacher=row['teacher'];ids=np.asarray(teacher['query_indices'],int);accept=np.asarray(teacher['accepted'],bool);accept_all=np.zeros(256,dtype=bool);accept_all[ids[accept]]=True;captures=[]
        original=adapter.QueryResidual
        class Tracked(original):
            def forward(self,features,boxes,scores):
                result=super().forward(features,boxes,scores)
                with torch.no_grad():
                    predicted,logits,_=result;arr=predicted.detach().numpy();center=np.linalg.norm(arr[:,:3]-z['boxes'][:,:3],axis=1);yaw=np.abs((arr[:,6]+np.pi/2)%np.pi-np.pi/2);chosen=int(ids[int(logits[ids].argmax())]);step={'step':len(captures),'chosen_query':chosen,'chosen_geometry_accepted':bool(accept_all[chosen])}
                    for label,subset in [('all256',~accept_all),('top8',~accept)]:
                        dc=center[subset] if label=='all256' else center[ids][subset];dy=yaw[subset] if label=='all256' else yaw[ids][subset]
                        step['unaccepted_'+label]={'count':len(dc),'changed_count_1e7_tolerance':int(((dc>1e-7)|(dy>1e-7)).sum()),'maximum_center_drift_m':float(dc.max()) if len(dc) else 0.,'maximum_modPI_yaw_drift_radians':float(dy.max()) if len(dy) else 0.}
                    captures.append(step)
                return result
        adapter.QueryResidual=Tracked;start=time.perf_counter()
        try:outputs,_=adapter.run(z['query_features'],z['boxes'],z['semantic_scores'],teacher)
        finally:adapter.QueryResidual=original
        sealed=np.load(out/name,allow_pickle=False)
        assert all(np.array_equal(outputs[i]['box7'],sealed['adapter'+str(i)]) for i in [0,1,3]),'instrumentation replay changed predictions'
        records.append({'id':row['id'],'platform':row['platform'],'steps':captures,'replay_adapter0_1_3_bitexact':True,'diagnostic_replay_seconds':time.perf_counter()-start})
    payload={'scope':'Read-only deterministic optimizer replay after prediction seal; shared-adapter spillover is measured, teacher rejection is not called network fallback','prediction_seal_sha256':sha(out/'sealed.json'),'diagnostic_lock_sha256':sha(R/'ADAPTER_DIAGNOSTIC_LOCK.json'),'rows':records,'all_replays_bitexact':True,'cost_scope':'additional diagnostic replay only, not included in deployed adaptationtiming'}
    dest=R/(a.cohort+'_shared_adapter_diagnostic.json');assert not dest.exists();dest.write_text(json.dumps(payload,indent=2));print(json.dumps({'n':len(records),'all_replays_bitexact':True}))
if __name__=='__main__':main()
