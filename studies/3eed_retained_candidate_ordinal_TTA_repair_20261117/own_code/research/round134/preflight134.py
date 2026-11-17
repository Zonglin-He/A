"""Check actionable AB ordinal evidence without reading any task labels."""
from common134 import *
import numpy as np,collections
def main():
 basecheck();reg=rd(R134/'research/round133/INPUT_REGISTRY_PRIVATE.json');rows=[]
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];D=R134/'crossplatform/round131/formal'/name
  with np.load(D/'features_PRIVATE'/(sid+'.npz')) as f:pA=f['pA'];pB=f['pB'];valid=f['valid_B']
  with np.load(R134/'crossplatform/round132/log_scopes_PRIVATE'/name/(sid+'.npz')) as f:ass=f['assignment'];S=f['S'];C=f['C']
  with np.load(R134/'crossplatform/round133/formal'/name/'samples_PRIVATE'/(sid+'.npz')) as f:p0=f['scores'][list(f['arm_names']).index('zero_F')]
  G=int(ass.max()+1);groups=[np.flatnonzero(ass==g) for g in range(G)]
  a=np.array([np.log(max(pA[g].max(),1e-300)) for g in groups]);b=np.array([np.log(max(pB[g].max(),1e-300)) for g in groups]);z=np.array([np.log(max(p0[g].max(),1e-300)) for g in groups]);margin=np.minimum(a[:,None]-a[None,:],b[:,None]-b[None,:]);edges=(margin>1e-10)&C[:,None]&C[None,:]&(z[:,None]<z[None,:]);q0=int(p0.argmax());g0=int(ass[q0]);wins=np.flatnonzero(edges[:,g0]);rows.append({'source':e['source'],'target':e['target'],'id_PRIVATE':sid,'old_partial_active':bool(e['active']),'B_available':bool(valid.any()),'S_groups':int(S.sum()),'C_groups':int(C.sum()),'top_group_supported':bool(S[g0]),'top_group_retained':bool(C[g0]),'all_reversed_pairs':int(edges.sum()),'top_reversed_pairs':len(wins),'smallest_top_margin':float(margin[wins,g0].min()) if len(wins) else 0.,'largest_top_margin':float(margin[wins,g0].max()) if len(wins) else 0.})
 summary={'n':len(rows),'GT_read':False,'FINAL_access':False,'B_available':sum(r['B_available'] for r in rows),'any_reversed_pairs':sum(r['all_reversed_pairs']>0 for r in rows),'top_reversed_pairs_inputs':sum(r['top_reversed_pairs']>0 for r in rows),'directions':{d:{'n':len(rr),'any_pair':sum(r['all_reversed_pairs']>0 for r in rr),'top_pair':sum(r['top_reversed_pairs']>0 for r in rr),'old_active_top_pair':sum(r['top_reversed_pairs']>0 and r['old_partial_active'] for r in rr)} for d,rr in [(d,[r for r in rows if r['source']+'_to_'+r['target']==d]) for d in sorted({r['source']+'_to_'+r['target'] for r in rows})]}}
 dump(H134/'PREFLIGHT_ROWS_PRIVATE.json',rows);dump(H134/'PREFLIGHT.json',summary);print(summary)
if __name__=='__main__':main()
