"""Postseal attribution of bridge gradient mass and matched-wrong decisions."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round113';B=R/'crossplatform/round113';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert (H/'SUPPLEMENT_SCALAR_AUDIT.json').exists();rows=[];same=0;n=0;mx=0.;groups=[]
 for src in ['waymo','drone','quad']:
  schema=rd(B/(src+'_THETA_SCHEMA_PRIVATE.json'));gi=sum(schema['sizes'][:schema['names'].index('gate_I')]);gate=float(np.load(R/('crossplatform/round91/trained_source_attempt001/states_PRIVATE/R76_full_disc_e1.npz' if src=='waymo' else 'crossplatform/round97/'+src+'/SELECTED_SOURCE_ONLY.npz'))['gate_I'])
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;sup=B/'supplement_wrong_natural_attempt001'/name
   for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
    sid=r['id'];p=np.load(sup/'samples_PRIVATE'/(sid+'.npz'));pp=dict(zip(p['arm_names'],p['scores']));a=pp['anchor_bridge_natural_raw_fusedABC'];b=pp['wrong_natural_sameL2_raw_fusedABC'];same+=int(a.argmax()==b.argmax());n+=1;mx=max(mx,float(abs(a-b).max()));meta=rd(B/name/'records_PRIVATE'/(sid+'.json'))
    if meta['protected_alias_fallback']:continue
    with np.load(B/name/'fits_PRIVATE'/(sid+'.npz')) as fits:g=float(fits['gradient_anchor'][gi]);newgate=float(fits['anchor_bridge_natural_raw_theta'][gi])
    norm=meta['fits']['anchor_bridge_natural']['scope_gradient_L2'];rows.append({'source':src,'target':tgt,'id_PRIVATE':sid,'gate_gradient_squared_fraction':g*g/max(norm*norm,1e-30),'source_gate_tanh':float(np.tanh(gate)),'updated_gate_tanh':float(np.tanh(newgate)),'absolute_gate_reduced':bool(abs(np.tanh(newgate))<abs(np.tanh(gate)))})
 for src in ['all','waymo','drone','quad']:
  rr=[r for r in rows if src=='all' or r['source']==src];groups.append({'source':src,'n':len(rr),'mean_gate_gradient_squared_fraction':float(np.mean([r['gate_gradient_squared_fraction'] for r in rr])),'median_gate_gradient_squared_fraction':float(np.median([r['gate_gradient_squared_fraction'] for r in rr])),'absolute_gate_reduced':sum(r['absolute_gate_reduced'] for r in rr)})
 out={'postseal_only':True,'n':n,'correct_vs_sameL2_wrong_top1_equal':same,'max_probability_difference':mx,'gradient_mass_not_a_semantic_identity_certificate':True,'groups':groups};(H/'GATE_ATTRIBUTION_PRIVATE.json').write_text(json.dumps(rows,indent=2)+'\n');(H/'GATE_ATTRIBUTION_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
