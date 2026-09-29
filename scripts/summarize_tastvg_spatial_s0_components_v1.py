"""Post-hoc, matched-source component readout from already sealed scalar rows."""
import sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write
from scripts.run_tastvg_spatial_expansion_s0_v1 import OUT
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats

def calculate(rows):
    result={}
    for name in ['corruption','clean']:
        rr=[r for r in rows if (r['condition']!='clean' if name=='corruption' else r['condition']=='clean')]
        d=collections.defaultdict(list)
        for r in rr:d[r['parent']].append(r)
        per=[]
        for parent,seq in sorted(d.items()):
            paired=[r for r in seq if r['observed_gain_at_s_oracle'] is not None and r['unobserved_gain_at_s_oracle'] is not None]
            expert=[r for r in seq if r['expert_observed_s'] is not None]
            per.append(dict(parent=parent,oracle_gain=float(np.mean([r['expanded_gain'] for r in seq])),step3_gain=float(np.mean([r['step3_gain'] for r in seq])),observed_gain=float(np.mean([r['observed_gain_at_s_oracle'] for r in paired])) if paired else None,unobserved_gain=float(np.mean([r['unobserved_gain_at_s_oracle'] for r in paired])) if paired else None,expert_minus_native=float(np.mean([r['expert_observed_s']-r['native_expert_observed_s'] for r in expert])) if expert else None))
        paired=[r for r in per if r['observed_gain'] is not None]
        result[name]=dict(source_rows=per,matched_observed_sources=len(paired),observed_gain=stats([r['observed_gain'] for r in paired]),unobserved_gain=stats([r['unobserved_gain'] for r in paired]),observed_minus_unobserved=stats([r['observed_gain']-r['unobserved_gain'] for r in paired]),expert_minus_native=stats([r['expert_minus_native'] for r in per]),expert_valid_GT_frame_cells=sum(r['expert_observed_valid_frames'] for r in rr),positive_oracle_sources=sum(r['oracle_gain']>1e-10 for r in per),step3_sources_improved=sum(r['step3_gain']>.001 for r in per),step3_sources_worse=sum(r['step3_gain']<-.001 for r in per))
    return result

if __name__=='__main__':write(OUT/'COMPONENTS.json',calculate(read(OUT/'ROWS.json')))
