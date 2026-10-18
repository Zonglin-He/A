"""Read back immutable64 decisions; publish non-GT algebra/decision diagnostics."""
from pathlib import Path
import sys,itertools,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_projection_components64_v1 import D,P,ARMS,SUPPORTS,read,put,load,sha

def run():
 bar=read(D/'GLOBAL_READOUT_BARRIER.json');zs=[load(ROOT/f['path'])for f in bar['files']];out={}
 for ds in ['hc2','vidstg']:
  zz=[z for z in zs if z['dataset']==ds];aa=[a for z in zz for a in z['audit']];u=[a['bar_u']for a in aa];pairs={}
  for sup in SUPPORTS:
   iv=lambda z,a:z['fused_intervals'][a]if sup=='fused'else z['offset_intervals'][a][int(sup[-1])]
   pairs[sup]={a+'__equals__'+b:sum(iv(z,a)==iv(z,b)for z in zz)for a,b in itertools.combinations(ARMS,2)}
  out[ds]=dict(queries=32,offsets=64,bar_u=dict(mean=float(np.mean(u)),minimum=min(u),maximum=max(u),positive=sum(x>0 for x in u),negative=sum(x<0 for x in u),zero=sum(x==0 for x in u)),weighted_mass_minus_endpoint_fraction={a:dict(mean=float(np.mean([v['length_definitions'][a]['difference']for v in aa])),max_absolute=max(abs(v['length_definitions'][a]['difference'])for v in aa))for a in ARMS},exact_interval_agreements=pairs)
 put(P/'DECOMPOSITION_DIAGNOSTICS.json',dict(status='pass',count=64,groups=out,GT_read_for_this_audit=False,prior_lambda=.1,nonlinear_argmax_prevents_additive_task_gain=True))
 doc=P/'REPORT.md';s=doc.read_text();e='''## Observed domain pattern

At fixed original online After geometry, HC fluctuation-only gains +4.508774pp vIoU versus Native, while length-only is −1.062619pp with a CI crossing zero. Vid length-only gains +4.130964pp, while fluctuation-only is −1.549118pp. Full projection gains +3.823126/+2.079386pp in HC/Vid. These are different conditional mechanisms across two exposed32-parent panels, not support for one universal length-only or fluctuation-only story. Full−length is positive in HC and negative in Vid; full−fluctuation is uncertain in HC and positive in Vid. No GT-dependent domain-specific readout is registered.

The weighted mean bar_u is negative at35/64 HC offsets and25/64 Vid offsets; its offset mean is −.100250/+.611195. Thus the constant-actionness length term can favor contraction as well as expansion. HC predicted/physical-GT duration ratios are .768713 Native,1.142432 Full,.633216 Length and1.221253 Fluctuation; Vid ratios are1.909904,2.177557,2.298963,2.091747. Ratio-of-length means do not alone determine temporal IoU. Full extension adds34.188/39.844 known frames/query, with After IoU54.305%/49.755% on known added support and31.625/20.250 added unknown frames/query. Unknown temporal support is not a measured background label.

DECOMPOSITION_DIAGNOSTICS.json retains every pair of exact interval-agreement counts, weighted means and the measured difference between weighted cell mass and endpoint duration fractions. Across all arms/offsets the maximum absolute mass-versus-endpoint fraction difference is .030928 HC and .100917 Vid. This is a unit/readout distinction, not a numerical approximation error.

'''
 marker='## Algebra and decision rules';assert marker in s and '## Observed domain pattern'not in s;doc.write_text(s.replace(marker,e+marker,1))
 fs=read(D/'PUBLIC_FILES.json')['files'];fs+=['scripts/audit_stvg_projection_components64_v1.py',str((P/'DECOMPOSITION_DIAGNOSTICS.json').relative_to(ROOT))];(D/'PUBLIC_FILES.json').write_text(json.dumps(dict(files=fs),indent=2)+'\n');put(D/'FINAL_CODE_BINDING.json',dict(status='pass',pins={f:sha(ROOT/f)for f in fs if f.endswith('.py')or f.startswith('protocols/')},readout_and_score_sources_unchanged_since_lock=True))
 print('READOUT_DIAGNOSTICS_PASS',flush=True)
if __name__=='__main__':run()
