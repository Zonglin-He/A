"""Report the frozen confirmation outcome; never choose a new winner here."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
from scripts.run_stvg_referdino_dualscale_confirm32_v1 import D,P,DEV,DP,verify

def main():
 p=verify(True);assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';z=read(P/'RESULTS_PUBLIC.json');cost=read(P/'COST_BREAKDOWN.json');sel=read(DEV/'SELECTION_LOCK.json');assert sel['selected_policy']==p['selected_policy'];tails={}
 def fmt(v):return f"{v['mean']:+.4f} [{v['CI95'][0]:+.4f},{v['CI95'][1]:+.4f}]"
 text=['# Frozen policy confirmation on reserved HC32','',f"Selected on exposed HC128 before confirmation: **{p['selected_policy']}**, scales{p['selected_scales']}, seeds41/42/43. No configuration switching or fitting to confirmation scores. Parent roster was reserved before HC128 and not used in these rounds; historical project Source/old-method evaluation exposure remains. This is confirmation for these policy/interface selections, not globally never-seen data.",'', 'Same original8 physical positions, native ReferDINO selected trajectory, E+finalMBR, eight independent content states, frozen TA-STVG,40Adamsteps/.01/tau.05/32candidates. Source and teacher acquired once per held parent, shared across controls. All32 outputs sealed before confirmationGT. Metrics average seeds within parent, then bootstrap32 parents, not96 independent videos.','', '| Comparison | Full vIoU pp [95%CI] | Outside8 actual sIoU pp [95%CI] |','|---|---:|---:|']
 for pair,v in z['primary']['paired'].items():text.append(f"| {pair} | {fmt(v['v'])} | {fmt(v['unobserved_actual_s'])} |")
 for seed,g in z['by_seed'].items():
  text+=['',f'## Seed {seed}','', '| Comparison | Full vIoU pp [95%CI] | Outside8 actual sIoU pp [95%CI] |','|---|---:|---:|']
  for pair,v in g['paired'].items():text.append(f"| {pair} | {fmt(v['v'])} | {fmt(v['unobserved_actual_s'])} |")
 for a in ['Direct','Selected_Fixed_seedmean','Selected_Live_seedmean','Old_Live_seedmean']:
  dd=sorted((100*(r['values'][a]['v']-r['values']['E_MBR']['v']),r['alias'])for r in z['rows']);v=np.array([x for x,a in dd]);tails[a]=dict(positive=int(sum(v>1e-12)),negative=int(sum(v< -1e-12)),unchanged=int(sum(abs(v)<=1e-12)),below_minus3pp=int(sum(v< -3)),below_minus5pp=int(sum(v< -5)),worst5=dd[:5],best5=dd[-5:])
 evidence={}
 for pair,v in z['primary']['paired'].items():evidence[pair]={ep:('positive_CI'if stat['CI95'][0]>0 else'negative_CI'if stat['CI95'][1]<0 else'inconclusive')for ep,stat in v.items()}
 text+=['','## Tails, cost and bounded conclusion','',json.dumps(tails,ensure_ascii=False,indent=2),'',f"GPUallocation{cost['total_GPU_seconds']:.3f}s; new Sourcequeries32/offsetforwards64; native ReferDINOqueries32/imageframes256. No extra observations per video, expert models, learned heads or pretrained-weight training. Actual state updates{cost['actual_state_updates']}. Prior HC128 cache costs excluded from new development but not free.",'', 'Direction and uncertainty are reported for selected Live−E,−Direct,−freshFixed,−oldLive. A more negative or uncertain confirmation is retained, not used to switch to another scale pair. All positive/negative/unknown parents and seeds remain. End structure iteration with the prelocked selection; CURRENT/MAIN/PAPER not promoted.']
 (P/'REPORT.md').write_text('\n'.join(text)+'\n');put(P/'DECISION_PUBLIC.json',dict(status='confirmation_completed_no_retuning',selected_policy=p['selected_policy'],selected_scales=p['selected_scales'],primary=z['primary'],by_seed=z['by_seed'],evidence=evidence,tails=tails,no_confirmation_based_switch=True,structure_iteration_finished=True,no_production_promotion=True))
 f=DP/'REPORT.md';f.write_text(f.read_text()+'\n\n## Frozen confirmation completed\n\nSelected policy remains'+p['selected_policy']+'. See [reserved32 confirmation](confirmation/REPORT.md). No confirmation-driven parameter switch or structure iteration.\n')
 print('CONFIRM REPORT READY',p['selected_policy'],evidence,flush=True)
if __name__=='__main__':main()
