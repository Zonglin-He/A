"""Synthetic-only R12 scalar reduction fixture including unequal32/18 scenes."""
from pathlib import Path
from types import SimpleNamespace
import tempfile,json,hashlib
import metrics as m
import audit_scored_feedback as a
D=Path(__file__).resolve().parent
rows=[]
for platform,quotas in [('drone',[32,18]),('quad',[32,32,32])]:
 for scene,n in enumerate(quotas):
  for i in range(n):
   base=.25 if scene%2==0 else .5;arms={}
   for j,name in enumerate(a.ARMS):
    changed=j>0 and i<(j%4+1);delta=(.02 if i%2==0 else -.01)*(j if changed else 0);value=max(0.,min(1.,base+delta));accepted=changed or (j>0 and i==5)
    arms[name]={'iou':value,'coordinate_errors':{'center_L2_m':1.-delta,'size_L2_m':.3,'abs_error6_m':[1.-delta,0,0,.3,0,0]},'accepted_any':accepted,'actual_changed':changed,'actual_changed_axes':[changed,False,False,False,False,False],'status':'synthetic_finite_nonconverged' if changed and i==0 else 'synthetic_NA' if not accepted else 'synthetic_accepted'}
   rows.append({'id':platform+'_'+str(scene)+'_'+str(i),'platform':platform,'scene':'synthetic_'+platform+'_'+str(scene),'arms':arms})
summary,pairs=m.summaries(rows)
source=[x for x in summary if x['platform']=='drone' and x['arm']=='source' and x['population']=='all'][0]
assert source['mean_iou']!=source['equal_scene_mean_iou'] and source['Acc25']!=source['equal_scene_Acc25'] and source['Acc50']==source['equal_scene_Acc50']==0
out={'state':'R12_146_independent_instance_multiview_scored_after_complete_seal','n':146,'summary':summary,'paired_contrasts':pairs,'rows_PRIVATE':rows}
with tempfile.TemporaryDirectory(prefix='R12_synthetic_scalar_',dir=D) as tmp:
 tmp=Path(tmp)
 for name,data in [('RESULTS_PRIVATE.json',out),('SCALAR_SUMMARY.json',{k:v for k,v in out.items() if k!='rows_PRIVATE'})]:(tmp/name).write_text(json.dumps(data,allow_nan=False))
 seal={'n_samples':146,'prediction_seal_sha256':'synthetic','release_sha256':'synthetic','files':{p.name:a.sha(p) for p in tmp.iterdir()}};(tmp/'SEALED.json').write_text(json.dumps(seal));a.main(SimpleNamespace(scored=tmp,output=tmp/'audit.json'));r=json.loads((tmp/'audit.json').read_text());assert r['state'].startswith('PASS')
receipt={'state':'PASS_R12_SYNTHETIC_SCALAR_REDUCTION_AUDITOR','n_synthetic_cases':146,'unequal_Drone32plus18_micro_and_equal_scene_differ_verified':True,'strict25_50_equality_rejected':True,'all8arms_allcommonandownpopulations_recomputed':True,'numeric_and_exact_checks':r['numeric_and_exact_checks'],'max_numeric_difference':r['max_numeric_difference'],'no_actual_GT_metadata_IoU_model_GPU_or_taskscoring':True,'code_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['audit_scored_feedback.py','metrics.py','check_scored_auditor_cpu.py']}};(D/'SCALAR_AUDITOR_CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
