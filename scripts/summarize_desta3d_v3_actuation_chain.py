"""Build a compact evidence-linked chain summary after all registered audits."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,verify_seal,total_prior

def main():
 names=['actuation_free002','actuation_full001','actuation_span001'];reports={}
 for name in names:
  verify_seal(OUT/name);reports[name]=read(OUT/name/'ROOT_DECISION.json')
  assert reports[name]['status']=='completed_independently_audited'
 mag=read(OUT/'actuation_span001/ROOT_EQUIVALENT_LATENT_MAGNITUDE.json');assert mag['status']=='passed'
 old=read(OUT/'diagnosis_cpu_v1/REPORT.json');cases=[]
 for j,(idx,branch,arm,primary) in enumerate([(7,'event','temporal','tIoU'),(3,'spatial','spatial','sIoU')]):
  base=reports[names[0]]['cases'][j]['baseline'];values={}
  for name,r in reports.items():
   c=r['cases'][j];assert c['baseline']['metrics']==base['metrics'] and c['branch']==branch
   values[name]={'final_percent':{k:100*c['final']['metrics'][k] for k in ['tIoU','sIoU','vIoU']},
    'CE_before':c['baseline']['CE'],'CE_after':c['final']['CE'],'format_failures':c['format_failures'],
    'fixed_endpoint_success':c['registered_fixed_endpoint_success'],'clip_count':c['clip_count']}
  cases.append({'branch':branch,'key':old['cases'][idx]['key'],'primary_metric':primary,
    'base_percent':{k:100*base['metrics'][k] for k in ['tIoU','sIoU','vIoU']},
    'old_correct_mask_percent':{k:100*old['cases'][idx]['arms'][arm]['metrics'][k] for k in ['tIoU','sIoU','vIoU']},
    'old_wrong_mask_percent':{k:100*old['cases'][idx]['arms']['wrong_'+arm]['metrics'][k] for k in ['tIoU','sIoU','vIoU']},
    'controls':values,'span_magnitude':mag['cases'][j]})
 stages=['actuation_free001','actuation_free002','layers001','actuation_full001','actuation_span001']
 receipts={s:read(OUT/s/'RECEIPT.json') for s in stages};fresh=len(list((OUT/'actuation_free001/episodes').glob('*/STEP_*.pt')))
 fresh+=sum(read(OUT/n/'COMPLETE.json')['new_model_gradient_steps'] for n in ['actuation_free002','actuation_full001','actuation_span001'])
 assert fresh==150
 native_paths=[p for n in names+['actuation_free001'] for p in (OUT/n/'episodes').glob('*/NATIVE_*.pt')]
 native_count=len({(p.stat().st_dev,p.stat().st_ino) for p in native_paths});assert native_count==156
 result={'status':'completed_audited_registered_chain','cases':cases,'source_training_queries':2,'outcome_selected':True,
  'source_GT_used':True,'target_read':False,'new_model_gradient_steps':fresh,'saved_gradient_Adam_reconstruction_steps':24,
  'fresh_native_two_pass_outputs':native_count,'old_mask_new_GPU':False,'layer_probe_new_native':0,'receipts':receipts,
  'cumulative_GPU_seconds':total_prior(),'cap':None,'earlier_location_test':'proposed_not_run','OPD_qualified':False,
  'interpretation':'Source actuation controls are not no-update-mask single-factor causal proof. QR absorbs the gate, so equivalent latent changes may be large. No learnability/generalization claim.',
  'pins':{str(OUT/n/'ROOT_DECISION.json'):sha(OUT/n/'ROOT_DECISION.json') for n in names}}
 write(OUT/'ROOT_CHAIN_SUMMARY.json',result)
 lines=['# 原生可控性诊断链：完成结果','',
 '同官方冻结PTD4B+B1、两个结果选出的源训练query，每例固定30步；source GT用于明确监督正控，未读target。所有原生失败与中间状态保留，不选best。',
 '', '|控制|时间源 tIoU / vIoU %|空间源 sIoU / vIoU %|','|---|---|---|']
 def fmt(x,a,b):return f"{x[a]:.6f} / {x[b]:.6f}"
 for label,key in [('B1原点','base_percent'),('旧正确mask（0步）','old_correct_mask_percent'),('旧错误mask（0步）','old_wrong_mask_percent')]:
  lines.append(f"|{label}|{fmt(cases[0][key],'tIoU','vIoU')}|{fmt(cases[1][key],'sIoU','vIoU')}|")
 for name in names:
  lines.append(f"|{name}|{fmt(cases[0]['controls'][name]['final_percent'],'tIoU','vIoU')}|{fmt(cases[1]['controls'][name]['final_percent'],'sIoU','vIoU')}|")
 lines+=['','受限坐标loss控制的空间0分来自原生格式失败，其event时间未改变。新完整词表控制只改归一化支持，不修生成token。时间命中登记GT token仍受原物理采样量化限制，不等于物理tIoU100%。',
  '', '## 解释范围','',result['interpretation'],
  '',f"新模型梯度步骤{fresh}，另24次保存梯度Adam算术重建；所有加载/失败/收尾实测入账，累计{result['cumulative_GPU_seconds']:.9f}秒，cap=null。",'',
  '旧16源oracle和所有正负保留。before-reader vs after-reader的同norm实验尚未执行；不能提前确认“证据太晚”或增gate收益，也未建立OPD/target效用。']
 for c in cases:
  m=c['span_magnitude'];lines.append(f"\n{c['branch']}列空间最终更新对应 ΔZ/Z L2={m['equivalent_latent_relative_L2']:.6f}，ΔF相对原F={m['token_delta_relative_to_base_F']:.6f}，约旧correct mask注入norm的{m['token_delta_to_old_correct_mask_L2_ratio']:.3f}倍。")
 (OUT/'ACTUATION_CHAIN_REPORT.md').write_text('\n'.join(lines)+'\n');print('chain summarized',result['cumulative_GPU_seconds'])
if __name__=='__main__':main()
