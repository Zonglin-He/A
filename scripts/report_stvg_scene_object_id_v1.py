"""Report sealed finite scene-ID measurements; no inference or method selection."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'artifacts/stvg_scene_object_id_referent_v1'
O=ROOT/'results/stvg_scene_object_id/2026-10-14'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,z):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 z=read(B/'ROOT_POSTSEAL_REPORT.json');a=read(O/'all_anonymous_results.json');t=read(B/'ROOT_TEACHER_READBACK.json');m=read(B/'method/ROOT_CPU_FIT_MATH_READBACK.json')
 assert m['status']=='pass' and m['fit_count']==24 and m['optimizer_steps']==120
 assert read(B/'ROOT_POSTSEAL_CONTROL_REUSE_READBACK.json')==z
 for k,v in z.items():
  if k!='GT_source_bindings':assert a[k]==v,k
 aliases={r['case_id']:('HC'if r['dataset']=='hc2'else'V')+f'{i%4+1:02d}'for i,r in enumerate(t['cases'])}
 public_teacher=json.loads(json.dumps(t))
 for container in ['rows','cases']:
  for r in public_teacher[container]:r['case_id']=aliases[r['case_id']]
 assert a['teacher']==public_teacher
 post=a['posthoc_readout_source_ID_NoUpdate'];pg=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in post['rows']if r['dataset']==ds]
  for arm in rr[0]['arms']:
   pg.append(dict(dataset=ds,arm=arm,metrics_percent={k:statistics.mean(r['arms'][arm][k]for r in rr)*100 for k in rr[0]['arms'][arm]}))
 save(O/'posthoc_source_ID_NoUpdate_aggregation.json',dict(status='pass',scope=post['scope'],groups=pg,new_model_or_optimizer_calls=0))
 roots=dict(status='pass',scope='finite eight previously exposed sources; actual source/candidate/teacher/math/metric readback, not universal method qualification',cases=8,teacher_calls=40,new_fits=24,new_optimizer_steps=120,new_temporal_fits=0,prior_Oracle_fits_reused=24,metric_readouts=120,independent_metric_max_abs_error=z['independent_metric_max_abs_error'],GT_used_for_method_update=False,source_parameter_state_and_full_readouts_exact=True,unchanged_CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),postseal_control_reuse_reader_sha256=sha(ROOT/'scripts/score_stvg_scene_object_id_postseal_v1.py'))
 assert roots['unchanged_CURRENT_sha256']=='146a2caf1633544a844ea33106c83c022c0bebad5f3f49a87bb5f81e6f8568d4'
 save(B/'ROOT_READBACK.json',roots)
 scalar=dict(status='pass',scope='all anonymous primary and posthoc metrics equal sealed readbacks; no private GT binding, query, RGB or model state',primary_metric_readouts=120,teacher_rows=40,candidate_rows=17,posthoc_case_rows=8,all_positive_negative_neutral_unknown_preserved=True,primary_rows_exact=True,posthoc_not_prelocked_primary_method=True)
 save(B/'ROOT_ANONYMOUS_SCALAR_READBACK.json',scalar);save(B/'PUBLIC_SCALAR_AUDIT.json',scalar)
 view=dict(status='pass',scope='root actual pixel inspection of all scene-ID controls and final decoded spatial updates',private_contacts=16,control_display_slots=160,final_update_display_slots=96,actual_root_pixel_view=True,GT_missing_positions_retained_unknown=True,private_media_excluded_from_public=True)
 save(B/'ROOT_VISUAL_REVIEW.json',view)
 decision=dict(status='finite_scene_ID_and_actual_updates_completed_mixed_evidence_no_automatic_expansion',scope='same-frame sparse student candidate-ID MVP only; one stable alternate and local positive full-tube result; no reliable bidomain OPD gain established',new_model_reference_not_independent_geometry_truth=True,source_only_cases=4,stable_nonSource_cases=1,posthoc_Source_ID_NoUpdate_requires_fresh_confirmation=True,current_deployment_changed=False,optical_flow_motion_loss_or_new_temporal_training_executed=False)
 save(B/'TECHNICAL_DECISION.json',decision)
 audit=read(B/'source_audit/LUNA_SCENE_INPUT_READBACK.json')
 compact=dict(status='pass_with_disclosed_audit_boundary_and_independent_decode_limit',root_actual_clean_decode_RGB_hash_and_original_frame_contract_verified=True,all40_input_arrays_reconstructed_by_independent_source_reader=True,source_candidates=17,strict_rejects_inherited=25,unknown_GT_not_added_to_candidates=True,Luna_boundary_deviation=audit['boundary_deviation']['description'],Luna_independent_clean_video_decode_skipped=True,root_teacher_runtime_job_and_all_response_file_hashes_verified=True,original_renderer_prelock_metadata_revision_preserved=True,private_geometry_and_paths_excluded=True)
 save(O/'source_custody_summary.json',compact)
 for n in ['ROOT_READBACK','ROOT_ANONYMOUS_SCALAR_READBACK','PUBLIC_SCALAR_AUDIT','ROOT_VISUAL_REVIEW','TECHNICAL_DECISION']:
  save(O/(n.lower()+'.json'),read(B/(n+'.json')))
 save(O/'method_math_readback.json',m)
 save(O/'actual_teacher_weight_custody.json',read(B/'ACTUAL_TEACHER_WEIGHT_CUSTODY.json'))
 cost=t['cost'];wall=read(B/'METHOD_WORKER_EXIT.json')['time']-read(B/'METHOD_LAUNCH.json')['time']
 lines=['# 同屏 Object-ID 空间验证｜固定八例真实结果','','完成40次冻结STVG-R1请求及24个Teacher参考实际fit（120步），全部full-model重新解码后才评分。每域4例均为历史暴露开发来源；0新时间fit，旧QA时间区间精确复用。这里的Teacher参考来自学生候选池，不是独立几何真值。','','## 预锁主结果','','vIoU单位为百分数。T+S为新实际空间解码与旧锁定QA区间组合。Direct仅是本次用户授权诊断，不是论文比较门槛。','','|方案|HC2|VidSTG|','|---|---:|---:|']
 for arm,label in [('F','Frozen Source'),('Projection','Saved Projection'),('T','Saved QA temporal'),('teacher_direct','Teacher Direct'),('teacher_iou','Teacher IoU OPD'),('teacher_diou','Teacher DIoU OPD'),('T_teacher_direct','QA + Teacher Direct'),('T_teacher_iou','QA + Teacher IoU OPD'),('T_teacher_diou','QA + Teacher DIoU OPD')]:
  vals=[next(g['metrics_percent']['m_vIoU']for g in z['groups']if g['dataset']==ds and g['arm']==arm)for ds in ['hc2','vidstg']];lines.append(f'|{label}|{vals[0]:.6f}|{vals[1]:.6f}|')
 lines+=['','## 候选及Teacher反馈','','原43平移运动假设18四帧合法、25整条越界拒绝，不裁剪、不用GT补候选。本轮保留Source+最多4个原顺序合法备选，共17条；4例只有Source。clean RGB同屏标稳定ID，保持原四观察位置和每帧Source宽高。冻结BF16/SDPA R1，不使用DINO/YOLO/SAM/新跟踪。非均匀真实时间以文字标注，视频encoder使用平均间隔均匀近似。','','40/40格式合法、0截断，ID换标8/8保持同几何；wrong-query 5/8偏好变化，其余不变。unmarked-none 8/8返回0，但它测试无可选ID，不表示画面没有对象。Source-removed仍可能存在正确备选，不能称负对象控制。7/8主选择Source，仅V03稳定选择ID2（旧窗口4）。C阶段触发来自GT-free稳定替代，随后固定全部8例×3算法，没有按GT筛例。','','V03完整Tube vIoU从6.006629提升为Direct8.349050、IoU8.246636、DIoU8.233786；QA+IoU从6.891628升到9.228748（+2.337121pp）。这是真实局部正结果，应保留。其原四监督位置仅一帧有GT，参考ID2该帧IoU .158103而Source .445717；其余三帧unknown，不足以断言整条参考对象错误。参考稀疏IoU与最终学生完整Tube效果必须分别评价。','','同屏提示同时改变多候选呈现、颜色/边框badge、候选上限和输出格式；不能把相对上一轮变化归因仅同屏。Source运动模板和固定宽高限制仍在，未运行光流/局部匹配/motion loss。','','## 实际更新、控制复用与事后消融','','每fit原1792 Query/LN接口、原域参数、同32 antithetic Gaussian种子、固定5步骤，无调参、GT最佳轮、carry或source cap。IoU和DIoU保留原OPD，DIoU仅替换奖励为IoU−center²/enclosingdiag²。7个Source-ID例仍按预锁主矩阵实际更新，因此存在自蒸馏漂移。','','旧24个真实GT Oracle fit精确复用，本轮0新增Oracle训练；已对旧文件SHA、Source模型/帧/输入绑定再次核验。GT Oracle仅离线可学习性控制，不算无标签方法成绩。全部120个官方/独立指标最大误差1.2212453270876722e-15，24fit/120步骤奖励、权重、likelihood、Adam、Source和final cached/full读回通过。','','附带Source-ID→NoUpdate只从现成输出作事后读出消融，在主矩阵指标读回后声明，不是预锁主方法或新确认；无新模型/优化调用。规则只用Teacher ID，不能把观察到的收益当新方法已验证。','','|事后读出（Source-ID保持F/T）|HC2|VidSTG|','|---|---:|---:|']
 for arm in ['direct','iou','diou','T_direct','T_iou','T_diou']:
  vals=[next(g['metrics_percent']['m_vIoU']for g in pg if g['dataset']==ds and g['arm']==arm)for ds in ['hc2','vidstg']];lines.append(f'|{arm}|{vals[0]:.6f}|{vals[1]:.6f}|')
 lines+=['','## 实际成本与证据边界','',f"Teacher input tokens={cost['input_tokens']}, visual={cost['visual_tokens']}, output={cost['output_tokens']}；40请求端到端{cost['end_to_end_request_seconds']:.6f}s，generate+sync {cost['model_generate_seconds']:.6f}s，GPU peak allocated={cost['GPU_peak_bytes']}bytes。上述请求时间不含cold load/官方资产hash/输入准备；学生阶段launch→exit={wall:.6f}s（包含阶段加载及读写），二者不能相加成每query部署延迟。",'', 'own cgroup 16/20/1GiB与30GiB磁盘底限，finite worker退出0，0观测OOM。Luna独立source输入审计在初始key库存时反序列化query-bearing JSON，未访问/打印/比较原Query，但因此未独立clean-video解码；这个窄审计偏差披露，根实际准备已核clean decode RGB hash，根重新核全部job/runtime/response字节与输入/数学/指标。它不是新科学或GPU故障。','','全部8例的正负neutralunknown、Teacher控制、监督/未监督帧、参数/输出位移、实际成本均附匿名JSON。16私人contact/256显示slots实际根像素查看，private RGB/Query/GT坐标/weights/state/optimizer不公开。保留Source/QA与所有历史失败，不恢复旧队列或更改CURRENT。结论仅限当前8历史例：ID提示协议可运行并出现一例局部增益；预锁OPD没有稳定双域平均收益，既不自动扩大，也不据此否定整个MLLM空间路线。']
 (ROOT/'docs/STVG_SCENE_OBJECT_ID_V1.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='pass',new_fits=24,steps=120,teacher_calls=40,posthoc_groups=pg)))
if __name__=='__main__':main()
