"""Write the verified other-recording table and close the local single-frame method registration."""
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STUDY='studies/3eed_single_frame_crossplatform_grounding_20261118'
ORDER=(('waymo','drone'),('waymo','quad'),('drone','waymo'),('drone','quad'),('quad','waymo'),('quad','drone'))


def read(path):
    return json.loads(path.read_text())


def atomic(path,data):
    tmp=path.with_suffix('.tmp159')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,path)


def table(which):
    with (HERE/'results'/which).open(newline='') as f:
        return list(csv.DictReader(f))


def evaluate():
    check=read(HERE/'RECORDING_SCALAR_AUDIT.json')
    assert check['status']=='PASS' and read(HERE/'RECORDING_AUDIT.json')['status']=='PASS'
    total={r['arm']:r for r in table('recording_overall.csv')}
    direction={(r['source'],r['target'],r['arm']):r for r in table('recording_directions.csv')}
    scores={k:float(v['mIoU']) for k,v in total.items()}
    gains=[]
    body=[]
    for s,t in ORDER:
        data=[direction[s,t,arm] for arm in ('old_axis','old_pca','r158_pca')]
        vals=[float(r['mIoU']) for r in data]
        gain=vals[-1]-vals[1];gains.append(gain)
        body.append(f"| {s.title()} → {t.title()} | {data[-1]['n']} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {gain:+.4f} |")
    head='## 锁定后的跨采集序列复核'
    part=[head,'',
          '另从此前已登记的普通 TRAIN 记录中取出三个目标平台各 512 条描述，每条由另外两个源模型独立预测。98 个原始采集目录（Waymo 94、Drone 2、Quad 2）与 R158 选型和附加评价的采集目录均无交集，保留了全部 1,536 条描述。规则、权重、三个对照和部署默认种子 0 均在预测前锁定。模型只读相机白名单字段；全部 3,072 条方向预测封存后，独立 CPU 程序才读取对应目标框。[锁定方案](experiment/RECORDING_CHECK.json) · [预测封存](RECORDING_PREDICTION_SEAL.json) · [传感器坐标复核](RECORDING_SCALAR_AUDIT.json)','',
          '| 源 → 目标 | 记录 | 原选框＋原朝向 | 原选框＋PCA | R158＋PCA | 选框增量（百分点） |',
          '|---|---:|---:|---:|---:|---:|',*body,
          f"| **六方向等权** | **{check['all_rows']:,}** | **{scores['old_axis']:.4f}** | **{scores['old_pca']:.4f}** | **{scores['r158_pca']:.4f}** | **{scores['r158_pca']-scores['old_pca']:+.4f}** |",'',
          f"在这批输入上，固定赢家后的 PCA 增量为 {scores['old_pca']-scores['old_axis']:+.4f} 个 mIoU 百分点。相同 PCA 下，R158 在 {sum(x>0 for x in gains)}/6 个方向提高了 mIoU；{check['improved']} 条改善、{check['worsened']} 条恶化、{check['tied']} 条持平。最终 Acc@25 为 {float(total['r158_pca']['Acc25']):.4f}%，Acc@50 为 {float(total['r158_pca']['Acc50']):.4f}%。[总体指标](results/recording_overall.csv) · [六方向三项指标](results/recording_directions.csv) · [逐采集序列](results/recording_scenes.csv) · [全部逐条结果](results/recording_all_rows.csv)",'',
          '该范围是相对 R158 选型的跨采集序列评价。Drone、Quad 的记录曾用于 R69 等早期研究。Waymo 记录曾用于 Waymo 源准备，在此仅由 Drone 或 Quad 源模型作预测。录制目录和传感器内容的检查无法认证地理地点完全独立。各列保留同一预设方法，六方向共用配置。','',
          '首次评分因两份历史清单的逗号分词约定不同而中止。对两端统一逗号和空白处理后，1,536 条描述均与原登记的物体索引匹配，其中 509 条 Waymo 描述需要这一处理。修复发生于预测封存后。模型输入、已封存的框、物体索引和分母均沿用原值。独立评分器在目标传感器坐标系复核了全部 9,216 个框。[首次评分锁](experiment/EVALUATOR_ATTEMPT001.json) · [评分修订](experiment/EVALUATOR_LOCK.json)','']
    p=HERE/'REPORT_ZH.md'
    report=p.read_text()
    if head in report:report=report[:report.index(head)]+report[report.index('## 与第二帧的关系'):]
    report=report.replace('## 与第二帧的关系','\n'.join(part)+'\n## 与第二帧的关系')
    report=report.replace('现有数字支持将单帧流程作为工作基线；面向未参与选型的目标场景仍待验证。','这 5,942 条度量同场景中的其他内容，跨采集序列证据单列于上表。')
    report=report.replace('上表按 R158 已封存、已评分的逐条记录重新汇总，沿用每条记录当时的采样种子。新入口默认种子为 0。这一批全量指标使用历史种子。','首张主结果表的 5,942 条记录使用 R158 当时封存的逐条采样种子。跨采集序列表的 3,072 条方向记录使用部署入口默认种子 0，并在本轮重新从原始传感器运行。')
    report=report.replace('；','。')
    p.write_text(report)
    result=dict(adopted_working_default='Frozen source-specific full matching, single target A, R158, sampled-A PCA',
                default_regular_direction_macro=12.970967520136298,old_regular_direction_macro=10.871772174003262,
                new_recordings=check['recording_directories_disjoint_from_R158'],new_recording_directions=check['all_rows'],
                seed0_minus_old_pca_pp=scores['r158_pca']-scores['old_pca'],seed0_pca_minus_old_axis_pp=scores['old_pca']-scores['old_axis'],
                seed0_mIoU=scores['r158_pca'],seed0_Acc25=float(total['r158_pca']['Acc25']),seed0_Acc50=float(total['r158_pca']['Acc50']),
                positive_directions=sum(x>0 for x in gains),matched_improved=check['improved'],
                matched_worsened=check['worsened'],matched_tied=check['tied'],target_updates=0,source_training=0,
                second_observation='separate registered evaluation',target_outcome_method_reselection=False,
                target_blind_method_selection=False,historically_exposed=True,certified_geographic_holdout=False,FINAL_access=False)
    atomic(HERE/'DECISION.json',result)
    return result


def register():
    result=read(HERE/'DECISION.json'); receipt=read(HERE/'publication/PUBLICATION_RECEIPT.json')
    sha=receipt['commit'];url='https://github.com/Zonglin-He/A/blob/'+sha+'/'+STUDY+'/REPORT_ZH.md'
    now=datetime.now(timezone.utc);local=now.astimezone().isoformat(timespec='seconds')
    record=read(HERE/'STATE.json')
    record.update(state='COMPLETED_PUBLISHED_MAIN_VERIFIED',recorded=local,decision='round159/DECISION.json',
                  default_regular_direction_macro=result['default_regular_direction_macro'],
                  seed0_other_recording_mIoU=result['seed0_mIoU'],other_recording_gain_vs_old_pca_pp=result['seed0_minus_old_pca_pp'],
                  public_report=url,public_commit=sha,receipt='round159/publication/PUBLICATION_RECEIPT.json')
    atomic(HERE/'STATE.json',record)
    path=ROOT/'research/STATE.json';s=read(path)
    s.update(R159=record,current_round='R159',latest_completed_round='R159',run_state='R159_COMPLETED_NO_ACTIVE_JOB',
             publication_state='R159_PUBLISHED_MAIN_VERIFIED',measurement_state='R159_RAW_SINGLE_FRAME_AND_RECORDING_EVALUATION_AUDITED',
             claim_state=f"Frozen single-frame R158/PCA is the default. Historical single-frame 12.9710%. New seed-0 recording check {result['seed0_mIoU']:.4f}%. No parameter TTA.",
             public_study=STUDY,publication_receipt=record['receipt'],updated_local=local,updated_utc=now.isoformat(),next_action='none_queued')
    d=s['directions']['crossplatform']
    d.update(state='R159_COMPLETED_SINGLE_FRAME_DEFAULT',active_round=None,active_inputs=0,active_diagnostic_round=None,
             active_report=record['report'],latest_completed_report=record['report'],latest_completed_publication=sha,
             latest_public_report=url,latest_public_commit=sha,latest_research_report='research/'+record['report'],
             current_candidate_scope='Single target frame; historical ordinary and recording-disjoint evaluations. Zero test-time parameter updates.',
             current_working_candidate_retained=True,current_robust_TTA_adopted=False,execution_open=False,FINAL_access=False)
    atomic(path,s)
    ledger=ROOT/'research/ARCHIVE.md';line=(f"R159 completed and published {sha}. Single-frame raw target LiDAR/RGB/calibration and English text → frozen source-specific R91 full_disc / A256 → R158 winner → PCA. "
        f"Ordinary 5942 mIoU 10.8718 / 12.5299 / 12.9710. Default seed zero on {result['new_recording_directions']} records from {result['new_recordings']} distinct other recording directories: "
        f"{result['seed0_mIoU']:.4f}% R158+PCA, {result['seed0_minus_old_pca_pp']:+.4f} pp versus old winner+PCA. No new source training, zero TTA or FINAL access. [Report](round159/REPORT_ZH.md) · [Entry](../crossplatform/ground_single_frame.py) · [Receipt](round159/publication/PUBLICATION_RECEIPT.json)\n")
    if line not in ledger.read_text():
        with ledger.open('a') as out:out.write('\n'+line)


if __name__=='__main__':
    if sys.argv[1:]==['register']:register()
    else:evaluate()
