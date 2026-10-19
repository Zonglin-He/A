# 完整上下文Source高分辨率时间反馈：HC128

同一冻结Vid Source，原65帧/全文本/subject/模型参数不变；仅实际原RGB重入448高的同规则预处理，保留宽高比cap1.4。原224为官方设置而非错误；高分辨率是新观察尺度，不是新模型或训练。原Source/E/Live空间输出不修改，新Source框丢弃。全部RC128重复曝光开发，首条实际224控制精确恢复原Source；全部新128预测先封存后GT。

| 读出 | 完整vIoU% | tIoU% | v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| HiRes | 28.613528 | 54.809109 | +0.685054 [-1.092755,+2.524703] |
| HiResMix | 29.161192 | 54.763483 | +1.232718 [+0.322089,+2.199435] |
| RootStart | 28.588835 | 53.868713 | +0.660362 [-0.464224,+1.771472] |
| RootEnd | 28.428876 | 52.896306 | +0.500402 [-0.581881,+1.600038] |
| StudentStartEnd | 27.798936 | 51.968480 | -0.129538 [-1.374044,+1.155770] |
| NativeFull | 28.172563 | 51.517960 | +0.244089 [-1.336667,+1.996112] |
| NativeFluct | 28.563299 | 52.070928 | +0.634825 [-0.888002,+2.301143] |
| HiResFull | 28.364432 | 54.766258 | +0.435958 [-1.771798,+2.735427] |
| HiResFluct | 29.484879 | 56.462219 | +1.556405 [-0.561314,+3.756410] |

HiResMix固定.5；RootStart替换start marginal但保留旧end conditional，RootEnd相反；StudentStartEnd在初始学生实际start条件下读取新teacher的end分布，空行回退原。Full/Fluct为既有同公式。所有原65决策不变，无GT在线选择或按视频拼赢家。条件读出本身非OPD，prefix和分布期望GT诊断也非部署成绩；不能将其增量与点读出增量相加。

```json
{
  "primary": "HiRes",
  "primary_delta": {
    "mean": 0.6850541291917636,
    "CI95": [
      -1.092754667706205,
      2.5247029036168263
    ],
    "parents": 128
  },
  "selected_development_route": "HiResFluct",
  "selected_delta": {
    "mean": 1.5564050797405091,
    "CI95": [
      -0.5613143937652367,
      3.756410372830896
    ],
    "parents": 128
  },
  "decision": "no_supported_meaningful_point_readout",
  "teacher_distribution_diagnostics": {
    "HiRes": {
      "v": {
        "mean": 0.3931735270155168,
        "CI95": [
          -1.1092696011978742,
          1.9116666679023444
        ],
        "parents": 128
      },
      "t": {
        "mean": 2.1658824126293386,
        "CI95": [
          -0.2235127845909632,
          4.597336300216631
        ],
        "parents": 128
      }
    },
    "HiResMix": {
      "v": {
        "mean": 0.1965867635077584,
        "CI95": [
          -0.5546348005989372,
          0.9558333339511718
        ],
        "parents": 128
      },
      "t": {
        "mean": 1.0829412063146677,
        "CI95": [
          -0.111756392295481,
          2.2986681501083135
        ],
        "parents": 128
      }
    },
    "RootStart": {
      "v": {
        "mean": 0.0834468118388923,
        "CI95": [
          -0.9093146770181076,
          1.0446136421756602
        ],
        "parents": 128
      },
      "t": {
        "mean": 1.1163716502770167,
        "CI95": [
          -0.436254728842251,
          2.697061809144912
        ],
        "parents": 128
      }
    },
    "RootEnd": {
      "v": {
        "mean": 0.3544292545184388,
        "CI95": [
          -0.5878032134173663,
          1.3248892459731125
        ],
        "parents": 128
      },
      "t": {
        "mean": 1.1973532702206513,
        "CI95": [
          -0.39585583517470746,
          2.875716265251114
        ],
        "parents": 128
      }
    },
    "StudentStartEnd": {
      "v": {
        "mean": 1.393539535461459,
        "CI95": [
          0.24625280851218626,
          2.540650261412755
        ],
        "parents": 128
      },
      "t": {
        "mean": 3.07357820471801,
        "CI95": [
          1.2871362257785888,
          4.92586833985542
        ],
        "parents": 128
      }
    }
  },
  "point_readout_failure_does_not_prove_OPD_failure": true,
  "not_OPD": true,
  "independent_confirmation": false
}
```

原RGB全部高于224，83至少448；其余448仍包含一定上采样，不能称每条有独立448像素。正结果亦不能区分细像素与模型scale行为，需匹配coarseRGB-upsample控制后才作该归因。七route开发选优偏差保留，点CI不是选优修正或独立证明。

## 成本与核验

```json
{
  "GPU_seconds": 725.2539238778409,
  "attempts": [
    {
      "GPU_seconds": 725.2539238778409,
      "highres_Source_attempts": 128,
      "native224_Source_attempts": 1,
      "Source_offset_forwards": 258,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "network_training": 0,
      "spatial_updates": 0
    }
  ],
  "highres_Source_attempts": 128,
  "native224_Source_attempts": 1
}
{
  "status": "pass",
  "official_DenseTube_checks": 5120,
  "spatial_arrays_exact": 512,
  "Source_CDF_checks": 128,
  "direct_MBR_utilities": 13819,
  "vectorized_quality_checks": 9728,
  "max_error": 7.771561172376096e-16,
  "native_actual224_control": {
    "status": "pass",
    "alias": "RC001",
    "batch224_exact": true,
    "Source_boxes_logits_q_exact": true,
    "seconds": 4.358014374040067,
    "GT_read": false
  },
  "max_GPU_allocated_bytes": 8672167936,
  "native_metrics_exact": true,
  "all128_before_GT": true,
  "weights_unchanged": true,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

Source448是额外实际编码，保留首条224控制/失败/重启成本，未新增DINO/Refer/训练/空间state。完整signed/unknown、全部负尾及Source额外计算见RESULTS_PUBLIC.json；所有生产/论文/研究登记保持原状。

## 补充解释控制（主结果之后添加）

StudentStartEnd的分布期望相对无条件p0同时改变了条件与教师，不能单凭它证明新教师的终点优势。另从旧p0在相同初始学生start读取NativeConditionalEnd，全部128新控制先封存再重用同GT评分；该控制是事后解释，不是预登记候选/独立确认，原预测和主选择不变。同起点下HiRes conditional相对Native conditional完整v点读出+.336016pp CI[−1.187530,+1.879930]，分布期望+.245446 CI[−.800654,+1.307019]，未建立独立终点优势。完整代码/标量见post_prefix_control.py及POSTHOC_PREFIX_CONTROL.json。冻结半数混合教师的条件目标随状态权重变化TV均值约.0503/.0505，只有target变化不等于正确性，也未运行OPD。
