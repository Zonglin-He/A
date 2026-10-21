# 内容边界CDF奖励资格

同一已曝光Vid128/八帧/32候选，固定tail=.025，零新教师/适应/参数搜索。Direct获得同样连续积分梯度；越界CDF饱和、不额外混几何惩罚。全量奖励封存后独立CPU标签评分。

{
  "status": "complete_signal_qualification",
  "queries": 128,
  "eligible_queries": 102,
  "rewards": {
    "Geometry": {
      "pair_accuracy": 0.7499521155096089,
      "spearman": 0.5898780305709308,
      "top1_IoU_regret": 0.12199660888339448,
      "GT_rectangle_coverage": 0.6122612402922519,
      "box_precision": 0.5692522269667472,
      "log_area_ratio": 0.2971831070434767,
      "area_smaller_than_GT": 0.4256185807656396
    },
    "Layout3": {
      "pair_accuracy": 0.7467712000450087,
      "spearman": 0.5784275832212136,
      "top1_IoU_regret": 0.12652465361667622,
      "GT_rectangle_coverage": 0.6104123594863284,
      "box_precision": 0.5585294178988623,
      "log_area_ratio": 0.33361960466714563,
      "area_smaller_than_GT": 0.416281512605042
    },
    "TransportDice": {
      "pair_accuracy": 0.7427400475536431,
      "spearman": 0.5848496480156758,
      "top1_IoU_regret": 0.14552464152292147,
      "GT_rectangle_coverage": 0.5876990718759012,
      "box_precision": 0.5740108344847987,
      "log_area_ratio": 0.24194061165466857,
      "area_smaller_than_GT": 0.49491348622782444
    },
    "MaskCDF": {
      "pair_accuracy": 0.7027976447969702,
      "spearman": 0.5182774231380146,
      "top1_IoU_regret": 0.12868379398305677,
      "GT_rectangle_coverage": 0.6561974492418661,
      "box_precision": 0.5418020288640863,
      "log_area_ratio": 0.4562510901425776,
      "area_smaller_than_GT": 0.36554621848739494
    },
    "TransportCDF": {
      "pair_accuracy": 0.7157908085255918,
      "spearman": 0.5469808166107261,
      "top1_IoU_regret": 0.140111709260016,
      "GT_rectangle_coverage": 0.6945281645033093,
      "box_precision": 0.5102597818471649,
      "log_area_ratio": 0.5760744345073873,
      "area_smaller_than_GT": 0.24087520424836603
    }
  },
  "qualification": {
    "MaskCDF": {
      "passed_development_signal_gate": false,
      "differences": {
        "pair_accuracy": {
          "queries": 93,
          "parents": 93,
          "mean_pp": -4.715447071263867,
          "CI95_pp": [
            -6.987532256201061,
            -2.3940832836740964
          ],
          "equal_parent_mean_pp": -4.715447071263866,
          "wins": 23,
          "losses": 70,
          "ties": 0,
          "below_minus5": 54
        },
        "top1_IoU_regret": {
          "queries": 102,
          "parents": 102,
          "mean_pp": 0.6687185099662288,
          "CI95_pp": [
            -1.6278736617886163,
            2.9325304295092716
          ],
          "equal_parent_mean_pp": 0.6687185099662288,
          "wins": 57,
          "losses": 35,
          "ties": 10,
          "below_minus5": 25
        }
      }
    },
    "TransportCDF": {
      "passed_development_signal_gate": false,
      "differences": {
        "pair_accuracy": {
          "queries": 93,
          "parents": 93,
          "mean_pp": -3.416130698401718,
          "CI95_pp": [
            -5.783012400753926,
            -1.0291939637988317
          ],
          "equal_parent_mean_pp": -3.416130698401718,
          "wins": 33,
          "losses": 60,
          "ties": 0,
          "below_minus5": 42
        },
        "top1_IoU_regret": {
          "queries": 102,
          "parents": 102,
          "mean_pp": 1.8115100376621518,
          "CI95_pp": [
            -0.4330246129639922,
            4.081594015903383
          ],
          "equal_parent_mean_pp": 1.8115100376621525,
          "wins": 54,
          "losses": 33,
          "ties": 15,
          "below_minus5": 20
        }
      }
    }
  },
  "empty_frames": {
    "MaskCDF": 0,
    "TransportCDF": 62
  },
  "groups": {
    "teacher_better": {
      "queries": 61,
      "rewards": {
        "Geometry": {
          "pair_accuracy": 0.8317732257506774,
          "spearman": 0.7594432332292956,
          "top1_IoU_regret": 0.046994641458505376,
          "GT_rectangle_coverage": 0.7198556281018587,
          "box_precision": 0.6115934133462053,
          "log_area_ratio": 0.37463716018072846,
          "area_smaller_than_GT": 0.32712724434035906
        },
        "Layout3": {
          "pair_accuracy": 0.8349945213410433,
          "spearman": 0.7624830591897575,
          "top1_IoU_regret": 0.04478151625019924,
          "GT_rectangle_coverage": 0.7202682556019139,
          "box_precision": 0.6195087653010167,
          "log_area_ratio": 0.3873363671956738,
          "area_smaller_than_GT": 0.3505464480874317
        },
        "TransportDice": {
          "pair_accuracy": 0.8178999595228411,
          "spearman": 0.737561894383923,
          "top1_IoU_regret": 0.09967860848186562,
          "GT_rectangle_coverage": 0.6554120158444633,
          "box_precision": 0.6042633920736638,
          "log_area_ratio": 0.2753234944027224,
          "area_smaller_than_GT": 0.45017564402810306
        },
        "MaskCDF": {
          "pair_accuracy": 0.7528614953680228,
          "spearman": 0.6354929745893818,
          "top1_IoU_regret": 0.10787510234811064,
          "GT_rectangle_coverage": 0.7031991086683345,
          "box_precision": 0.5579465601979213,
          "log_area_ratio": 0.48843892268467287,
          "area_smaller_than_GT": 0.30308352849336456
        },
        "TransportCDF": {
          "pair_accuracy": 0.7681882831102981,
          "spearman": 0.6616324997396211,
          "top1_IoU_regret": 0.11189140079314272,
          "GT_rectangle_coverage": 0.7542090482252665,
          "box_precision": 0.5274732192307104,
          "log_area_ratio": 0.6022076165564656,
          "area_smaller_than_GT": 0.19635050741608118
        }
      }
    },
    "teacher_not_better": {
      "queries": 41,
      "rewards": {
        "Geometry": {
          "pair_accuracy": 0.6143628471101239,
          "spearman": 0.30888426616564046,
          "top1_IoU_regret": 0.23358490188140024,
          "GT_rectangle_coverage": 0.4521817852584465,
          "box_precision": 0.5062568033290169,
          "log_area_ratio": 0.1819465889612242,
          "area_smaller_than_GT": 0.5721544715447154
        },
        "Layout3": {
          "pair_accuracy": 0.6005725533258658,
          "spearman": 0.27342136590191257,
          "top1_IoU_regret": 0.24814249213753214,
          "GT_rectangle_coverage": 0.4469682213631403,
          "box_precision": 0.46780404737370573,
          "log_area_ratio": 0.2536995433442132,
          "area_smaller_than_GT": 0.5140824622531939
        },
        "TransportDice": {
          "pair_accuracy": 0.6181893362904001,
          "spearman": 0.3257015935725902,
          "top1_IoU_regret": 0.21373459312058993,
          "GT_rectangle_coverage": 0.4869554235324306,
          "box_precision": 0.5290009317306341,
          "log_area_ratio": 0.19227339585878375,
          "area_smaller_than_GT": 0.5614746660859465
        },
        "MaskCDF": {
          "pair_accuracy": 0.619834692422083,
          "spearman": 0.32403450930432026,
          "top1_IoU_regret": 0.15964306690334248,
          "GT_rectangle_coverage": 0.586268151070779,
          "box_precision": 0.5177821163917952,
          "log_area_ratio": 0.40836187587263056,
          "area_smaller_than_GT": 0.45847851335656203
        },
        "TransportCDF": {
          "pair_accuracy": 0.6289607077852211,
          "spearman": 0.35814275028078113,
          "top1_IoU_regret": 0.1820980218570714,
          "GT_rectangle_coverage": 0.6057346545755191,
          "box_precision": 0.48464954573993874,
          "log_area_ratio": 0.5371933587758314,
          "area_smaller_than_GT": 0.3071192653890824
        }
      }
    },
    "no_GT_observation": {
      "queries": 26,
      "rewards": {
        "Geometry": {
          "pair_accuracy": null,
          "spearman": null,
          "top1_IoU_regret": null,
          "GT_rectangle_coverage": null,
          "box_precision": null,
          "log_area_ratio": null,
          "area_smaller_than_GT": null
        },
        "Layout3": {
          "pair_accuracy": null,
          "spearman": null,
          "top1_IoU_regret": null,
          "GT_rectangle_coverage": null,
          "box_precision": null,
          "log_area_ratio": null,
          "area_smaller_than_GT": null
        },
        "TransportDice": {
          "pair_accuracy": null,
          "spearman": null,
          "top1_IoU_regret": null,
          "GT_rectangle_coverage": null,
          "box_precision": null,
          "log_area_ratio": null,
          "area_smaller_than_GT": null
        },
        "MaskCDF": {
          "pair_accuracy": null,
          "spearman": null,
          "top1_IoU_regret": null,
          "GT_rectangle_coverage": null,
          "box_precision": null,
          "log_area_ratio": null,
          "area_smaller_than_GT": null
        },
        "TransportCDF": {
          "pair_accuracy": null,
          "spearman": null,
          "top1_IoU_regret": null,
          "GT_rectangle_coverage": null,
          "box_precision": null,
          "log_area_ratio": null,
          "area_smaller_than_GT": null
        }
      }
    }
  },
  "tail_probability": 0.025,
  "no_GPU": true,
  "no_adaptation": true,
  "no_parameter_search": true,
  "GT_only_after_complete_seal": true,
  "not_independent_confirmation": true
}

资格通过不等于适应收益、OPD优势或独立确认；教师分组仅诊断，不能GT门控。
