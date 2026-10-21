# TubeDETR 奖励与真实decoder单步机制探针

{
  "status": "complete",
  "cpu_queries": 128,
  "network_queries": 16,
  "chart": {
    "initial": {
      "OPD_Direct_cosine": 0.6973998973667677,
      "OPD_seed_cosine": 0.7179970327451466,
      "direct_reward_gain": 0.0017207594319188502,
      "OPD_reward_gain": 0.0012579953484722257,
      "direct_gradient_fraction_outside_event": 0.5729197834626685,
      "OPD_gradient_fraction_outside_event": 0.5743390076651336
    },
    "Layout3OPD": {
      "OPD_Direct_cosine": 0.6719291616562428,
      "OPD_seed_cosine": 0.706401413228656,
      "direct_reward_gain": 0.00147213868694962,
      "OPD_reward_gain": 0.0010444117591153677,
      "direct_gradient_fraction_outside_event": 0.5800106384005688,
      "OPD_gradient_fraction_outside_event": 0.5767639630803172
    },
    "Layout3Direct": {
      "OPD_Direct_cosine": 0.6419677479673613,
      "OPD_seed_cosine": 0.7049202079283627,
      "direct_reward_gain": 0.0012794966569344979,
      "OPD_reward_gain": 0.0008837193696914862,
      "direct_gradient_fraction_outside_event": 0.5843677510447886,
      "OPD_gradient_fraction_outside_event": 0.5832144106231274
    }
  },
  "network": {
    "state_gradient_cosine": 0.6952425437048078,
    "OPD": {
      "m_vIoU_change_pp": 0.0016768423776707883,
      "m_sIoU_change_pp": 0.0012641898653996246,
      "observed_center_displacement": 0.0006138190625505728,
      "unobserved_center_displacement": 3.1278177314147205e-05
    },
    "Direct": {
      "m_vIoU_change_pp": 0.0021065322645112384,
      "m_sIoU_change_pp": 0.0016779667386506542,
      "observed_center_displacement": 0.0005986834478496669,
      "unobserved_center_displacement": 3.520392759008869e-05
    }
  },
  "annotation_only_after_probe_seals": true,
  "one_step_not_60_step_evidence": true,
  "not_a_new_reward_or_full_evaluation": true,
  "chart_gradients_not_decoder_state_gradients": true,
  "GT_groups_not_adaptation_gates": true
}

128条缓存输出空间探针和16条输入哈希选择的真实decoder单步探针；不是60步方法评价，不按结果选择query，不证明新奖励收益。
