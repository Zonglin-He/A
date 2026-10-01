# Matched best-configuration Frozen/Ours comparison

Fixed old P1/P5 sources and streams; selected v3 parameters. Historically exposed. 
All predictions were sealed before GT diagnosis. No full official-query claim or parameter reselection.

| Dataset / cohort | Frozen vIoU % | Ours vIoU % | Delta pp | Paired source 95% CI pp |
|---|---:|---:|---:|---|
| vidstg / full / corruption | 21.0217 | 22.3684 | +1.3468 | [+0.9485, +1.7582] |
| vidstg / full / clean | 21.7122 | 23.1453 | +1.4331 | [+1.0024, +1.8733] |
| vidstg / outside_tuning_sources / corruption | 21.0053 | 22.4188 | +1.4135 | [+0.9865, +1.8471] |
| vidstg / outside_tuning_sources / clean | 21.6477 | 23.0978 | +1.4501 | [+0.9976, +1.9271] |

## vidstg pipeline diagnosis

The decomposition measures inherited boxes, inherited interval and current temporal rerank separately. 
Spatial step utility fixes the original time support; current-query post-update gains are diagnostic, not online output.

### corruption

Temporal correct-support/missed-selection and correct-destroyed/rescued:
```json
{
  "0.3": {
    "scheduled": 1680,
    "correct_candidate_exists": 698,
    "missed_correct_candidate": 159,
    "no_correct_candidate": 982,
    "correct_destroyed": 57,
    "correct_rescued": 65,
    "good_teacher_bad_contributor": 50
  },
  "0.5": {
    "scheduled": 1680,
    "correct_candidate_exists": 403,
    "missed_correct_candidate": 104,
    "no_correct_candidate": 1277,
    "correct_destroyed": 33,
    "correct_rescued": 41,
    "good_teacher_bad_contributor": 20
  }
}
```

Spatial per-step support, teacher errors and harmful updates:
```json
{
  "0": {
    "cells": 1680,
    "empty_evidence": 211,
    "flat_rewards": 109,
    "loss_down_but_GT_harm": 422,
    "transitions": {
      "cells": 1680,
      "degraded": 434,
      "improved": 730,
      "unchanged": 516,
      "gross_loss_pp": 0.4464263456612454,
      "gross_gain_pp": 0.7573510510411725,
      "0.3": {
        "correct_before": 531,
        "correct_to_wrong": 13,
        "conditional_damage_rate": 0.02448210922787194,
        "wrong_before": 1149,
        "wrong_to_correct": 30
      },
      "0.5": {
        "correct_before": 291,
        "correct_to_wrong": 10,
        "conditional_damage_rate": 0.03436426116838488,
        "wrong_before": 1389,
        "wrong_to_correct": 14
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 547,
        "expert_top_has_no_correct": 3,
        "empty_evidence_with_correct_support": 35,
        "useful_top_but_update_wrong": 15
      },
      "0.5": {
        "support_correct": 299,
        "expert_top_has_no_correct": 6,
        "empty_evidence_with_correct_support": 19,
        "useful_top_but_update_wrong": 9
      }
    }
  }
}
```

### clean

Temporal correct-support/missed-selection and correct-destroyed/rescued:
```json
{
  "0.3": {
    "scheduled": 336,
    "correct_candidate_exists": 143,
    "missed_correct_candidate": 32,
    "no_correct_candidate": 193,
    "correct_destroyed": 13,
    "correct_rescued": 10,
    "good_teacher_bad_contributor": 11
  },
  "0.5": {
    "scheduled": 336,
    "correct_candidate_exists": 79,
    "missed_correct_candidate": 17,
    "no_correct_candidate": 257,
    "correct_destroyed": 6,
    "correct_rescued": 8,
    "good_teacher_bad_contributor": 4
  }
}
```

Spatial per-step support, teacher errors and harmful updates:
```json
{
  "0": {
    "cells": 336,
    "empty_evidence": 42,
    "flat_rewards": 25,
    "loss_down_but_GT_harm": 84,
    "transitions": {
      "cells": 336,
      "degraded": 85,
      "improved": 154,
      "unchanged": 97,
      "gross_loss_pp": 0.45644603980617515,
      "gross_gain_pp": 0.7439946514560641,
      "0.3": {
        "correct_before": 114,
        "correct_to_wrong": 5,
        "conditional_damage_rate": 0.043859649122807015,
        "wrong_before": 222,
        "wrong_to_correct": 3
      },
      "0.5": {
        "correct_before": 60,
        "correct_to_wrong": 3,
        "conditional_damage_rate": 0.05,
        "wrong_before": 276,
        "wrong_to_correct": 5
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 118,
        "expert_top_has_no_correct": 3,
        "empty_evidence_with_correct_support": 7,
        "useful_top_but_update_wrong": 5
      },
      "0.5": {
        "support_correct": 64,
        "expert_top_has_no_correct": 2,
        "empty_evidence_with_correct_support": 4,
        "useful_top_but_update_wrong": 3
      }
    }
  }
}
```

| hc2 / full / corruption | 29.1984 | 29.4118 | +0.2134 | [-1.0299, +1.4268] |
| hc2 / full / clean | 29.9901 | 30.3016 | +0.3115 | [-0.9127, +1.5096] |
| hc2 / outside_tuning_sources / corruption | 30.0704 | 29.4230 | -0.6474 | [-2.9974, +1.0913] |
| hc2 / outside_tuning_sources / clean | 31.0008 | 30.4026 | -0.5982 | [-3.0367, +1.2544] |

## hc2 pipeline diagnosis

The decomposition measures inherited boxes, inherited interval and current temporal rerank separately. 
Spatial step utility fixes the original time support; current-query post-update gains are diagnostic, not online output.

### corruption

Temporal correct-support/missed-selection and correct-destroyed/rescued:
```json
{
  "0.3": {
    "scheduled": 160,
    "correct_candidate_exists": 77,
    "missed_correct_candidate": 4,
    "no_correct_candidate": 83,
    "correct_destroyed": 0,
    "correct_rescued": 4,
    "good_teacher_bad_contributor": 0
  },
  "0.5": {
    "scheduled": 160,
    "correct_candidate_exists": 35,
    "missed_correct_candidate": 14,
    "no_correct_candidate": 125,
    "correct_destroyed": 8,
    "correct_rescued": 6,
    "good_teacher_bad_contributor": 3
  }
}
```

Spatial per-step support, teacher errors and harmful updates:
```json
{
  "0": {
    "cells": 160,
    "empty_evidence": 20,
    "flat_rewards": 5,
    "loss_down_but_GT_harm": 34,
    "transitions": {
      "cells": 160,
      "degraded": 34,
      "improved": 96,
      "unchanged": 30,
      "gross_loss_pp": 0.07328536046159888,
      "gross_gain_pp": 0.05975107322449654,
      "0.3": {
        "correct_before": 69,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 91,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 23,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 137,
        "wrong_to_correct": 1
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 70,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 10,
        "useful_top_but_update_wrong": 1
      },
      "0.5": {
        "support_correct": 25,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  },
  "1": {
    "cells": 140,
    "empty_evidence": 0,
    "flat_rewards": 5,
    "loss_down_but_GT_harm": 36,
    "transitions": {
      "cells": 140,
      "degraded": 36,
      "improved": 91,
      "unchanged": 13,
      "gross_loss_pp": 0.05017680725459134,
      "gross_gain_pp": 0.06310322585113923,
      "0.3": {
        "correct_before": 59,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 81,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 24,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 116,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 60,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      },
      "0.5": {
        "support_correct": 26,
        "expert_top_has_no_correct": 1,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  },
  "2": {
    "cells": 140,
    "empty_evidence": 0,
    "flat_rewards": 5,
    "loss_down_but_GT_harm": 30,
    "transitions": {
      "cells": 140,
      "degraded": 31,
      "improved": 95,
      "unchanged": 14,
      "gross_loss_pp": 0.0253360869454434,
      "gross_gain_pp": 0.06813145700901033,
      "0.3": {
        "correct_before": 59,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 81,
        "wrong_to_correct": 1
      },
      "0.5": {
        "correct_before": 24,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 116,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 60,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 26,
        "expert_top_has_no_correct": 1,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  },
  "3": {
    "cells": 140,
    "empty_evidence": 0,
    "flat_rewards": 5,
    "loss_down_but_GT_harm": 26,
    "transitions": {
      "cells": 140,
      "degraded": 27,
      "improved": 99,
      "unchanged": 14,
      "gross_loss_pp": 0.013198679089994662,
      "gross_gain_pp": 0.06100275922758299,
      "0.3": {
        "correct_before": 60,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 80,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 24,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 116,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 60,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 26,
        "expert_top_has_no_correct": 1,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  },
  "4": {
    "cells": 140,
    "empty_evidence": 0,
    "flat_rewards": 5,
    "loss_down_but_GT_harm": 34,
    "transitions": {
      "cells": 140,
      "degraded": 37,
      "improved": 90,
      "unchanged": 13,
      "gross_loss_pp": 0.014215375961888573,
      "gross_gain_pp": 0.04958947883804188,
      "0.3": {
        "correct_before": 60,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 80,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 24,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 116,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 60,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 26,
        "expert_top_has_no_correct": 1,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  },
  "5": {
    "cells": 140,
    "empty_evidence": 0,
    "flat_rewards": 5,
    "loss_down_but_GT_harm": 43,
    "transitions": {
      "cells": 140,
      "degraded": 45,
      "improved": 82,
      "unchanged": 13,
      "gross_loss_pp": 0.019313571376325736,
      "gross_gain_pp": 0.04869128885764901,
      "0.3": {
        "correct_before": 60,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 80,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 24,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 116,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 60,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 26,
        "expert_top_has_no_correct": 1,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  },
  "6": {
    "cells": 140,
    "empty_evidence": 0,
    "flat_rewards": 5,
    "loss_down_but_GT_harm": 44,
    "transitions": {
      "cells": 140,
      "degraded": 45,
      "improved": 82,
      "unchanged": 13,
      "gross_loss_pp": 0.015576869370095663,
      "gross_gain_pp": 0.05006768918938819,
      "0.3": {
        "correct_before": 60,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 80,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 24,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 116,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 60,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 26,
        "expert_top_has_no_correct": 1,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  },
  "7": {
    "cells": 140,
    "empty_evidence": 0,
    "flat_rewards": 5,
    "loss_down_but_GT_harm": 45,
    "transitions": {
      "cells": 140,
      "degraded": 45,
      "improved": 82,
      "unchanged": 13,
      "gross_loss_pp": 0.020759211553981895,
      "gross_gain_pp": 0.0460204696826597,
      "0.3": {
        "correct_before": 60,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 80,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 24,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 116,
        "wrong_to_correct": 1
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 60,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 26,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  }
}
```

### clean

Temporal correct-support/missed-selection and correct-destroyed/rescued:
```json
{
  "0.3": {
    "scheduled": 32,
    "correct_candidate_exists": 14,
    "missed_correct_candidate": 0,
    "no_correct_candidate": 18,
    "correct_destroyed": 0,
    "correct_rescued": 0,
    "good_teacher_bad_contributor": 0
  },
  "0.5": {
    "scheduled": 32,
    "correct_candidate_exists": 7,
    "missed_correct_candidate": 3,
    "no_correct_candidate": 25,
    "correct_destroyed": 1,
    "correct_rescued": 1,
    "good_teacher_bad_contributor": 1
  }
}
```

Spatial per-step support, teacher errors and harmful updates:
```json
{
  "0": {
    "cells": 32,
    "empty_evidence": 4,
    "flat_rewards": 1,
    "loss_down_but_GT_harm": 7,
    "transitions": {
      "cells": 32,
      "degraded": 7,
      "improved": 19,
      "unchanged": 6,
      "gross_loss_pp": 0.02579636433464281,
      "gross_gain_pp": 0.06075258213043854,
      "0.3": {
        "correct_before": 14,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 18,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 4,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 28,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 14,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 2,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 4,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      }
    }
  },
  "1": {
    "cells": 28,
    "empty_evidence": 0,
    "flat_rewards": 1,
    "loss_down_but_GT_harm": 8,
    "transitions": {
      "cells": 28,
      "degraded": 8,
      "improved": 18,
      "unchanged": 2,
      "gross_loss_pp": 0.16263377722516906,
      "gross_gain_pp": 0.07055799590386326,
      "0.3": {
        "correct_before": 12,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 16,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 4,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 24,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 12,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 6,
        "expert_top_has_no_correct": 1,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  },
  "2": {
    "cells": 28,
    "empty_evidence": 0,
    "flat_rewards": 1,
    "loss_down_but_GT_harm": 7,
    "transitions": {
      "cells": 28,
      "degraded": 7,
      "improved": 19,
      "unchanged": 2,
      "gross_loss_pp": 0.031657563536486036,
      "gross_gain_pp": 0.05881090805135498,
      "0.3": {
        "correct_before": 12,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 16,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 4,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 24,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 12,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 6,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 2
      }
    }
  },
  "3": {
    "cells": 28,
    "empty_evidence": 0,
    "flat_rewards": 1,
    "loss_down_but_GT_harm": 7,
    "transitions": {
      "cells": 28,
      "degraded": 7,
      "improved": 19,
      "unchanged": 2,
      "gross_loss_pp": 0.026375077398678008,
      "gross_gain_pp": 0.051837461769639756,
      "0.3": {
        "correct_before": 12,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 16,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 4,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 24,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 12,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 6,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 2
      }
    }
  },
  "4": {
    "cells": 28,
    "empty_evidence": 0,
    "flat_rewards": 1,
    "loss_down_but_GT_harm": 6,
    "transitions": {
      "cells": 28,
      "degraded": 6,
      "improved": 20,
      "unchanged": 2,
      "gross_loss_pp": 0.011634025506322244,
      "gross_gain_pp": 0.07675280747546154,
      "0.3": {
        "correct_before": 12,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 16,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 4,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 24,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 12,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 6,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 2
      }
    }
  },
  "5": {
    "cells": 28,
    "empty_evidence": 0,
    "flat_rewards": 1,
    "loss_down_but_GT_harm": 8,
    "transitions": {
      "cells": 28,
      "degraded": 8,
      "improved": 18,
      "unchanged": 2,
      "gross_loss_pp": 0.019762819203160264,
      "gross_gain_pp": 0.06505618887474855,
      "0.3": {
        "correct_before": 12,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 16,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 4,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 24,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 12,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 6,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 2
      }
    }
  },
  "6": {
    "cells": 28,
    "empty_evidence": 0,
    "flat_rewards": 1,
    "loss_down_but_GT_harm": 8,
    "transitions": {
      "cells": 28,
      "degraded": 8,
      "improved": 18,
      "unchanged": 2,
      "gross_loss_pp": 0.012817486288314316,
      "gross_gain_pp": 0.07140054827547884,
      "0.3": {
        "correct_before": 12,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 16,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 4,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 24,
        "wrong_to_correct": 1
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 12,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 6,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  },
  "7": {
    "cells": 28,
    "empty_evidence": 0,
    "flat_rewards": 1,
    "loss_down_but_GT_harm": 8,
    "transitions": {
      "cells": 28,
      "degraded": 8,
      "improved": 18,
      "unchanged": 2,
      "gross_loss_pp": 0.019278233729969826,
      "gross_gain_pp": 0.05403790351240674,
      "0.3": {
        "correct_before": 12,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 16,
        "wrong_to_correct": 0
      },
      "0.5": {
        "correct_before": 5,
        "correct_to_wrong": 0,
        "conditional_damage_rate": 0.0,
        "wrong_before": 23,
        "wrong_to_correct": 0
      }
    },
    "thresholds": {
      "0.3": {
        "support_correct": 12,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 0
      },
      "0.5": {
        "support_correct": 6,
        "expert_top_has_no_correct": 0,
        "empty_evidence_with_correct_support": 0,
        "useful_top_but_update_wrong": 1
      }
    }
  }
}
```

## Scope and next decision

The two panels differ in dataset, checkpoint and stream length/order count. Inspect negative and positive cases before a model change. 
A smaller observed delta does not prove the mechanism impossible; a positive panel does not promote the configuration. 
The previous all-query run is preserved and paused. No automatic continuation or new model is launched by this report.
