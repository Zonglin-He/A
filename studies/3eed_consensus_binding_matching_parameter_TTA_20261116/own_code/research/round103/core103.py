"""Same R99 fitter; only target evidence aggregation changes."""
from pathlib import Path
import sys
R103=Path(__file__).resolve().parents[2];H103=R103/'research/round103';B103=R103/'crossplatform/round103'
sys.path.insert(0,str(R103/'research/round99'));from core99 import *
FITS103=['common3','scaledmean3','shufflecommon3']
REFS103=['original_source','prepared_frozen','binding_direct','direct_C','direct_FC','direct_four','lm_FC','joint3_FC','kl3_FC','kl3_shuffle_FC']+[n+'_'+v for n in ['single0','single1','signcommon','scaledmean','shufflecommon'] for v in ['F','C','FC']]
ARMS103=REFS103+[n+'_'+v for n in FITS103 for v in ['F','C','FC']]
