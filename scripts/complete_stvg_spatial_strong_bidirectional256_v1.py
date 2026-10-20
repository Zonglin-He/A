"""Execute independent spatial audit then sealed official evaluation."""
from scripts.run_stvg_spatial_strong_bidirectional256_v1 import R,D,read,sha
from scripts.audit_stvg_spatial_strong_bidirectional256_v1 import main as audit
from scripts.score_stvg_spatial_strong_bidirectional256_v1 import main as score
if __name__=='__main__':
 initial=read(D/'SUPPLEMENTAL_PRE_GT_PINS.json')['pins'];revision=read(D/'SUPPLEMENTAL_PRE_GT_PINS_revision001.json')['pins'];initial.update(revision)
 for f,h in initial.items():assert sha(R/f)==h
 audit();assert read(D/'INDEPENDENT_SPATIAL_AUDIT.json')['status']=='pass';score()
