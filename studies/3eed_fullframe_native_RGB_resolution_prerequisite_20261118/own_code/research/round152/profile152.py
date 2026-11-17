"""Conservative bound includes retained engineering and streamed formal payload."""
from common152 import *
from storage152 import storage152
def main():
 check();assert rd(H/'PREPARATION_AUDIT.json')['status']=='PASS';used,free=storage152();pred=1798*(7*256*8+256*6*4+1500);feature=2*256*288*4+48*64*4;graphs=1024*18*12+12288*8*12;cnn=2*64*64*96*4;engineering=6*3*(feature+graphs+cnn);formal=2*3*(feature+graphs);estimate=used+pred+engineering+formal+10*1024**2;assert estimate<.18*1024**3,(estimate,.18*1024**3);dump(H/'RESOURCE_PROFILE.json',dict(status='PASS_BEFORE_MODEL',existing_payload_bytes=used,free_bytes=free,conservative_peak_bytes=estimate,prediction_uncompressed_upper_bytes=pred,retained_engineering_upper_bytes=engineering,formal_chunk_upper_bytes=formal,chunk_cases=2,cap_GiB=.18,free_floor_GiB=12,safety_GiB=1,actual_GPU_peak_to_be_recorded_in_engineering=True));print('PROFILE152',estimate,flush=True)
if __name__=='__main__':main()
