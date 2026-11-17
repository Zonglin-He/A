"""Conservative new-arm-only prediction and eight-case native-stream budget."""
from common151 import *
from storage151 import storage151

def main():
 check();assert rd(H/'PREPARATION_AUDIT.json')['status']=='PASS';used,free=storage151();pred=1798*(11*256*8+256*6*4+1500);chunk=8*(5*(256*288*4*2+48*64*4+5*1024*45*12+5*1536*8*12)+5*5*64*32*48*4);est=used+pred+chunk+30*1024**2;assert est<.4*1024**3;dump(H/'RESOURCE_PROFILE.json',dict(status='PASS_BEFORE_MODEL',existing_payload_bytes=used,free_bytes=free,conservative_peak_bytes=est,prediction_uncompressed_upper_bytes=pred,engineering_chunk_upper_bytes=chunk,chunk_cases=8,cap_GiB=.4,free_floor_GiB=12,safety_GiB=1,new_temporary_only_deleted_after_CPU_PASS=True));print('PROFILE151',est,flush=True)
if __name__=='__main__':main()
