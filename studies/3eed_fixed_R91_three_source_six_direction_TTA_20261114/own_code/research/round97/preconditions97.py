"""All registered camera prerequisites, without model execution or numeric task labels."""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round97';sys.path.insert(0,str(R/'crossplatform/round7'))
from camera_projection import CameraSpec,read_camera_allowlist
def main():
 rows={r['id']:r for p in (H/'input').glob('*.json') for r in json.loads(p.read_text())};checked={};errors=[]
 for r in rows.values():
  key=(r['metapath'],r['platform'],r['suffix'])
  if key in checked:continue
  try:
   fields=read_camera_allowlist(r['metapath']);cam=CameraSpec.from_camera_fields(r['platform'],fields,waymo_view_suffix=int(r.get('suffix') or 0) if r['platform']=='waymo' else 0,missing_distortion_policy='assume_zero_declared');checked[key]=True
  except Exception as e:errors.append({'row_id_PRIVATE':r['id'],'type':type(e).__name__,'message':str(e)})
 out={'state':'PASS_ALL_R97_REGISTERED_CAMERA_PREREQUISITES' if not errors else 'R97_CAMERA_PRECONDITION_FAILURE_REQUIRES_EXPLICIT_UNKNOWN_HANDLING','expressions':len(rows),'unique_camera_frames':len(checked),'errors_PRIVATE':errors,'no_GT_fields_used':True,'FINAL_access':False};(H/'CAMERA_PRECONDITIONS_PRIVATE.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='errors_PRIVATE'},'errors',len(errors));assert not errors
if __name__=='__main__':main()
