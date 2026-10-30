"""Independent GT-free SE3 audit of sealed camera arrays only.
No captions, metadata containers, image pixels, source predictions, or GT are read.
"""
import pathlib,json,hashlib,numpy as np
R=pathlib.Path(__file__).resolve().parent;B=R.parents[2];C=B/'crossplatform/round4/source/calibration/CORRECTION'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 expected='0833a3c830566685bba8813444c948e30d9bf30d9b8a249efce406819e10a1b3';seal=C/'ADAPTER_INPUT_SEAL_v2.json';assert sha(seal)==expected;binding=json.loads(seal.read_text());assert sha(C/'camera_helper.py')==binding['helper_sha256'];assert sha(C/'sealed.json')==binding['parent_correction_seal_sha256']
 A=np.eye(4);A[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]];assert np.array_equal(A[:3,:3]@np.array([1.,0,0]),[0,0,1]) and np.array_equal(A[:3,:3]@np.array([0.,1,0]),[-1,0,0]) and np.array_equal(A[:3,:3]@np.array([0.,0,1]),[0,-1,0]);counts={};matrix_errors=[];origin_errors=[];ortho_errors=[];auxerrors=[]
 for cohort in ['dev','confirm']:
  mf=C/(cohort+'_adapter_manifest.json');assert sha(mf)==binding['manifests'][cohort]['sha256'];rows=json.loads(mf.read_text());assert len(rows)==binding['manifests'][cohort]['n']
  for row in rows:
   p=pathlib.Path(row['camera_path']);assert sha(p)==row['camera_sha256'];counts[row['platform']]=counts.get(row['platform'],0)+1
   with np.load(p,allow_pickle=False) as z:
    actual=z['source_to_camera'];T=z['source_to_raw'];K=z['image_intrinsic'];size=z['image_size_wh'];assert actual.shape==T.shape==(4,4) and K.shape[0]>=3 and np.all(size>0) and K[0,0]>0 and K[1,1]>0
    if row['platform']=='waymo':
     E=z['image_extrinsic_raw'];rr=E[:3,:3];tt=E[:3,3];ortho_errors.append(float(np.max(abs(rr.T@rr-np.eye(3)))));invE=np.eye(4);invE[:3,:3]=rr.T;invE[:3,3]=-rr.T@tt
     # Explicit rigid inverse independent from producer np.linalg.inv(E).
     wanted=A@invE@T;matrix_errors.append(float(np.max(abs(wanted-actual))));assert np.array_equal(z['waymo_camera_to_opencv'],A)
     # Source-frame camera origin: inverse(source_to_raw) applied to E's camera origin.
     invT=np.eye(4);invT[:3,:3]=T[:3,:3].T;invT[:3,3]=-T[:3,:3].T@T[:3,3];origin=(invT@np.r_[tt,1])[:3]
     projected=actual@np.r_[origin,1];origin_errors.append(float(np.max(abs(projected[:3]))))
    else:
     auxerrors.append(float(np.max(abs(actual-z['image_extrinsic']@T))))
 assert counts=={'waymo':72,'drone':12,'quad':12};assert max(matrix_errors)<1e-12 and max(origin_errors)<1e-12 and max(auxerrors)==0
 out={'status':'passed_GTfree_physical_coordinate_prerequisite','camera_input_seal_sha256':expected,'camera_arrays_checked':counts,'mapping':'source=F*vehicle; OpenCV=A*inverse(E_camera_to_vehicle)*inverse(F)*source. No global pose for vehicle-frame raw points.','A_axis_basis_fixture_passed':True,'max_R_orthogonality_error':max(ortho_errors),'max_explicit_SE3_inverse_vs_cached_transform_error':max(matrix_errors),'max_camera_origin_roundtrip_error_m':max(origin_errors),'DroneQuad_original_mapping_continuity_error':max(auxerrors),'official_primary_sources':[{'url':'https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/dataset.proto','evidence':'CameraCalibration.extrinsic documented camera-to-vehicle, pose vehicle-to-global. Independently fetched.'},{'url':'https://waymo.com/intl/fil/open/data/e2e/','evidence':'Official camera frame centered at lens, x out of lens and z upward, right handed. Independently fetched.'}],'limits':['Algebra/SE3/FOV prerequisite is not observed pixel reprojection accuracy or semantic target identification','Metadata absent lens-distortion coefficients are treated as zero per existing explicit convention; no rolling-shutter reconstruction','DroneQuad direct-E continuity is checked, no new independent camera convention established for those platforms','No teacher task correctness, caption parser accuracy, or TTA effectiveness asserted'],'target_or_caption_files_read':0,'image_pixels_or_model_inputs_read':0,'new_model_GPU_calls':0,'new_GT_scores':0,'source_hash':sha(pathlib.Path(__file__))}
 (R/'STAGE2_CAMERA_REVIEW.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
