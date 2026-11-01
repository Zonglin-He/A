from pathlib import Path
import sys,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round25'))
from common25 import *
failed={}
for name in ['capture25.py','rgb25.py']:
 p=BASE/name;shutil.copy2(p,p.with_name(p.stem+'_attempt001.py'));failed[name]=sha(p)
dump(EXP/'COORDINATE_REPAIR.json',{'state':'engineering_error_detected_before_any_fusion_training','problem':'Waymo source points lacked official camera-view rotation; target-only generic sampler had been reused','affected':'R25 source features001,source labels001,and partialRGB001; not earlier R24 target or source checkpoint','invalid_for_performance':True,'source_checkpoint_unchanged':True,'no_fusion_training_or_target_scoring_had_started':True,'failed_code_sha256':failed,'old_source_seal_sha256':sha(BASE/'source_features_attempt001/SEALED.json'),'old_label_seal_sha256':sha(EXP/'source_labels_attempt001/SEALED.json'),'partial_RGB_count':len(list((BASE/'rgb_features_attempt001/samples_PRIVATE').glob('*.npz'))),'repair':'official transform_to_front_view after exact point sampling; official comma/whitespace source text formatting; entirely new002 source and RGB caches; all negative/invalid artifacts retained'})
p=BASE/'capture25.py';s=p.read_text().replace('pilot32_source_attempt001','pilot32_source_attempt002').replace('source_features_attempt001','source_features_attempt002');s=s.replace("pc,_,_=sample_points(r);text=r['caption']+' . not mentioned';mask,_=mask_for", "pc,_,_=sample_points(r)\n   if r['platform']=='waymo':\n    from utils.transform_waymo import transform_to_front_view\n    pc,_=transform_to_front_view(pc,np.array([[0.,0.,0.,1.,1.,1.,0.]]),['F','FL','FR','SL','SR'][r['suffix']]);pc=np.ascontiguousarray(pc,np.float32)\n   text=(' '.join(r['caption'].replace(',', ' ,').split()) if r['platform']=='waymo' else r['caption'])+' . not mentioned';mask,_=mask_for")
p.write_text(s)
p=BASE/'rgb25.py';s=p.read_text().replace('pilot32_source_attempt001','pilot32_source_attempt002').replace('pilot32_RGB_attempt001','pilot32_RGB_attempt002').replace('rgb_features_attempt001','rgb_features_attempt002');p.write_text(s)
p=BASE/'common25.py';s=p.read_text().replace('source_features_attempt001','source_features_attempt002').replace('rgb_features_attempt001','rgb_features_attempt002');p.write_text(s)
p=BASE/'fusion25.py';s=p.read_text().replace('source_features_attempt001','source_features_attempt002');p.write_text(s)
p=BASE/'train25.py';s=p.read_text().replace('source_features_attempt001','source_features_attempt002').replace('rgb_features_attempt001','rgb_features_attempt002').replace('source_labels_attempt001','source_labels_attempt002');p.write_text(s)
p=EXP.parent/'source_labels25.py';s=p.read_text().replace('source_features_attempt001','source_features_attempt002').replace('source_labels_attempt001','source_labels_attempt002');p.write_text(s)
p=EXP.parent/'pilot_audit25.py';s=p.read_text().replace('pilot32_source_attempt001','pilot32_source_attempt002').replace('pilot32_RGB_attempt001','pilot32_RGB_attempt002').replace("'PILOT_AUDIT.json'","'PILOT_REPAIRED_AUDIT.json'");p.write_text(s)
print('REPAIR_REGISTERED',failed)
