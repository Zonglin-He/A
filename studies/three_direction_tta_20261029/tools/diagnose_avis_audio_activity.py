"""Post-seal diagnostic of a physically normalized RMS activity signal.

No predictor, parameter update, threshold tuning, or prediction modification.
The fixed -50dBFS rule and raw/video-centered scores are written before label use.
"""
import json
import hashlib
import wave
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from scipy.io import wavfile
from audit_avis_frames import rle

ROOT=Path(__file__).resolve().parents[1]


def auc(score,y):
    n=int(y.sum());m=len(y)-n
    return float((rankdata(score)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def main():
    out=ROOT/'research/avis_audio_activity_v2'
    out.mkdir(exist_ok=False)
    plan={'scope':'Post-seal exploratory signal diagnosis on the same exposed8full videos',
          'observable':'Per-second RMS across both PCM24 channels, divided by2**23;20log10 with1e-12 floor',
          'reader':'scipy WAV reader representsPCM24 left-justified in int32, so divide its output by2**31, equivalent to signed24/2**23; Python3.11 wave cannot read WAVEFORMATEXTENSIBLE',
          'scores':['raw_dBFS','dBFS minus per-video median'],
          'fixed_activity_rule':'active iff RMS_dBFS >= -50; no sweep or GT calibration',
          'labels':'Any nonempty annotated sounding-instance mask in that frame',
          'not_claimed':'An adoptedTTA method, correct object identity, causal audiovisual synchronization, or population generalization'}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2))
    inputs=json.loads((ROOT/'avis/studies/avis_full_context_v2/inputs.json').read_text())
    all_data=[];by_video=[]
    # Acquire observable before opening annotations.
    for v in inputs['videos']:
        path=ROOT/'avis/datasets/test/WAVAudios'/f"{v['video_id']}.wav"
        rate,pcm=wavfile.read(path)
        assert pcm.dtype==np.int32 and pcm.ndim==2 and pcm.shape[1]==2
        samples=pcm.astype(np.float64)/(2**31)
        rms=np.array([np.sqrt(np.mean(samples[t*rate:min((t+1)*rate,len(samples))]**2)) for t in v['frame_indices']])
        db=20*np.log10(np.maximum(rms,1e-12))
        all_data.append((v,db))
    gt=json.loads((ROOT/'research/avis_tail_recovery/test.json.recovered').read_text())
    videos={Path(v['file_names'][0]).parts[0]:v for v in gt['videos']}
    rows=[]
    for v,db in all_data:
        ref=videos[v['video_id']];annotations=[x for x in gt['annotations'] if x['video_id']==ref['id']]
        y=np.array([any(rle(x['segmentations'][t],v['height'],v['width']) is not None for x in annotations) for t in v['frame_indices']])
        predicted=db>=-50
        key=hashlib.sha256(v['video_id'].encode()).hexdigest()[:16]
        by_video.append({'video_key':key,'n':len(y),'silent_frames':int((~y).sum()),'RMS_activity_AUC':auc(db,y),
                         'fixed_gate_false_silence_on_active':int((~predicted&y).sum()),
                         'fixed_gate_detected_silence':int((~predicted&~y).sum()),
                         'dbFS_min':float(db.min()),'dbFS_median':float(np.median(db))})
        rows.extend({'video_key':key,'frame':i,'dbFS':float(z),'centered_dbFS':float(z-np.median(db)),
                     'sounding_GT':bool(active)} for i,z,active in zip(v['frame_indices'],db,y))
    y=np.array([r['sounding_GT'] for r in rows]);score=np.array([r['dbFS'] for r in rows]);centered=np.array([r['centered_dbFS'] for r in rows])
    result={'n_frames':len(rows),'silent_frames':int((~y).sum()),'raw_RMS_AUC':auc(score,y),
            'video_centered_RMS_AUC':auc(centered,y),'fixed_rule_predicted_silent':int((score< -50).sum()),
            'fixed_rule_false_silence_on_active':int(((score< -50)&y).sum()),
            'fixed_rule_detected_silence':int(((score< -50)&~y).sum()),
            'scope':plan['scope'],'no_predictions_modified':True,'per_video':by_video,'rows':rows}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','per_video']},indent=2))


if __name__=='__main__':main()
