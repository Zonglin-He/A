"""Check the sealed RGB/LiDAR bridge and render local projection evidence."""
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import OUT, PLATFORMS, PRIVATE, digest, key, read, write
from geometry import calibrated_grid, calibration, pixel_rays, project_camera
from questions import lateral_axes


def run():
    panel=read(OUT/"panel_seal.json")
    assert panel["frames"]==digest(PRIVATE/"frames.json")
    assert panel["ground_objects"]==digest(PRIVATE/"candidate_truth.json")
    frames=read(PRIVATE/"frames.json")
    gold=read(PRIVATE/"candidate_truth.json")
    counts=defaultdict(list)
    for frame in frames:
        uid=frame["uid"]
        path=PRIVATE/"observations"/(uid+".npz")
        assert digest(path)==frame["observation_sha"]
        spec,size,extrinsic=calibration(frame)
        lateral_axes(extrinsic)
        with np.load(path,allow_pickle=False) as obs:
            cell=obs["cells"]
            fit=~obs["hold"]
            point=obs["camera"]
            uv=obs["uv"]
            assert len(cell)==len(np.unique(cell,axis=0))
            assert (cell==np.floor(uv.astype(float)*448/size)).all() or np.all(
                abs((uv.astype(float)*448/size).clip(0,448)-cell)<=1.0001)
            assert not ({tuple(c) for c in cell[fit]//8} & {tuple(c) for c in cell[~fit]//8})
            check=project_camera(point.astype(float),spec)
            expected=pixel_rays(uv,spec)*point[:,2:3]
            record=dict(n_fit=int(fit.sum()),n_heldout=int((~fit).sum()),projected=int(obs["projected_points"]),
                 occupied=len(point),multiple=int(obs["multi_occupied_cells"]),ambiguous_depth=int(obs["conflict_cells"]),
                 max_pixel_error=float(np.max(abs(check-uv))), max_lift_error=float(np.max(abs(expected-point))),
                 distortion_assumed_zero=spec.validated()[3])
        record.update({k:v for k,v in gold[uid].items() if k!="objects"})
        counts[frame["target"]].append(record)
    result={}
    for target,r in counts.items():
        result[target]=dict(
            frames=len(r),fit_returns=int(sum(x["n_fit"] for x in r)),heldout_returns=int(sum(x["n_heldout"] for x in r)),
            single_grid_frame_min=min(x["n_heldout"] for x in r),
            projected_before_zbuffer=int(sum(x["projected"] for x in r)),
            occupied=int(sum(x["occupied"] for x in r)),multiple=int(sum(x["multiple"] for x in r)),
            ambiguous_depth=int(sum(x["ambiguous_depth"] for x in r)),
            max_pixel_error=max(x["max_pixel_error"] for x in r),max_lift_error=max(x["max_lift_error"] for x in r),
            distortion_assumed_zero=sum(x["distortion_assumed_zero"] for x in r),
            shared_metadata_timestamp=sum(x["shared_metadata_timestamp"] for x in r),
            matching_file_stem=sum(x["rgb_lidar_filename_stem_match"] for x in r),
            has_independent_sensor_clock_fields=sum(x["independent_rgb_and_scan_clock_fields"]==2 for x in r))
    write(OUT/"bridge_audit.json",dict(platforms=result,paired_roles_verified=True,fit_holdout_block_overlap=0,
          separate_rgb_and_lidar_time_offset="not measured from the release metadata"))
    folder=PRIVATE/"review"
    folder.mkdir(exist_ok=True,mode=0o700)
    sheet=Image.new("RGB",(1136,870),"#20242a")
    font=ImageFont.truetype("DejaVuSans.ttf",16)
    for row,target in enumerate(PLATFORMS):
        shown=sorted((r for r in frames if r["target"]==target),key=lambda r:key(r["uid"],"alignment"))[:2]
        for col,frame in enumerate(shown):
            path=PRIVATE/"observations"/(frame["uid"]+".npz")
            with Image.open(frame["rgb"]) as source:
                image=source.convert("RGB")
            image.thumbnail((558,262))
            with np.load(path,allow_pickle=False) as data:
                uv=data["uv"]*np.array(image.size)/np.array(frame["size"])
                z=data["camera"][:,2]
                held=data["hold"]
            draw=ImageDraw.Draw(image)
            keep=np.arange(0,len(uv),max(1,len(uv)//2800))
            for j in keep:
                x,y=uv[j]
                color="#ffb037" if held[j] else "#70e0e2"
                draw.point((round(x),round(y)),fill=color)
            x,y=col*568,row*290
            sheet.paste(image,(x+(568-image.width)//2,y+24))
            ImageDraw.Draw(sheet).text((x+9,y+4),f"{target.upper()} / RGB and projected fit (cyan), held-out (orange)",font=font,fill="white")
    sheet.save(folder/"lidar_projection.jpg",quality=91)
    print([(t,x["frames"],x["heldout_returns"]) for t,x in sorted(result.items())])


if __name__=="__main__":
    run()
