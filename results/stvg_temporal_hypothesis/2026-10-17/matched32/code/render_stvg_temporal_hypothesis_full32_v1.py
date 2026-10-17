"""Actual final B trajectories and pseudo references on private fixed input panels."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_temporal_semantic_view32_v1 as a
from scripts.stvg_temporal_hypothesis32_v1 import BASE,read,write,sha

def main():
    src=inspect.getsource(a.render);needle="j=read(ROOT/jm[cid]['path']);frames=";assert src.count(needle)==1
    src=src.replace(needle,"j=read(ROOT/jm[cid]['path']);j=dict(j,video_path=j['selection']['hypotheses'][0]['video_path'],sampled_physical_frame_ids=j['selection']['hypotheses'][0]['sampled_physical_frame_ids']);frames=")
    def put(path,data,**kw):
        if Path(path).name=='ROOT_PIXEL_VIEW_BINDING.json':path=BASE/'ROOT_FULL_PIXEL_VIEW_BINDING.json'
        return write(path,data,**kw)
    ns=dict(a.__dict__);ns.update(BASE=BASE,write=put);exec(compile(src,__file__,'exec'),ns);ns['render']()
if __name__=='__main__':main()
