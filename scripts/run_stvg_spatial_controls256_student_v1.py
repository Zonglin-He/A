"""Ten frozen reference/update recipes, actual episodic full Student decode."""
from pathlib import Path
import ast,copy,gc,inspect,json,sys,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import BASE,read,put,sha,SMALL
from scripts.run_stvg_B_transfer_v1 import extent
ARM_FROM={'Full_E1':'Full_E1','A':'A','B':'B','C':'C','FULL_ORDER':'FULL_ORDER','A_ORDER':'A_ORDER','A_BT':'A','B_BT':'B','B_G10':'B','B_GR11':'B'}
def recipe(a):return'lite_backtrack'if a in ('A_BT','B_BT','B_GR11')else'lite_fullstep'
def verify():
    for p,h in read(BASE/'NATIVE_DISPATCH_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    assert read(BASE/'SPATIAL_INPUT_HANDOFF.json')['count']==256
def prepare(arm):
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.prepare_stvg_lite_reclocked_inputs002 import _anchor_from_teacher_selection
    verify();teacher=ARM_FROM[arm];assert read(BASE/'teacher'/teacher/'ROOT_READBACK.json')['status']=='pass';jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];refs=[]
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=rec['probe_alias'];c,p=load_case(rec);jj=[j for j in jobs if j['case_id']==cid];anchors=[];decisions=[];pool=read(BASE/'spatial/E1/proposals'/f'{cid}.json')if jj else None
        for j in jj:
            p=BASE/'teacher'/teacher/(j['job_id']+'.json');d=read(p);w=next(x for x in pool['windows']if x['slot']==j['center_slot']);a,reason=_anchor_from_teacher_selection(j,d,c['frame_ids'],w['source_grid_position'],teacher_output_sha256=sha(p))
            if a is not None:
                if arm in ('B_G10','B_GR11'):a=copy.deepcopy(a);a['box_cxcywh']=extent(a['box_cxcywh'])
                anchors.append(a)
            decisions.append(dict(slot=j['center_slot'],selected_candidate_id=d['parsed'].get('selected_candidate_id'),reason=reason,teacher_sha256=sha(p),GT_read=False))
        refs.append(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=c['source_case_sha256'],anchors=anchors,decisions=decisions,GT_read=False));del c;gc.collect()
    b=BASE/'trials'/arm;b.mkdir(parents=True,exist_ok=False);put(b/'REFERENCE_LOCK.json',dict(status='locked',count=256,records=refs,presence_threshold=.6,source_candidate_forbidden=True,GT_read=False));(b/'TEMPORAL_LOCK.json').write_bytes((BASE/'TEMPORAL_LOCK.json').read_bytes());d=copy.deepcopy(read(SMALL/'trials/A/DESIGN_LOCK.json'));d.update(trial_id=arm,cohort='fixed_parent128_per_domain',count=256,scope='all current controlled recipes; fixed references and original parameter settings',recipe=recipe(arm),family_reuse_from=None);put(b/'DESIGN_LOCK.json',d)
    rt=copy.deepcopy(read(SMALL/'trials/A/RUNTIME.json'));rt.update(trial_id=arm,scope=d['scope'])
    for p in [Path(__file__),BASE/'NATIVE_DISPATCH_RUNTIME.json',BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'DESIGN_LOCK.json',BASE/'teacher'/teacher/'ROOT_READBACK.json',BASE/'teacher'/teacher/'TEACHER_GLOBAL_BARRIER.json',b/'REFERENCE_LOCK.json',b/'TEMPORAL_LOCK.json',b/'DESIGN_LOCK.json']:rt['pins'][str(p.relative_to(ROOT))]=sha(p)
    put(b/'RUNTIME.json',rt);print('STUDENT256_PREPARED',arm,sum(len(r['anchors'])for r in refs))
def check(a):
    verify();r=read(BASE/'trials'/a/'RUNTIME.json')
    for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
def execute(mode,arm):
    from scripts import run_stvg_instance_student32_v1 as old
    s=inspect.getsource(old.worker if mode=='worker'else old.root);tree=ast.parse(s)
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(256),n)if type(n.value)is int and n.value==32 else n
    s=ast.unparse(ast.fix_missing_locations(Count().visit(tree)));assert s.count("'lite_fullstep'")==1;s=s.replace("'lite_fullstep'",repr(recipe(arm)))
    if mode=='root':assert s.count("bb = arr(o['boxes'])")==1;s=s.replace("bb = arr(o['boxes'])","bb = arr(o['boxes']).astype(float)")
    ns=dict(old.__dict__);ns.update(BASE=BASE,check=check,modes=lambda _:('M2',));exec(compile(s,str(Path(__file__)),'exec'),ns);ns[mode](arm)
if __name__=='__main__':
    m,a=sys.argv[1:3]
    if m=='prepare':prepare(a)
    else:execute(m,a)
