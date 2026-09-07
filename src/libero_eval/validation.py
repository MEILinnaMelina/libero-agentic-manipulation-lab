import numpy as np
from scipy.spatial.transform import Rotation
from .bootstrap import ROOT
from .io import dump
from .env import LiberoEnvAdapter, LiberoActionAdapter
from .scene import LiberoSceneAdapter
from .runner import Recorder
from .skills import Skills

def control(manifest,config):
    directory=ROOT/'runs/control_validation'
    env=LiberoEnvAdapter(manifest['tasks'][0],config)
    recorder=Recorder(directory)
    env.on_step=recorder.step
    reports=[]
    try:
        env.reset(0,config['seed'])
        action=LiberoActionAdapter(env)
        scene=LiberoSceneAdapter(env)
        dump(directory/'initial_scene.json',scene.snapshot())
        for axis in range(3):
            before=env.obs['robot0_eef_pos'].copy()
            target=before.copy();target[axis]+=.025
            ok=action.move(target,max_steps=40,tolerance=.005)
            actual=env.obs['robot0_eef_pos']-before
            reports.append({'test':'translation_'+str(axis),'expected_m':(target-before).tolist(),'actual_m':actual.tolist(),'passed':bool(ok and actual[axis]>.015)})
        before=Rotation.from_quat(env.obs['robot0_eef_quat']).as_matrix()
        target=Rotation.from_rotvec([0,0,.15]).as_matrix()@before
        action.move(env.obs['robot0_eef_pos'].copy(),target,max_steps=45)
        actual=Rotation.from_matrix(Rotation.from_quat(env.obs['robot0_eef_quat']).as_matrix()@before.T).as_rotvec()
        reports.append({'test':'rotation_world_z','expected_rad':[0,0,.15],'actual_rad':actual.tolist(),'passed':bool(actual[2]>.08)})
        apertures={}
        for value in [-1,1,-1]:
            action.hold(20,value)
            apertures[str(value)]=float(np.sum(np.abs(env.obs['robot0_gripper_qpos'])))
        reports.append({'test':'gripper_sign','apertures':apertures,'passed':apertures['-1']>apertures['1']+.03})
        skills=Skills(env)
        before_qpos=env.raw.sim.data.qpos.copy()
        rot=Rotation.from_quat(env.obs['robot0_eef_quat']).as_matrix()
        ik=skills.motion.solve(env.obs['robot0_eef_pos'],rot)
        reports.append({'test':'isolated_ik','result':ik,'passed':bool(np.array_equal(before_qpos,env.raw.sim.data.qpos) and ik['feasible'])})
        goal=next(g for g in scene.goals() if g[0].lower()=='in')
        for skill in ['grasp','lift','transport','place']:
            feedback=skills.execute({'skill':skill,'object':goal[1],'goal':goal[2],'strategy':'top'})
            reports.append({'test':skill,**feedback})
        dump(directory/'final_scene.json',scene.snapshot())
    finally:
        dump(directory/'results.json',reports)
        recorder.close();env.close()
    print(reports,flush=True)
