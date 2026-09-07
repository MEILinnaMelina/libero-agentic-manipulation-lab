"""Multi-start damped-least-squares IK and collision checking on private MjData.

The simulation model is read-only shared; mutable dynamics state is never shared.
This replaces RoboEval dm_control/two-arm IK. Actual motion remains OSC env.step.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

class MotionPlanner:
    def __init__(self, env):
        self.env=env
        self.model=env.raw.sim.model._model
        self.data=mujoco.MjData(self.model)
        robot=env.raw.robots[0]
        self.qidx=np.asarray(robot._ref_joint_pos_indexes)
        self.vidx=np.asarray(robot._ref_joint_vel_indexes)
        self.site=robot.eef_site_id
        self.robot_geoms={g for g in range(self.model.ngeom) if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or '').startswith(('robot0','gripper0'))}
        self.finger_geoms={g for g in self.robot_geoms if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or '').startswith('gripper0')}
        self.limits=np.asarray([self.model.jnt_range[j] for j in robot._ref_joint_indexes])
        self.carried=None
        self.carried_geoms=set()

    def carry(self,name):
        self.carried=name
        obj=self.env.raw.get_object(name) if name else None
        self.carried_geoms=set(self.env.raw.sim.model.geom_name2id(g) for g in obj.contact_geoms) if obj else set()

    def forward(self):
        mujoco.mj_forward(self.model,self.data)
        if self.carried:
            sim=self.env.raw.sim
            obj=self.env.raw.get_object(self.carried)
            joint=next(j for j in obj.joints if self.model.jnt_type[sim.model.joint_name2id(j)]==0)
            adr=self.model.jnt_qposadr[sim.model.joint_name2id(joint)]
            live=sim.data._data
            oldrot=live.site_xmat[self.site].reshape(3,3)
            newrot=self.data.site_xmat[self.site].reshape(3,3)
            delta=newrot@oldrot.T
            self.data.qpos[adr:adr+3]=self.data.site_xpos[self.site]+delta@(live.qpos[adr:adr+3]-live.site_xpos[self.site])
            q=live.qpos[adr+3:adr+7][[1,2,3,0]]
            newq=Rotation.from_matrix(delta@Rotation.from_quat(q).as_matrix()).as_quat()
            self.data.qpos[adr+3:adr+7]=newq[[3,0,1,2]]
            mujoco.mj_forward(self.model,self.data)

    def sync(self):
        live=self.env.raw.sim.data._data
        mujoco.mj_copyData(self.data,self.model,live) if hasattr(mujoco,'mj_copyData') else self._copy(live)

    def _copy(self, live):
        for name in ['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied']:
            getattr(self.data,name)[:]=getattr(live,name)
        self.data.time=live.time

    def collisions(self, allowed=()):
        allowed=set(allowed)
        bad=[]
        for c in self.data.contact[:self.data.ncon]:
            a,b=int(c.geom1),int(c.geom2)
            if c.dist<-.002 and ((a in self.carried_geoms) != (b in self.carried_geoms)):
                other=b if a in self.carried_geoms else a
                if other not in self.robot_geoms:
                    bad.append({'carried_object':self.carried,'other_geom':mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,other),'penetration_m':float(-c.dist)})
            if c.dist>-.002 or not ((a in self.robot_geoms) != (b in self.robot_geoms)):
                continue
            r,o=(a,b) if a in self.robot_geoms else (b,a)
            if r in self.finger_geoms and o in allowed:
                continue
            # Fixed robot pedestal contacts with its table are not arm collisions.
            rn=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,r) or ''
            if 'link0' in rn or 'base' in rn:
                continue
            bad.append({'robot_geom':rn,'other_geom':mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,o),'penetration_m':float(-c.dist)})
        return bad

    def solve(self, position, rotation, allowed=()):
        live_before=self.env.raw.sim.data.qpos.copy()
        current=live_before[self.qidx]
        rng=np.random.RandomState(42)
        seeds=[current,np.clip(current+rng.normal(0,.15,len(current)),self.limits[:,0],self.limits[:,1])]
        reports=[]
        for seed in seeds:
            self.sync()
            self.data.qpos[self.qidx]=seed
            for _ in range(100):
                self.forward()
                pos_err=np.asarray(position)-self.data.site_xpos[self.site]
                rot_err=Rotation.from_matrix(np.asarray(rotation) @ self.data.site_xmat[self.site].reshape(3,3).T).as_rotvec()
                if np.linalg.norm(pos_err)<.003 and np.linalg.norm(rot_err)<.035:
                    break
                jp=np.zeros((3,self.model.nv)); jr=jp.copy()
                mujoco.mj_jacSite(self.model,self.data,jp,jr,self.site)
                jac=np.vstack([jp[:,self.vidx],jr[:,self.vidx]*.25])
                err=np.r_[pos_err,rot_err*.25]
                step=jac.T@np.linalg.solve(jac@jac.T+.003*np.eye(6),err)
                self.data.qpos[self.qidx]=np.clip(self.data.qpos[self.qidx]+np.clip(step,-.15,.15),self.limits[:,0],self.limits[:,1])
            self.forward()
            error=float(np.linalg.norm(np.asarray(position)-self.data.site_xpos[self.site]))
            orient=float(Rotation.from_matrix(np.asarray(rotation) @ self.data.site_xmat[self.site].reshape(3,3).T).magnitude())
            contacts=self.collisions(allowed)
            reports.append({'position_error_m':error,'orientation_error_rad':orient,'collisions':contacts})
            if error<.02 and orient<.22 and not contacts:
                assert np.array_equal(live_before,self.env.raw.sim.data.qpos),'IK changed live state'
                return {'feasible':True,'qpos':self.data.qpos[self.qidx].tolist(),'attempts':reports}
        assert np.array_equal(live_before,self.env.raw.sim.data.qpos),'IK changed live state'
        return {'feasible':False,'attempts':reports}

    def validate_path(self, positions, rotation, allowed=()):
        reports=[]
        previous=np.asarray(positions[0])
        for target in positions:
            target=np.asarray(target)
            count=max(1,int(np.ceil(np.linalg.norm(target-previous)/.04)))
            for alpha in np.linspace(0,1,count+1)[1:]:
                p=previous+(target-previous)*alpha
                report=self.solve(p,rotation,allowed)
                reports.append(report)
                if not report['feasible']:
                    return {'feasible':False,'waypoints':reports}
            previous=target
        return {'feasible':True,'waypoints':reports}
