"""World-frame meters/radians; quaternions exported consistently as xyzw."""
import numpy as np
from scipy.spatial.transform import Rotation

class LiberoSceneAdapter:
    def __init__(self, env):
        self.env = env
        self.raw = env.raw

    def pose(self, name):
        state = self.raw.object_states_dict[name]
        geom = state.get_geom_state()
        quat = np.asarray(geom['quat'])
        if state.object_state_type == 'object':
            quat = quat[[1,2,3,0]]
        return np.asarray(geom['pos']).copy(), Rotation.from_quat(quat).as_matrix()

    def bounds(self, name):
        sim = self.raw.sim
        if name in self.raw.object_sites_dict:
            sid = sim.model.site_name2id(name)
            pos = sim.data.site_xpos[sid].copy()
            extent = np.abs(sim.data.site_xmat[sid].reshape(3,3)) @ sim.model.site_size[sid]
            return pos-extent,pos+extent
        obj=self.raw.get_object(name)
        geoms=[sim.model.geom_name2id(n) for n in obj.contact_geoms]
        if not geoms:
            p,_=self.pose(name)
            return p-.025,p+.025
        lows,highs=[],[]
        for g in geoms:
            typ=int(sim.model.geom_type[g]); size=sim.model.geom_size[g]
            # For mesh geoms MuJoCo geom_size stores the local bounding-box half sizes.
            if typ == 2:
                size=np.repeat(size[0],3)
            elif typ in (3,5):
                size=np.array([size[0],size[0],size[1]+(size[0] if typ==3 else 0)])
            extent=np.abs(sim.data.geom_xmat[g].reshape(3,3)) @ size
            p=sim.data.geom_xpos[g]
            lows.append(p-extent); highs.append(p+extent)
        return np.min(lows,axis=0),np.max(highs,axis=0)

    def held(self, name):
        if name not in self.raw.objects_dict:
            return False
        robot=self.raw.robots[0]
        # Panda's stock helper checks only pad geoms. Thin-object fingertip
        # pinches contact finger meshes; require contact on BOTH actual fingers.
        groups=[[g for g in robot.gripper.contact_geoms if 'finger%d_'%i in g] for i in (1,2)]
        return bool(self.raw._check_grasp(groups,self.raw.get_object(name).contact_geoms))

    def goals(self):
        return self.raw.parsed_problem['goal_state']

    def goal_status(self):
        from libero.libero.envs.predicates import eval_predicate_fn
        return [{'predicate':g, 'satisfied':bool(eval_predicate_fn(g[0],*[self.raw.object_states_dict[n] for n in g[1:]]))} for g in self.goals()]

    def insertion_requirements(self):
        """Skill feasibility constraints, separate from official goal predicates."""
        result=[]
        sim=self.raw.sim
        for goal in self.goals():
            if len(goal)!=3 or goal[0].lower()!='in' or 'microwave' not in goal[2]:
                continue
            state=self.raw.object_states_dict[goal[2]]
            parent=getattr(state,'parent_name',None)
            obj=self.raw.get_object(parent)
            if obj is None: continue
            articulation=obj.object_properties.get('articulation',{})
            target=float(np.mean(articulation['default_open_ranges']))
            closed=float(np.mean(articulation['default_close_ranges']))
            joints=[sim.model.joint_name2id(j) for j in obj.joints]
            joints=[j for j in joints if int(sim.model.jnt_type[j])==3]
            if not joints: continue
            q=float(sim.data.qpos[sim.model.jnt_qposadr[joints[0]]])
            ready=bool(np.sign(target-closed)*(q-target)>=-.05)
            result.append({'object':goal[1],'goal':goal[2],'mechanism':parent,
                           'required_grasp_strategy':'handle','opening_ready':ready,
                           'preparation_skill':'open_door',
                           'reason':'Before grasping for cavity insertion, open the door to the skill target with an empty hand. A partially open door can satisfy Open but still obstruct insertion. Use the mug handle to keep the palm outside the opening.'})
        return result

    def snapshot(self):
        sim=self.raw.sim
        objects={}
        for name,state in self.raw.object_states_dict.items():
            pos,rot=self.pose(name)
            low,high=self.bounds(name)
            objects[name]={'position_m':pos.tolist(),'quaternion_xyzw':Rotation.from_matrix(rot).as_quat().tolist(), 'aabb_m':[low.tolist(),high.tolist()], 'type':state.object_state_type, 'fixed':bool(state.is_fixture), 'parent':getattr(state,'parent_name',None), 'held':self.held(name)}
        joints={}
        for j in range(sim.model.njnt):
            name=sim.model.joint_id2name(j)
            if not name or name.startswith(('robot','gripper')) or int(sim.model.jnt_type[j]) not in (2,3):
                continue
            qadr=sim.model.jnt_qposadr[j]
            joints[name]={'qpos':float(sim.data.qpos[qadr]),'type':'slide' if sim.model.jnt_type[j]==2 else 'hinge','range':sim.model.jnt_range[j].tolist(),'axis_world':sim.data.xaxis[j].tolist(),'anchor_world':sim.data.xanchor[j].tolist()}
        contacts=[]
        for c in sim.data.contact[:sim.data.ncon]:
            if c.dist <= .001:
                contacts.append([sim.model.geom_id2name(c.geom1),sim.model.geom_id2name(c.geom2)])
        return {'frame':'world','length_unit':'meter','angle_unit':'radian','objects':objects,'joints':joints,'contacts':contacts,'eef_position_m':self.env.obs['robot0_eef_pos'].tolist(),'eef_quaternion_xyzw':self.env.obs['robot0_eef_quat'].tolist(),'gripper_qpos':self.env.obs['robot0_gripper_qpos'].tolist(),'robot_qpos':self.env.obs['robot0_joint_pos'].tolist(),'goal_status':self.goal_status(),'insertion_requirements':self.insertion_requirements(),'official_success':self.env.success(),'remaining_steps':self.env.config['max_env_steps']-self.env.steps}
