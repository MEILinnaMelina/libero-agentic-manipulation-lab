"""All evaluated dynamics go through env.step; no state editing after reset."""
import time
import numpy as np
from .bootstrap import ROOT, configure

class StepBudget(RuntimeError):
    pass

class OfficialSuccess(RuntimeError):
    """Stop exactly at the first official successful policy action."""
    pass

class LiberoEnvAdapter:
    def __init__(self, task, config):
        configure()
        from libero.libero.envs import OffScreenRenderEnv
        self.config, self.task = config, task
        # MuJoCo 2.3's automatic stack size scales quadratically with njmax=5000,
        # reserving >1GB per MjData for these small scenes. Bound workspace allocation
        # only: contact/constraint capacities and all physical/controller values stay intact.
        from robosuite.utils.binding_utils import MjSim
        import xml.etree.ElementTree as ET
        original=MjSim.__dict__['from_xml_string']
        loader=MjSim.from_xml_string
        def bounded_load(xml):
            tree=ET.fromstring(xml)
            size=tree.find('size')
            if size is None: size=ET.SubElement(tree,'size')
            size.set('nstack',str(config.get('mujoco_stack_doubles',2000000)))
            return loader(ET.tostring(tree,encoding='unicode'))
        MjSim.from_xml_string=staticmethod(bounded_load)
        try:
            self.env = OffScreenRenderEnv(bddl_file_name=str(ROOT/task['bddl_path']), camera_heights=config['camera_size'], camera_widths=config['camera_size'], control_freq=config['control_freq'], horizon=config['horizon'], hard_reset=False)
        finally:
            MjSim.from_xml_string=original
        self.raw = self.env.env
        self.steps = 0
        self.settle_steps = 0
        self.on_step = None
        self.obs = None
        self.started = time.monotonic()

    def normalize_observation(self):
        from scipy.spatial.transform import Rotation
        # robosuite 1.4 combines grip-site position with WRIST-body quaternion.
        # OSC and IK both operate on the grip site; use its rotation consistently.
        self.obs['robot0_wrist_quat']=self.obs['robot0_eef_quat'].copy()
        site=self.raw.robots[0].eef_site_id
        self.obs['robot0_eef_quat']=Rotation.from_matrix(self.raw.sim.data.site_xmat[site].reshape(3,3)).as_quat()

    def reset(self, state_id, seed):
        import torch
        self.env.seed(seed)
        # Avoid upstream ControlEnv.reset's unbounded retry loop.
        from robosuite.utils.errors import RandomizationError
        for attempt in range(5):
            try:
                self.obs = self.raw.reset()
                break
            except RandomizationError:
                if attempt == 4:
                    raise
        states = torch.load(str(ROOT/self.task['init_path']), map_location='cpu')
        self.obs = self.env.set_init_state(np.asarray(states[state_id]))
        self.steps = self.settle_steps = 0
        self.started = time.monotonic()
        for _ in range(self.config['settle_steps']):
            self.obs, _, done, _ = self.env.step(np.asarray(self.config['settle_action'], dtype=float))
            self.normalize_observation()
            self.settle_steps += 1
            if self.on_step:
                self.on_step(self, self.config['settle_action'], done, 'settle')
        return self.obs

    def step(self, action):
        if self.steps >= self.config['max_env_steps']:
            raise StepBudget('step_budget')
        if time.monotonic()-self.started > self.config['max_episode_seconds']:
            raise StepBudget('wall_time_budget')
        a = np.asarray(action, dtype=float)
        lo,hi = self.raw.action_spec
        if a.shape != (7,) or not np.isfinite(a).all() or np.any(a<lo-1e-6) or np.any(a>hi+1e-6):
            raise ValueError('Invalid OSC_POSE action')
        self.obs, reward, done, info = self.env.step(a)
        self.normalize_observation()
        self.steps += 1
        data=self.raw.sim.data._data
        if any(int(w.number)>0 for w in data.warning):
            raise RuntimeError('MuJoCo warning/constraint capacity/numerical failure')
        if bool(done) != self.success():
            raise RuntimeError('Unexpected done/check_success disagreement')
        if self.on_step:
            self.on_step(self, a, done, 'action')
        if done:
            raise OfficialSuccess('success')
        return self.obs, reward, done, info

    def success(self):
        return bool(self.env.check_success())

    def render(self):
        return {name: np.ascontiguousarray(self.obs[name+'_image'][::-1]) for name in ['agentview','robot0_eye_in_hand']}

    def close(self):
        self.env.close()

class LiberoActionAdapter:
    def __init__(self, env):
        self.env = env
        self.controller = env.raw.robots[0].controller
        c = self.controller
        if c.name != 'OSC_POSE' or not c.use_delta:
            raise ValueError('Expected delta OSC_POSE')
        self.gripper = -1.

    def action(self, position, rotation=None, gripper=None):
        from scipy.spatial.transform import Rotation
        c = self.controller
        # OSC uses world-frame translation and left-multiplied axis-angle rotation.
        delta = np.zeros(6)
        delta[:3] = np.asarray(position)-self.env.obs['robot0_eef_pos']
        if rotation is not None:
            current = Rotation.from_quat(self.env.obs['robot0_eef_quat']).as_matrix()
            delta[3:] = Rotation.from_matrix(np.asarray(rotation) @ current.T).as_rotvec()
        lo,hi = np.asarray(c.output_min),np.asarray(c.output_max)
        imin,imax = np.asarray(c.input_min),np.asarray(c.input_max)
        normalized = (np.clip(delta,lo,hi)-(lo+hi)/2)/((hi-lo)/2)*(imax-imin)/2+(imax+imin)/2
        if gripper is not None:
            self.gripper = float(gripper)
        return np.r_[normalized, self.gripper]

    def move(self, target, rotation=None, gripper=None, max_steps=65, tolerance=.012):
        from scipy.spatial.transform import Rotation
        target = np.asarray(target)
        for _ in range(max_steps):
            self.env.step(self.action(target,rotation,gripper))
            pos_error = float(np.linalg.norm(target-self.env.obs['robot0_eef_pos']))
            ori_error = 0. if rotation is None else float(Rotation.from_matrix(np.asarray(rotation) @ Rotation.from_quat(self.env.obs['robot0_eef_quat']).as_matrix().T).magnitude())
            if self.env.success() or (pos_error<tolerance and ori_error<.10):
                return True
        return False

    def hold(self, steps=12, gripper=None):
        target = self.env.obs['robot0_eef_pos'].copy()
        for _ in range(steps):
            self.env.step(self.action(target,gripper=gripper))
