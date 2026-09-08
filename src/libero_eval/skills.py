"""Geometric manipulation skills; all actions use the public OSC step interface."""
import numpy as np
from scipy.spatial.transform import Rotation
from .env import LiberoActionAdapter
from .scene import LiberoSceneAdapter
from .motion import MotionPlanner

class SkillFailure(RuntimeError):
    def __init__(self, code, message, diagnostics=None):
        super().__init__(message)
        self.code,self.diagnostics=code,diagnostics or {}

class Skills:
    def __init__(self, env):
        self.env=env
        self.scene=LiberoSceneAdapter(env)
        self.action=LiberoActionAdapter(env)
        self.motion=MotionPlanner(env)
        self.held_name=None
        self.hold_rotation=None
        self.last_diagnostics={}

    def allowed(self,name):
        obj=self.env.raw.get_object(name)
        return [self.env.raw.sim.model.geom_name2id(g) for g in obj.contact_geoms] if obj else []

    def require_hold(self,name):
        if self.held_name!=name or not self.scene.held(name):
            raise SkillFailure('grasp_slip','No verified two-finger hold on '+str(name))
        self.motion.carry(name)

    def move(self, pos, rot=None, grip=None, steps=65, tolerance=.012):
        if not self.action.move(pos,rot,grip,max_steps=steps,tolerance=tolerance):
            raise SkillFailure('control_error','End effector did not reach target',{'target':np.asarray(pos).tolist(),'actual':self.env.obs['robot0_eef_pos'].tolist()})

    def grasp(self,name,strategy=None):
        self.motion.carry(None)
        if name not in self.env.raw.objects_dict:
            raise SkillFailure('precondition','Cannot grasp fixture/site')
        if self.held_name and self.scene.held(self.held_name):
            raise SkillFailure('precondition','Hand already occupied')
        # Official reset contains airborne objects; observe after settling actions
        # counted within the episode budget, before committing geometric targets.
        self.action.hold(12,-1)
        low,high=self.scene.bounds(name)
        center=(low+high)/2
        # Alternate top-down closing axes; aperture determines geometric candidate order.
        widths=high-low
        # Panda finger bodies are rotated 90 degrees: closing axis is EEF x, not y.
        angles=[np.pi/2,-np.pi/2,0,np.pi] if widths[1]<widths[0] else [0,np.pi,np.pi/2,-np.pi/2]
        if strategy=='side':
            angles.reverse()
        if strategy=='handle':
            sim=self.env.raw.sim
            obj=self.env.raw.get_object(name)
            gids=[sim.model.geom_name2id(g) for g in obj.contact_geoms]
            # Prefer a narrow protruding collision part, retaining actual geometric coordinates.
            slender=[g for g in gids if .002<min(sim.model.geom_size[g])<.01 and max(sim.model.geom_size[g])<.04]
            if slender:
                bodypos,_=self.scene.pose(name)
                g=max(slender,key=lambda g:float(np.linalg.norm(sim.data.geom_xpos[g][:2]-bodypos[:2])))
                center=sim.data.geom_xpos[g].copy()
        candidates=[]
        knob=None
        rim=None
        if ('bowl' in name or 'mug' in name) and strategy!='handle':
            bodypos,_=self.scene.pose(name)
            rim=np.array([bodypos[0]+(high[0]-low[0])*.36,bodypos[1],high[2]-.012])
            angles=[0,np.pi]
        if name.startswith('moka_pot'):
            sim=self.env.raw.sim
            gids=self.allowed(name)
            small=[g for g in gids if max(sim.model.geom_size[g])<.02]
            if small: knob=sim.data.geom_xpos[max(small,key=lambda g:sim.data.geom_xpos[g][2])].copy()
        orientations=[(yaw,tilt) for tilt in [0,.4,-.4,.7,-.7] for yaw in angles]
        for yaw,tilt in orientations:
            rot=Rotation.from_euler('z',yaw).as_matrix() @ Rotation.from_euler('x',tilt).as_matrix() @ np.diag([1,-1,-1])
            pos=center.copy()
            if strategy!='handle':
                pos[2]=low[2]+max(.013,min(widths[2]*.50,widths[2]-.012))
                if widths[2]>.08:
                    pos[2]=high[2]-.025
            if knob is not None: pos=knob.copy()
            if rim is not None: pos=rim.copy()
            if rim is not None and strategy=='side':
                bodypos,_=self.scene.pose(name)
                pos=np.array([bodypos[0],bodypos[1]-.037,high[2]-.015])
                rot=Rotation.from_euler('x',.7).as_matrix()@Rotation.from_euler('z',np.pi/2).as_matrix()@np.diag([1,-1,-1])
            approach=pos+np.array([0,0,.13])
            if rim is not None and strategy=='side':approach=pos-rot[:,2]*.13
            report=self.motion.validate_path([approach,pos],rot,self.allowed(name))
            candidates.append({'yaw':yaw,'tilt':tilt,'target':pos.tolist(),'report':report})
            if report['feasible']:
                self.last_diagnostics={'candidates':candidates}
                self.move(approach,rot,-1)
                self.move(pos,rot,-1,tolerance=.003 if widths[2]<.025 else .008)
                self.action.hold(14,gripper=1)
                if not self.scene.held(name):
                    raise SkillFailure('grasp_miss','Two-finger object contact absent',self.last_diagnostics)
                self.held_name=name
                self.hold_rotation=rot
                return
        raise SkillFailure('ik_or_collision','No feasible grasp candidate',{'candidates':candidates})

    def lift(self,name):
        self.require_hold(name)
        before=self.scene.pose(name)[0]
        pos=self.env.obs['robot0_eef_pos']+np.array([0,0,.13])
        self.move(pos,self.hold_rotation,1)
        self.require_hold(name)
        actual=self.scene.pose(name)[0]-before
        self.last_diagnostics={'object_displacement_m':actual.tolist()}
        if actual[2]<.07:
            raise SkillFailure('grasp_slip','Object failed to lift',self.last_diagnostics)

    def clear_obstruction(self,name):
        self.grasp(name)
        self.lift(name)
        start=self.env.obs['robot0_eef_pos'].copy()
        points=[self.scene.pose(n)[0] for n in self.env.raw.objects_dict if n!=name]
        choices=[np.array([x,y,start[2]]) for x in [-.18,.18] for y in [-.32,.12,.25]]
        free=[p for p in choices if min(np.linalg.norm((p-q)[:2]) for q in points)>.13]
        target=min(free or choices,key=lambda p:np.linalg.norm(p-start))
        self.move(target,self.hold_rotation,1)
        target[2]-=.12
        self.move(target,self.hold_rotation,1)
        self.action.hold(14,-1)
        self.held_name=None; self.motion.carry(None)
        self.move(target+np.array([0,0,.15]),self.hold_rotation,-1)

    def placement(self,name,goal):
        if goal not in self.env.raw.object_states_dict:
            raise SkillFailure('precondition','Unknown target region '+str(goal))
        low,high=self.scene.bounds(goal)
        target=(low+high)/2
        objlow,objhigh=self.scene.bounds(name)
        objpos,_=self.scene.pose(name)
        bottom_offset=objpos[2]-objlow[2]
        state=self.env.raw.object_states_dict[goal]
        if state.object_state_type=='site':
            target,_=self.scene.pose(goal)
            target=target.copy()
            target[2]=max(low[2]+bottom_offset+.012,target[2])
        else:
            # Containment objects have in_box; use their interior lower surface.
            is_in=any(g[0].lower()=='in' and g[1]==name and g[2]==goal for g in self.scene.goals())
            target[2]=(low[2]+.025 if is_in else high[2]) + bottom_offset+.015
        # Avoid already placed movable objects with deterministic candidate offsets.
        choices=[np.array([0,0,0]),np.array([0,.04,0]),np.array([0,-.04,0]),np.array([.04,0,0]),np.array([-.04,0,0])]
        parent=getattr(state,'parent_name',goal)
        occupied=[self.scene.pose(n)[0] for n in self.env.raw.objects_dict if n not in (name,goal,parent)]
        occupied=[p for p in occupied if np.all(p[:2]>=low[:2]) and np.all(p[:2]<=high[:2]) and abs(p[2]-target[2])<.15]
        margin=(objhigh[:2]-objlow[:2])/2+.012
        valid=[o for o in choices if np.all(target[:2]+o[:2]>=low[:2]+margin) and np.all(target[:2]+o[:2]<=high[:2]-margin)] or [choices[0]]
        offset=max(valid,key=lambda o:min([np.linalg.norm((target+o-p)[:2]) for p in occupied] or [1.]))
        siblings=[g[1] for g in self.scene.goals() if len(g)==3 and g[0].lower()=='on' and g[2]==goal and g[1].startswith('moka_pot')]
        if name in siblings and len(siblings)==2:
            # Old 8 cm center spacing overlapped ~8.1 cm bodies. Reserve positive
            # body clearance even when the model chooses the opposite pot order.
            half_width=max((self.scene.bounds(n)[1][0]-self.scene.bounds(n)[0][0])/2 for n in siblings)
            half_spacing=min((high[0]-low[0])/2-.015,max(.055,half_width+.014))
            offset=np.array([-half_spacing if siblings.index(name)==0 else half_spacing,0.,0.])
        target+=offset
        if 'microwave' in goal:
            parentpos,_=self.scene.pose(parent)
            target[0]=parentpos[0]-.04
            target[1]=low[1]+.055
            target[2]=low[2]+bottom_offset+.018
        ee_offset=self.env.obs['robot0_eef_pos']-objpos
        return target+ee_offset

    def transport(self,name,goal):
        self.require_hold(name)
        if 'book' in name:
            sim=self.env.raw.sim;g=max(self.allowed(name),key=lambda g:np.prod(sim.model.geom_size[g]))
            axes=sim.data.geom_xmat[g].reshape(3,3)
            extent=sim.model.geom_size[g]*np.linalg.norm(axes[:2,:],axis=0)
            direction=axes[:,np.argmax(extent)]
            a,b=self.scene.bounds(goal); desired=np.pi/2 if (b-a)[1]>(b-a)[0] else 0.
            angle=(desired-np.arctan2(direction[1],direction[0])+np.pi/2)%np.pi-np.pi/2
            self.hold_rotation=Rotation.from_euler('z',angle).as_matrix()@self.hold_rotation
            self.move(self.env.obs['robot0_eef_pos']+np.array([0,0,.04]),self.hold_rotation,1)
            self.require_hold(name)
        target=self.placement(name,goal)
        current=self.env.obs['robot0_eef_pos'].copy()
        if 'microwave' in goal:
            # Approach through the front opening; descending over the roof is blocked.
            target=self.placement(name,goal)
            low,_=self.scene.bounds(goal)
            target[1]=low[1]-.18
            target[2]+= .08
            self.move(target,self.hold_rotation,1)
            self.require_hold(name)
            return
        target[2]=max(target[2]+(.23 if name.startswith('moka_pot') else .14),current[2])
        mid=current.copy(); mid[2]=target[2]
        report=self.motion.validate_path([mid,(mid+target)/2,target],self.hold_rotation,self.allowed(name))
        if not report['feasible']:
            raise SkillFailure('ik_or_collision','Transport path rejected',report)
        self.move(mid,self.hold_rotation,1)
        self.move(target,self.hold_rotation,1)
        self.require_hold(name)

    def place(self,name,goal):
        self.require_hold(name)
        target=self.placement(name,goal)
        if 'microwave' in goal:
            pre=target.copy();pre[1]=self.env.obs['robot0_eef_pos'][1]
            self.move(pre,self.hold_rotation,1,tolerance=.008)
        # Allow bounded vertical support contact before opening the fingers.
        reached=self.action.move(target,self.hold_rotation,1,max_steps=45,tolerance=.015)
        error=self.env.obs['robot0_eef_pos']-target
        satisfied=[s for s in self.scene.goal_status() if len(s['predicate'])==3 and s['predicate'][1:]==[name,goal]]
        in_region=bool(satisfied) and all(s['satisfied'] for s in satisfied)
        if not reached and not in_region and not (np.linalg.norm(error[:2])<.02 and -.01<error[2]<.04):
            raise SkillFailure('control_error','Placement approach blocked',{'target':target.tolist(),'actual':self.env.obs['robot0_eef_pos'].tolist()})
        self.action.hold(14,gripper=-1)
        self.held_name=None
        self.motion.carry(None)
        retreat=self.env.obs['robot0_eef_pos']-self.hold_rotation[:,2]*.12 if 'microwave' in goal else target+np.array([0,0,.11])
        self.move(retreat,self.hold_rotation,-1)
        self.action.hold(12,-1)
        states=[s for s in self.scene.goal_status() if len(s['predicate'])==3 and s['predicate'][1:]==[name,goal]]
        if states and not all(s['satisfied'] for s in states):
            raise SkillFailure('placement_unstable','Official placement predicate false after release',{'goals':states})

    def articulate(self,name,skill):
        self.motion.carry(None)
        raw=self.env.raw; sim=raw.sim
        if self.held_name and self.scene.held(self.held_name):
            raise SkillFailure('precondition','Release held object before mechanism interaction')
        state=raw.object_states_dict[name]
        obj=raw.object_sites_dict[name] if state.object_state_type=='site' else raw.get_object(name)
        parent=raw.get_object(getattr(state,'parent_name',name))
        joints=[sim.model.joint_name2id(j) for j in obj.joints if int(sim.model.jnt_type[sim.model.joint_name2id(j)]) in (2,3)]
        if not joints:
            raise SkillFailure('precondition','No matching articulated joint')
        # Site regions scope drawers to one joint; appliance objects usually have one hinge.
        j=joints[0]
        q=float(sim.data.qpos[sim.model.jnt_qposadr[j]])
        key='default_turnon_ranges' if skill=='turn_on' else ('default_open_ranges' if skill.startswith('open') else 'default_close_ranges')
        limits=parent.object_properties['articulation'][key]
        target=float(np.mean(limits))
        axis=sim.data.xaxis[j].copy(); anchor=sim.data.xanchor[j].copy()
        body=int(sim.model.jnt_bodyid[j])
        children={body}
        for b in range(body+1,sim.model.nbody):
            if int(sim.model.body_parentid[b]) in children:
                children.add(b)
        geoms=[g for g in range(sim.model.ngeom) if int(sim.model.geom_bodyid[g]) in children and (sim.model.geom_contype[g] or sim.model.geom_conaffinity[g])]
        if not geoms:
            raise SkillFailure('precondition','Mechanism has no contact geoms')
        if skill=='close_drawer':
            g=min(geoms,key=lambda g:float(np.dot(sim.data.geom_xpos[g],axis)))
            p=sim.data.geom_xpos[g].copy()-axis*.018
            z=axis*np.sin(.45)-np.array([0.,0.,1.])*np.cos(.45); x=np.cross(axis,np.array([0.,0.,1.])); y=np.cross(z,x)
            rot=np.column_stack([x,y,z])
            safe=self.env.obs['robot0_eef_pos'].copy();safe[2]+=.15;safe-=axis*.15
            self.move(safe,None,-1)
            self.move(p-axis*.055,rot,1)
            for offset in np.linspace(0,target-q+.025,25):
                self.action.move(p+axis*offset,rot,1,max_steps=6,tolerance=.009)
                if state.is_close(): return
            raise SkillFailure('mechanism_contact','Drawer push did not close',{'before':q,'after':float(sim.data.qpos[sim.model.jnt_qposadr[j]])})
        if skill=='turn_on':
            if state.turn_on(): return
            # Grasp the raised rotary fin, not the smallest base collision box.
            bodyrot=sim.data.body_xmat[body].reshape(3,3).copy()
            yaw=np.arctan2(bodyrot[1,0],bodyrot[0,0])
            rot=Rotation.from_euler('z',yaw).as_matrix()@np.diag([1,-1,-1])
            p=anchor+axis*.035
            self.move(p+np.array([0,0,.12]),rot,-1)
            self.move(p,rot,-1,tolerance=.006)
            self.action.hold(14,1)
            for angle in np.linspace(0,1.1,20)[1:]:
                r=Rotation.from_rotvec(axis*angle).as_matrix()
                self.action.move(anchor+r@(p-anchor),r@rot,1,max_steps=6,tolerance=.01)
                if state.turn_on():
                    self.action.hold(10,-1)
                    self.move(self.env.obs['robot0_eef_pos']+np.array([0,0,.12]),None,-1)
                    return
            raise SkillFailure('mechanism_contact','Rotary fin did not reach on threshold',{'before':q,'after':float(sim.data.qpos[sim.model.jnt_qposadr[j]])})
        # Handle/knob contact candidates from actual geometry, scored by narrow cross-section.
        ranked=sorted(geoms,key=lambda g:(float(np.prod(sim.model.geom_size[g])), -float(np.linalg.norm(sim.data.geom_xpos[g]-anchor))))
        reports=[]
        selected=None
        if sim.model.jnt_type[j]==2:
            g=min(geoms,key=lambda g:float(np.dot(sim.data.geom_xpos[g],axis)))
            p=sim.data.geom_xpos[g].copy()
            yaw=np.arctan2(axis[1],axis[0])
            rot=Rotation.from_euler('z',yaw).as_matrix()@np.diag([1,-1,-1])
            selected=(p,p+np.array([0,0,.13]),rot)
        for g in ([] if selected is not None else ranked[:8]):
            p=sim.data.geom_xpos[g].copy()
            rot=Rotation.from_euler('z',np.pi/2 if skill=='close_door' else 0).as_matrix()@np.diag([1,-1,-1])
            pre=p+np.array([0,0,.10])
            report=self.motion.validate_path([pre,p],rot,geoms)
            reports.append({'geom':sim.model.geom_id2name(g),'position':p.tolist(),'report':report})
            if report['feasible']:
                selected=(p,pre,rot); break
        if selected is None:
            raise SkillFailure('ik_or_collision','No reachable mechanism contact',{'candidates':reports})
        p,pre,rot=selected
        self.move(pre,rot,-1)
        self.move(p,rot,-1,tolerance=.012)
        self.action.hold(12,1)
        delta=target-q
        for v in np.linspace(0,delta,15)[1:]:
            if sim.model.jnt_type[j]==2:
                nextp=p+axis*v; nextr=rot
            else:
                r=Rotation.from_rotvec(axis*v).as_matrix()
                nextp=anchor+r@(p-anchor); nextr=r@rot
            self.move(nextp,nextr,1,steps=12,tolerance=.018)
        if skill=='close_door':
            # Contact compliance leaves a small hinge-angle lag; maintain a
            # bounded closing push until the official closed threshold is reached.
            r=Rotation.from_rotvec(axis*(delta+.06)).as_matrix()
            self.action.move(anchor+r@(p-anchor),r@rot,1,max_steps=25,tolerance=.006)
        self.action.hold(10,-1)
        self.move(self.env.obs['robot0_eef_pos']+np.array([0,0,.10]),None,-1)
        after=float(sim.data.qpos[sim.model.jnt_qposadr[j]])
        self.last_diagnostics={'joint':sim.model.joint_id2name(j),'before':q,'target':target,'after':after,'candidates':reports}
        valid=state.turn_on() if skill=='turn_on' else (state.is_open() if skill.startswith('open') else state.is_close())
        if not valid:
            raise SkillFailure('mechanism_contact','Mechanism predicate not reached',self.last_diagnostics)

    def execute(self,request):
        self.last_diagnostics={}
        skill,name,goal=request['skill'],request['object'],request['goal']
        try:
            if skill=='grasp':
                for rule in self.scene.insertion_requirements():
                    if rule['object']!=name: continue
                    if not rule['opening_ready']:
                        raise SkillFailure('precondition','Open '+rule['mechanism']+' to the insertion-ready position before grasping the mug.',rule)
                    if request['strategy']!=rule['required_grasp_strategy']:
                        raise SkillFailure('precondition','Cavity insertion requires a handle grasp, not a rim pinch.',rule)
                self.grasp(name,request['strategy'])
            elif skill=='clear_obstruction': self.clear_obstruction(name)
            elif skill=='lift': self.lift(name)
            elif skill=='transport': self.transport(name,goal)
            elif skill=='place': self.place(name,goal)
            elif skill in ('open_drawer','close_drawer','open_door','close_door','turn_on'): self.articulate(name,skill)
            elif skill!='finish': raise SkillFailure('skill_missing',skill)
            return {'request':request,'success':True,'failure_code':None,'diagnostics':self.last_diagnostics}
        except SkillFailure as exc:
            return {'request':request,'success':False,'failure_code':exc.code,'message':str(exc),'diagnostics':exc.diagnostics}

def fixed_plan(goals,scene=None):
    """Explicit BDDL-informed diagnostic baseline, never labelled GPT planning."""
    steps=[]
    def add(skill,obj,goal=''):
        steps.append({'skill':skill,'object':obj,'goal':goal,'strategy':'top' if skill=='grasp' else None})
    if scene is not None:
        cleared=set()
        for g in goals:
            if g[0].lower() not in ('in','on'): continue
            low,high=scene.bounds(g[1]); center=(low+high)/2
            if high[2]-low[2]>.025: continue
            for n in scene.raw.objects_dict:
                if n==g[1] or n in cleared: continue
                a,b=scene.bounds(n)
                if b[2]>high[2]+.12 and np.linalg.norm(((a+b)/2-center)[:2])<.125:
                    add('clear_obstruction',n);cleared.add(n)
    for goal in goals:
        if goal[0].lower()=='close':
            add('open_drawer' if 'cabinet' in goal[1] else 'open_door',goal[1])
        elif goal[0].lower()=='turnon':
            add('turn_on',goal[1])
    for goal in goals:
        if goal[0].lower() in ('in','on'):
            for skill in ['grasp','lift','transport','place']:
                add(skill,goal[1],goal[2])
                if skill=='grasp' and 'microwave' in goal[2]:steps[-1]['strategy']='handle'
    for goal in goals:
        if goal[0].lower()=='close':
            add('close_drawer' if 'cabinet' in goal[1] else 'close_door',goal[1])
    add('finish',None)
    return steps
