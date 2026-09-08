import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from libero_eval.env import LiberoEnvAdapter, LiberoActionAdapter, StepBudget, OfficialSuccess
from libero_eval.planner import parse_response, PlannerError, VERSION
from libero_eval.report import wilson
from libero_eval.io import dump

def response(**kw):
    r={'status':'completed','model':'gpt-6-astra','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({'schema_version':VERSION,'thought':'Grasp first.','request':{'skill':'grasp','object':'can','goal':'basket','strategy':'top'}})}]}]}
    r.update(kw)
    return r

def test_response_rejects_truncation_and_model_substitution():
    assert parse_response(response())['request']['skill']=='grasp'
    for r in [response(status='incomplete',incomplete_details={'reason':'max_output_tokens'}),response(model='gpt-5.6-terra'),response(output=[])]:
        with pytest.raises(PlannerError):parse_response(r)

def test_low_level_authority_rejected():
    r=response();v=json.loads(r['output'][0]['content'][0]['text']);v['request']['position']=[1,2,3]
    r['output'][0]['content'][0]['text']=json.dumps(v)
    with pytest.raises(PlannerError):parse_response(r)

def test_planner_can_request_physical_obstacle_clearance():
    r=response()
    value=json.loads(r['output'][0]['content'][0]['text'])
    value['request'].update(skill='clear_obstruction',object='milk_1',goal='',strategy=None)
    r['output'][0]['content'][0]['text']=json.dumps(value)
    assert parse_response(r)['request']['skill']=='clear_obstruction'

def test_diagnostic_budget_is_separate_and_enforced(tmp_path,monkeypatch):
    from libero_eval import budget
    monkeypatch.setattr(budget,'ROOT',tmp_path)
    (tmp_path/'runs').mkdir()
    config={'calibrated':True,'pricing_usd_per_million':{'uncached_input':10,'cached_input':1,'cache_write':12.5,'output':50},'max_output_tokens':4096,'formal_budget_usd':.3,'budget_ledger':'runs/check.sqlite'}
    reservation=budget.reserve({'input':'x'},config)
    with pytest.raises(RuntimeError,match='campaign_cost_budget'):
        budget.reserve({'input':'x'},config)
    budget.settle(reservation,{'input_tokens':100,'output_tokens':10},config)
    assert budget.reserve({'input':'x'},config)
    assert not (tmp_path/'runs/budget.sqlite').exists()

def test_two_pot_targets_clear_bodies_in_either_selection_order():
    from libero_eval.skills import Skills
    names=['moka_pot_1','moka_pot_2']
    positions={names[0]:np.array([-.02,.23,.966]),names[1]:np.array([.04,.08,.966]),'cook':np.array([0.,0.,.905])}
    def bounds(name):
        half=np.array([.075,.075,.0025]) if name=='cook' else np.array([.041,.074,.076])
        return positions[name]-half,positions[name]+half
    scene=SimpleNamespace(bounds=bounds,pose=lambda n:(positions[n],np.eye(3)),goals=lambda:[['on',n,'cook'] for n in names])
    skill=Skills.__new__(Skills);skill.scene=scene
    skill.env=SimpleNamespace(raw=SimpleNamespace(object_states_dict={'cook':SimpleNamespace(object_state_type='site',parent_name='stove')},objects_dict=dict.fromkeys(names)),obs={'robot0_eef_pos':np.array([0.,0.,1.2])})
    for order in [names,list(reversed(names))]:
        targets={n:skill.placement(n,'cook')-(skill.env.obs['robot0_eef_pos']-positions[n]) for n in order}
        # Two 8.2 cm bodies need positive space, not merely distinct centers.
        assert abs(targets[names[0]][0]-targets[names[1]][0])-.082>=.02
        assert all(abs(p[0])<.075 for p in targets.values())

def test_cavity_grasp_requires_opening_clearance_and_handle_strategy():
    from libero_eval.scene import LiberoSceneAdapter
    from libero_eval.skills import Skills
    qpos=np.array([-1.432])  # Official Open is true at q < -1.3, but not ready for insertion.
    appliance=SimpleNamespace(joints=['hinge'],object_properties={'articulation':{'default_open_ranges':[-2.094,-1.3],'default_close_ranges':[-.005,0.]}})
    raw=SimpleNamespace(parsed_problem={'goal_state':[['in','mug','microwave_region']]},object_states_dict={'microwave_region':SimpleNamespace(parent_name='microwave')},get_object=lambda n:appliance,sim=SimpleNamespace(model=SimpleNamespace(joint_name2id=lambda n:0,jnt_type=[3],jnt_qposadr=[0]),data=SimpleNamespace(qpos=qpos)))
    scene=LiberoSceneAdapter(SimpleNamespace(raw=raw))
    assert not scene.insertion_requirements()[0]['opening_ready']
    skill=Skills.__new__(Skills);skill.scene=scene
    calls=[];skill.grasp=lambda *args:calls.append(args)
    req=dict(skill='grasp',object='mug',goal='microwave_region',strategy='handle')
    assert skill.execute(req)['failure_code']=='precondition' and not calls
    qpos[0]=-1.68
    assert scene.insertion_requirements()[0]['opening_ready']
    assert skill.execute(dict(req,strategy='top'))['failure_code']=='precondition' and not calls
    assert skill.execute(req)['success'] and calls==[('mug','handle')]

def test_action_scaling_and_gripper_persistence():
    controller=SimpleNamespace(name='OSC_POSE',use_delta=True,output_min=np.array([-.05]*3+[-.5]*3),output_max=np.array([.05]*3+[.5]*3),input_min=-np.ones(6),input_max=np.ones(6))
    env=SimpleNamespace(raw=SimpleNamespace(robots=[SimpleNamespace(controller=controller)]),obs={'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':np.array([0,0,0,1])})
    adapter=LiberoActionAdapter(env)
    np.testing.assert_allclose(adapter.action([.025,0,0],gripper=1),[.5,0,0,0,0,0,1])
    assert adapter.action([2,0,0])[-1]==1
    assert adapter.action([2,0,0])[0]==1

def test_step_budget_blocks_before_physics():
    env=LiberoEnvAdapter.__new__(LiberoEnvAdapter)
    env.config={'max_env_steps':600};env.steps=600
    with pytest.raises(StepBudget):env.step(np.zeros(7))

def test_official_success_stops_immediately_after_recording():
    import time
    env=LiberoEnvAdapter.__new__(LiberoEnvAdapter)
    env.config={'max_env_steps':600,'max_episode_seconds':900};env.steps=0;env.started=time.monotonic()
    env.raw=SimpleNamespace(action_spec=(-np.ones(7),np.ones(7)),robots=[SimpleNamespace(eef_site_id=0)],sim=SimpleNamespace(data=SimpleNamespace(site_xmat=np.eye(3).reshape(1,9),_data=SimpleNamespace(warning=[]))))
    env.env=SimpleNamespace(step=lambda a:({'robot0_eef_quat':np.array([0,0,0,1])},1.,True,{}),check_success=lambda:True)
    events=[];env.on_step=lambda *a:events.append(a)
    with pytest.raises(OfficialSuccess):env.step(np.zeros(7))
    assert env.steps==1 and len(events)==1

def test_observation_uses_grip_site_rotation_for_osc():
    from scipy.spatial.transform import Rotation
    env=LiberoEnvAdapter.__new__(LiberoEnvAdapter)
    site_rotation=Rotation.from_euler('z',np.pi/2).as_matrix()
    env.raw=SimpleNamespace(robots=[SimpleNamespace(eef_site_id=0)],sim=SimpleNamespace(data=SimpleNamespace(site_xmat=site_rotation.reshape(1,9))))
    env.obs={'robot0_eef_quat':np.array([0.,0.,0.,1.])}
    env.normalize_observation()
    np.testing.assert_allclose(Rotation.from_quat(env.obs['robot0_eef_quat']).as_matrix(),site_rotation,atol=1e-12)
    np.testing.assert_array_equal(env.obs['robot0_wrist_quat'],[0,0,0,1])

def test_zero_success_interval_not_zero_width():
    low,high=wilson(0,47)
    assert low<1e-10 and .07<high<.08

def test_manifest_disjoint_unique_and_complete():
    from libero_eval.bootstrap import ROOT
    from libero_eval.io import read
    manifest=read(ROOT/'configs/task_manifest.json')
    assert [t['task_id'] for t in manifest['tasks']]==list(range(10))
    for t in manifest['tasks']:
        assert not set(t['dev_state_ids'])&set(t['formal_state_ids'])
        hashes=[t['state_sha256'][i] for i in t['dev_state_ids']+t['formal_state_ids']]
        assert len(hashes)==len(set(hashes))

def test_three_state_schedule_has_exactly_thirty_distinct_pairs():
    import importlib.util
    from libero_eval.bootstrap import ROOT
    from libero_eval.io import read
    spec=importlib.util.spec_from_file_location('run_gpt_check',ROOT/'scripts/run_gpt_check.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=read(ROOT/'configs/task_manifest.json')
    schedule=module.build_schedule(manifest,list(range(10)),[4,5,6])
    pairs={(e['task_id'],e['init_state_id']) for e in schedule}
    assert len(schedule)==30 and pairs=={(t,s) for t in range(10) for s in [4,5,6]}
    for states in [[4,4,5],[0,4,5],[50],[]]:
        with pytest.raises(ValueError):module.build_schedule(manifest,[8,9],states)

def test_summary_keeps_failed_denominator_and_rejects_mixed_protocol(tmp_path,monkeypatch):
    import libero_eval.report as report
    monkeypatch.setattr(report,'ROOT',tmp_path)
    run=tmp_path/'run'
    identity={'run_id':'test','method':'gpt6','split':'formal','code_hash':'a','config_hash':'b','manifest_hash':'c','schedule':[{'task_id':0,'init_state_id':3},{'task_id':0,'init_state_id':4}], 'config':{'max_env_steps':600}}
    dump(run/'run.json',identity)
    base={'task_id':0,'init_state_id':3,'method':'gpt6','split':'formal','code_hash':'a','config_hash':'b','manifest_hash':'c','env_steps':0,'success':False,'termination_reason':'api_error','skill_failures':[],'llm_calls':1,'replans':0,'llm_input_tokens':0,'llm_output_tokens':0,'llm_latency_seconds':1.,'wall_seconds':1.}
    dump(run/'task00_state003/result.json',base)
    result=report.summarize(run)
    assert result['scheduled']==2 and result['completed']==1 and not result['complete']
    assert result['per_task'][0]['success_rate']==0
    changed=dict(base,init_state_id=4,code_hash='different')
    dump(run/'task00_state004/result.json',changed)
    with pytest.raises(ValueError):report.summarize(run)
