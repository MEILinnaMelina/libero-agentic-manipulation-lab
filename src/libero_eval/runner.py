import hashlib
import time
import traceback
from pathlib import Path
import numpy as np
from .bootstrap import ROOT
from .io import dump, read, append, code_hash, digest
from .env import LiberoEnvAdapter, StepBudget, OfficialSuccess
from .planner import GPTPlanner, PlannerError
from .scene import LiberoSceneAdapter
from .skills import Skills, fixed_plan

class Recorder:
    def __init__(self, directory):
        import imageio.v2 as imageio
        self.directory=directory
        directory.mkdir(parents=True,exist_ok=True)
        self.writer=imageio.get_writer(str(directory/'video.mp4'),fps=20,codec='libx264',quality=6,macro_block_size=16)
        self.actions=[]
        self.states=[]
        self.frames=0

    def step(self,env,action,done,phase):
        frames=env.render()
        self.writer.append_data(np.concatenate(list(frames.values()),axis=1))
        self.frames+=1
        self.actions.append(np.asarray(action))
        self.states.append(env.env.get_sim_state())
        append(self.directory/'trajectory.jsonl',{'phase':phase,'step':env.steps,'settle_steps':env.settle_steps,'action':action,'eef_position_m':env.obs['robot0_eef_pos'],'eef_quaternion_xyzw':env.obs['robot0_eef_quat'],'gripper_qpos':env.obs['robot0_gripper_qpos'],'success':bool(done)})

    def close(self):
        self.writer.close()
        np.savez_compressed(str(self.directory/'trajectory.npz'),actions=np.asarray(self.actions),sim_states=np.asarray(self.states))

def episode(task,state_id,config,directory,run_id,method,split,fingerprint):
    result={'run_id':run_id,'method':method,'requested_model':config['model'] if method.startswith('gpt6') else None,'returned_models':[], 'task_id':task['task_id'],'task_name':task['name'],'seed':config['seed'],'init_state_id':state_id,'init_state_sha256':task['state_sha256'][state_id], 'split':split,'success':False,'termination_reason':'not_started','env_steps':0,'settle_steps':0,'llm_calls':0,'llm_input_tokens':0,'llm_output_tokens':0,'llm_latency_seconds':0.,'replans':0,'skill_failures':[],'wall_seconds':0.,'code_hash':fingerprint,'config_hash':digest(ROOT/'configs/protocol.json'),'manifest_hash':digest(ROOT/'configs/task_manifest.json'),'video_path':str(directory/'video.mp4'),'protocol_id':config['protocol_id'],'budget_steps':config['max_env_steps'],'attempt':0}
    dump(directory/'started.json',result)
    env=recorder=planner=None
    history=[]
    start=time.monotonic()
    try:
        env=LiberoEnvAdapter(task,config)
        recorder=Recorder(directory)
        env.on_step=recorder.step
        env.reset(state_id,config['seed'])
        scene=LiberoSceneAdapter(env)
        dump(directory/'initial_scene.json',scene.snapshot())
        if method=='noop':
            for _ in range(config['max_env_steps']):
                env.step(np.zeros(7))
                if env.success(): break
            result['termination_reason']='success' if env.success() else 'step_budget'
        else:
            skills=Skills(env)
            fixed=fixed_plan(scene.goals(),scene if method=='fixed' else None)
            if method.startswith('gpt6'): planner=GPTPlanner(config,directory)
            result['termination_reason']='skill_request_budget'
            for idx in range(config['max_skill_requests']):
                if env.success():
                    result['termination_reason']='success'; break
                if history and not history[-1]['success']:
                    if method=='gpt6_no_replan':
                        result['termination_reason']='failure_replan_disabled';break
                    if method=='gpt6':
                        if result['replans']>=config['max_replans']:
                            result['termination_reason']='replan_budget';break
                        result['replans']+=1
                snapshot=scene.snapshot()
                dump(directory/('scene_%02d.json'%idx),snapshot)
                if planner:
                    decision=planner.decide({'language':task['language'],'goals':scene.goals()},snapshot,history)
                    req=decision['request']
                else:
                    req=fixed[idx] if idx<len(fixed) else {'skill':'finish','object':None,'goal':'','strategy':None}
                    decision={'request':req,'source':'fixed_bddl_diagnostic'}
                append(directory/'decisions.jsonl',decision)
                if req['skill']=='finish':
                    result['termination_reason']='planner_finish';break
                step_start=env.steps
                feedback=skills.execute(req)
                feedback.update(env_steps=env.steps-step_start,official_success=env.success())
                history.append(feedback)
                append(directory/'skills.jsonl',feedback)
                if not feedback['success']:
                    result['skill_failures'].append(feedback['failure_code'])
                    if method=='fixed' and config.get('fixed_stop_on_failure'):
                        result['termination_reason']='fixed_skill_failure'
                        break
        result['success']=env.success()
        if result['success']: result['termination_reason']='success'
        dump(directory/'final_scene.json',scene.snapshot())
    except OfficialSuccess:
        result['success']=True
        result['termination_reason']='success'
        if 'req' in locals():
            append(directory/'skills.jsonl',{'request':req,'success':True,'failure_code':None,'official_success':True,'env_steps':env.steps-step_start,'message':'Episode stopped at first official success.'})
    except StepBudget as exc:
        result['termination_reason']=str(exc)
        result['success']=env.success() if env else False
    except PlannerError as exc:
        result['termination_reason']='api_error'
        result['error']=str(exc)
    except Exception:
        result['termination_reason']='simulation_error'
        result['error']=traceback.format_exc()
    finally:
        if env:
            result.update(env_steps=env.steps,settle_steps=env.settle_steps,mujoco_stack_doubles=int(env.raw.sim.model._model.nstack),mujoco_maxuse_stack=int(env.raw.sim.data._data.maxuse_stack),mujoco_maxuse_contacts=int(env.raw.sim.data._data.maxuse_con),mujoco_maxuse_constraints=int(env.raw.sim.data._data.maxuse_efc))
            if 'scene' in locals() and not (directory/'final_scene.json').exists():
                try:dump(directory/'final_scene.json',scene.snapshot())
                except Exception:result['final_scene_error']=traceback.format_exc()
        if result['success']:result['termination_reason']='success'
        if planner:
            result.update(llm_calls=planner.calls,returned_models=sorted({m['returned_model'] for m in planner.metadata if m.get('returned_model')}),llm_input_tokens=sum(m['usage'].get('input_tokens',0) for m in planner.metadata),llm_output_tokens=sum(m['usage'].get('output_tokens',0) for m in planner.metadata),llm_latency_seconds=sum(m['latency_seconds'] for m in planner.metadata))
        if recorder:
            try: recorder.close()
            except Exception: result['artifact_error']=traceback.format_exc()
            result['video_frames']=recorder.frames
        if env:
            try: env.close()
            except Exception: result['close_error']=traceback.format_exc()
        # Robosuite owns reference cycles. Collect between episodes to release MjData/model memory.
        env=recorder=planner=None
        if 'skills' in locals(): skills=None
        if 'scene' in locals(): scene=None
        import gc
        gc.collect()
        result['wall_seconds']=time.monotonic()-start
        dump(directory/'result.json',result)
    return result

def execute(args,manifest,config):
    fingerprint=code_hash(ROOT)
    if args.command=='freeze':
        if not config['calibrated'] or config['formal_budget_usd'] is None:
            raise RuntimeError('Run pilot and calibrate cost/budget before formal freeze')
        import zipfile
        archive=ROOT/'docs/provenance/frozen_source.zip'
        with zipfile.ZipFile(str(archive),'w',compression=zipfile.ZIP_DEFLATED) as z:
            for folder in ['src','scripts','configs','tests']:
                for p in sorted((ROOT/folder).rglob('*')):
                    if p.is_file() and '__pycache__' not in p.parts and p.name!='frozen.json':
                        z.write(str(p),p.relative_to(ROOT).as_posix())
        dump(ROOT/'configs/frozen.json',{'code_hash':fingerprint,'config_hash':digest(ROOT/'configs/protocol.json'),'manifest_hash':digest(ROOT/'configs/task_manifest.json'),'source_archive_sha256':digest(archive),'source_manifest_sha256':digest(ROOT/'docs/provenance/sources.json'),'created_at':time.time()})
        return
    if args.command=='control':
        from .validation import control
        control(manifest,config)
        return
    if args.split=='formal':
        frozen=read(ROOT/'configs/frozen.json')
        if frozen['code_hash']!=fingerprint or frozen['config_hash']!=digest(ROOT/'configs/protocol.json') or frozen['manifest_hash']!=digest(ROOT/'configs/task_manifest.json'):
            raise RuntimeError('Frozen code/config/manifest changed')
    if not args.run_id or Path(args.run_id).name!=args.run_id:
        raise ValueError('run-id must be a simple directory name')
    run=ROOT/'runs'/args.run_id
    run.mkdir(parents=True,exist_ok=True)
    ids=[int(x) for x in args.tasks.split(',')]
    if len(ids)!=len(set(ids)) or any(i not in range(10) for i in ids): raise ValueError('Invalid/duplicate task IDs')
    schedule=[]
    for task in manifest['tasks']:
        if task['task_id'] not in ids: continue
        allowed=task[args.split+'_state_ids']
        states=allowed if args.states is None else [int(s) for s in args.states.split(',')]
        if len(states)!=len(set(states)) or not set(states)<=set(allowed): raise ValueError('Invalid, duplicate or split-leaking initial states')
        for state in states: schedule.append({'task_id':task['task_id'],'init_state_id':state})
    identity={'run_id':args.run_id,'method':args.method,'split':args.split,'code_hash':fingerprint,'config_hash':digest(ROOT/'configs/protocol.json'),'manifest_hash':digest(ROOT/'configs/task_manifest.json'),'schedule':schedule,'config':config}
    if (run/'run.json').exists():
        if read(run/'run.json')!=identity: raise RuntimeError('Resume identity mismatch; use a distinct run ID')
    else: dump(run/'run.json',identity)
    pending=[]
    for entry in schedule:
        directory=run/('task%02d_state%03d'%(entry['task_id'],entry['init_state_id']))
        if (directory/'result.json').exists(): continue
        if (directory/'started.json').exists():
            # No favourable rerun selection. Interrupted attempts stay in denominator.
            result=read(directory/'started.json')
            result['termination_reason']='interrupted'
            dump(directory/'result.json',result)
            continue
        pending.append((manifest['tasks'][entry['task_id']],entry['init_state_id'],config,directory,args.run_id,args.method,args.split,fingerprint))
    def show(result):
        print({k:result[k] for k in ['task_id','init_state_id','success','termination_reason','env_steps','llm_calls','wall_seconds']},flush=True)
    if args.workers==1:
        for arguments in pending: show(episode(*arguments))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        if not 1<=args.workers<=4: raise ValueError('Workers must be 1..4 with bounded MuJoCo workspace')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(episode,*arguments) for arguments in pending]
            for future in as_completed(futures): show(future.result())
    from .report import summarize
    summarize(run)
