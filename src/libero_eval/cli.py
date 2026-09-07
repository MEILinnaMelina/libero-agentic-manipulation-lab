import argparse
import traceback
from .bootstrap import ROOT
from .io import dump, read

def smoke(manifest, config, ids):
    import imageio.v2 as imageio
    import numpy as np
    from .env import LiberoEnvAdapter, LiberoActionAdapter
    results = []
    for task in manifest['tasks']:
        if task['task_id'] not in ids:
            continue
        env = None
        result = {'task_id':task['task_id'],'name':task['name'],'ok':False}
        directory = ROOT/'runs/smoke'/str(task['task_id'])
        directory.mkdir(parents=True,exist_ok=True)
        try:
            env = LiberoEnvAdapter(task,config)
            env.reset(0,config['seed'])
            ctrl = LiberoActionAdapter(env)
            for _ in range(3):
                env.step(np.zeros(7))
            for camera,frame in env.render().items():
                imageio.imwrite(str(directory/(camera+'.png')),frame)
            c=ctrl.controller
            result.update(ok=True, action_spec=[v.tolist() for v in env.raw.action_spec], output_min=c.output_min.tolist(), output_max=c.output_max.tolist(),input_min=c.input_min.tolist(), input_max=c.input_max.tolist(), eef_pos=env.obs['robot0_eef_pos'].tolist(), eef_quat_xyzw=env.obs['robot0_eef_quat'].tolist(),success=env.success(),env_steps=env.steps,settle_steps=env.settle_steps,nstack=int(env.raw.sim.model._model.nstack),maxuse_stack=int(env.raw.sim.data._data.maxuse_stack))
        except Exception:
            result['error']=traceback.format_exc()
        finally:
            if env:
                env.close()
        results.append(result)
        dump(directory/'result.json',result)
        print(result,flush=True)
    dump(ROOT/'runs/smoke/results.json',results)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['manifest','smoke','run','summarize','control','freeze'])
    p.add_argument('--tasks',default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--states',default=None)
    p.add_argument('--split',choices=['dev','formal'],default='dev')
    p.add_argument('--method',choices=['fixed','gpt6','gpt6_no_replan','noop'],default='gpt6')
    p.add_argument('--run-id',default='dev-gpt6-v1')
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    config=read(ROOT/'configs/protocol.json')
    if args.command=='manifest':
        from .manifest import build
        manifest=build(config)
        print([(t['task_id'],t['available_init_states'],len(t['formal_state_ids'])) for t in manifest['tasks']])
        return
    manifest=read(ROOT/'configs/task_manifest.json')
    if args.command=='smoke':
        smoke(manifest,config,[int(x) for x in args.tasks.split(',')])
    elif args.command in ['run','control','freeze']:
        from .runner import execute
        execute(args,manifest,config)
    else:
        from .report import summarize
        summarize(ROOT/'runs'/args.run_id)
