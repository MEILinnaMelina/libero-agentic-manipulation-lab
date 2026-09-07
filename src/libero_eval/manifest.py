from .bootstrap import configure, ROOT
from .io import dump, digest

def build(config):
    paths = configure()
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()['libero_10'](task_order_index=0)
    tasks = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        bddl = paths['bddl_files']/task.problem_folder/task.bddl_file
        init = paths['init_states']/task.problem_folder/task.init_states_file
        states = suite.get_task_init_states(i)
        dev = config['dev_state_ids']
        formal = [j for j in range(len(states)) if j not in dev][:config['formal_states_per_task']]
        # Reject duplicate initial state contents as well as duplicate indices.
        import hashlib
        state_hashes = [hashlib.sha256(s.numpy().tobytes() if hasattr(s,'numpy') else s.tobytes()).hexdigest() for s in states]
        seen = {state_hashes[j] for j in dev}
        formal = []
        for j,h in enumerate(state_hashes):
            if h not in seen and len(formal)<config['formal_states_per_task']:
                formal.append(j)
                seen.add(h)
        tasks.append({'task_id':i, **task._asdict(), 'bddl_path':str(bddl.relative_to(ROOT)), 'init_path':str(init.relative_to(ROOT)), 'bddl_sha256':digest(bddl), 'init_sha256':digest(init), 'available_init_states':len(states), 'state_sha256':state_hashes, 'dev_state_ids':dev, 'formal_state_ids':formal})
    result = {'suite':'libero_10','task_order_index':0,'tasks':tasks}
    dump(ROOT/'configs/task_manifest.json', result)
    return result
