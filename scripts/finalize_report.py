"""Generate bilingual findings from real results only; validate completeness first."""
import argparse
from pathlib import Path
import sys
import json
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from libero_eval.io import read,dump

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',default='frozen-v1')
    args=parser.parse_args()
    main_summary=read(ROOT/'reports'/(args.prefix+'-gpt6')/'summary.json')
    comparison=read(ROOT/'reports'/args.prefix/'paired_comparison.json')
    if not main_summary['complete'] or len(main_summary['per_task'])!=10:
        raise RuntimeError('Formal 10-task results are incomplete')
    records=read(ROOT/'reports'/(args.prefix+'-gpt6')/'raw_results.json')
    examples={}
    diagnoses=Counter()
    def contains_collision(value):
        if isinstance(value,dict):
            return bool(value.get('collisions')) or any(contains_collision(v) for v in value.values())
        if isinstance(value,list):return any(contains_collision(v) for v in value)
        return False
    for r in records:
        for code in (['success'] if r['success'] else r['skill_failures'] or [r['termination_reason']]):
            if code not in examples:
                examples[code]={k:r[k] for k in ['task_id','init_state_id','seed','termination_reason','video_path']}
        episode_dir=ROOT/'runs'/(args.prefix+'-gpt6')/('task%02d_state%03d'%(r['task_id'],r['init_state_id']))
        skill_file=episode_dir/'skills.jsonl'
        if skill_file.exists():
            for text in skill_file.read_text(encoding='utf-8').splitlines():
                skill=json.loads(text)
                if skill.get('success'):continue
                code=skill.get('failure_code','unknown')
                if code=='ik_or_collision':code='collision_candidate_rejection' if contains_collision(skill.get('diagnostics',{})) else 'ik_unreachable'
                diagnoses[code]+=1
                if code not in examples:
                    examples[code]={**{k:r[k] for k in ['task_id','init_state_id','seed','termination_reason','video_path']},'skill_request':skill['request'],'diagnostics_file':str(skill_file)}
        if r['termination_reason'] in ['api_error','simulation_error','planner_finish']:
            diagnoses[r['termination_reason']]+=1
    dump(ROOT/'reports'/args.prefix/'examples.json',examples)
    dump(ROOT/'reports'/args.prefix/'failure_diagnoses.json',{'counts':dict(diagnoses),'interpretation':{'collision_candidate_rejection':'Private-state planner rejected contact; not necessarily an executed physical collision.','ik_unreachable':'No candidate met IK pose tolerance, with no collision rejection recorded.','control_error':'Executed motion failed the end-effector target tolerance.','grasp_miss':'No verified two-finger object contact.','grasp_slip':'Held-object precondition or contact was lost.','placement_unstable':'Official placement predicate false after release.','mechanism_contact':'Executed joint manipulation did not reach official mechanism predicate.','precondition':'Invalid symbolic state/object/mechanism affordance.','skill_missing':'Unsupported semantic skill.','planner_finish':'Planner stopped with unsatisfied official goal; semantic omission versus exhausted recovery requires inspection of request rationale.','api_error':'Provider/timeout/format/model/budget error.','simulation_error':'Environment or numerical/capacity error.'}})
    output=ROOT/'reports'/args.prefix/'FINAL_REPORT.md'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks=main_summary['per_task']
    rates=[t['success_rate'] for t in tasks]
    errors=[[max(0,r-t['wilson95'][0]) for r,t in zip(rates,tasks)],[max(0,t['wilson95'][1]-r) for r,t in zip(rates,tasks)]]
    fig,ax=plt.subplots(figsize=(9,4.6))
    ax.bar(range(10),rates,color='#226b91',yerr=errors,capsize=4)
    ax.set(xlabel='LIBERO-10 task ID',ylabel='Official success rate',ylim=(0,1.08),xticks=range(10),title='GPT-6 Astra + state-assisted Agentic v2 | 47 held-out states per task')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.spines[['top','right']].set_visible(False)
    for i,t in enumerate(tasks):ax.text(i,max(rates[i]+.02,.03),'%d/47'%t['successes'],ha='center',fontsize=9)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout()
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(output.parent/'per_task_success.png'),dpi=180)
    fig.savefig(str(output.parent/'per_task_success.svg'))
    plt.close(fig)
    lines=['# LIBERO-10 + GPT-6：正式评测 / Formal evaluation','',
           '本次评测已完成全部 10 个任务、每任务 47 个未用于调参的官方初始状态，共 470 episodes。',
           '条件：模拟器状态与 BDDL 目标辅助；单臂 Panda，OSC_POSE，20 Hz；5 步零动作稳定操作后最多 600 步；模型为 gpt-6-astra；无跨 episode 记忆。',
           '', 'This evaluation uses simulator state and BDDL goals, a single Panda arm, OSC_POSE at 20 Hz, five zero settling actions followed by at most 600 policy actions, and gpt-6-astra with no cross-episode memory. All 470 held-out episodes are included, including failures.', '',
           '宏平均成功率 / Macro success rate: **%.2f%%**.'%(100*main_summary['macro_success_rate']),
           '分层 bootstrap 95%% 区间 / Stratified bootstrap 95%% interval: **%.2f%%–%.2f%%**.'%tuple(100*x for x in main_summary['macro_bootstrap95']),
           '总体 Wilson 95%% 区间 / Pooled Wilson 95%% interval: **%.2f%%–%.2f%%**.'%tuple(100*x for x in main_summary['micro_wilson95']),
           '', '| ID | Success / n | Rate | 95% Wilson CI |','|---|---|---|---|']
    for t in main_summary['per_task']:
        lines.append('| %d | %d / %d | %.2f%% | %.2f%%–%.2f%% |'%(t['task_id'],t['successes'],t['scheduled'],100*t['success_rate'],100*t['wilson95'][0],100*t['wilson95'][1]))
    lines+=['', '![Per-task results](per_task_success.png)', '', '配对诊断 / Paired diagnostics: each alternative uses official indices 3–7 for every task; comparisons restrict GPT-6 main results to those identical 50 states.']
    for c in comparison:
        lines.append('- %s: GPT-6 %d/50, alternative %d/50; paired difference %.2f percentage points, bootstrap 95%% CI [%.2f, %.2f].'%(c['alternative'],c['gpt6_successes'],c['alternative_successes'],100*c['paired_macro_difference'],100*c['paired_bootstrap95'][0],100*c['paired_bootstrap95'][1]))
    lines+=['', '平均模型调用 / Mean model calls: %.2f. 平均重规划 / Mean replans: %.2f.'%(main_summary['mean_llm_calls'],main_summary['mean_replans']), 'Tokens: input %d; output %d.'%(main_summary['total_input_tokens'],main_summary['total_output_tokens']), 'Usage-priced cost estimate: '+str(main_summary.get('estimated_cost_usd'))+' USD (not an invoice).', '', '技能失败分类 / Skill failure taxonomy: '+json.dumps(main_summary['failure_counts'],ensure_ascii=False), '',
            '可复现例子及视频 / Reproducible examples and videos: [examples.json](examples.json). Detailed failure categories distinguish candidate collision rejection from unreachable IK: [failure_diagnoses.json](failure_diagnoses.json). Original results, request/response logs, scene states, actions and MP4 files remain under runs/.', '',
            '结论限定：成功只取自官方 check_success。开发成功覆盖不能代表稳定解决；此表给出正式状态上的实际完成水平。负面结果保留在分母中。本实验不是 RGB-only VLA 同条件比较，也不是终身学习训练/遗忘协议复现。',
            'Interpretation: only official check_success determines success. Development coverage does not establish reliability. Failures remain in the denominator. State/goal privilege prevents direct comparison to RGB-only VLA policies; no lifelong-learning training or forgetting protocol was reproduced.', '',
            '方法、偏离、版本与协议详见 ../../docs/METHOD.md、../../docs/CHANGELOG.md、../../configs/frozen.json。']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(output)

if __name__=='__main__':main()
