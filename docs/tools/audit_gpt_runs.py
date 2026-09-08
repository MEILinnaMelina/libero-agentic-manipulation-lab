# -*- coding: utf-8 -*-
"""Offline audit of a completed or stopped GPT run; no API calls or simulation."""
import argparse
import json
import sys
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from libero_eval.budget import actual_cost

NAMES=['汤罐和番茄酱放篮子','奶酪和黄油放篮子','打开炉灶并放摩卡壶','碗放抽屉并关闭','两个杯子放左右盘子','书放入收纳盒后槽','杯子放盘上、布丁放右侧','汤罐和奶酪放篮子','两个摩卡壶放炉灶','杯子放微波炉并关门']

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def save(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_id');args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id:raise ValueError('Invalid run ID')
    run=ROOT/'runs'/args.run_id
    identity=read(run/'run.json');status=read(run/'check_status.json')
    out=ROOT/'reports'/run.name;out.mkdir(parents=True,exist_ok=True)
    schedule=identity['schedule'];prices=identity['config']['pricing_usd_per_million']
    rows=[];videos=[];thumbs={};usage_totals=dict(input_tokens=0,output_tokens=0,cached_input_tokens=0,cache_write_tokens=0,ordinary_input_tokens=0)
    for planned in schedule:
        t,s=planned['task_id'],planned['init_state_id']
        ep=run/('task%02d_state%03d'%(t,s))
        if not (ep/'result.json').exists():continue
        r=read(ep/'result.json')
        assert all(r[k]==identity[k] for k in ['method','split','code_hash','config_hash','manifest_hash'])
        assert r['method']=='gpt6' and r['env_steps']<=identity['config']['max_env_steps']
        metas=[read(p) for p in sorted(ep.glob('llm_*.metadata.json'))]
        assert len(metas)==r['llm_calls']
        assert r['task_id']==t and r['init_state_id']==s
        returned={m['returned_model'] for m in metas if m.get('returned_model')}
        assert all(m=='gpt-6-astra' or m.startswith('gpt-6-astra-') for m in returned)
        r['estimated_usd']=sum(actual_cost(m.get('usage',{}),prices) for m in metas)
        r['unknown_usage_requests']=sum(not m.get('usage') for m in metas)
        for m in metas:
            u=m.get('usage',{});d=u.get('input_tokens_details') or {}
            i=u.get('input_tokens',0);c=d.get('cached_tokens',0);w=d.get('cache_write_tokens',0)
            for k,v in dict(input_tokens=i,output_tokens=u.get('output_tokens',0),cached_input_tokens=c,cache_write_tokens=w,ordinary_input_tokens=max(0,i-c-w)).items():usage_totals[k]+=v
        rows.append(r)
        if not (ep/'video.mp4').exists():
            videos.append(dict(task_id=t,state_id=s,decoded=False,reason='video missing'));continue
        assert not r.get('artifact_error')
        scene=read(ep/'final_scene.json')
        assert bool(scene['official_success'])==r['success']
        if r['success']:assert all(g['satisfied'] for g in scene['goal_status'])
        trace=[json.loads(line) for line in (ep/'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(trace)==r['env_steps']+r['settle_steps']
        reader=imageio.get_reader(str(ep/'video.mp4'))
        count=reader.count_frames();meta=reader.get_meta_data();last=reader.get_data(count-1);reader.close()
        assert count==len(trace) and last.shape[:2]==(256,512) and meta['fps']==20
        if r['success']:assert trace[-1]['success']
        videos.append(dict(task_id=t,state_id=s,frames=count,fps=meta['fps'],decoded=True,official_success=r['success'],returned_models=sorted(returned)))
        im=Image.fromarray(last);draw=ImageDraw.Draw(im);draw.rectangle((0,0,512,20),fill='black')
        draw.text((6,5),'Task %d / state %d | %s | %d steps'%(t,s,'SUCCESS' if r['success'] else r['termination_reason'],r['env_steps']),fill='white')
        thumbs.setdefault(s,[]).append(im)
    tasks=sorted({x['task_id'] for x in schedule});states=sorted({x['init_state_id'] for x in schedule})
    cost=sum(r['estimated_usd'] for r in rows)
    assert len(rows)==status['completed']
    assert sum(r['success'] for r in rows)==status['successes']
    assert sum(r['llm_calls'] for r in rows)==status['llm_calls']
    observed=sum(item['amount_usd'] for item in status['ledger'] if item['state']=='observed')
    assert abs(cost-observed)<1e-8
    per_task=[dict(task_id=t,scheduled=sum(x['task_id']==t for x in schedule),completed=sum(r['task_id']==t for r in rows),successes=sum(r['success'] for r in rows if r['task_id']==t),llm_calls=sum(r['llm_calls'] for r in rows if r['task_id']==t),estimated_usd=sum(r['estimated_usd'] for r in rows if r['task_id']==t)) for t in tasks]
    audit=dict(run_id=run.name,code_hash=identity['code_hash'],formal_benchmark=False,scheduled=len(schedule),completed=len(rows),successes=sum(r['success'] for r in rows),llm_calls=sum(r['llm_calls'] for r in rows),estimated_usd=cost,unknown_usage_requests=sum(r['unknown_usage_requests'] for r in rows),api_errors=sum(r['termination_reason']=='api_error' for r in rows),usage=usage_totals,episode_wall_seconds=sum(r['wall_seconds'] for r in rows),per_task=per_task,video_audit=videos,cost_note='Usage-priced estimate, not invoice; unknown usage is not zero cost.')
    save(out/'audit.json',audit)
    for s,ims in thumbs.items():
        sheet=Image.new('RGB',(1024,256*((len(ims)+1)//2)),(235,235,235))
        for i,im in enumerate(ims):sheet.paste(im,((i%2)*512,(i//2)*256))
        sheet.save(str(out/('final_frames_state%03d.jpg'%s)),quality=90)
    lines=['# GPT-6 多状态运行结果','',f'运行 `{run.name}`。任务 {", ".join(map(str,tasks))}；官方初始状态 {", ".join(map(str,states))}，每个预定任务与状态组合执行一次。','',f'完成 **{len(rows)}/{len(schedule)}** 回合，成功 **{audit["successes"]}**。这是一批限定状态的检查，不是完整 470 回合正式评测，不与旧版本结果混合。','',f'共 **{audit["llm_calls"]}** 次 GPT-6 Astra 调用，费用按已记录用量估算 **${cost:.4f}**，不是账单。API 错误 {audit["api_errors"]} 回合，未知用量请求 {audit["unknown_usage_requests"]} 条。','', '| 任务 | '+' | '.join('状态 '+str(s) for s in states)+' | 成功次数 | API 次数 | 估计美元 |','|---|'+'---|'*len(states)+'---:|---:|---:|']
    for summary in per_task:
        t=summary['task_id'];cells=[]
        for s in states:
            r=next((r for r in rows if r['task_id']==t and r['init_state_id']==s),None)
            if r is None:cells.append('未执行');continue
            video='../../runs/%s/task%02d_state%03d/video.mp4'%(run.name,t,s)
            cells.append('[%s · %d 步](%s)'%('成功' if r['success'] else '失败',r['env_steps'],video))
        lines.append('| %d %s | %s | %d/%d | %d | %.4f |'%(t,NAMES[t],' | '.join(cells),summary['successes'],summary['scheduled'],summary['llm_calls'],summary['estimated_usd']))
    lines+=['','## 失败与停止原因','']
    failures=[r for r in rows if not r['success']]
    for r in failures:
        lines.append('- 任务 %d，状态 %d：%s；技能失败：%s。'%(r['task_id'],r['init_state_id'],r['termination_reason'],', '.join(r['skill_failures']) or '无'))
    if not failures:lines.append('所有已执行回合均成功。')
    lines+=['',f'整批停止原因：{status["stop_reason"] or "计划回合全部结束"}。','', '## 协议与录像核验','',f'沿用官方成功判定、600 动作步、3 次重规划与首次成功终止。独立预算上限 ${identity["config"]["formal_budget_usd"]:g}。不改变物理、初始状态或成功条件，不在运行中调参，不重跑失败回合。','',f'已解码核验 {sum(v["decoded"] for v in videos)} 段双视角视频，512×256、20 fps；帧数与逐步轨迹一致。成功回合最终官方目标均为真。视频在首次成功时结束，不包含额外松手稳定性测试。','',f'输入 {usage_totals["input_tokens"]:,} token，输出 {usage_totals["output_tokens"]:,} token；其中缓存输入 {usage_totals["cached_input_tokens"]:,}、缓存写入 {usage_totals["cache_write_tokens"]:,}、普通输入 {usage_totals["ordinary_input_tokens"]:,}。计价单价随运行配置保存。','',f'执行代码 SHA-256：`{identity["code_hash"]}`。完整执行源码与配置见运行目录 `source.zip`；每回合保留技能、请求响应、初末场景、逐步动作和状态。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['run_id','scheduled','completed','successes','llm_calls','estimated_usd','unknown_usage_requests','api_errors','episode_wall_seconds','per_task']},ensure_ascii=True))

if __name__=='__main__':main()
